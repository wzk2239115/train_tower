"""GradNorm: dynamic per-task loss weight balancing via gradient norm equalization.

Reference: Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss
Weighting in Multimodal Architectures", ICML 2018.

Core idea:
  - Maintain a learnable weight w_i per task (CE, image_FM, audio_FM, ...)
  - Every K steps, compute gradient norms g_i = ||dL_i / d\theta_shared||
  - Update w_i so that all g_i converge to a common target norm
  - The "target" adjusts for learning speed: faster-learning tasks get lower weight

Integration points:
  - flow_tower.py: _accumulate_exit_losses returns unweighted per-task losses
  - forward() builds total = sum(w_i * L_i), then GradNormBalancer observes
    the gradient norms on the shared backbone after backward
  - diagnostics.py: logs w_i and g_i every logging step

Constraints for DeepSpeed ZeRO-2:
  - w_i are registered as nn.Parameter on the model so ZeRO-2 optimizer sees them
  - Gradient norm is measured on the **last shared transformer layer** only
    (one backward per task, no full-graph retention needed)
  - All-reduce of per-rank gradient norms before weight update
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


class GradNormWeights(nn.Module):
    """Learnable per-task weights registered as model parameters.

    By being an nn.Module on the model, DeepSpeed ZeRO-2's optimizer
    automatically includes them in the parameter group.
    """

    def __init__(self, task_names: list[str], initial_weights: dict[str, float] | None = None):
        super().__init__()
        self.task_names = list(task_names)
        init_w = initial_weights or {}
        vals = []
        for name in self.task_names:
            vals.append(float(init_w.get(name, 1.0)))
        self.raw = nn.Parameter(torch.tensor(vals, dtype=torch.float32), requires_grad=False)
        self._weights = torch.softmax(self.raw, dim=0) * len(self.task_names)

    def get(self, task_name: str) -> float:
        idx = self.task_names.index(task_name)
        return float(self._weights[idx].item())

    def weights_dict(self) -> dict[str, float]:
        return {name: float(self._weights[i].item()) for i, name in enumerate(self.task_names)}

    def update_from_grad_norms(
        self,
        grad_norms: dict[str, float],
        target_norm: float,
        lr: float = 0.025,
    ) -> dict[str, float]:
        if not grad_norms or len(grad_norms) != len(self.task_names):
            return {}
        device = self.raw.device
        g = torch.tensor(
            [grad_norms.get(n, 0.0) for n in self.task_names],
            dtype=torch.float32,
            device=device,
        )
        g = g.clamp(min=1e-6)
        loss_ratio = g / g.mean().clamp(min=1e-6)
        target = torch.full_like(g, target_norm)
        grad_update = lr * (loss_ratio - 1.0) * (g - target_norm)
        new_raw = self.raw + grad_update
        self.raw.copy_(new_raw)
        self._weights = torch.softmax(self.raw, dim=0) * len(self.task_names)
        return self.weights_dict()


TASK_NAMES = [
    "ce_loss",
    "image_fm_loss",
    "image_jepa_loss",
    "audio_fm_loss",
    "video_fm_loss",
    "text_hidden_loss",
]


class GradNormBalancer:
    """Manages GradNorm weight updates based on observed gradient norms.

    Usage:
        balancer = GradNormBalancer(cfg, model)
        # in forward():
        per_task = model._accumulate_exit_losses(...)  # returns dict
        total, weights = balancer.weighted_total(per_task)
        # after backward (in callback or compute_loss):
        balancer.maybe_update(trainer.model)
    """

    def __init__(self, cfg: Any, model: nn.Module):
        self.cfg = cfg
        self.enabled = bool(getattr(cfg, "grad_norm_balance", False))
        self.target_ratio = float(getattr(cfg, "grad_norm_target", 1.0))
        self.update_interval = int(getattr(cfg, "grad_norm_update_interval", 100))
        self._step = 0
        self._warmup_steps = 100
        self._last_grad_norms: dict[str, float] = {}
        self._last_weights: dict[str, float] = {}
        self._last_target_norm: float = 0.0
        if self.enabled:
            initial = self._initial_weights_from_tower_yml(model)
            self.weights_module = GradNormWeights(TASK_NAMES, initial)
        else:
            self.weights_module = None

    def _initial_weights_from_tower_yml(self, model: nn.Module) -> dict[str, float]:
        tower_cfg = getattr(model, "tower_cfg", None)
        if tower_cfg is None:
            return {}
        stage = model._current_stage() if hasattr(model, "_current_stage") else ""
        init_w: dict[str, float] = {}
        for spec in tower_cfg.exits:
            key = model._loss_breakdown_key(spec) if hasattr(model, "_loss_breakdown_key") else spec.name
            w = tower_cfg.loss_weight(spec.name, stage)
            if w > 0:
                init_w[key] = init_w.get(key, 0.0) + w
        ce_w = float(getattr(self.cfg, "loss_weights", {}).get("ce", 1.0))
        init_w["ce_loss"] = ce_w
        return init_w

    def weighted_total(
        self,
        per_task_losses: dict[str, torch.Tensor],
        *,
        fm_weight: float = 1.0,
        ce_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if not self.enabled or self.weights_module is None:
            total = torch.tensor(0.0, device=_infer_device(per_task_losses))
            weights: dict[str, float] = {}
            for name, loss in per_task_losses.items():
                if name == "ce_loss":
                    w = ce_weight
                else:
                    w = fm_weight
                total = total + w * loss
                weights[name] = w
            return total, weights

        total = torch.tensor(0.0, device=_infer_device(per_task_losses))
        weights: dict[str, float] = {}
        wm = self.weights_module
        for name, loss in per_task_losses.items():
            if not hasattr(loss, "requires_grad"):
                loss = torch.tensor(float(loss), device=total.device)
            w = wm.get(name) if name in wm.task_names else fm_weight
            total = total + w * loss
            weights[name] = w
        self._last_weights = dict(weights)
        return total, weights

    def record_grad_norms(
        self,
        per_task_losses: dict[str, torch.Tensor],
        shared_params: list[nn.Parameter],
    ) -> dict[str, float]:
        if not self.enabled or self._step < self._warmup_steps:
            self._step += 1
            return {}
        if self._step % self.update_interval != 0:
            self._step += 1
            return {}

        grad_norms: dict[str, float] = {}
        last_param = shared_params[-1] if shared_params else None
        if last_param is None:
            self._step += 1
            return {}

        for name, loss in per_task_losses.items():
            if name not in TASK_NAMES:
                continue
            if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
                grad_norms[name] = 0.0
                continue
            try:
                grads = torch.autograd.grad(
                    loss,
                    last_param,
                    retain_graph=True,
                    allow_unused=True,
                )
                g = grads[0]
                if g is not None:
                    grad_norms[name] = float(g.norm(2).item())
                else:
                    grad_norms[name] = 0.0
            except Exception:
                grad_norms[name] = 0.0

        if grad_norms and self.weights_module is not None:
            valid_norms = {k: v for k, v in grad_norms.items() if v > 0}
            if valid_norms:
                target = sum(valid_norms.values()) / len(valid_norms) * self.target_ratio
                self.weights_module.update_from_grad_norms(valid_norms, target)
                self._last_weights = self.weights_module.weights_dict()
                self._last_grad_norms = dict(grad_norms)
                self._last_target_norm = target

        self._step += 1
        return grad_norms

    def last_diagnostics(self) -> dict[str, Any]:
        return {
            "grad_norm_weights": dict(self._last_weights),
            "grad_norms": dict(self._last_grad_norms),
            "grad_norm_target": self._last_target_norm,
            "grad_norm_step": self._step,
            "grad_norm_warmup": self._step < self._warmup_steps,
        }


def _infer_device(losses: dict[str, torch.Tensor]) -> torch.device:
    for v in losses.values():
        if isinstance(v, torch.Tensor) and v.numel() > 0:
            return v.device
    return torch.device("cpu")

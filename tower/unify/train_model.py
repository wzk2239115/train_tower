from __future__ import annotations

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from tower.train.config import TrainConfig
from tower.train.losses import compute_resolution_noise_scale, rectified_flow_velocity_loss, sample_flow_batch


def _build_indexes_with_hw(model, input_ids, indexes, image_grid_hw):
    """Augment NEO temporal indexes with spatial h/w for image context tokens."""
    from sensenova_u1.models.neo_unify.modeling_neo_vit import build_abs_positions_from_grid_hw

    if indexes.ndim == 1:
        indexes = indexes.unsqueeze(0)
    if indexes.shape[0] == 1:
        t_idx = indexes[0]
        pos_h = torch.zeros_like(t_idx)
        pos_w = torch.zeros_like(t_idx)
        indexes = torch.stack([t_idx, pos_h, pos_w], dim=0)
    else:
        t_idx, pos_h, pos_w = indexes[0], indexes[1], indexes[2]

    if image_grid_hw and len(image_grid_hw) > 0:
        grid_hw = image_grid_hw[0]
        if not isinstance(grid_hw, torch.Tensor):
            grid_hw = torch.tensor(grid_hw, device=input_ids.device)
        merge = int(1 / model.downsample_ratio)
        abs_pos_w, abs_pos_h = build_abs_positions_from_grid_hw(
            grid_hw // merge, device=input_ids.device
        )
        selected = input_ids[0] == model.img_context_token_id
        if selected.any():
            pos_h = pos_h.clone()
            pos_w = pos_w.clone()
            pos_h[selected] = abs_pos_h.to(dtype=pos_h.dtype)
            pos_w[selected] = abs_pos_w.to(dtype=pos_w.dtype)
        indexes = torch.stack([t_idx, pos_h, pos_w], dim=0)
    return indexes


def _weighted_ce(logits, labels, loss_weight, vocab_size):
    shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
    shift_labels = labels[..., 1:].contiguous().view(-1)
    if loss_weight is not None:
        if isinstance(loss_weight, list):
            loss_weight = loss_weight[0]
        if not isinstance(loss_weight, torch.Tensor):
            loss_weight = torch.tensor(loss_weight, device=shift_labels.device, dtype=torch.float32)
        else:
            loss_weight = loss_weight.to(device=shift_labels.device, dtype=torch.float32)
        if loss_weight.dim() > 1:
            loss_weight = loss_weight.view(-1)
        if loss_weight.numel() == labels.numel():
            loss_weight = loss_weight[1:].contiguous()
    else:
        loss_weight = (shift_labels != -100).float()

    per_token = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="none")
    denom = loss_weight.sum().clamp_min(1e-5)
    return (per_token * loss_weight).sum() / denom


class SenseNovaTrainModel(nn.Module):
    """Training wrapper for SenseNova MoT NEOChatModel (CE + FM)."""

    def __init__(self, model, cfg: TrainConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self._patch_fm_head_for_scale()

    def _patch_fm_head_for_scale(self) -> None:
        hidden = self.model.config.llm_config.hidden_size
        if hidden == 4096 or self.model.config.fm_head_layers > 2:
            return
        patch = self.model.patch_size
        merge = int(1 / self.model.downsample_ratio)
        out_dim = 3 * (patch * merge) ** 2
        self.model.fm_modules["fm_head"] = nn.Sequential(
            nn.Linear(hidden, hidden * 2, bias=True),
            nn.GELU(),
            nn.Linear(hidden * 2, out_dim, bias=True),
        )

    @property
    def config(self):
        c = self.model.config
        if getattr(c, "hidden_size", None) is None and hasattr(c, "llm_config"):
            c.hidden_size = c.llm_config.hidden_size
        return c

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(**kwargs)
        elif hasattr(self.model.language_model, "gradient_checkpointing_enable"):
            self.model.language_model.gradient_checkpointing_enable(**kwargs)

    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def save_pretrained(self, output_dir: str, **kwargs):
        kwargs.setdefault("safe_serialization", False)
        self.model.save_pretrained(output_dir, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.model.load_state_dict(state_dict, strict=strict, assign=assign)

    def _parse_grid_hw(self, grid_hw, *, num_patches: int) -> tuple[int, int]:
        if grid_hw is not None:
            if isinstance(grid_hw, torch.Tensor):
                if grid_hw.dim() == 2:
                    return int(grid_hw[0, 0].item()), int(grid_hw[0, 1].item())
                return int(grid_hw[0].item()), int(grid_hw[1].item())
            return int(grid_hw[0][0]), int(grid_hw[0][1])
        merge = int(1 / self.model.downsample_ratio)
        side = max(1, int(round(math.sqrt(num_patches))))
        return side * merge, side * merge

    def _fm_noise_scale(self, grid_h: int, grid_w: int) -> float:
        cfg = self.model.config
        merge = int(1 / self.model.downsample_ratio)
        return compute_resolution_noise_scale(
            grid_h,
            grid_w,
            merge_size=merge,
            noise_scale=float(getattr(cfg, "noise_scale", 1.0)),
            noise_scale_mode=str(getattr(cfg, "noise_scale_mode", "resolution")),
            base_image_seq_len=int(getattr(cfg, "noise_scale_base_image_seq_len", 64)),
            max_value=float(getattr(cfg, "noise_scale_max_value", 8.0)),
        )

    def _should_cfg_label_drop(self, batch) -> bool:
        prob = self.cfg.cfg_label_drop_prob
        if prob <= 0:
            return False
        is_gen = batch.get("is_gen")
        if is_gen is None:
            return False
        if isinstance(is_gen, (list, tuple)):
            if not any(is_gen):
                return False
        elif not is_gen:
            return False
        return random.random() < prob

    def _apply_cfg_label_drop(self, hidden, selected: torch.Tensor) -> torch.Tensor:
        """Zero caption/text embeddings for classifier-free guidance training."""
        hidden = hidden.clone()
        cond_mask = ~selected
        if cond_mask.any():
            hidden[0, cond_mask] = 0
        return hidden

    def _fm_time_schedule(self) -> str:
        schedule = str(getattr(self.model.config, "time_schedule", "logit_normal"))
        if schedule == "standard":
            return "logit_normal"
        return schedule

    def _inject_vision(self, input_ids, hidden_states, pixel_values, image_grid_hw, *, gen=False):
        if pixel_values is None or len(pixel_values) == 0 or pixel_values[0] is None:
            grid_size = int(1 / self.model.downsample_ratio)
            pixel_values_flat = torch.rand(
                grid_size**2,
                3 * self.model.patch_size * self.model.patch_size,
                device=self.device,
                dtype=self.dtype,
            )
            grid_hw = torch.tensor([[grid_size, grid_size]], device=self.device)
        else:
            pixel_values_flat = pixel_values[0].to(device=self.device, dtype=self.dtype)
            grid_hw = image_grid_hw[0]
            if not isinstance(grid_hw, torch.Tensor):
                grid_hw = torch.tensor(grid_hw, device=self.device)

        if gen:
            vit_embeds = self.model.extract_feature(pixel_values_flat, gen_model=True, grid_hw=grid_hw)
        else:
            vit_embeds = self.model.extract_feature(pixel_values_flat, gen_model=False, grid_hw=grid_hw)

        selected = input_ids[0] == self.model.img_context_token_id
        vit_embeds = vit_embeds.reshape(-1, vit_embeds.shape[-1])
        hidden_states = hidden_states.clone()
        n = min(selected.sum().item(), vit_embeds.shape[0])
        if n > 0:
            hidden_states[0, selected][ :n] = vit_embeds[:n]
        return hidden_states

    def _ce_forward(self, batch):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        indexes = _build_indexes_with_hw(
            self.model, input_ids, batch["indexes"], batch.get("image_grid_hw")
        )
        hidden = self.model.language_model.get_input_embeddings()(input_ids)
        hidden = self._inject_vision(
            input_ids, hidden, batch.get("pixel_values"), batch.get("image_grid_hw"), gen=False
        )

        from sensenova_u1.models.neo_unify.modeling_qwen3 import create_block_causal_mask

        attn = {"full_attention": create_block_causal_mask(indexes[0])}
        indicators = batch.get("image_gen_indicators")
        if indicators is None:
            indicators = torch.zeros(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)

        outputs = self.model.language_model(
            inputs_embeds=hidden,
            indexes=indexes,
            attention_mask=attn,
            image_gen_indicators=indicators.unsqueeze(0),
        )
        return _weighted_ce(
            outputs.logits,
            labels,
            batch.get("loss_weight"),
            self.model.language_model.config.vocab_size,
        )

    def _fm_forward(self, batch):
        input_ids = batch["input_ids"]
        pixel_values = batch.get("pixel_values")
        if pixel_values is None or len(pixel_values) == 0 or pixel_values[0] is None:
            return torch.tensor(0.0, device=self.device)

        clean = pixel_values[0].to(device=self.device, dtype=self.dtype)
        grid_hw_raw = batch.get("image_grid_hw", [None])[0]
        grid_h, grid_w = self._parse_grid_hw(grid_hw_raw, num_patches=clean.shape[0])
        noise_scale = self._fm_noise_scale(grid_h, grid_w)

        model_cfg = self.model.config
        z, t, _ = sample_flow_batch(
            clean.unsqueeze(0),
            t_eps=float(getattr(model_cfg, "t_eps", 0.05)),
            p_mean=float(getattr(model_cfg, "P_mean", -0.8)),
            p_std=float(getattr(model_cfg, "P_std", 0.8)),
            time_schedule=self._fm_time_schedule(),
            noise_scale=noise_scale,
        )
        z = z.squeeze(0)
        t_scalar = t.squeeze(0)

        if grid_hw_raw is None:
            merge = int(1 / self.model.downsample_ratio)
            grid_size = max(grid_h // merge, 1)
            grid_hw = torch.tensor([[grid_size * merge, grid_size * merge]], device=self.device)
        elif not isinstance(grid_hw_raw, torch.Tensor):
            grid_hw = torch.tensor(grid_hw_raw, device=self.device)
        else:
            grid_hw = grid_hw_raw.to(device=self.device)

        # Noisy patch embedding through gen vision tower
        vit_noisy = self.model.extract_feature(z, gen_model=True, grid_hw=grid_hw)
        seq_len = input_ids.shape[1]
        hidden = self.model.language_model.get_input_embeddings()(input_ids)
        hidden = hidden.clone()
        selected = input_ids[0] == self.model.img_context_token_id
        vit_noisy = vit_noisy.reshape(-1, vit_noisy.shape[-1])
        n = min(selected.sum().item(), vit_noisy.shape[0])
        if n > 0:
            hidden[0, selected][:n] = vit_noisy[:n]

        if n > 0:
            t_value = float(t_scalar.mean().item()) if t_scalar.numel() else float(t_scalar.item())
            t_expanded = torch.full((n,), t_value, device=self.device, dtype=self.dtype)
            timestep_embeddings = self.model.fm_modules["timestep_embedder"](t_expanded)
            hidden[0, selected][:n] = hidden[0, selected][:n] + timestep_embeddings.to(hidden.dtype)

            if getattr(model_cfg, "add_noise_scale_embedding", False) and "noise_scale_embedder" in self.model.fm_modules:
                max_value = float(getattr(model_cfg, "noise_scale_max_value", 8.0))
                noise_scale_tensor = torch.full(
                    (n,),
                    noise_scale / max_value,
                    device=self.device,
                    dtype=self.dtype,
                )
                noise_embeddings = self.model.fm_modules["noise_scale_embedder"](noise_scale_tensor)
                hidden[0, selected][:n] = hidden[0, selected][:n] + noise_embeddings.to(hidden.dtype)

        if self._should_cfg_label_drop(batch):
            hidden = self._apply_cfg_label_drop(hidden, selected)

        indexes = _build_indexes_with_hw(self.model, input_ids, batch["indexes"], batch.get("image_grid_hw"))
        from sensenova_u1.models.neo_unify.modeling_qwen3 import create_block_causal_mask

        attn = {"full_attention": create_block_causal_mask(indexes[0])}
        indicators = torch.ones(seq_len, dtype=torch.bool, device=self.device)

        out = self.model.language_model.model(
            inputs_embeds=hidden,
            indexes=indexes,
            attention_mask=attn,
            image_gen_indicators=indicators.unsqueeze(0),
        )
        h = out.last_hidden_state[0, selected][:n]
        x_pred = self.model.fm_modules["fm_head"](h)
        target = clean[:n]
        z_part = z[:n]
        t_part = t_scalar[: min(n, t_scalar.shape[0])]
        if t_part.numel() == 1 and n > 1:
            t_part = t_part.expand(n)
        return rectified_flow_velocity_loss(
            x_pred, z_part, t_part, target, t_eps=float(getattr(model_cfg, "t_eps", 0.05))
        )

    def forward(self, **batch):
        ce = torch.tensor(0.0, device=self.device)
        fm = torch.tensor(0.0, device=self.device)

        if self.cfg.ce_weight > 0:
            ce = self._ce_forward(batch)
        if self.cfg.fm_weight > 0:
            fm = self._fm_forward(batch)

        total = self.cfg.ce_weight * ce + self.cfg.fm_weight * fm
        return CausalLMOutputWithPast(loss=total)

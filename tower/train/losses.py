from __future__ import annotations

import torch
import torch.nn.functional as F


def rectified_flow_velocity_loss(
    x_pred: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
    x_clean: torch.Tensor,
    *,
    t_eps: float = 0.02,
) -> torch.Tensor:
    """MSE between predicted and target velocity (SenseNova rectified flow)."""
    denom = (1.0 - t).clamp_min(t_eps)
    while denom.ndim < z.ndim:
        denom = denom.unsqueeze(-1)
    v_pred = (x_pred - z) / denom
    v_target = (x_clean - z) / denom
    return F.mse_loss(v_pred, v_target)


def sample_flow_batch(clean: torch.Tensor, t_eps: float = 0.02) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(clean)
    t = torch.rand(clean.shape[0], device=clean.device, dtype=clean.dtype).clamp(t_eps, 1.0 - t_eps)
    view = t.view(-1, *([1] * (clean.ndim - 1)))
    z = (1.0 - view) * noise + view * clean
    return z, t, noise

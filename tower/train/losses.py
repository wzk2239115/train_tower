from __future__ import annotations

import math

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


def sample_logit_normal_timesteps(
    batch_size: int,
    *,
    p_mean: float = -0.8,
    p_std: float = 0.8,
    t_eps: float = 0.05,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    z = torch.randn(batch_size, device=device, dtype=dtype) * p_std + p_mean
    return torch.sigmoid(z).clamp(t_eps, 1.0 - t_eps)


def compute_resolution_noise_scale(
    grid_h: int,
    grid_w: int,
    *,
    merge_size: int,
    noise_scale: float = 1.0,
    noise_scale_mode: str = "resolution",
    base_image_seq_len: int = 64,
    max_value: float = 8.0,
) -> float:
    """Match SenseNova inference noise scaling for multi-resolution images."""
    if noise_scale_mode not in ("resolution", "dynamic", "dynamic_sqrt"):
        return float(noise_scale)
    num_tokens = (grid_h * grid_w) / (merge_size**2)
    scale = math.sqrt(num_tokens / base_image_seq_len) * float(noise_scale)
    if noise_scale_mode == "dynamic_sqrt":
        scale = math.sqrt(scale)
    return min(scale, max_value)


def sample_flow_batch(
    clean: torch.Tensor,
    *,
    t_eps: float = 0.02,
    p_mean: float = -0.8,
    p_std: float = 0.8,
    time_schedule: str = "logit_normal",
    noise_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = clean.shape[0]
    device, dtype = clean.device, clean.dtype
    if time_schedule == "logit_normal":
        t = sample_logit_normal_timesteps(
            batch, p_mean=p_mean, p_std=p_std, t_eps=t_eps, device=device, dtype=dtype
        )
    elif time_schedule in ("uniform", "standard"):
        t = torch.rand(batch, device=device, dtype=dtype).clamp(t_eps, 1.0 - t_eps)
    else:
        t = sample_logit_normal_timesteps(
            batch, p_mean=p_mean, p_std=p_std, t_eps=t_eps, device=device, dtype=dtype
        )
    noise = torch.randn_like(clean) * noise_scale
    view = t.view(-1, *([1] * (clean.ndim - 1)))
    z = (1.0 - view) * noise + view * clean
    return z, t, noise

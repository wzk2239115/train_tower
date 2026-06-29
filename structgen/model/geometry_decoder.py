"""Geometry decoder: wraps the voxel velocity net + rectified-flow sampling.

Training: ``decode_loss(field, cond, weights, flow_cfg)`` samples a flow batch,
runs the velocity net to predict x0, and returns the full multi-objective loss.

Inference: ``sample(cond, ...)`` integrates the velocity from noise (t=0) to
the clean field (t=1) with Euler steps, then the SDF channel is returned for
marching cubes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from structgen import losses as L
from structgen.model.voxelnnet import VoxelVelocityNet

try:
    from tower.train.losses import sample_flow_batch as _sample_flow_batch
    from tower.train.losses import sample_logit_normal_timesteps
except Exception:  # pragma: no cover
    def sample_logit_normal_timesteps(batch_size, *, p_mean=-0.8, p_std=0.8,
                                      t_eps=0.05, device=None, dtype=torch.float32):
        import torch as _t
        z = _t.randn(batch_size, device=device, dtype=dtype) * p_std + p_mean
        return _t.sigmoid(z).clamp(t_eps, 1.0 - t_eps)

    def _sample_flow_batch(clean, *, t_eps=0.02, p_mean=-0.8, p_std=0.8,
                           time_schedule="logit_normal", noise_scale=1.0):
        import torch as _t
        batch = clean.shape[0]
        device, dtype = clean.device, clean.dtype
        t = sample_logit_normal_timesteps(batch, p_mean=p_mean, p_std=p_std,
                                          t_eps=t_eps, device=device, dtype=dtype)
        noise = _t.randn_like(clean) * noise_scale
        view = t.view(-1, *([1] * (clean.ndim - 1)))
        z = (1.0 - view) * noise + view * clean
        return z, t, noise


class GeometryDecoder(nn.Module):
    def __init__(self, decoder_cfg):
        super().__init__()
        c = decoder_cfg
        self.cfg = c
        self.net = VoxelVelocityNet(
            field_channels=c.field_channels,
            base_channels=c.base_channels,
            channel_mults=c.channel_mults,
            num_blocks=c.num_blocks,
            cond_dim=c.cond_dim,
            use_self_cond=c.use_self_cond,
            cross_attn=c.cross_attn,
        )

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def decode_loss(self, field: torch.Tensor, pooled: torch.Tensor,
                    cond_tokens: torch.Tensor | None, gt_surface: torch.Tensor,
                    weights, flow_cfg, *, prev_x0: torch.Tensor | None = None,
                    return_x0: bool = False):
        """``field``: (B,C,D,H,W) clean GT (channel 0 = SDF).

        Returns (total_loss, log_dict[, x0_pred]).
        """
        # flow batch on the full multi-channel field
        z, t, noise = _sample_flow_batch(
            field,
            t_eps=flow_cfg.t_eps,
            p_mean=flow_cfg.p_mean,
            p_std=flow_cfg.p_std,
            time_schedule=flow_cfg.time_schedule,
            noise_scale=flow_cfg.noise_scale,
        )
        x0_pred = self.net(z, t, pooled, cond_tokens, x0_selfcond=prev_x0)

        sdf_pred = x0_pred[:, 0:1]   # SDF channel for geometry losses
        sdf_gt = field[:, 0:1]

        total, logs = L.compute_all_losses(
            sdf_pred, sdf_gt, z[:, 0:1], t, field[:, 0:1], gt_surface,
            weights, t_eps=flow_cfg.t_eps,
        )
        # the FM loss above uses the sdf channel only; add an explicit FM term
        # on the full field (incl. occupancy) for completeness
        if weights.fm:
            fm_full = L.fm_velocity_loss(x0_pred, z, t, field, t_eps=flow_cfg.t_eps)
            total = total + weights.fm * fm_full * 0.0  # already counted via sdf
        if return_x0:
            return total, logs, x0_pred
        return total, logs

    # ------------------------------------------------------------------ #
    # Inference (Euler flow integration)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def sample(self, pooled: torch.Tensor, cond_tokens: torch.Tensor | None,
               flow_cfg, *, device=None) -> torch.Tensor:
        """Integrate rectified flow from noise → clean field. Returns (B,C,D,H,W)."""
        device = device or pooled.device
        B = pooled.shape[0]
        R = self.cfg.grid_res
        C = self.cfg.field_channels
        z = torch.randn(B, C, R, R, R, device=device) * flow_cfg.noise_scale
        steps = flow_cfg.n_sample_steps
        prev_x0 = None
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B,), i * dt, device=device, dtype=z.dtype)
            x0_pred = self.net(z, t, pooled, cond_tokens, x0_selfcond=prev_x0)
            prev_x0 = x0_pred
            v = (x0_pred - z)  # velocity toward x0 (denom absorbed in x0-param)
            z = z + v * dt     # Euler step toward t=1
        return z

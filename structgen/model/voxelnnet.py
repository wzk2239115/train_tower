"""3D voxel velocity network (rectified flow on the SDF/occupancy field).

A 3D U-Net whose input is the noisy field ``z_t`` and whose output is the
predicted clean field ``x0`` (matching the project convention used by
``rectified_flow_velocity_loss``).

Condition injection (the backbone is a *pretrained multimodal component*):
* timestep ``t`` + pooled condition → **adaLN** modulation at every res-block;
* condition tokens → **cross-attention** at the bottleneck (cheap, effective);
* optional **self-conditioning** (ELF-style): the previous x0 estimate is
  concatenated to the input.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _silu(x):
    return F.silu(x)


def _num_groups(ch: int) -> int:
    g = min(8, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return max(1, g)


class _TimestepCondMod(nn.Module):
    """Produce per-channel (scale, shift, gate) from t-embed + pooled cond."""

    def __init__(self, cond_dim: int, out_ch: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_ch * 3),
        )

    def forward(self, pooled: torch.Tensor):
        x = self.mlp(pooled)
        return x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).chunk(3, dim=1)


class _ResBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.mod = _TimestepCondMod(cond_dim, out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, pooled):
        h = self.norm1(x)
        s, sh, g = self.mod(pooled)
        # norm1 output has in_ch channels; modulation has out_ch channels.
        # Only apply modulation after conv1 (channels == out_ch).
        h = _silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = h * (1 + s) + sh
        h = _silu(h)
        h = self.conv2(h)
        return self.skip(x) + g * h


class _CrossAttn3d(nn.Module):
    """Reshape voxel tokens → attend to cond tokens → reshape back."""

    def __init__(self, channels: int, cond_dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.q = nn.Conv3d(channels, channels, 1)
        self.norm_c = nn.LayerNorm(cond_dim)
        self.kv = nn.Linear(cond_dim, channels * 2)
        self.out = nn.Conv3d(channels, channels, 1)
        self.scale = (channels // heads) ** -0.5

    def forward(self, x, cond_tokens):
        B, C, D, H, W = x.shape
        heads, dh = self.heads, C // self.heads
        N = D * H * W
        h = self.norm(x)
        q = self.q(h).reshape(B, heads, dh, N).permute(0, 1, 3, 2)  # (B,heads,N,dh)
        kv = self.kv(self.norm_c(cond_tokens)).reshape(B, -1, 2, heads, dh)
        k = kv[:, :, 0].permute(0, 2, 1, 3)  # (B,heads,T,dh)
        v = kv[:, :, 1].permute(0, 2, 1, 3)  # (B,heads,T,dh)
        attn = (q @ k.transpose(-1, -2)) * self.scale  # (B,heads,N,T)
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B,heads,N,dh)
        out = out.permute(0, 1, 3, 2).reshape(B, C, D, H, W)
        return x + self.out(out)


class VoxelVelocityNet(nn.Module):
    """3D U-Net velocity/x0 predictor.

    out = net(z_t, t, pooled, cond_tokens[, x0_selfcond])  →  predicted x0
    """

    def __init__(self, field_channels: int = 2, base_channels: int = 64,
                 channel_mults: tuple[int, ...] = (1, 2, 4, 8),
                 num_blocks: int = 2, cond_dim: int = 768,
                 t_dim: int = 256, use_self_cond: bool = True,
                 cross_attn: bool = True):
        super().__init__()
        self.field_channels = field_channels
        self.use_self_cond = use_self_cond
        self.cross_attn = cross_attn
        in_ch = field_channels * (2 if use_self_cond else 1)
        self.t_dim = t_dim

        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))

        dims = [base_channels * m for m in channel_mults]
        n_stages = len(dims)

        self.in_conv = nn.Conv3d(in_ch, dims[0], 3, padding=1)

        # encoder: per stage, (num_blocks) resblocks [dims[i]->dims[i]], then
        # a downsample [dims[i]->dims[i+1]] (except after the last stage).
        self.down_blocks = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        for i in range(n_stages):
            blocks = nn.ModuleList(
                [_ResBlock3d(dims[i], dims[i], cond_dim) for _ in range(num_blocks)])
            self.down_blocks.append(blocks)
            if i + 1 < n_stages:
                self.down_sample.append(nn.Conv3d(dims[i], dims[i + 1], 3, stride=2, padding=1))
            else:
                self.down_sample.append(nn.Identity())

        # bottleneck
        bot_c = dims[-1]
        self.bot_blocks = nn.ModuleList(
            [_ResBlock3d(bot_c, bot_c, cond_dim) for _ in range(num_blocks)])
        self.bot_attn = _CrossAttn3d(bot_c, cond_dim) if cross_attn else None

        # decoder: per stage (reversed), optional upsample then resblocks.
        # The first decoder resblock of each stage takes the skip concat.
        self.up_blocks = nn.ModuleList()
        self.up_sample = nn.ModuleList()
        for i in range(n_stages):
            # upsample maps dims[i+1] -> dims[i]; None for the lowest stage
            if i + 1 < n_stages:
                self.up_sample.append(
                    nn.ConvTranspose3d(dims[i + 1], dims[i], 4, stride=2, padding=1))
            else:
                self.up_sample.append(nn.Identity())
            blocks = nn.ModuleList()
            for j in range(num_blocks):
                in_c = (2 * dims[i]) if j == 0 else dims[i]
                blocks.append(_ResBlock3d(in_c, dims[i], cond_dim))
            self.up_blocks.append(blocks)

        self.out_norm = nn.GroupNorm(_num_groups(dims[0]), dims[0])
        self.out_conv = nn.Conv3d(dims[0], field_channels, 3, padding=1)

    def _t_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.t_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half,
                     device=t.device, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.t_mlp(emb)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, pooled: torch.Tensor,
                cond_tokens: torch.Tensor | None = None,
                x0_selfcond: torch.Tensor | None = None) -> torch.Tensor:
        pooled = self._t_embedding(t) + pooled
        x = z_t
        if self.use_self_cond:
            sc = x0_selfcond
            if sc is None:
                sc = torch.zeros_like(z_t)
            elif sc.shape != z_t.shape:
                sc = F.interpolate(sc, size=z_t.shape[2:], mode="trilinear",
                                   align_corners=False)
            x = torch.cat([z_t, sc], dim=1)
        x = self.in_conv(x)

        skips = []
        for i in range(len(self.down_blocks)):
            for blk in self.down_blocks[i]:
                x = blk(x, pooled)
            skips.append(x)
            x = self.down_sample[i](x)

        for blk in self.bot_blocks:
            x = blk(x, pooled)
        if self.bot_attn is not None and cond_tokens is not None:
            x = self.bot_attn(x, cond_tokens)

        # decoder in reverse order (lowest stage first)
        n = len(self.up_blocks)
        for k in range(n - 1, -1, -1):
            x = self.up_sample[k](x)
            x = torch.cat([x, skips[k]], dim=1)
            for blk in self.up_blocks[k]:
                x = blk(x, pooled)

        x = _silu(self.out_norm(x))
        return self.out_conv(x)

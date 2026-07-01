"""Stage A (TRELLIS/SparseFlex-style): a voxel VAE with KL regularization.

Encodes an occupancy grid (1, R, R, R) into a compact Gaussian latent
(latent_ch, L, L, L) and decodes it back. The KL term pulls the latent toward
N(0,I) so that flow matching IN the latent space is well-behaved (no
low-timestep collapse — the failure mode of raw-voxel flow matching).

Train this FIRST; then run flow matching on its latent (structgen/model/dit.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ng(ch):
    g = min(8, ch)
    while ch % g:
        g -= 1
    return max(1, g)


class _Enc(nn.Module):
    def __init__(self, in_ch, latent_ch, base, n_down):
        super().__init__()
        layers = [nn.Conv3d(in_ch, base, 3, padding=1)]
        ch = base
        for i in range(n_down):
            out = base * (2 ** (i + 1))
            layers += [nn.GroupNorm(_ng(ch), ch), nn.SiLU(),
                       nn.Conv3d(ch, out, 4, stride=2, padding=1)]
            ch = out
        layers += [nn.GroupNorm(_ng(ch), ch), nn.SiLU(),
                   nn.Conv3d(ch, latent_ch, 3, padding=1)]
        self.net = nn.Sequential(*layers)
        self.mu = nn.Conv3d(latent_ch, latent_ch, 3, padding=1)
        self.logvar = nn.Conv3d(latent_ch, latent_ch, 3, padding=1)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class _Dec(nn.Module):
    def __init__(self, latent_ch, out_ch, base, n_down):
        super().__init__()
        layers = [nn.Conv3d(latent_ch, base * (2 ** n_down), 3, padding=1)]
        ch = base * (2 ** n_down)
        for i in range(n_down, 0, -1):
            out = base * (2 ** (i - 1))
            layers += [nn.GroupNorm(_ng(ch), ch), nn.SiLU(),
                       nn.ConvTranspose3d(ch, out, 4, stride=2, padding=1)]
            ch = out
        layers += [nn.GroupNorm(_ng(ch), ch), nn.SiLU(),
                   nn.Conv3d(ch, out_ch, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class VoxelVAE(nn.Module):
    """Occupancy (1,R,R,R) <-> latent (latent_ch, L,L,L), R = L * 2**n_down."""

    def __init__(self, grid_res=64, latent_res=16, latent_ch=32, base=24, n_down=None):
        super().__init__()
        if n_down is None:
            n_down = 0
            r = grid_res
            while r > latent_res:
                r //= 2
                n_down += 1
            assert r == latent_res, f"{grid_res} -> {latent_res} not a power-of-2 downsample"
        self.grid_res = grid_res
        self.latent_res = latent_res
        self.latent_ch = latent_ch
        self.n_down = n_down
        self.enc = _Enc(1, latent_ch, base, n_down)
        self.dec = _Dec(latent_ch, 1, base, n_down)

    def encode(self, occ):
        mu, logvar = self.enc(occ)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.dec(z)

    def forward(self, occ):
        mu, logvar = self.enc(occ)
        z = self.reparam(mu, logvar) if self.training else mu
        recon = self.dec(z)
        return recon, mu, logvar


def vae_loss(recon, occ, mu, logvar, beta=1e-3):
    """BCE reconstruction + KL to N(0,I). beta balances reconstruction vs latent
    smoothness; small beta keeps shape fidelity while still regularizing."""
    bce = F.binary_cross_entropy_with_logits(recon, occ)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta * kl, {"loss/bce": float(bce), "loss/kl": float(kl)}

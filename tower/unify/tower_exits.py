"""Flow Tower exit modules — JEPA + stacked ELF heads at each floor."""

from __future__ import annotations

import torch
import torch.nn as nn

from tower.train.losses import rectified_flow_velocity_loss


class _ElfBlock(nn.Module):
    """Lightweight ELF-style residual block (LayerNorm + FFN)."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        mlp_dim = int(hidden_size * mlp_ratio)
        self.norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class ElfFlowTowerExit(nn.Module):
    """Stacked ELF flow head at a tower floor (rectified flow velocity MSE)."""

    def __init__(
        self,
        hidden_size: int,
        out_dim: int,
        *,
        elf_depth: int = 2,
        t_embedder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        if t_embedder is not None:
            self.t_embedder = t_embedder
        else:
            from sensenova_u1.models.neo_unify.modeling_fm_modules import TimestepEmbedder

            self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList([_ElfBlock(hidden_size) for _ in range(max(elf_depth, 1))])
        self.fm_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, out_dim),
        )
        self.self_cond_proj = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, hidden_size),
        )

    def _encode_hidden(
        self,
        hidden: torch.Tensor,
        t: torch.Tensor,
        *,
        noise_scale_emb: torch.Tensor | None = None,
        self_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_emb = self.t_embedder(t.to(hidden.dtype))
        x = hidden + t_emb
        if noise_scale_emb is not None:
            x = x + noise_scale_emb.to(x.dtype)
        if self_cond is not None:
            x = x + self.self_cond_proj(self_cond.to(x.dtype))
        for block in self.blocks:
            x = block(x)
        return x

    def predict_x(
        self,
        hidden: torch.Tensor,
        t: torch.Tensor,
        *,
        noise_scale_emb: torch.Tensor | None = None,
        self_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden.numel() == 0:
            return hidden
        n = hidden.shape[0]
        if t.numel() == 1:
            t = t.expand(n)
        x = self._encode_hidden(hidden, t, noise_scale_emb=noise_scale_emb, self_cond=self_cond)
        return self.fm_head(x)

    def forward(
        self,
        hidden: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        x_clean: torch.Tensor,
        *,
        t_eps: float = 0.05,
        noise_scale_emb: torch.Tensor | None = None,
        self_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden.numel() == 0:
            return hidden.sum() * 0.0

        x_pred = self.predict_x(hidden, t, noise_scale_emb=noise_scale_emb, self_cond=self_cond)
        return rectified_flow_velocity_loss(x_pred, z, t, x_clean, t_eps=t_eps)


class JepaTowerExit(nn.Module):
    """JEPA predictor head with optional EMA target projector."""

    def __init__(
        self,
        hidden_size: int,
        *,
        ema_momentum: float = 0.996,
    ) -> None:
        super().__init__()
        self.ema_momentum = float(ema_momentum)
        self.predictor = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.target_projector = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        for p_tgt, p_pred in zip(self.target_projector.parameters(), self.predictor.parameters()):
            p_tgt.data.copy_(p_pred.data)
        for p in self.target_projector.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def ema_update_target(self) -> None:
        """EMA update target projector from predictor prefix layers."""
        m = self.ema_momentum
        for p_tgt, p_pred in zip(self.target_projector.parameters(), self.predictor.parameters()):
            p_tgt.data.mul_(m).add_(p_pred.data, alpha=1.0 - m)

    def forward(
        self,
        hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.numel() == 0 or target_hidden.numel() == 0:
            return hidden.sum() * 0.0
        if pred_mask.numel() == 0 or not pred_mask.any():
            return hidden.sum() * 0.0

        n = min(hidden.shape[0], target_hidden.shape[0], pred_mask.shape[0])
        h = hidden[:n]
        tgt = target_hidden[:n]
        pm = pred_mask[:n]
        if not pm.any():
            return hidden.sum() * 0.0

        pred = self.predictor(h[pm])
        with torch.no_grad():
            target = self.target_projector(tgt[pm]).detach()
        return torch.mean((pred - target) ** 2)

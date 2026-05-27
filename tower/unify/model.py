from __future__ import annotations

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from tower.train.config import TrainConfig
from tower.train.losses import rectified_flow_velocity_loss, sample_flow_batch


class UnifiedNeoChatModel(nn.Module):
    """Unified training wrapper: NEO understanding + optional FM generation head."""

    def __init__(self, neo_model, cfg: TrainConfig):
        super().__init__()
        self.neo = neo_model
        self.cfg = cfg
        hidden = neo_model.language_model.config.hidden_size
        patch = neo_model.patch_size
        merge = int(1 / neo_model.downsample_ratio)
        out_dim = 3 * (patch * merge) ** 2
        self.fm_timestep = nn.Sequential(
            nn.Linear(256, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.fm_head = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, out_dim),
        )
        self._register_freq_buffer()

    def _register_freq_buffer(self) -> None:
        dim = 256
        half = dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half) / half)
        self.register_buffer("fm_freqs", freqs, persistent=False)

    def _timestep_embed(self, t: torch.Tensor) -> torch.Tensor:
        args = t[:, None].float() * self.fm_freqs[None].to(t.device)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < 256:
            emb = torch.nn.functional.pad(emb, (0, 256 - emb.shape[-1]))
        return self.fm_timestep(emb.to(self.neo.dtype))

    @property
    def config(self):
        return self.neo.config

    @property
    def device(self):
        return self.neo.device

    @property
    def dtype(self):
        return self.neo.dtype

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.neo, "gradient_checkpointing_enable"):
            self.neo.gradient_checkpointing_enable(**kwargs)

    def get_input_embeddings(self):
        return self.neo.get_input_embeddings()

    def save_pretrained(self, output_dir: str, **kwargs):
        self.neo.save_pretrained(output_dir, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.neo.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.neo.load_state_dict(state_dict, strict=strict, assign=assign)

    def _compute_fm_loss(self, pixel_values: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        # pixel_values: (N_patches, patch_dim)
        clean = pixel_values
        z, t, _ = sample_flow_batch(clean.unsqueeze(0))
        z = z.squeeze(0)
        t_scalar = t.squeeze(0)
        # Pool hidden states to match patch count
        h = hidden[-clean.shape[0] :]
        if h.shape[0] != clean.shape[0]:
            h = hidden.mean(dim=0, keepdim=True).expand(clean.shape[0], -1)
        t_emb = self._timestep_embed(t_scalar.unsqueeze(0).expand(h.shape[0]))
        x_pred = self.fm_head(h + t_emb)
        return rectified_flow_velocity_loss(x_pred, z, t_scalar, clean)

    def forward(self, **batch):
        outputs: CausalLMOutputWithPast = self.neo(**{k: v for k, v in batch.items() if k not in ("tasks", "is_gen")})
        loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=self.device)
        fm_loss = torch.tensor(0.0, device=self.device)

        if self.cfg.fm_weight > 0 and batch.get("is_gen") and any(batch["is_gen"]):
            if batch.get("pixel_values") is not None and outputs.hidden_states is None:
                # Re-run with hidden states if needed — use logits path approximation
                pass
            pv = batch.get("pixel_values")
            if pv is not None and isinstance(pv, torch.Tensor) and pv.numel() > 0:
                hidden = self.neo.language_model.get_input_embeddings()(batch["input_ids"])
                fm_loss = self._compute_fm_loss(pv, hidden[0])

        ce_w = self.cfg.ce_weight
        fm_w = self.cfg.fm_weight
        if ce_w == 0:
            total = fm_w * fm_loss
        elif fm_w == 0:
            total = ce_w * loss
        else:
            total = ce_w * loss + fm_w * fm_loss

        outputs.loss = total
        return outputs

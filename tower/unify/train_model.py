from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from tower.train.config import TrainConfig
from tower.train.losses import rectified_flow_velocity_loss, sample_flow_batch


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
        z, t, _ = sample_flow_batch(clean.unsqueeze(0), t_eps=self.model.config.t_eps)
        z = z.squeeze(0)
        t_scalar = t.squeeze(0)

        grid_hw = batch.get("image_grid_hw", [None])[0]
        if grid_hw is None:
            grid_size = int(1 / self.model.downsample_ratio)
            grid_hw = torch.tensor([[grid_size, grid_size]], device=self.device)
        elif not isinstance(grid_hw, torch.Tensor):
            grid_hw = torch.tensor(grid_hw, device=self.device)

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

        t_emb = self.model.fm_modules["timestep_embedder"].timestep_embedding(
            t_scalar.mean().view(1), self.model.config.llm_config.hidden_size
        )
        if n > 0:
            hidden[0, selected][:n] = hidden[0, selected][:n] + t_emb.to(hidden.dtype)

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
        return rectified_flow_velocity_loss(x_pred, z_part, t_part, target, t_eps=self.model.config.t_eps)

    def forward(self, **batch):
        ce = torch.tensor(0.0, device=self.device)
        fm = torch.tensor(0.0, device=self.device)

        if self.cfg.ce_weight > 0:
            ce = self._ce_forward(batch)
        if self.cfg.fm_weight > 0:
            fm = self._fm_forward(batch)

        total = self.cfg.ce_weight * ce + self.cfg.fm_weight * fm
        return CausalLMOutputWithPast(loss=total)

from __future__ import annotations

from typing import List, Optional

import torch

from .utils import build_abs_positions_from_grid_hw as _build_abs_pos


class KVCacheMixin:
    """KV-cache manipulation helpers for prefix caching and interleaved generation."""

    def get_thw_indexes(self, input_ids, grid_hw=None):
        img_start_shift = torch.cat([torch.zeros(1, dtype=torch.long).to(input_ids.device),
                                     (input_ids == self.img_start_token_id).long()], dim=0)[:-1]
        not_img_token = (input_ids != self.img_context_token_id).long()
        t_indexes = ((img_start_shift + not_img_token).cumsum(0) - 1)
        h_indexes = torch.zeros_like(t_indexes).to(t_indexes.device)
        w_indexes = torch.zeros_like(t_indexes).to(t_indexes.device)

        if grid_hw is not None:
            selected = (input_ids == self.img_context_token_id)
            if selected.long().sum() > 0:
                abs_pos_w, abs_pos_h = _build_abs_pos(
                    grid_hw // int(1 / self.downsample_ratio), device=t_indexes.device)
                h_indexes[selected] = abs_pos_h.to(t_indexes.device, t_indexes.dtype)
                w_indexes[selected] = abs_pos_w.to(t_indexes.device, t_indexes.dtype)
        return torch.stack([t_indexes, h_indexes, w_indexes], dim=0)

    def _append_text_tokens_to_cache(self, cache, t_idx, input_ids):
        if input_ids.shape[1] == 0:
            return t_idx

        device = input_ids.device
        seq_len = input_ids.shape[1]
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        t_indexes = torch.arange(t_idx + 1, t_idx + 1 + seq_len, dtype=torch.long, device=device)
        h_indexes = torch.zeros(seq_len, dtype=torch.long, device=device)
        w_indexes = torch.zeros(seq_len, dtype=torch.long, device=device)
        indexes = torch.stack([t_indexes, h_indexes, w_indexes], dim=0)

        past_len = cache.get_seq_length()
        mask = torch.zeros(1, 1, seq_len, past_len + seq_len, device=device)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        causal_mask = torch.where(causal_mask == 1, 0.0, float('-inf'))
        mask[:, :, :, past_len:] = causal_mask
        attention_mask_dict = {"full_attention": mask}

        self.language_model(
            inputs_embeds=inputs_embeds,
            indexes=indexes,
            attention_mask=attention_mask_dict,
            past_key_values=cache,
            use_cache=True
        )
        return t_idx + seq_len

    def _append_text_to_cache(self, cache, t_idx, text_ids, device):
        if text_ids.shape[-1] == 0:
            return t_idx
        seq_len = text_ids.shape[-1]
        inputs_embeds = self.language_model.get_input_embeddings()(text_ids)
        t_indexes = torch.arange(t_idx + 1, t_idx + 1 + seq_len, dtype=torch.long, device=device)
        h_indexes = torch.zeros(seq_len, dtype=torch.long, device=device)
        w_indexes = torch.zeros(seq_len, dtype=torch.long, device=device)
        indexes = torch.stack([t_indexes, h_indexes, w_indexes], dim=0)
        past_len = cache.get_seq_length()
        mask = torch.zeros(1, 1, seq_len, past_len + seq_len, device=device)
        causal = torch.tril(torch.ones(seq_len, seq_len, device=device))
        causal = torch.where(causal == 1, 0.0, float('-inf'))
        mask[:, :, :, past_len:] = causal
        self.language_model(
            inputs_embeds=inputs_embeds,
            indexes=indexes,
            attention_mask={"full_attention": mask},
            past_key_values=cache,
            use_cache=True,
        )
        return t_idx + seq_len

    def _append_modality_to_cache(
        self, cache, t_idx, context_embeds, n_tokens, abs_pos_h, abs_pos_w, end_token_embed, device
    ):
        tgt_len = n_tokens + 1
        inputs = torch.cat([context_embeds[:, :n_tokens], end_token_embed.unsqueeze(0).unsqueeze(0)], dim=1)

        t_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
        t_indexes[:n_tokens] = t_idx + 1
        t_indexes[n_tokens] = t_idx + 2

        h_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
        w_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
        h_indexes[:n_tokens] = abs_pos_h
        w_indexes[:n_tokens] = abs_pos_w

        indexes = torch.stack([t_indexes, h_indexes, w_indexes], dim=0)
        past_len = cache.get_seq_length()
        mask = torch.zeros(1, 1, tgt_len, past_len + tgt_len, device=device)
        mask[0, 0, :n_tokens, past_len + n_tokens] = float('-inf')

        outputs = self.language_model(
            inputs_embeds=inputs,
            indexes=indexes,
            attention_mask={"full_attention": mask},
            past_key_values=cache,
            use_cache=True,
        )
        return outputs, t_idx + 2

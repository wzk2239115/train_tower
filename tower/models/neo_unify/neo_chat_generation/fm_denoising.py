from __future__ import annotations

import torch
import torch.nn.functional as F


class FMDenoisingMixin:
    """Flow-matching denoising primitives: velocity prediction and time scheduling."""

    def _euler_step(self, v_pred, z, t, t_next):
        z_next = z + (t_next - t) * v_pred
        return z_next

    def _calculate_dynamic_mu(self, image_seq_len: int) -> float:
        denom = self.max_image_seq_len - self.base_image_seq_len
        if denom == 0:
            return float(self.base_shift)
        m = (self.max_shift - self.base_shift) / denom
        b = self.base_shift - m * self.base_image_seq_len
        return float(image_seq_len) * m + b

    def _apply_time_schedule(self, t: torch.Tensor, image_seq_len: int, timestep_shift: float) -> torch.Tensor:
        import math

        self.time_schedule = "standard"
        sigma = 1 - t
        if timestep_shift != 1:
            self.time_schedule = "standard"
        if self.time_schedule == "standard":
            shift = timestep_shift
            sigma = shift * sigma / (1 + (shift - 1) * sigma)
        elif self.time_schedule == "dynamic":
            mu = self._calculate_dynamic_mu(image_seq_len)
            mu_t = t.new_tensor(mu)
            if self.time_shift_type == "exponential":
                shift = torch.exp(mu_t)
                sigma = shift * sigma / (1 + (shift - 1) * sigma)
            elif self.time_shift_type == "linear":
                sigma = mu_t / (mu_t + (1 / sigma - 1))
            else:
                raise ValueError(f"Unsupported time_shift_type: {self.time_shift_type}")
        else:
            raise ValueError(f"Unsupported time_schedule: {self.time_schedule}")
        return 1 - sigma

    def _t2i_predict_v(self, input_embeds, indexes_image, attn_mask, past_key_values, t, z, image_token_num, timestep_embeddings=None, image_size=None):
        B, L = z.shape[0], z.shape[1]

        outputs = self.language_model.model(
            inputs_embeds=input_embeds,
            image_gen_indicators=torch.ones((input_embeds.shape[0], input_embeds.shape[1]), dtype=torch.bool, device=input_embeds.device),
            indexes=indexes_image,
            attention_mask=attn_mask,
            past_key_values=past_key_values,
            update_cache=False,
            use_cache=True,
        )

        if self.use_pixel_head:
            merge_size = int(1 / self.downsample_ratio)
            token_h = image_size[1] // (self.patch_size * merge_size)
            token_w = image_size[0] // (self.patch_size * merge_size)

            img_reshaped = outputs.last_hidden_state[:, -image_token_num:].view(B, token_h, token_w, -1)
            img_2d = torch.einsum("b h w c -> b c h w", img_reshaped)
            img_2d = img_2d.contiguous().view(B, -1, token_h, token_w)

            smoothed_img_2d = self.fm_modules['fm_head'](img_2d)

            smoothed_reshaped = smoothed_img_2d.view(B, 3, token_h, self.patch_size * merge_size, token_w, self.patch_size * merge_size)
            smoothed_reshaped = torch.einsum("b c h p w q -> b h w p q c", smoothed_reshaped)
            out_1d = smoothed_reshaped.contiguous().view(B, L, self.patch_size * merge_size * self.patch_size * merge_size * 3)
            x_pred = out_1d
        else:
            if self.use_deep_fm_head:
                x_pred = self.fm_modules["fm_head"](
                outputs.last_hidden_state[:, -image_token_num:].view(B*L, -1), t.repeat(B*L)
                ).view(B, L, -1)
            else:
                x_pred = self.fm_modules["fm_head"](
                    outputs.last_hidden_state[:, -image_token_num:].view(B, L, -1)
                ).view(B, L, -1)

        v_pred = (x_pred - z) / (1 - t).clamp_min(self.config.t_eps)
        return v_pred

    def _t2a_predict_v(
        self,
        audio_embeds: torch.Tensor,
        indexes_audio: torch.Tensor,
        attn_mask,
        past_key_values,
        t: torch.Tensor,
        z: torch.Tensor,
        audio_token_num: int,
    ) -> torch.Tensor:
        B, L = z.shape[0], z.shape[1]
        outputs = self.language_model.model(
            inputs_embeds=audio_embeds,
            image_gen_indicators=torch.ones(
                (audio_embeds.shape[0], audio_embeds.shape[1]),
                dtype=torch.bool,
                device=audio_embeds.device,
            ),
            indexes=indexes_audio,
            attention_mask=attn_mask,
            past_key_values=past_key_values,
            update_cache=False,
            use_cache=True,
        )
        x_pred = self.audio_latent_proj(
            outputs.last_hidden_state[:, -audio_token_num:].view(B * L, -1)
        ).view(B, L, -1)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.config.t_eps)
        return v_pred

    def _t2v_predict_v(
        self,
        video_embeds: torch.Tensor,
        indexes_video: torch.Tensor,
        attn_mask,
        past_key_values,
        t: torch.Tensor,
        z: torch.Tensor,
        video_token_num: int,
    ) -> torch.Tensor:
        B, L = z.shape[0], z.shape[1]
        outputs = self.language_model.model(
            inputs_embeds=video_embeds,
            image_gen_indicators=torch.ones(
                (video_embeds.shape[0], video_embeds.shape[1]),
                dtype=torch.bool,
                device=video_embeds.device,
            ),
            indexes=indexes_video,
            attention_mask=attn_mask,
            past_key_values=past_key_values,
            update_cache=False,
            use_cache=True,
        )
        x_pred = self.fm_modules["fm_head"](
            outputs.last_hidden_state[:, -video_token_num:].view(B * L, -1)
        ).view(B, L, -1)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.config.t_eps)
        return v_pred

    def _apply_cfg(
        self,
        out_cond: torch.Tensor,
        cfg_scale: float,
        img_cfg_scale: float,
        cfg_norm: str,
        use_cfg: bool,
        out_text_uncond: torch.Tensor | None = None,
        out_img_uncond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not use_cfg or (cfg_scale <= 1 and img_cfg_scale <= 1):
            return out_cond
        if out_text_uncond is None and out_img_uncond is None:
            return out_cond
        if out_img_uncond is None:
            v_pred = out_text_uncond + cfg_scale * (out_cond - out_text_uncond)
        elif cfg_scale == img_cfg_scale:
            v_pred = out_img_uncond + cfg_scale * (out_cond - out_img_uncond)
        else:
            v_pred = (
                out_img_uncond
                + cfg_scale * (out_cond - out_text_uncond)
                + img_cfg_scale * (out_text_uncond - out_img_uncond)
            )
        if cfg_scale > 1 or img_cfg_scale > 1:
            if cfg_norm == "global":
                nc = torch.norm(out_cond, dim=(1, 2), keepdim=True)
                nv = torch.norm(v_pred, dim=(1, 2), keepdim=True)
                v_pred = v_pred * (nc / (nv + 1e-8)).clamp(min=0, max=1.0)
            elif cfg_norm == "channel":
                nc = torch.norm(out_cond, dim=-1, keepdim=True)
                nv = torch.norm(v_pred, dim=-1, keepdim=True)
                v_pred = v_pred * (nc / (nv + 1e-8)).clamp(min=0, max=1.0)
        return v_pred

    @torch.no_grad()
    def _denoise_image_core(
        self,
        past_kv_cond,
        past_kv_tu,
        past_kv_iu,
        t_start_cond: int,
        t_start_tu: int | None,
        t_start_iu: int | None,
        *,
        token_h: int,
        token_w: int,
        image_size: tuple[int, int],
        num_steps: int = 30,
        cfg_scale: float = 1.0,
        img_cfg_scale: float = 1.0,
        cfg_norm: str = "none",
        cfg_interval: tuple[float, float] = (0, 1),
        enable_timestep_shift: bool = True,
        timestep_shift: float = 1.0,
        batch_size: int = 1,
        seed: int = 0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        import math
        from .utils import prepare_flash_kv_cache, clear_flash_kv_cache

        device = self.device
        merge_size = int(1 / self.downsample_ratio)
        itn = token_h * token_w

        idx_c = self._build_t2i_image_indexes(token_h, token_w, t_start_cond, device=device)
        idx_tu = self._build_t2i_image_indexes(token_h, token_w, t_start_tu, device=device) if t_start_tu is not None else None
        idx_iu = self._build_t2i_image_indexes(token_h, token_w, t_start_iu, device=device) if t_start_iu is not None else None

        grid_h = image_size[1] // self.patch_size
        grid_w = image_size[0] // self.patch_size
        gen_ghw = torch.tensor([[grid_h, grid_w]] * batch_size, device=device)

        ns = self.noise_scale
        if self.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
            base = float(self.noise_scale_base_image_seq_len)
            ns = math.sqrt((grid_h * grid_w) / (merge_size ** 2) / base) * float(self.noise_scale)
            if self.noise_scale_mode == "dynamic_sqrt":
                ns = math.sqrt(ns)
        ns = min(ns, self.noise_scale_max_value)

        if generator is None:
            generator = torch.Generator(device).manual_seed(seed)
        img_pred = ns * torch.randn(
            (batch_size, 3, image_size[1], image_size[0]),
            device=device, dtype=torch.bfloat16, generator=generator,
        )

        prepare_flash_kv_cache(past_kv_cond, current_len=itn, batch_size=batch_size)
        if past_kv_tu is not None:
            prepare_flash_kv_cache(past_kv_tu, current_len=itn, batch_size=batch_size)
        if past_kv_iu is not None:
            prepare_flash_kv_cache(past_kv_iu, current_len=itn, batch_size=batch_size)

        attn_none = {"full_attention": None}
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        if enable_timestep_shift:
            timesteps = self._apply_time_schedule(timesteps, itn, timestep_shift)

        for si in range(num_steps):
            t = timesteps[si]
            t_next = timesteps[si + 1]

            z = self.patchify(img_pred, self.patch_size * merge_size)
            img_in = self.patchify(img_pred, self.patch_size, channel_first=True)
            embeds = self.extract_feature(
                img_in.view(batch_size * grid_h * grid_w, -1), gen_model=True, grid_hw=gen_ghw,
            ).view(batch_size, itn, -1)

            t_exp = t.expand(batch_size * itn)
            ts_emb = self.fm_modules["timestep_embedder"](t_exp).view(batch_size, itn, -1)
            if self.add_noise_scale_embedding:
                ns_t = torch.full_like(t_exp, ns / self.noise_scale_max_value)
                ts_emb = ts_emb + self.fm_modules["noise_scale_embedder"](ns_t).view(batch_size, itn, -1)
            embeds = embeds + ts_emb

            use_cfg = (t > cfg_interval[0] and t < cfg_interval[1]) or cfg_interval[0] == 0

            out_c = self._t2i_predict_v(
                embeds, idx_c, attn_none, past_kv_cond, t, z,
                image_token_num=itn, timestep_embeddings=ts_emb, image_size=image_size,
            )
            out_tu = (
                self._t2i_predict_v(
                    embeds, idx_tu, attn_none, past_kv_tu, t, z,
                    image_token_num=itn, timestep_embeddings=ts_emb, image_size=image_size,
                ) if past_kv_tu is not None and idx_tu is not None else None
            )
            out_iu = (
                self._t2i_predict_v(
                    embeds, idx_iu, attn_none, past_kv_iu, t, z,
                    image_token_num=itn, timestep_embeddings=ts_emb, image_size=image_size,
                ) if past_kv_iu is not None and idx_iu is not None else None
            )

            v = self._apply_cfg(out_c, cfg_scale, img_cfg_scale, cfg_norm, use_cfg, out_tu, out_iu)
            z = z + (t_next - t) * v
            img_pred = self.unpatchify(z, self.patch_size * merge_size, image_size[1], image_size[0])

        clear_flash_kv_cache(past_kv_cond)
        if past_kv_tu is not None:
            clear_flash_kv_cache(past_kv_tu)
        if past_kv_iu is not None:
            clear_flash_kv_cache(past_kv_iu)

        return img_pred

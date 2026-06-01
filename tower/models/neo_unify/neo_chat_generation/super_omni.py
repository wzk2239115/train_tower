from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..conversation import get_conv_template
from ..modeling_omni_decoders import AudioPatchEncoder
from ..modeling_neo_vit import build_abs_positions_from_grid_hw
from ..utils import load_image_native
from .utils import clear_flash_kv_cache, prepare_flash_kv_cache


class OmniGeneratorMixin:

    @torch.no_grad()
    def _denoise_audio(
        self,
        past_key_values_cond,
        past_key_values_tu,
        t_index_cond: int,
        t_index_tu: int,
        *,
        audio_duration_ms: int = 5000,
        audio_sample_rate: int = 24000,
        audio_n_mels: int = 80,
        audio_hop_size: int = 256,
        audio_patch_t: int = 4,
        num_steps: int = 30,
        cfg_scale: float = 1.0,
        t_eps: float = 0.02,
        seed: int = 0,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ):
        """Run rectified-flow denoising for audio in mel-patch latent space.

        Returns (mel_spectrogram, t_index_cond, t_index_tu, audio_token_num).
        """
        if device is None:
            device = self.device

        T_mel = int(audio_duration_ms / 1000.0 * audio_sample_rate / audio_hop_size)
        T_p = T_mel // audio_patch_t
        audio_token_num = T_p

        n_freq = audio_n_mels // 10
        # TODO: 训练数据里 audio token 的 position encoding 是顺序 t 索引,
        #       推理里用 t = t_index + 1 + t_audio // n_freq (按频段分组).
        #       两者需要一致, 否则 audio 生成质量会下降.
        t_audio = torch.arange(audio_token_num, dtype=torch.long, device=device)
        t_audio = t_audio_cond + 1 + t_audio // n_freq
        h_audio = torch.arange(audio_token_num, dtype=torch.long, device=device)
        h_audio = h_audio % n_freq
        w_audio = torch.zeros(audio_token_num, dtype=torch.long, device=device)
        indexes_audio_cond = torch.stack([t_audio, h_audio, w_audio], dim=0)

        t_audio_tu = torch.arange(audio_token_num, dtype=torch.long, device=device)
        t_audio_tu = t_index_tu + 1 + t_audio_tu // n_freq
        indexes_audio_tu = torch.stack([t_audio_tu, h_audio, w_audio], dim=0)

        patch_dim = audio_n_mels * audio_patch_t
        generator = torch.Generator(device).manual_seed(seed) if generator is None else generator
        z_patches = torch.randn(1, audio_token_num, patch_dim, device=device, dtype=torch.bfloat16, generator=generator)

        proj_weight = self.audio_latent_proj[0].weight.data[:, :patch_dim]
        proj_bias = self.audio_latent_proj[0].bias.data

        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

        for step_i in range(num_steps):
            t = timesteps[step_i]
            t_next = timesteps[step_i + 1]

            t_expanded = t.expand(audio_token_num)
            ts_emb = self.fm_modules["timestep_embedder"](t_expanded).view(1, audio_token_num, -1)

            flat = z_patches.view(audio_token_num, -1)
            embeds = F.linear(flat, proj_weight, proj_bias).view(1, audio_token_num, -1)
            embeds = embeds + ts_emb

            attn_mask = {"full_attention": None}

            out_cond = self._t2a_predict_v(
                embeds, indexes_audio_cond, attn_mask,
                past_key_values_cond, t, z_patches, audio_token_num,
            )

            if cfg_scale > 1.0:
                out_tu = self._t2a_predict_v(
                    embeds, indexes_audio_tu, attn_mask,
                    past_key_values_tu, t, z_patches, audio_token_num,
                )
                v_pred = out_tu + cfg_scale * (out_cond - out_tu)
            else:
                v_pred = out_cond

            z_patches = z_patches + (t_next - t) * v_pred

        mel_patches = z_patches.view(audio_token_num, patch_dim)
        pe = AudioPatchEncoder(n_mels=audio_n_mels, patch_t=audio_patch_t)
        mel = pe.unpatchify(mel_patches, T_mel)
        return mel, t_index_cond, t_index_tu, audio_token_num

    @torch.no_grad()
    def _denoise_video(
        self,
        past_key_values_cond,
        past_key_values_tu,
        t_index_cond: int,
        t_index_tu: int,
        *,
        num_frames: int = 16,
        frame_size: tuple[int, int] = (256, 256),
        num_steps: int = 30,
        cfg_scale: float = 1.0,
        t_eps: float = 0.02,
        seed: int = 0,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ):
        """Run rectified-flow denoising for video as per-frame pixel patches.

        Returns (video_frames_tensor, t_index_cond, t_index_tu, video_token_num).
        """
        if device is None:
            device = self.device

        merge_size = int(1 / self.downsample_ratio)
        token_h = frame_size[1] // (self.patch_size * merge_size)
        token_w = frame_size[0] // (self.patch_size * merge_size)
        patches_per_frame = token_h * token_w
        video_token_num = num_frames * patches_per_frame
        output_dim = 3 * (self.patch_size * merge_size) ** 2

        grid_h = frame_size[1] // self.patch_size
        grid_w = frame_size[0] // self.patch_size
        gen_grid_hw = torch.tensor([[grid_h, grid_w]], device=device)

        indexes_video_cond_list = []
        indexes_video_tu_list = []
        for f in range(num_frames):
            t_base_cond = t_index_cond + 1 + f
            t_base_tu = t_index_tu + 1 + f
            idx = torch.arange(patches_per_frame, device=device)
            h_idx = idx // token_w
            w_idx = idx % token_w
            indexes_video_cond_list.append(torch.stack([
                torch.full((patches_per_frame,), t_base_cond, dtype=torch.long, device=device),
                h_idx, w_idx,
            ], dim=0))
            indexes_video_tu_list.append(torch.stack([
                torch.full((patches_per_frame,), t_base_tu, dtype=torch.long, device=device),
                h_idx, w_idx,
            ], dim=0))
        indexes_video_cond = torch.cat(indexes_video_cond_list, dim=1)
        indexes_video_tu = torch.cat(indexes_video_tu_list, dim=1)

        generator = torch.Generator(device).manual_seed(seed) if generator is None else generator
        z_video = torch.randn(1, video_token_num, output_dim, device=device, dtype=torch.bfloat16, generator=generator)

        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

        for step_i in range(num_steps):
            t = timesteps[step_i]
            t_next = timesteps[step_i + 1]

            pixel_flat = z_video.view(num_frames * patches_per_frame, output_dim)

            c = 3
            ps = self.patch_size
            ms = merge_size
            frame_channels = c
            patch_px = ps * ms
            pixel_reshaped = pixel_flat.view(num_frames * patches_per_frame, c, ps, ms, ps, ms)
            pixel_reshaped = pixel_reshaped.permute(0, 2, 1, 3, 4, 5).reshape(
                num_frames * patches_per_frame, c * ps * ms * ps * ms
            )
            image_input = pixel_flat

            vit_embeds = self.extract_feature(
                image_input, gen_model=True, grid_hw=gen_grid_hw.repeat(num_frames, 1)
            ).view(1, video_token_num, -1)

            t_expanded = t.expand(video_token_num)
            ts_emb = self.fm_modules["timestep_embedder"](t_expanded).view(1, video_token_num, -1)
            if self.add_noise_scale_embedding:
                ns = min(
                    math.sqrt(video_token_num / 64.0),
                    float(self.noise_scale_max_value),
                )
                ns_tensor = torch.full_like(t_expanded, ns / self.noise_scale_max_value)
                ns_emb = self.fm_modules["noise_scale_embedder"](ns_tensor).view(1, video_token_num, -1)
                ts_emb = ts_emb + ns_emb
            video_embeds = vit_embeds + ts_emb

            attn_mask = {"full_attention": None}

            out_cond = self._t2v_predict_v(
                video_embeds, indexes_video_cond, attn_mask,
                past_key_values_cond, t, z_video, video_token_num,
            )

            if cfg_scale > 1.0:
                out_tu = self._t2v_predict_v(
                    video_embeds, indexes_video_tu, attn_mask,
                    past_key_values_tu, t, z_video, video_token_num,
                )
                v_pred = out_tu + cfg_scale * (out_cond - out_tu)
            else:
                v_pred = out_cond

            z_video = z_video + (t_next - t) * v_pred

        frames = self.video_frame_decoder.decode_frames(
            z_video.view(video_token_num, -1),
            num_frames=num_frames,
            frame_h=frame_size[1],
            frame_w=frame_size[0],
        )
        return frames, t_index_cond, t_index_tu, video_token_num

    @torch.no_grad()
    def super_omni_gen(
        self,
        tokenizer,
        prompt,
        images=None,
        cfg_scale: float = 1.0,
        img_cfg_scale: float = 1.0,
        image_size: tuple[int, int] | list[tuple[int, int]] = (256, 256),
        audio_duration_ms: int = 5000,
        audio_sample_rate: int = 24000,
        audio_n_mels: int = 80,
        audio_hop_size: int = 256,
        audio_num_steps: int = 30,
        video_num_frames: int = 16,
        video_frame_size: tuple[int, int] = (256, 256),
        video_num_steps: int = 30,
        num_steps: int = 30,
        max_images: int = 10,
        max_audios: int = 10,
        max_videos: int = 10,
        max_new_tokens: int = 8192,
        enable_timestep_shift: bool = True,
        timestep_shift: float = 1.0,
        t_eps: float = 0.02,
        cfg_interval: tuple[float, float] = (0, 1),
        cfg_norm: str = 'none',
        think_mode: bool = False,
        seed: int = 0,
        verbose: bool = False,
        IMG_START_TOKEN='<img>',
        IMG_END_TOKEN='</img>',
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
        AUDIO_START_TOKEN='<audio>',
        AUDIO_END_TOKEN='</audio>',
        AUDIO_CONTEXT_TOKEN='<AUDIO_CONTEXT>',
        VIDEO_START_TOKEN='<video>',
        VIDEO_END_TOKEN='</video>',
        VIDEO_CONTEXT_TOKEN='<VIDEO_CONTEXT>',
        system_message='',
    ):
        """Unified super-omni interleaved generation.

        Autoregressively generates text, and upon encountering special start
        tokens (``<img>``, ``<audio>``, ``<video>``) switches to the
        corresponding FM denoising loop, then injects the result back into
        the KV cache and continues AR generation.

        Returns a dict with keys:
            text: str — interleaved text with ``<image>``, ``<audio>``, ``<video>`` placeholders
            images: list[Tensor] — generated image tensors (B, 3, H, W)
            audios: list[Tensor] — generated mel spectrograms (B, n_mels, T)
            videos: list[Tensor] — generated video frame tensors (F, 3, H, W)
        """
        self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.audio_context_token_id = tokenizer.convert_tokens_to_ids(AUDIO_CONTEXT_TOKEN)
        self.audio_start_token_id = tokenizer.convert_tokens_to_ids(AUDIO_START_TOKEN)
        self.video_context_token_id = tokenizer.convert_tokens_to_ids(VIDEO_CONTEXT_TOKEN)
        self.video_start_token_id = tokenizer.convert_tokens_to_ids(VIDEO_START_TOKEN)
        self.config.t_eps = t_eps

        device = self.device
        merge_size = int(1 / self.downsample_ratio)
        generator = torch.Generator(device).manual_seed(seed)

        if isinstance(image_size, tuple):
            image_size_list = [image_size] * max_images
        elif isinstance(image_size, list):
            image_size_list = image_size + [image_size[-1]] * max(0, max_images - len(image_size))
        else:
            image_size_list = [(256, 256)] * max_images

        img_end_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        audio_end_id = tokenizer.convert_tokens_to_ids(AUDIO_END_TOKEN)
        video_end_id = tokenizer.convert_tokens_to_ids(VIDEO_END_TOKEN)
        audio_patch_t = int(getattr(self.config, 'audio_patch_t', 4))

        template = get_conv_template(self.template)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        if images is None:
            images = []

        image_token_count = prompt.count('<image>')
        if len(images) > image_token_count:
            prompt = "<image>\n" * (len(images) - image_token_count) + prompt

        pixel_values = []
        grid_hw = []
        for image in images:
            pv, ghw = load_image_native(
                image, self.patch_size, self.downsample_ratio,
                min_pixels=512 * 512,
                max_pixels=min(2048 * 2048, (4096 * 4096) // max(1, len(images))),
                upscale=False,
            )
            grid_hw.append(ghw.to(device))
            pixel_values.append(pv.to(device).to(torch.bfloat16))
        pv_tensor = torch.cat(pixel_values) if pixel_values else None
        ghw_tensor = torch.cat(grid_hw) if grid_hw else None

        template_cond = get_conv_template(self.template)
        template_cond.system_message = system_message
        template_cond.append_message(template_cond.roles[0], prompt)
        template_cond.append_message(template_cond.roles[1], None)
        query_cond = template_cond.get_prompt()
        if not think_mode:
            query_cond += 'tières\n\n</thinkèvres\n\n'

        def replace_img_tokens(query, ghw_list):
            for i in range(len(ghw_list)):
                npt = int(ghw_list[i][0, 0] * ghw_list[i][0, 1] * self.downsample_ratio ** 2)
                tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * npt + IMG_END_TOKEN
                query = query.replace('<image>', tokens, 1)
            return query

        query_cond = replace_img_tokens(query_cond, grid_hw)
        input_embeds_cond, indexes_cond, attention_mask_cond = self._build_it2i_inputs(
            tokenizer, query_cond, pv_tensor, ghw_tensor,
        )

        outputs_cond = self.language_model(
            inputs_embeds=input_embeds_cond, indexes=indexes_cond,
            attention_mask=attention_mask_cond, use_cache=True,
        )
        past_kv_cond = outputs_cond.past_key_values
        t_cond = indexes_cond[0].max().item()

        question_text_uncond = '<image>' * len(images)
        template_tu = get_conv_template(self.template)
        template_tu.system_message = self.system_message
        template_tu.append_message(template_tu.roles[0], question_text_uncond)
        template_tu.append_message(template_tu.roles[1], None)
        query_tu = template_tu.get_prompt()
        query_tu = replace_img_tokens(query_tu, grid_hw)
        ie_tu, idx_tu, am_tu = self._build_it2i_inputs(tokenizer, query_tu, pv_tensor, ghw_tensor)
        outputs_tu = self.language_model(inputs_embeds=ie_tu, indexes=idx_tu, attention_mask=am_tu, use_cache=True)
        past_kv_tu = outputs_tu.past_key_values
        t_tu = idx_tu[0].max().item()

        generated_text = ""
        generated_images = []
        generated_audios = []
        generated_videos = []
        img_count = 0
        audio_count = 0
        video_count = 0
        total_tokens = 0

        next_token = torch.argmax(outputs_cond.logits[:, -1, :], dim=-1)

        while True:
            gen_tokens = []
            hit_max = False
            last_decoded = 0

            while True:
                tok = next_token.item()
                trigger_tokens = {self.img_start_token_id, eos_token_id}
                if self.audio_start_token_id is not None:
                    trigger_tokens.add(self.audio_start_token_id)
                if self.video_start_token_id is not None:
                    trigger_tokens.add(self.video_start_token_id)

                if tok in trigger_tokens:
                    break

                gen_tokens.append(tok)
                total_tokens += 1

                self.language_model.model.current_index = t_cond
                outputs_cond = self.language_model(
                    input_ids=next_token.unsqueeze(0),
                    past_key_values=past_kv_cond, use_cache=True,
                )
                past_kv_cond = outputs_cond.past_key_values
                t_cond += 1
                next_token = torch.argmax(outputs_cond.logits[:, -1, :], dim=-1)

                if verbose and len(gen_tokens) - last_decoded >= 16:
                    partial = tokenizer.decode(gen_tokens[last_decoded:], skip_special_tokens=True)
                    print(partial, end='', flush=True)
                    last_decoded = len(gen_tokens)

                if total_tokens >= max_new_tokens:
                    hit_max = True
                    break

            if gen_tokens:
                chunk = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                generated_text += chunk
                if verbose:
                    remaining = tokenizer.decode(gen_tokens[last_decoded:], skip_special_tokens=True)
                    if remaining:
                        print(remaining, end='', flush=True)

            tok = next_token.item()
            if tok == eos_token_id or hit_max:
                break

            # ---------- IMAGE generation ----------
            if tok == self.img_start_token_id and img_count < max_images:
                generated_text += "<image>"
                if verbose:
                    print(f"\n[image {img_count + 1}] denoising...", flush=True)

                self.language_model.model.current_index = t_cond
                o = self.language_model(input_ids=next_token.unsqueeze(0), past_key_values=past_kv_cond, use_cache=True)
                past_kv_cond = o.past_key_values
                t_cond += 1
                self.language_model.model.current_index = t_tu
                o_tu = self.language_model(input_ids=next_token.unsqueeze(0), past_key_values=past_kv_tu, use_cache=True)
                past_kv_tu = o_tu.past_key_values
                t_tu += 1

                cur_size = image_size_list[img_count]
                token_h = cur_size[1] // (self.patch_size * merge_size)
                token_w = cur_size[0] // (self.patch_size * merge_size)
                idx_img_cond = self._build_t2i_image_indexes(token_h, token_w, t_cond + 1, device=device)
                idx_img_tu = self._build_t2i_image_indexes(token_h, token_w, t_tu + 1, device=device)

                grid_h = cur_size[1] // self.patch_size
                grid_w = cur_size[0] // self.patch_size
                gen_ghw = torch.tensor([[grid_h, grid_w]], device=device)
                ns = self.noise_scale
                if self.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
                    base = float(self.noise_scale_base_image_seq_len)
                    ns = math.sqrt((grid_h * grid_w) / (merge_size ** 2) / base) * float(self.noise_scale)
                    if self.noise_scale_mode == 'dynamic_sqrt':
                        ns = math.sqrt(ns)
                ns = min(ns, self.noise_scale_max_value)

                img_pred = ns * torch.randn(
                    (1, 3, cur_size[1], cur_size[0]),
                    device=device, dtype=torch.bfloat16, generator=generator,
                )

                prepare_flash_kv_cache(past_kv_cond, current_len=token_h * token_w, batch_size=1)
                prepare_flash_kv_cache(past_kv_tu, current_len=token_h * token_w, batch_size=1)

                timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
                if enable_timestep_shift:
                    timesteps = self._apply_time_schedule(timesteps, token_h * token_w, timestep_shift)

                for si in range(num_steps):
                    t = timesteps[si]
                    t_next = timesteps[si + 1]
                    z = self.patchify(img_pred, self.patch_size * merge_size)
                    img_in = self.patchify(img_pred, self.patch_size, channel_first=True)
                    embeds = self.extract_feature(
                        img_in.view(grid_h * grid_w, -1), gen_model=True, grid_hw=gen_ghw,
                    ).view(1, token_h * token_w, -1)
                    t_exp = t.expand(token_h * token_w)
                    ts_emb = self.fm_modules['timestep_embedder'](t_exp).view(1, token_h * token_w, -1)
                    if self.add_noise_scale_embedding:
                        ns_t = torch.full_like(t_exp, ns / self.noise_scale_max_value)
                        ts_emb = ts_emb + self.fm_modules['noise_scale_embedder'](ns_t).view(1, token_h * token_w, -1)
                    embeds = embeds + ts_emb

                    use_cfg = (t > cfg_interval[0] and t < cfg_interval[1]) or cfg_interval[0] == 0
                    out_c = self._t2i_predict_v(
                        embeds, idx_img_cond, {"full_attention": None},
                        past_kv_cond, t, z, image_token_num=token_h * token_w,
                    )
                    if use_cfg and cfg_scale > 1:
                        out_tu2 = self._t2i_predict_v(
                            embeds, idx_img_tu, {"full_attention": None},
                            past_kv_tu, t, z, image_token_num=token_h * token_w,
                        )
                        v = out_tu2 + cfg_scale * (out_c - out_tu2)
                    else:
                        v = out_c

                    z = z + (t_next - t) * v
                    img_pred = self.unpatchify(z, self.patch_size * merge_size, cur_size[1], cur_size[0])

                generated_images.append(img_pred)
                clear_flash_kv_cache(past_kv_cond)
                clear_flash_kv_cache(past_kv_tu)
                img_count += 1

                pred_img = img_pred[0].unsqueeze(0).to(torch.bfloat16)
                raw = pred_img * 0.5 + 0.5
                img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=raw.dtype, device=device).view(1, 3, 1, 1)
                img_std = torch.tensor([0.229, 0.224, 0.225], dtype=raw.dtype, device=device).view(1, 3, 1, 1)
                und = (raw - img_mean) / img_std
                c_dim, h_px, w_px = und[0].shape
                ps = self.patch_size
                p_gh = h_px // ps
                p_gw = w_px // ps
                flat_pv = und[0].view(c_dim, p_gh, ps, p_gw, ps).permute(1, 3, 0, 2, 4).reshape(p_gh * p_gw, c_dim * ps ** 2)
                vit_e = self.extract_feature(flat_pv, grid_hw=gen_ghw[:1]).unsqueeze(0)

                end_embed = self.language_model.get_input_embeddings()(torch.tensor([[img_end_id]], device=device))
                n_img_tok = vit_e.shape[1]
                abs_w, abs_h = build_abs_positions_from_grid_hw(
                    gen_ghw[:1] // merge_size, device=device,
                )

                o_c2, t_cond = self._append_modality_to_cache(
                    past_kv_cond, t_cond, vit_e, n_img_tok, abs_h, abs_w, end_embed[0, 0], device,
                )
                o_tu2, t_tu = self._append_modality_to_cache(
                    past_kv_tu, t_tu, vit_e, n_img_tok, abs_h, abs_w, end_embed[0, 0], device,
                )
                next_token = torch.argmax(o_c2.logits[:, -1, :], dim=-1)

            # ---------- AUDIO generation ----------
            elif tok == self.audio_start_token_id and audio_count < max_audios and self.audio_context_token_id is not None:
                generated_text += "<audio>"
                if verbose:
                    print(f"\n[audio {audio_count + 1}] denoising...", flush=True)

                self.language_model.model.current_index = t_cond
                o = self.language_model(input_ids=next_token.unsqueeze(0), past_key_values=past_kv_cond, use_cache=True)
                past_kv_cond = o.past_key_values
                t_cond += 1
                self.language_model.model.current_index = t_tu
                o_tu = self.language_model(input_ids=next_token.unsqueeze(0), past_key_values=past_kv_tu, use_cache=True)
                past_kv_tu = o_tu.past_key_values
                t_tu += 1

                mel, _, _, audio_tok_n = self._denoise_audio(
                    past_kv_cond, past_kv_tu, t_cond, t_tu,
                    audio_duration_ms=audio_duration_ms,
                    audio_sample_rate=audio_sample_rate,
                    audio_n_mels=audio_n_mels,
                    audio_hop_size=audio_hop_size,
                    audio_patch_t=audio_patch_t,
                    num_steps=audio_num_steps,
                    cfg_scale=cfg_scale,
                    t_eps=t_eps,
                    seed=seed + audio_count + 1,
                    device=device,
                    generator=generator,
                )
                generated_audios.append(mel)
                audio_count += 1

                audio_patch_dim = audio_n_mels * audio_patch_t
                audio_hid = self.config.llm_config.hidden_size
                context_embeds_raw = mel.view(-1, audio_patch_dim)[:audio_tok_n]
                w_proj = self.audio_latent_proj[0].weight.data[:, :audio_patch_dim]
                b_proj = self.audio_latent_proj[0].bias.data
                context_embeds = (context_embeds_raw @ w_proj.T + b_proj).unsqueeze(0)

                t_cond += audio_tok_n
                t_tu += audio_tok_n

                end_embed = self.language_model.get_input_embeddings()(torch.tensor([[audio_end_id]], device=device))
                h_z = torch.zeros(audio_tok_n, dtype=torch.long, device=device)
                w_z = torch.zeros(audio_tok_n, dtype=torch.long, device=device)

                o_c3, t_cond = self._append_modality_to_cache(
                    past_kv_cond, t_cond, context_embeds, audio_tok_n, h_z, w_z, end_embed[0, 0], device,
                )
                o_tu3, t_tu = self._append_modality_to_cache(
                    past_kv_tu, t_tu, context_embeds, audio_tok_n, h_z, w_z, end_embed[0, 0], device,
                )
                next_token = torch.argmax(o_c3.logits[:, -1, :], dim=-1)

            # ---------- VIDEO generation ----------
            elif tok == self.video_start_token_id and video_count < max_videos and self.video_context_token_id is not None:
                generated_text += "<video>"
                if verbose:
                    print(f"\n[video {video_count + 1}] denoising...", flush=True)

                self.language_model.model.current_index = t_cond
                o = self.language_model(input_ids=next_token.unsqueeze(0), past_key_values=past_kv_cond, use_cache=True)
                past_kv_cond = o.past_key_values
                t_cond += 1
                self.language_model.model.current_index = t_tu
                o_tu = self.language_model(input_ids=next_token.unsqueeze(0), past_key_values=past_kv_tu, use_cache=True)
                past_kv_tu = o_tu.past_key_values
                t_tu += 1

                frames, _, _, video_tok_n = self._denoise_video(
                    past_kv_cond, past_kv_tu, t_cond, t_tu,
                    num_frames=video_num_frames,
                    frame_size=video_frame_size,
                    num_steps=video_num_steps,
                    cfg_scale=cfg_scale,
                    t_eps=t_eps,
                    seed=seed + video_count + 1,
                    device=device,
                    generator=generator,
                )
                generated_videos.append(frames)
                video_count += 1

                grid_h = video_frame_size[1] // self.patch_size
                grid_w = video_frame_size[0] // self.patch_size
                gen_ghw = torch.tensor([[grid_h, grid_w]], device=device)
                token_h = video_frame_size[1] // (self.patch_size * merge_size)
                token_w = video_frame_size[0] // (self.patch_size * merge_size)
                patches_per_frame = token_h * token_w

                raw = frames * 0.5 + 0.5
                img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=raw.dtype, device=device).view(1, 3, 1, 1)
                img_std = torch.tensor([0.229, 0.224, 0.225], dtype=raw.dtype, device=device).view(1, 3, 1, 1)
                und_frames = (raw - img_mean) / img_std

                all_vit_embeds = []
                for f_i in range(frames.shape[0]):
                    c_dim, h_px, w_px = und_frames[f_i].shape
                    ps = self.patch_size
                    p_gh = h_px // ps
                    p_gw = w_px // ps
                    flat = und_frames[f_i].view(c_dim, p_gh, ps, p_gw, ps).permute(1, 3, 0, 2, 4).reshape(p_gh * p_gw, c_dim * ps ** 2)
                    ve = self.extract_feature(flat, grid_hw=gen_ghw[:1])
                    all_vit_embeds.append(ve)
                all_vit = torch.cat(all_vit_embeds, dim=0).unsqueeze(0)

                t_cond += video_tok_n
                t_tu += video_tok_n

                end_embed = self.language_model.get_input_embeddings()(torch.tensor([[video_end_id]], device=device))
                n_v = all_vit.shape[1]
                abs_w_v, abs_h_v = build_abs_positions_from_grid_hw(
                    gen_ghw[:1] // merge_size, device=device,
                )
                abs_h_v_expanded = abs_h_v.repeat(video_num_frames)
                abs_w_v_expanded = abs_w_v.repeat(video_num_frames)

                o_c4, t_cond = self._append_modality_to_cache(
                    past_kv_cond, t_cond, all_vit, n_v, abs_h_v_expanded, abs_w_v_expanded, end_embed[0, 0], device,
                )
                o_tu4, t_tu = self._append_modality_to_cache(
                    past_kv_tu, t_tu, all_vit, n_v, abs_h_v_expanded, abs_w_v_expanded, end_embed[0, 0], device,
                )
                next_token = torch.argmax(o_c4.logits[:, -1, :], dim=-1)

            else:
                break

        return {
            "text": generated_text,
            "images": generated_images,
            "audios": generated_audios,
            "videos": generated_videos,
        }

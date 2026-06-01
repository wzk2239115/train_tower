from __future__ import annotations
import math
import torch
from ..conversation import get_conv_template
from ..modeling_qwen3 import create_block_causal_mask
from ..utils import load_image_native, SYSTEM_MESSAGE_FOR_GEN
from .utils import prepare_flash_kv_cache, clear_flash_kv_cache


class IT2IGeneratorMixin:

    def _it2i_prefix_forward(self, input_imbeds, indexes, attention_mask, gen_indicators=None):
        out = self.language_model.model(
            inputs_embeds=input_imbeds,
            indexes=indexes,
            attention_mask=attention_mask,
            use_cache=True,
            image_gen_indicators=gen_indicators.view(1, -1) if gen_indicators is not None else None
        )
        return out.past_key_values, out.last_hidden_state

    def _build_it2i_inputs(self, tokenizer, query, pixel_values=None, grid_hw=None):
        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)

        indexes = self.get_thw_indexes(input_ids[0], grid_hw)

        attention_mask = {"full_attention": create_block_causal_mask(indexes[0])}

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        if pixel_values is not None:
            vit_embeds = self.extract_feature(pixel_values, grid_hw=grid_hw)
            input_embeds = input_embeds.reshape(B * N, C)
            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            input_embeds = input_embeds.reshape(B, N, C)

        return input_embeds, indexes, attention_mask

    @torch.no_grad()
    def it2i_generate(self, tokenizer, prompt, images, cfg_scale=1, img_cfg_scale=1, cfg_norm='none', enable_timestep_shift=True, timestep_shift=1, image_size=(256, 256), num_steps=30, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', method='euler', cfg_interval=(0, 1), batch_size=1, t_eps=0.02, think_mode=False, seed=0):
        assert cfg_norm in ['none', 'global', 'channel']

        self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.config.t_eps = t_eps

        image_token_count = prompt.count('<image>')
        assert len(images) >= image_token_count
        if len(images) > image_token_count:
            if image_token_count == 0 and len(images) > 1:
                prompt = "".join(f"Image-{i + 1}:<image>\n" for i in range(len(images))) + prompt
            else:
                prompt = "<image>\n" * (len(images) - image_token_count) + prompt

        pixel_values = []
        grid_hw = []
        for image in images:
            cur_pixel_values, cur_grid_hw = load_image_native(
                image,
                self.patch_size,
                self.downsample_ratio,
                min_pixels=512 * 512,
                max_pixels=min(2048*2048, (4096 * 4096) // len(images)),
                upscale=False,
            )
            cur_grid_hw = cur_grid_hw.to(self.device)
            cur_pixel_values = cur_pixel_values.to(self.device).to(torch.bfloat16)
            pixel_values.append(cur_pixel_values)
            grid_hw.append(cur_grid_hw)
        pixel_values = torch.cat(pixel_values)
        grid_hw = torch.cat(grid_hw)

        merge_size = int(1 / self.downsample_ratio)
        question_condition = f"{prompt}"
        think_text = ""
        needs_cfg = not (cfg_scale == 1 and img_cfg_scale == 1)
        needs_img_condition = needs_cfg and (img_cfg_scale == 1 or cfg_scale != img_cfg_scale)
        needs_uncondition = needs_cfg and img_cfg_scale != 1

        think_content = '<think>\n' if think_mode else '<think>\n\n</think>\n\n' + IMG_START_TOKEN
        query_condition = self._build_t2i_query(question_condition, system_message=SYSTEM_MESSAGE_FOR_GEN, append_text=think_content)
        query_img_condition = (
            self._build_t2i_query('<image>' * len(images), append_text=IMG_START_TOKEN)
            if needs_img_condition
            else None
        )
        query_uncondition = self._build_t2i_query("", append_text=IMG_START_TOKEN) if needs_uncondition else None

        for i in range(grid_hw.shape[0]):
            num_patch_token = int(grid_hw[i, 0] * grid_hw[i, 1] * self.downsample_ratio**2)
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_patch_token + IMG_END_TOKEN
            query_condition = query_condition.replace('<image>', image_tokens, 1)
            if query_img_condition is not None:
                query_img_condition = query_img_condition.replace('<image>', image_tokens, 1)

        input_embeds_condition, indexes_condition, attention_mask_condition_prefix = self._build_it2i_inputs(
            tokenizer, query_condition, pixel_values, grid_hw
        )
        if query_img_condition is not None:
            input_embeds_img_condition, indexes_img_condition, attention_mask_img_condition_prefix = self._build_it2i_inputs(
                tokenizer, query_img_condition, pixel_values, grid_hw
            )
        else:
            input_embeds_img_condition = indexes_img_condition = attention_mask_img_condition_prefix = None
        if query_uncondition is not None:
            input_embeds_uncondition, indexes_uncondition, attention_mask_uncondition_prefix = self._build_it2i_inputs(
                tokenizer, query_uncondition
            )
        else:
            input_embeds_uncondition = indexes_uncondition = attention_mask_uncondition_prefix = None

        token_h = image_size[1] // (self.patch_size * merge_size)
        token_w = image_size[0] // (self.patch_size * merge_size)

        indexes_image_condition = self._build_t2i_image_indexes(
            token_h, token_w, indexes_condition[0].max() + 1, device=input_embeds_condition.device
        )
        indexes_image_img_condition = (
            self._build_t2i_image_indexes(
                token_h, token_w, indexes_img_condition[0].max() + 1, device=input_embeds_img_condition.device
            )
            if indexes_img_condition is not None
            else None
        )
        indexes_image_uncondition = (
            self._build_t2i_image_indexes(
                token_h, token_w, indexes_uncondition[0].max() + 1, device=input_embeds_uncondition.device
            )
            if indexes_uncondition is not None
            else None
        )

        if think_mode:
            outputs_condition = self.language_model(
                inputs_embeds=input_embeds_condition,
                indexes=indexes_condition,
                attention_mask=attention_mask_condition_prefix,
                use_cache=True,
                output_hidden_states=True,
            )
            past_key_values_condition = outputs_condition.past_key_values
            hidden_states_condition = outputs_condition.hidden_states[-1]
            t_index_condition = indexes_condition[0].max().item()
            past_key_values_condition, t_index_condition, think_text = self._generate_think(
                tokenizer,
                outputs_condition,
                past_key_values_condition,
                t_index_condition,
                IMG_START_TOKEN,
            )
            indexes_image_condition = self._build_t2i_image_indexes(
                token_h, token_w, t_index_condition + 1, device=input_embeds_condition.device
            )
        else:
            past_key_values_condition, hidden_states_condition = self._it2i_prefix_forward(
                input_embeds_condition, indexes_condition, attention_mask_condition_prefix
            )
        past_key_values_img_condition = None
        if input_embeds_img_condition is not None:
            past_key_values_img_condition, _ = self._it2i_prefix_forward(
                input_embeds_img_condition, indexes_img_condition, attention_mask_img_condition_prefix
            )
        past_key_values_uncondition = None
        if input_embeds_uncondition is not None:
            past_key_values_uncondition, _ = self._it2i_prefix_forward(
                input_embeds_uncondition, indexes_uncondition, attention_mask_uncondition_prefix
            )

        device = hidden_states_condition.device
        dtype = hidden_states_condition.dtype

        del pixel_values, grid_hw
        del input_embeds_condition, indexes_condition, attention_mask_condition_prefix
        if input_embeds_img_condition is not None:
            del input_embeds_img_condition, indexes_img_condition, attention_mask_img_condition_prefix
        if input_embeds_uncondition is not None:
            del input_embeds_uncondition, indexes_uncondition, attention_mask_uncondition_prefix
        del hidden_states_condition

        for layer_idx in range(len(past_key_values_condition.layers)):
            past_key_values_condition.layers[layer_idx].keys = past_key_values_condition.layers[layer_idx].keys.expand(
                batch_size, *past_key_values_condition.layers[layer_idx].keys.shape[1:]
            )
            past_key_values_condition.layers[layer_idx].values = past_key_values_condition.layers[layer_idx].values.expand(
                batch_size, *past_key_values_condition.layers[layer_idx].values.shape[1:]
            )
            if past_key_values_img_condition is not None:
                past_key_values_img_condition.layers[layer_idx].keys = past_key_values_img_condition.layers[layer_idx].keys.expand(
                    batch_size, *past_key_values_img_condition.layers[layer_idx].keys.shape[1:]
                )
                past_key_values_img_condition.layers[layer_idx].values = past_key_values_img_condition.layers[layer_idx].values.expand(
                    batch_size, *past_key_values_img_condition.layers[layer_idx].values.shape[1:]
                )
            if past_key_values_uncondition is not None:
                past_key_values_uncondition.layers[layer_idx].keys = past_key_values_uncondition.layers[layer_idx].keys.expand(
                    batch_size, *past_key_values_uncondition.layers[layer_idx].keys.shape[1:]
                )
                past_key_values_uncondition.layers[layer_idx].values = past_key_values_uncondition.layers[layer_idx].values.expand(
                    batch_size, *past_key_values_uncondition.layers[layer_idx].values.shape[1:]
                )

        prepare_flash_kv_cache(
            past_key_values_condition,
            current_len=token_h * token_w,
            batch_size=batch_size,
        )
        if past_key_values_img_condition is not None:
            prepare_flash_kv_cache(
                past_key_values_img_condition,
                current_len=token_h * token_w,
                batch_size=batch_size,
            )
        if past_key_values_uncondition is not None:
            prepare_flash_kv_cache(
                past_key_values_uncondition,
                current_len=token_h * token_w,
                batch_size=batch_size,
            )

        grid_h = image_size[1] // self.patch_size
        grid_w = image_size[0] // self.patch_size
        grid_hw = torch.tensor([[grid_h, grid_w]] * batch_size, device=device)

        noise_scale = self.noise_scale
        if self.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
            base = float(self.noise_scale_base_image_seq_len)
            scale = math.sqrt((grid_h * grid_w) / (merge_size**2) / base)
            noise_scale = scale * float(self.noise_scale)
            if self.noise_scale_mode == 'dynamic_sqrt':
                noise_scale = math.sqrt(noise_scale)
        noise_scale = min(noise_scale, self.noise_scale_max_value)
        generator = torch.Generator(device).manual_seed(seed)
        image_prediction = noise_scale * torch.randn(
            (batch_size, 3, image_size[1], image_size[0]), device=device, dtype=dtype, generator=generator
        )

        attention_mask_condition = {"full_attention": None}
        attention_mask_img_condition = {"full_attention": None}
        attention_mask_uncondition = {"full_attention": None}

        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        if enable_timestep_shift:
            timesteps = self._apply_time_schedule(timesteps, token_h * token_w, timestep_shift)

        for step_i in range(num_steps):
            t = timesteps[step_i]
            t_next = timesteps[step_i + 1]
            use_cfg = (t > cfg_interval[0] and t < cfg_interval[1]) or cfg_interval[0] == 0

            z = self.patchify(image_prediction, self.patch_size * merge_size)
            image_input = self.patchify(image_prediction, self.patch_size, channel_first=True)
            image_embeds = self.extract_feature(
                image_input.view(batch_size * grid_h * grid_w, -1),
                gen_model=True,
                grid_hw=grid_hw,
            ).view(batch_size, token_h * token_w, -1)
            t_expanded = t.expand(batch_size * token_h * token_w)
            timestep_embeddings = self.fm_modules['timestep_embedder'](t_expanded).view(batch_size, token_h * token_w, -1)
            if self.add_noise_scale_embedding:
                noise_scale_tensor = torch.full_like(t_expanded, noise_scale / self.noise_scale_max_value)
                noise_embeddings = self.fm_modules['noise_scale_embedder'](noise_scale_tensor).view(batch_size, token_h * token_w, -1)
                timestep_embeddings += noise_embeddings
            image_embeds = image_embeds + timestep_embeddings

            out_cond = self._t2i_predict_v(
                image_embeds,
                indexes_image_condition,
                attention_mask_condition,
                past_key_values_condition,
                t,
                z,
                image_token_num=token_h * token_w,
                timestep_embeddings=timestep_embeddings,
                image_size=image_size,
            )

            if not use_cfg:
                v_pred = out_cond
            elif cfg_scale == 1 and img_cfg_scale == 1:
                v_pred = out_cond
            elif img_cfg_scale == 1:
                out_img_cond = self._t2i_predict_v(
                    image_embeds,
                    indexes_image_img_condition,
                    attention_mask_img_condition,
                    past_key_values_img_condition,
                    t,
                    z,
                    image_token_num=token_h * token_w,
                    timestep_embeddings=timestep_embeddings,
                    image_size=image_size,
                )
                v_pred = out_img_cond + cfg_scale * (out_cond - out_img_cond)
            elif cfg_scale == img_cfg_scale:
                out_uncond = self._t2i_predict_v(
                    image_embeds,
                    indexes_image_uncondition,
                    attention_mask_uncondition,
                    past_key_values_uncondition,
                    t,
                    z,
                    image_token_num=token_h * token_w,
                    timestep_embeddings=timestep_embeddings,
                    image_size=image_size,
                )
                v_pred = out_uncond + cfg_scale * (out_cond - out_uncond)
            else:
                out_img_cond = self._t2i_predict_v(
                    image_embeds,
                    indexes_image_img_condition,
                    attention_mask_img_condition,
                    past_key_values_img_condition,
                    t,
                    z,
                    image_token_num=token_h * token_w,
                    timestep_embeddings=timestep_embeddings,
                    image_size=image_size,
                )
                out_uncond = self._t2i_predict_v(
                    image_embeds,
                    indexes_image_uncondition,
                    attention_mask_uncondition,
                    past_key_values_uncondition,
                    t,
                    z,
                    image_token_num=token_h * token_w,
                    timestep_embeddings=timestep_embeddings,
                    image_size=image_size,
                )
                v_pred = (
                    out_uncond
                    + cfg_scale * (out_cond - out_img_cond)
                    + img_cfg_scale * (out_img_cond - out_uncond)
                )
            if (cfg_scale > 1 or img_cfg_scale > 1) and use_cfg:
                if cfg_norm == 'global':
                    norm_v_condition = torch.norm(out_cond, dim=(1, 2), keepdim=True)
                    norm_v_cfg = torch.norm(v_pred, dim=(1, 2), keepdim=True)
                    scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                    v_pred = v_pred * scale
                elif cfg_norm == 'channel':
                    norm_v_condition = torch.norm(out_cond, dim=-1, keepdim=True)
                    norm_v_cfg = torch.norm(v_pred, dim=-1, keepdim=True)
                    scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                    v_pred = v_pred * scale

            z = z + (t_next - t) * v_pred
            image_prediction = self.unpatchify(z, self.patch_size * merge_size, image_size[1], image_size[0])

        clear_flash_kv_cache(past_key_values_condition)
        if past_key_values_img_condition is not None:
            clear_flash_kv_cache(past_key_values_img_condition)
        if past_key_values_uncondition is not None:
            clear_flash_kv_cache(past_key_values_uncondition)

        self.last_think_content = think_text
        if think_mode:
            return image_prediction, think_text
        return image_prediction


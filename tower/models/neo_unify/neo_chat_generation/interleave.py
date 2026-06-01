from __future__ import annotations

import math

import torch

from ..conversation import get_conv_template
from ..modeling_qwen3 import create_block_causal_mask
from ..modeling_neo_vit import build_abs_positions_from_grid_hw
from ..utils import load_image_native
from .utils import prepare_flash_kv_cache, clear_flash_kv_cache


class InterleavedGeneratorMixin:

    @torch.no_grad()
    def interleave_gen_image_only(
            self,
            tokenizer,
            prompt,
            gt_text,
            images=None,
            gt_images=None,
            cfg_scale=1.0,
            img_cfg_scale=1.0,
            cfg_norm='none',
            max_images=10,
            enable_timestep_shift=True,
            timestep_shift=1.0,
            image_size=(256, 256),
            num_steps=30,
            IMG_START_TOKEN='<img>',
            IMG_END_TOKEN='</img>',
            IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
            method='euler',
            cfg_interval=(0, 1),
            t_eps=0.02,
            verbose=False,
            system_message='',
    ):
        self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.config.t_eps = t_eps

        if isinstance(image_size, tuple):
            image_size_list = [image_size] * max_images
        elif isinstance(image_size, list) and isinstance(image_size[0], tuple):
            image_size_list = image_size
            if len(image_size) < max_images:
                image_size_list += [image_size_list[-1]] * (max_images - len(image_size_list))
        else:
            assert False, "image size should be a tuple or a list of tuple"

        if images is None:
            images =[]

        image_token_count = prompt.count('<image>')
        assert len(images) >= image_token_count
        if len(images) > image_token_count:
            prompt = "<image>\n" * (len(images) - image_token_count) + prompt

        pixel_values =[]
        grid_hw =[]
        for image in images:
            cur_pixel_values, cur_grid_hw = load_image_native(image, self.patch_size, self.downsample_ratio, min_pixels=512*512, max_pixels=min(2048*2048, (4096*4096)//max(1, len(images))), upscale=False)
            grid_hw.append(cur_grid_hw.to(self.device))
            pixel_values.append(cur_pixel_values.to(self.device).to(torch.bfloat16))

        merge_size = int(1 / self.downsample_ratio)
        pv_tensor = torch.cat(pixel_values) if pixel_values else None
        ghw_tensor = torch.cat(grid_hw) if grid_hw else None

        # Condition Initial Cache
        template_cond = get_conv_template(self.template)
        template_cond.system_message = system_message
        template_cond.append_message(template_cond.roles[0], prompt)
        template_cond.append_message(template_cond.roles[1], None)
        query_cond = template_cond.get_prompt() + ' оку\n\n ок\n\n'

        def replace_image_tokens(query, grid_hw_list):
            for i in range(len(grid_hw_list)):
                num_patch_token = int(grid_hw_list[i][0, 0] * grid_hw_list[i][0, 1] * self.downsample_ratio**2)
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_patch_token + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)
            return query

        query_cond = replace_image_tokens(query_cond, grid_hw)
        input_embeds_cond, indexes_cond, attention_mask_cond = self._build_it2i_inputs(tokenizer, query_cond, pv_tensor, ghw_tensor)
        
        outputs_cond = self.language_model(inputs_embeds=input_embeds_cond, indexes=indexes_cond, attention_mask=attention_mask_cond, use_cache=True)
        past_key_values_cond = outputs_cond.past_key_values
        t_index_cond = indexes_cond[0].max().item()

        # Text Uncondition Cache Initial
        question_text_uncondition = '<image>' * len(images)
        template_tu = get_conv_template(self.template)
        template_tu.system_message = self.system_message
        template_tu.append_message(template_tu.roles[0], question_text_uncondition)
        template_tu.append_message(template_tu.roles[1], None)
        query_text_uncond = template_tu.get_prompt()
        query_text_uncond = replace_image_tokens(query_text_uncond, grid_hw)

        input_embeds_tu, indexes_tu, attention_mask_tu = self._build_it2i_inputs(tokenizer, query_text_uncond, pv_tensor, ghw_tensor)
        outputs_tu = self.language_model(inputs_embeds=input_embeds_tu, indexes=indexes_tu, attention_mask=attention_mask_tu, use_cache=True)
        past_key_values_tu = outputs_tu.past_key_values
        t_index_tu = indexes_tu[0].max().item()

        # Img Uncondition Cache Initial
        query_img_uncond = self._build_t2i_query("", append_text=IMG_START_TOKEN)
        input_embeds_iu, indexes_iu, attention_mask_iu = self._build_it2i_inputs(tokenizer, query_img_uncond)
        outputs_iu = self.language_model(inputs_embeds=input_embeds_iu, indexes=indexes_iu, attention_mask=attention_mask_iu, use_cache=True)
        past_key_values_iu = outputs_iu.past_key_values


        generated_images =[]
        img_count = 0
        device = self.device

        def append_ids_to_cache(cache, t_idx, input_ids):
            if input_ids.shape[1] == 0:
                return t_idx
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

        def append_image_to_cache(cache, t_idx, inputs_embeds_img, N_img_tokens, abs_pos_w, abs_pos_h):
            past_len = cache.get_seq_length()
            tgt_len = N_img_tokens + 1
            
            t_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
            t_indexes[:N_img_tokens] = t_idx + 1
            t_indexes[N_img_tokens] = t_idx + 2
            
            h_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
            w_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
            h_indexes[:N_img_tokens] = abs_pos_h
            w_indexes[:N_img_tokens] = abs_pos_w
            
            indexes = torch.stack([t_indexes, h_indexes, w_indexes], dim=0)
            
            mask = torch.zeros(1, 1, tgt_len, past_len + tgt_len, device=device)
            mask[0, 0, :N_img_tokens, past_len + N_img_tokens] = float('-inf')
            attention_mask_dict = {"full_attention": mask}
            
            self.language_model(
                inputs_embeds=inputs_embeds_img,
                indexes=indexes,
                attention_mask=attention_mask_dict,
                past_key_values=cache,
                use_cache=True
            )
            return t_idx + 2

        parts = gt_text.split('<image>')
        img_start_id_tensor = torch.tensor([[self.img_start_token_id]], device=device)

        for i, part in enumerate(parts):
            if len(part) > 0:
                if verbose:
                    print(part, end='', flush=True)
                part_ids = tokenizer(part, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
                t_index_cond = append_ids_to_cache(past_key_values_cond, t_index_cond, part_ids)

            if i < len(parts) - 1:
                if img_count >= max_images:
                    break
                    
                if verbose:
                    print("<image>", end='', flush=True)

                t_index_cond = append_ids_to_cache(past_key_values_cond, t_index_cond, img_start_id_tensor)
                t_index_tu = append_ids_to_cache(past_key_values_tu, t_index_tu, img_start_id_tensor)

                cur_image_size = image_size_list[img_count]
                token_h = cur_image_size[1] // (self.patch_size * merge_size)
                token_w = cur_image_size[0] // (self.patch_size * merge_size)

                indexes_image_condition = self._build_t2i_image_indexes(token_h, token_w, t_index_cond + 1, device=device)
                indexes_image_text_uncondition = self._build_t2i_image_indexes(token_h, token_w, t_index_tu + 1, device=device)
                indexes_image_img_uncondition = self._build_t2i_image_indexes(token_h, token_w, indexes_iu[0].max() + 1, device=device)

                grid_h = cur_image_size[1] // self.patch_size
                grid_w = cur_image_size[0] // self.patch_size
                gen_grid_hw = torch.tensor([[grid_h, grid_w]], device=device)

                noise_scale = self.noise_scale
                if self.noise_scale_mode in ("resolution", "dynamic", 'dynamic_sqrt'):
                    noise_scale = math.sqrt((grid_h*grid_w)/(merge_size**2) / self.noise_scale_base_image_seq_len)
                    base = float(self.noise_scale_base_image_seq_len)
                    noise_scale = math.sqrt((grid_h*grid_w)/(merge_size**2)/base) * float(self.noise_scale)
                    if self.noise_scale_mode == 'dynamic_sqrt':
                        noise_scale = math.sqrt(noise_scale)
                noise_scale = min(noise_scale, self.noise_scale_max_value)
                image_prediction = noise_scale * torch.randn((1, 3, cur_image_size[1], cur_image_size[0]), device=device, dtype=outputs_cond.logits.dtype)

                past_key_values_cond_cfg = past_key_values_cond
                past_key_values_tu_cfg = past_key_values_tu
                past_key_values_iu_cfg = past_key_values_iu

                # attention_mask_condition = {"full_attention": torch.zeros(1, 1, token_h*token_w, past_key_values_cond.get_seq_length() + token_h*token_w, device=device)}
                # attention_mask_text_uncondition = {"full_attention": torch.zeros(1, 1, token_h*token_w, past_key_values_tu.get_seq_length() + token_h*token_w, device=device)}
                # attention_mask_img_uncondition = {"full_attention": torch.zeros(1, 1, token_h*token_w, past_key_values_iu.get_seq_length() + token_h*token_w, device=device)}
                attention_mask_condition = {"full_attention": None}
                attention_mask_text_uncondition = {"full_attention": None}
                attention_mask_img_uncondition = {"full_attention": None}

                prepare_flash_kv_cache(
                    past_key_values_cond_cfg,
                    current_len=token_h * token_w,
                    batch_size=1,
                )
                prepare_flash_kv_cache(
                    past_key_values_tu_cfg,
                    current_len=token_h * token_w,
                    batch_size=1,
                )
                prepare_flash_kv_cache(
                    past_key_values_iu_cfg,
                    current_len=token_h * token_w,
                    batch_size=1,
                )

                timesteps = torch.linspace(0.0, 1.0, num_steps+1, device=device)
                if enable_timestep_shift:
                    timesteps = self._apply_time_schedule(timesteps, token_h*token_w, timestep_shift)

                step_iter = range(num_steps)
                if verbose:
                    try:
                        from tqdm import tqdm as _tqdm
                        step_iter = _tqdm(
                            step_iter,
                            desc=f"image {img_count + 1} ({image_size[0]}x{image_size[1]})",
                            total=num_steps,
                            leave=False,
                        )
                    except ImportError:
                        pass
                for step_i in step_iter:
                    t = timesteps[step_i]
                    t_next = timesteps[step_i + 1]

                    z = self.patchify(image_prediction, self.patch_size * merge_size)
                    image_input = self.patchify(image_prediction, self.patch_size, channel_first=True)
                    image_embeds = self.extract_feature(image_input.view(1 * grid_h*grid_w, -1), gen_model=True, grid_hw=gen_grid_hw).view(1, token_h*token_w, -1)
                    t_expanded = t.expand(token_h*token_w)
                    timestep_embeddings = self.fm_modules['timestep_embedder'](t_expanded).view(1, token_h*token_w, -1)
                    if self.add_noise_scale_embedding:
                        noise_scale_tensor = torch.full_like(t_expanded, noise_scale/self.noise_scale_max_value)
                        noise_embeddings = self.fm_modules['noise_scale_embedder'](noise_scale_tensor).view(1, token_h*token_w, -1)
                        timestep_embeddings += noise_embeddings
                    image_embeds = image_embeds + timestep_embeddings

                    use_cfg = (t > cfg_interval[0] and t < cfg_interval[1]) or cfg_interval[0] == 0
                    out_cond = self._t2i_predict_v(image_embeds, indexes_image_condition, attention_mask_condition, past_key_values_cond_cfg, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings)
                    if not use_cfg:
                        v_pred = out_cond
                    elif cfg_scale == 1 and img_cfg_scale == 1:
                        v_pred = out_cond
                    elif img_cfg_scale == 1:
                        out_img_cond = self._t2i_predict_v(image_embeds, indexes_image_text_uncondition, attention_mask_text_uncondition, past_key_values_tu_cfg, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings)
                        v_pred = out_img_cond + cfg_scale * (out_cond - out_img_cond)
                    elif cfg_scale == img_cfg_scale:
                        out_uncond = self._t2i_predict_v(image_embeds, indexes_image_img_uncondition, attention_mask_img_uncondition, past_key_values_iu_cfg, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings)
                        v_pred = out_uncond + cfg_scale * (out_cond - out_uncond)
                    else:
                        out_img_cond = self._t2i_predict_v(image_embeds, indexes_image_text_uncondition, attention_mask_text_uncondition, past_key_values_tu_cfg, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings)
                        out_uncond = self._t2i_predict_v(image_embeds, indexes_image_img_uncondition, attention_mask_img_uncondition, past_key_values_iu_cfg, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings)
                        v_pred = (
                            out_uncond
                            + cfg_scale * (out_cond - out_img_cond)
                            + img_cfg_scale * (out_img_cond - out_uncond)
                        )
                    if (cfg_scale > 1 or img_cfg_scale > 1) and use_cfg:
                        if cfg_norm == 'global':
                            norm_v_condition = torch.norm(out_cond, dim=(1,2), keepdim=True)
                            norm_v_cfg = torch.norm(v_pred, dim=(1,2), keepdim=True)
                            scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                            v_pred = v_pred * scale
                        elif cfg_norm == 'channel':
                            norm_v_condition = torch.norm(out_cond, dim=-1, keepdim=True)
                            norm_v_cfg = torch.norm(v_pred, dim=-1, keepdim=True)
                            scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                            v_pred = v_pred * scale

                    z = z + (t_next - t) * v_pred
                    image_prediction = self.unpatchify(z, self.patch_size * merge_size, cur_image_size[1], cur_image_size[0])

                generated_images.append(image_prediction)

                clear_flash_kv_cache(past_key_values_cond_cfg)
                clear_flash_kv_cache(past_key_values_tu_cfg)
                clear_flash_kv_cache(past_key_values_iu_cfg)

                if gt_images is not None and img_count < len(gt_images):
                    gt_img_pil = gt_images[img_count]
                    gt_pixel_values, gt_grid_hw = load_image_native(gt_img_pil, self.patch_size, self.downsample_ratio, min_pixels=512*512, max_pixels=(2048*2048), upscale=False)
                    gt_pixel_values = gt_pixel_values.to(device).to(torch.bfloat16)
                    
                    flatten_pixel_values = gt_pixel_values
                    gen_grid_hw_und = gt_grid_hw
                else:
                    pred_img = image_prediction[0].unsqueeze(0).to(torch.bfloat16)
                    raw_img = pred_img * 0.5 + 0.5
                    img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=raw_img.dtype, device=device).view(1, 3, 1, 1)
                    img_std = torch.tensor([0.229, 0.224, 0.225], dtype=raw_img.dtype, device=device).view(1, 3, 1, 1)
                    und_img = (raw_img - img_mean) / img_std
                    
                    c, h, w = und_img[0].shape
                    ps = self.patch_size
                    p_grid_h = h // ps
                    p_grid_w = w // ps
                    flatten_pixel_values = (
                        und_img[0].view(c, p_grid_h, ps, p_grid_w, ps)
                        .permute(1, 3, 0, 2, 4)
                        .reshape(p_grid_h * p_grid_w, c * ps ** 2)
                    )
                    gen_grid_hw_und = torch.tensor([[p_grid_h, p_grid_w]], device=device)

                vit_embeds = self.extract_feature(flatten_pixel_values, grid_hw=gen_grid_hw_und[:1]).unsqueeze(0)
                
                img_end_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
                img_end_embed = self.language_model.get_input_embeddings()(torch.tensor([[img_end_id]], device=device))
                inputs_embeds_img = torch.cat([vit_embeds, img_end_embed], dim=1) # (1, N + 1, C)
                
                N_img_tokens = vit_embeds.shape[1]
                abs_pos_w, abs_pos_h = build_abs_positions_from_grid_hw(gen_grid_hw_und[:1] // int(1 / self.downsample_ratio), device=device)

                t_index_cond = append_image_to_cache(past_key_values_cond, t_index_cond, inputs_embeds_img, N_img_tokens, abs_pos_w, abs_pos_h)
                t_index_tu = append_image_to_cache(past_key_values_tu, t_index_tu, inputs_embeds_img, N_img_tokens, abs_pos_w, abs_pos_h)

                img_count += 1

        return generated_images

    @torch.no_grad()
    def interleave_gen(
            self,
            tokenizer,
            prompt,
            images=None,
            generation_config=None,
            cfg_scale=1.0,
            img_cfg_scale=1.0,
            cfg_norm='none',
            max_images=10,
            enable_timestep_shift=True,
            timestep_shift=1.0,
            image_size=(256, 256),
            num_steps=30,
            IMG_START_TOKEN='<img>',
            IMG_END_TOKEN='</img>',
            IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
            method='euler',
            cfg_interval=(0, 1),
            t_eps=0.02,
            verbose=False,
            system_message='',
            think_mode=False,
            seed=0,
    ):
        self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.config.t_eps = t_eps

        if isinstance(image_size, tuple):
            image_size_list = [image_size] * max_images
        elif isinstance(image_size, list) and isinstance(image_size[0], tuple):
            image_size_list = image_size
            if len(image_size) < max_images:
                image_size_list += [image_size_list[-1]] * (max_images - len(image_size_list))
        else:
            assert False, "image size should be a tuple or a list of tuple"

        if generation_config and hasattr(generation_config, 'max_new_tokens') and generation_config.max_new_tokens is not None:
            max_new_tokens = generation_config.max_new_tokens
        else:
            max_new_tokens = 8192

        current_generated_tokens = 0

        if images is None:
            images = []

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        image_token_count = prompt.count('<image>')
        assert len(images) >= image_token_count
        if len(images) > image_token_count:
            prompt = "<image>\n" * (len(images) - image_token_count) + prompt

        pixel_values =[]
        grid_hw =[]
        for image in images:
            cur_pixel_values, cur_grid_hw = load_image_native(image, self.patch_size, self.downsample_ratio, min_pixels=512*512, max_pixels=min(2048*2048, (4096*4096)//max(1, len(images))), upscale=False)
            grid_hw.append(cur_grid_hw.to(self.device))
            pixel_values.append(cur_pixel_values.to(self.device).to(torch.bfloat16))

        merge_size = int(1 / self.downsample_ratio)
        pv_tensor = torch.cat(pixel_values) if pixel_values else None
        ghw_tensor = torch.cat(grid_hw) if grid_hw else None

        # Condition
        template_cond = get_conv_template(self.template)
        template_cond.system_message = system_message
        template_cond.append_message(template_cond.roles[0], prompt)
        template_cond.append_message(template_cond.roles[1], None)
        query_cond = template_cond.get_prompt()

        if not think_mode:
            query_cond = query_cond + ' оку\n\n ок\n\n'

        def replace_image_tokens(query, grid_hw_list):
            for i in range(len(grid_hw_list)):
                num_patch_token = int(grid_hw_list[i][0, 0] * grid_hw_list[i][0, 1] * self.downsample_ratio**2)
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_patch_token + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)
            return query

        query_cond = replace_image_tokens(query_cond, grid_hw)
        input_embeds_cond, indexes_cond, attention_mask_cond = self._build_it2i_inputs(tokenizer, query_cond, pv_tensor, ghw_tensor)
        
        outputs_cond = self.language_model(inputs_embeds=input_embeds_cond, indexes=indexes_cond, attention_mask=attention_mask_cond, use_cache=True)
        past_key_values_cond = outputs_cond.past_key_values
        t_index_cond = indexes_cond[0].max().item()

        # Initialize Text Uncondition Cache
        question_text_uncondition = '<image>' * len(images)
        template_tu = get_conv_template(self.template)
        template_tu.system_message = self.system_message
        template_tu.append_message(template_tu.roles[0], question_text_uncondition)
        template_tu.append_message(template_tu.roles[1], None)
        query_text_uncond = template_tu.get_prompt()
        query_text_uncond = replace_image_tokens(query_text_uncond, grid_hw)

        input_embeds_tu, indexes_tu, attention_mask_tu = self._build_it2i_inputs(tokenizer, query_text_uncond, pv_tensor, ghw_tensor)
        outputs_tu = self.language_model(inputs_embeds=input_embeds_tu, indexes=indexes_tu, attention_mask=attention_mask_tu, use_cache=True)
        past_key_values_tu = outputs_tu.past_key_values
        t_index_tu = indexes_tu[0].max().item()

        # Initialize Img (ALL) Uncondition Cache
        query_img_uncond = self._build_t2i_query("", append_text=IMG_START_TOKEN)
        input_embeds_iu, indexes_iu, attention_mask_iu = self._build_it2i_inputs(tokenizer, query_img_uncond)
        outputs_iu = self.language_model(inputs_embeds=input_embeds_iu, indexes=indexes_iu, attention_mask=attention_mask_iu, use_cache=True)
        past_key_values_iu = outputs_iu.past_key_values


        generated_text = ""
        generated_images =[]
        max_images = 10
        img_count = 0

        next_token = torch.argmax(outputs_cond.logits[:, -1, :], dim=-1)

        generator = torch.Generator(self.device).manual_seed(seed)
        while True:
            # text generation
            gen_tokens = []
            hit_max_tokens = False
            last_decoded = 0
            while True:
                token_item = next_token.item()
                if token_item == eos_token_id or token_item == self.img_start_token_id:
                    break
                gen_tokens.append(token_item)
                current_generated_tokens += 1

                self.language_model.model.current_index = t_index_cond
                outputs_cond = self.language_model(
                    input_ids=next_token.unsqueeze(0),
                    past_key_values=past_key_values_cond,
                    use_cache=True
                )
                past_key_values_cond = outputs_cond.past_key_values
                t_index_cond += 1
                next_token = torch.argmax(outputs_cond.logits[:, -1, :], dim=-1)

                # Stream partial text so users see liveness during long runs
                # (e.g. low VRAM offload). Decode in 16-token chunks.
                if verbose and len(gen_tokens) - last_decoded >= 16:
                    partial = tokenizer.decode(gen_tokens[last_decoded:], skip_special_tokens=True)
                    print(partial, end='', flush=True)
                    last_decoded = len(gen_tokens)

                if current_generated_tokens >= max_new_tokens:
                    hit_max_tokens = True
                    break

            if len(gen_tokens) > 0:
                chunk_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                generated_text += chunk_text
                if verbose:
                    remaining = tokenizer.decode(gen_tokens[last_decoded:], skip_special_tokens=True)
                    if remaining:
                        print(remaining, end='', flush=True)

            if next_token.item() == eos_token_id or hit_max_tokens:
                break

            if next_token.item() == self.img_start_token_id:
                if img_count >= max_images:
                    break

                generated_text += "<image>"
                if verbose:
                    print(f"\n[image {img_count + 1}] preparing diffusion...", flush=True)

                # Add the img_start_token for condition and text_uncondition branch
                self.language_model.model.current_index = t_index_cond
                outputs_cond = self.language_model(input_ids=next_token.unsqueeze(0), past_key_values=past_key_values_cond, use_cache=True)
                past_key_values_cond = outputs_cond.past_key_values
                t_index_cond += 1

                self.language_model.model.current_index = t_index_tu
                outputs_tu = self.language_model(input_ids=next_token.unsqueeze(0), past_key_values=past_key_values_tu, use_cache=True)
                past_key_values_tu = outputs_tu.past_key_values
                t_index_tu += 1

                image_size = image_size_list[img_count]
                # Image Generation
                token_h = image_size[1] // (self.patch_size * merge_size)
                token_w = image_size[0] // (self.patch_size * merge_size)
                device = self.device

                indexes_image_condition = self._build_t2i_image_indexes(token_h, token_w, t_index_cond + 1, device=device)
                indexes_image_text_uncondition = self._build_t2i_image_indexes(token_h, token_w, t_index_tu + 1, device=device)
                indexes_image_img_uncondition = self._build_t2i_image_indexes(token_h, token_w, indexes_iu[0].max() + 1, device=device)

                grid_h = image_size[1] // self.patch_size
                grid_w = image_size[0] // self.patch_size
                gen_grid_hw = torch.tensor([[grid_h, grid_w]], device=device)

                noise_scale = self.noise_scale
                if self.noise_scale_mode in ("resolution", "dynamic", 'dynamic_sqrt'):
                    base = float(self.noise_scale_base_image_seq_len)
                    noise_scale = math.sqrt((grid_h*grid_w)/(merge_size**2)/base) * float(self.noise_scale)
                    if self.noise_scale_mode == 'dynamic_sqrt':
                        noise_scale = math.sqrt(noise_scale)
                noise_scale = min(noise_scale, self.noise_scale_max_value)
                image_prediction = noise_scale * torch.randn((1, 3, image_size[1], image_size[0]), device=device, dtype=outputs_cond.logits.dtype, generator=generator)

                past_key_values_cond_cfg = past_key_values_cond
                past_key_values_tu_cfg = past_key_values_tu
                past_key_values_iu_cfg = past_key_values_iu

                attention_mask_condition = {"full_attention": None}
                attention_mask_text_uncondition = {"full_attention": None}
                attention_mask_img_uncondition = {"full_attention": None}

                prepare_flash_kv_cache(
                    past_key_values_cond_cfg,
                    current_len=token_h * token_w,
                    batch_size=1,
                )
                prepare_flash_kv_cache(
                    past_key_values_tu_cfg,
                    current_len=token_h * token_w,
                    batch_size=1,
                )
                prepare_flash_kv_cache(
                    past_key_values_iu_cfg,
                    current_len=token_h * token_w,
                    batch_size=1,
                )

                timesteps = torch.linspace(0.0, 1.0, num_steps+1, device=device)
                if enable_timestep_shift:
                    timesteps = self._apply_time_schedule(timesteps, token_h*token_w, timestep_shift)

                step_iter = range(num_steps)
                if verbose:
                    try:
                        from tqdm import tqdm as _tqdm
                        step_iter = _tqdm(
                            step_iter,
                            desc=f"image {img_count + 1} ({image_size[0]}x{image_size[1]})",
                            total=num_steps,
                            leave=False,
                        )
                    except ImportError:
                        pass
                for step_i in step_iter:
                    t = timesteps[step_i]
                    t_next = timesteps[step_i + 1]

                    z = self.patchify(image_prediction, self.patch_size * merge_size)
                    image_input = self.patchify(image_prediction, self.patch_size, channel_first=True)
                    image_embeds = self.extract_feature(image_input.view(1 * grid_h*grid_w, -1), gen_model=True, grid_hw=gen_grid_hw).view(1, token_h*token_w, -1)
                    t_expanded = t.expand(token_h*token_w)
                    timestep_embeddings = self.fm_modules['timestep_embedder'](t_expanded).view(1, token_h*token_w, -1)
                    if self.add_noise_scale_embedding:
                        noise_scale_tensor = torch.full_like(t_expanded, noise_scale/self.noise_scale_max_value)
                        noise_embeddings = self.fm_modules['noise_scale_embedder'](noise_scale_tensor).view(1, token_h*token_w, -1)
                        timestep_embeddings += noise_embeddings
                    image_embeds = image_embeds + timestep_embeddings

                    use_cfg = (t > cfg_interval[0] and t < cfg_interval[1]) or cfg_interval[0] == 0
                    out_cond = self._t2i_predict_v(image_embeds, indexes_image_condition, attention_mask_condition, past_key_values_cond_cfg, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings)
                    if not use_cfg:
                        v_pred = out_cond
                    elif cfg_scale == 1 and img_cfg_scale == 1:
                        v_pred = out_cond
                    elif img_cfg_scale == 1:
                        out_img_cond = self._t2i_predict_v(image_embeds, indexes_image_text_uncondition, attention_mask_text_uncondition, past_key_values_tu_cfg, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings)
                        v_pred = out_img_cond + cfg_scale * (out_cond - out_img_cond)
                    elif cfg_scale == img_cfg_scale:
                        out_uncond = self._t2i_predict_v(image_embeds, indexes_image_img_uncondition, attention_mask_img_uncondition, past_key_values_iu_cfg, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings)
                        v_pred = out_uncond + cfg_scale * (out_cond - out_uncond)
                    else:
                        out_img_cond = self._t2i_predict_v(image_embeds, indexes_image_text_uncondition, attention_mask_text_uncondition, past_key_values_tu_cfg, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings)
                        out_uncond = self._t2i_predict_v(image_embeds, indexes_image_img_uncondition, attention_mask_img_uncondition, past_key_values_iu_cfg, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings)
                        v_pred = (
                            out_uncond
                            + cfg_scale * (out_cond - out_img_cond)
                            + img_cfg_scale * (out_img_cond - out_uncond)
                        )
                    if (cfg_scale > 1 or img_cfg_scale > 1 and use_cfg):
                        if cfg_norm == 'global':
                            norm_v_condition = torch.norm(out_cond, dim=(1,2), keepdim=True)
                            norm_v_cfg = torch.norm(v_pred, dim=(1,2), keepdim=True)
                            scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                            v_pred = v_pred * scale
                        elif cfg_norm == 'channel':
                            norm_v_condition = torch.norm(out_cond, dim=-1, keepdim=True)
                            norm_v_cfg = torch.norm(v_pred, dim=-1, keepdim=True)
                            scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                            v_pred = v_pred * scale

                    z = z + (t_next - t) * v_pred
                    image_prediction = self.unpatchify(z, self.patch_size * merge_size, image_size[1], image_size[0])

                generated_images.append(image_prediction)

                clear_flash_kv_cache(past_key_values_cond_cfg)
                clear_flash_kv_cache(past_key_values_tu_cfg)
                clear_flash_kv_cache(past_key_values_iu_cfg)

                img_count += 1

                # re-encode the generated image using the und-branch
                pred_img = image_prediction[0].unsqueeze(0).to(torch.bfloat16)
                # re-normalize the image
                raw_img = pred_img * 0.5 + 0.5
                img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=raw_img.dtype, device=device).view(1, 3, 1, 1)
                img_std = torch.tensor([0.229, 0.224, 0.225], dtype=raw_img.dtype, device=device).view(1, 3, 1, 1)
                und_img = (raw_img - img_mean) / img_std
                c, h, w = und_img[0].shape
                ps = self.patch_size
                p_grid_h = h // ps
                p_grid_w = w // ps
                flatten_pixel_values = (
                    und_img[0].view(c, p_grid_h, ps, p_grid_w, ps)
                    .permute(1, 3, 0, 2, 4)  # [grid_h, grid_w, c, patch_size, patch_size]
                    .reshape(p_grid_h * p_grid_w, c * ps ** 2)
                )
                vit_embeds = self.extract_feature(flatten_pixel_values, grid_hw=gen_grid_hw[:1]).unsqueeze(0)
                
                img_end_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
                img_end_embed = self.language_model.get_input_embeddings()(torch.tensor([[img_end_id]], device=device))
                inputs_embeds_img = torch.cat([vit_embeds, img_end_embed], dim=1) # (1, N + 1, C)
                
                N_img_tokens = vit_embeds.shape[1]
                abs_pos_w, abs_pos_h = build_abs_positions_from_grid_hw(gen_grid_hw[:1] // int(1 / self.downsample_ratio), device=device)

                def append_image_to_cache(cache, t_idx):
                    past_len = cache.get_seq_length()
                    tgt_len = N_img_tokens + 1
                    
                    t_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
                    t_indexes[:N_img_tokens] = t_idx + 1
                    t_indexes[N_img_tokens] = t_idx + 2
                    
                    h_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
                    w_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
                    h_indexes[:N_img_tokens] = abs_pos_h
                    w_indexes[:N_img_tokens] = abs_pos_w
                    
                    indexes = torch.stack([t_indexes, h_indexes, w_indexes], dim=0)
                    
                    mask = torch.zeros(1, 1, tgt_len, past_len + tgt_len, device=device)
                    mask[0, 0, :N_img_tokens, past_len + N_img_tokens] = float('-inf')
                    attention_mask_dict = {"full_attention": mask}
                    
                    outputs = self.language_model(
                        inputs_embeds=inputs_embeds_img,
                        indexes=indexes,
                        attention_mask=attention_mask_dict,
                        past_key_values=cache,
                        use_cache=True
                    )
                    return outputs, t_idx + 2

                outputs_cond, t_index_cond = append_image_to_cache(past_key_values_cond, t_index_cond)
                outputs_tu, t_index_tu = append_image_to_cache(past_key_values_tu, t_index_tu)

                next_token = torch.argmax(outputs_cond.logits[:, -1, :], dim=-1)

        return generated_text, generated_images

from __future__ import annotations

import math

import torch

from ..conversation import get_conv_template
from ..modeling_qwen3 import create_block_causal_mask
from ..utils import load_image_native, SYSTEM_MESSAGE_FOR_GEN
from .utils import prepare_flash_kv_cache, clear_flash_kv_cache, optimized_scale


class T2IGeneratorMixin:

    def _build_t2i_query(self, prompt_text, system_message=None, append_text=None):
        template = get_conv_template(self.template)
        template.system_message = self.system_message if system_message is None else system_message
        template.append_message(template.roles[0], prompt_text)
        template.append_message(template.roles[1], None)
        if append_text is not None:
            return template.get_prompt() + append_text
        return template.get_prompt()

    def _build_t2i_text_inputs(self, tokenizer, query: str):
        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)

        t_idx = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        h_idx = torch.zeros_like(t_idx)
        w_idx = torch.zeros_like(t_idx)
        indexes = torch.stack([t_idx, h_idx, w_idx], dim=0)

        attention_mask = {"full_attention": create_block_causal_mask(indexes[0])}
        return input_ids, indexes, attention_mask
    
    def _build_t2i_image_indexes(self, token_h, token_w, text_len, device):
        t_image = torch.full((token_h * token_w,), text_len, dtype=torch.long, device=device)
        idx = torch.arange(token_h * token_w, device=device, dtype=torch.long)
        h_image = idx // token_w
        w_image = idx % token_w
        return torch.stack([t_image, h_image, w_image], dim=0)
    
    def _t2i_prefix_forward(self, input_ids, indexes, attention_mask):
        out = self.language_model.model(
            input_ids=input_ids,
            indexes=indexes,
            attention_mask=attention_mask,
            use_cache=True,
        )
        return out.past_key_values, out.last_hidden_state

    @torch.no_grad()
    def t2i_generate(self, tokenizer, prompt, cfg_scale=1, timestep_shift=1, enable_timestep_shift=True, cfg_norm='none', image_size=(256, 256), num_steps=30, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', method='euler', cfg_interval=(0, 1), batch_size=1, t_eps=0.02, think_mode=False, seed=0):
        assert self.concat_time_token_num == 0
        assert cfg_norm in ['cfg_zero_star', 'global', 'none', 'channel']
        merge_size = int(1 / self.downsample_ratio)

        self.config.t_eps = t_eps
        # question_condition = f"Please generate an image based on the following description: {prompt}"
        question_condition = f"{prompt}"
        # question_condition += f"\nThe resolution of the image should be {image_size}"

        think_text = ""
        needs_cfg = cfg_scale > 1

        think_content = '<think>\n' if think_mode else '<think>\n\n</think>\n\n' + IMG_START_TOKEN
        query_condition = self._build_t2i_query(question_condition, system_message=SYSTEM_MESSAGE_FOR_GEN, append_text=think_content)
        query_uncondition = self._build_t2i_query("", append_text=IMG_START_TOKEN) if needs_cfg else None

        input_ids_condition, indexes_condition, attention_mask_condition_prefix = self._build_t2i_text_inputs(tokenizer, query_condition)
        if query_uncondition is not None:
            input_ids_uncondition, indexes_uncondition, attention_mask_uncondition_prefix = self._build_t2i_text_inputs(tokenizer, query_uncondition)
        else:
            input_ids_uncondition = indexes_uncondition = attention_mask_uncondition_prefix = None
       
        token_h = image_size[1] // (self.patch_size * merge_size)
        token_w = image_size[0] // (self.patch_size * merge_size)

        indexes_image_condition = self._build_t2i_image_indexes(token_h, token_w, indexes_condition.shape[1], device=input_ids_condition.device)
        indexes_image_uncondition = (
            self._build_t2i_image_indexes(token_h, token_w, indexes_uncondition.shape[1], device=input_ids_uncondition.device)
            if indexes_uncondition is not None
            else None
        )

        if think_mode:
            outputs_condition = self.language_model(
                input_ids=input_ids_condition,
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
                token_h, token_w, t_index_condition + 1, device=input_ids_condition.device
            )
        else:
            past_key_values_condition, hidden_states_condition = self._t2i_prefix_forward(input_ids_condition, indexes_condition, attention_mask_condition_prefix)
        past_key_values_uncondition = None
        if input_ids_uncondition is not None:
            past_key_values_uncondition, _ = self._t2i_prefix_forward(input_ids_uncondition, indexes_uncondition, attention_mask_uncondition_prefix)

        device = hidden_states_condition.device
        dtype = hidden_states_condition.dtype

        del input_ids_condition, indexes_condition, attention_mask_condition_prefix
        if input_ids_uncondition is not None:
            del input_ids_uncondition, indexes_uncondition, attention_mask_uncondition_prefix
        del hidden_states_condition

        for layer_idx in range(len(past_key_values_condition.layers)):
            past_key_values_condition.layers[layer_idx].keys = past_key_values_condition.layers[layer_idx].keys.expand(batch_size, *past_key_values_condition.layers[layer_idx].keys.shape[1:])
            past_key_values_condition.layers[layer_idx].values = past_key_values_condition.layers[layer_idx].values.expand(batch_size, *past_key_values_condition.layers[layer_idx].values.shape[1:])
            if past_key_values_uncondition is not None:
                past_key_values_uncondition.layers[layer_idx].keys = past_key_values_uncondition.layers[layer_idx].keys.expand(batch_size, *past_key_values_uncondition.layers[layer_idx].keys.shape[1:])
                past_key_values_uncondition.layers[layer_idx].values = past_key_values_uncondition.layers[layer_idx].values.expand(batch_size, *past_key_values_uncondition.layers[layer_idx].values.shape[1:])

        # prepare flash cache once
        prepare_flash_kv_cache(
            past_key_values_condition,
            current_len=token_h * token_w,
            batch_size=batch_size,
        )
        if past_key_values_uncondition is not None:
            prepare_flash_kv_cache(
                past_key_values_uncondition,
                current_len=token_h * token_w,
                batch_size=batch_size,
            )

        # init noise image tokens
        grid_h = image_size[1] // self.patch_size
        grid_w = image_size[0] // self.patch_size
        grid_hw = torch.tensor([[grid_h, grid_w]]*batch_size, device=device)

        noise_scale = self.noise_scale
        if self.noise_scale_mode in ("resolution", "dynamic", 'dynamic_sqrt'):
            base = float(self.noise_scale_base_image_seq_len)
            scale = math.sqrt((grid_h*grid_w)/(merge_size**2)/base)
            noise_scale = scale * float(self.noise_scale)
            if self.noise_scale_mode == 'dynamic_sqrt':
                noise_scale = math.sqrt(noise_scale)
        noise_scale = min(noise_scale, self.noise_scale_max_value)
        generator = torch.Generator(device).manual_seed(seed)
        image_prediction = noise_scale * torch.randn((batch_size, 3, image_size[1], image_size[0]), device=device, dtype=dtype, generator=generator)

        attention_mask_condition = {"full_attention": None}
        attention_mask_uncondition = {"full_attention": None}

        timesteps = torch.linspace(0.0, 1.0, num_steps+1, device=device)
        if enable_timestep_shift:
            timesteps = self._apply_time_schedule(timesteps, token_h*token_w, timestep_shift)

        for step_i in range(num_steps):
            t = timesteps[step_i]
            t_next = timesteps[step_i + 1]

            z = self.patchify(image_prediction, self.patch_size * merge_size)
            image_input = self.patchify(image_prediction, self.patch_size, channel_first=True)
            image_embeds = self.extract_feature(image_input.view(batch_size * grid_h*grid_w, -1), gen_model=True, grid_hw=grid_hw).view(batch_size, token_h*token_w, -1)
            t_expanded = t.expand(batch_size*token_h*token_w)
            timestep_embeddings = self.fm_modules['timestep_embedder'](t_expanded).view(batch_size, token_h*token_w, -1)
            if self.add_noise_scale_embedding:
                noise_scale_tensor = torch.full_like(t_expanded, noise_scale / self.noise_scale_max_value)
                noise_embeddings = self.fm_modules['noise_scale_embedder'](noise_scale_tensor).view(batch_size, token_h*token_w, -1)
                timestep_embeddings += noise_embeddings
            image_embeds = image_embeds + timestep_embeddings

            v_pred_condition = self._t2i_predict_v(image_embeds, indexes_image_condition, attention_mask_condition, past_key_values_condition, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings, image_size=image_size)
            
            if t >= cfg_interval[0] and t <= cfg_interval[1] and cfg_scale > 1:
                v_pred_uncondition = self._t2i_predict_v(image_embeds, indexes_image_uncondition, attention_mask_uncondition, past_key_values_uncondition, t, z, image_token_num=token_h*token_w, timestep_embeddings=timestep_embeddings, image_size=image_size)
                if cfg_norm == 'cfg_zero_star':
                    positive_flat = v_pred_condition.view(batch_size, -1)  
                    negative_flat = v_pred_uncondition.view(batch_size, -1)  

                    alpha = optimized_scale(positive_flat,negative_flat)
                    alpha = alpha.view(batch_size, *([1] * (len(v_pred_condition.shape) - 1)))
                    alpha = alpha.to(positive_flat.dtype)

                    if (step_i <= 0):
                        v_pred = v_pred_condition*0.
                    else:
                        v_pred = v_pred_uncondition * alpha + cfg_scale * (v_pred_condition - v_pred_uncondition * alpha)
                else: 
                    v_pred = v_pred_uncondition + cfg_scale * (v_pred_condition - v_pred_uncondition)
                    if cfg_norm == 'global':
                        norm_v_condition = torch.norm(v_pred_condition, dim=(1,2), keepdim=True)
                        norm_v_cfg = torch.norm(v_pred, dim=(1,2), keepdim=True)
                        scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                        v_pred = v_pred * scale
                    elif cfg_norm == 'channel':
                        norm_v_condition = torch.norm(v_pred_condition, dim=-1, keepdim=True)
                        norm_v_cfg = torch.norm(v_pred, dim=-1, keepdim=True)
                        scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                        v_pred = v_pred * scale
            else:
                v_pred = v_pred_condition

            z = z + (t_next - t) * v_pred

            image_prediction = self.unpatchify(z, self.patch_size * merge_size, image_size[1], image_size[0])

        clear_flash_kv_cache(past_key_values_condition)
        if past_key_values_uncondition is not None:
            clear_flash_kv_cache(past_key_values_uncondition)

        self.last_think_content = think_text
        if think_mode:
            return image_prediction, think_text
        return image_prediction

    def _generate_think(
        self,
        tokenizer,
        prefix_outputs,
        past_key_values,
        t_idx,
        IMG_START_TOKEN,
        max_think_tokens=1024,
    ):
        template = get_conv_template(self.template)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        think_end_token_id = tokenizer.convert_tokens_to_ids('</think')
        think_token_ids = []
        next_token = torch.argmax(prefix_outputs.logits[:, -1, :], dim=-1)

        for _ in range(max_think_tokens):
            token_item = next_token.item()
            if token_item == eos_token_id:
                break
            if token_item == think_end_token_id:
                self.language_model.model.current_index = t_idx
                outputs = self.language_model(
                    input_ids=next_token.unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True
                )
                past_key_values = outputs.past_key_values
                t_idx += 1
                think_token_ids.append(token_item)
                break

            think_token_ids.append(token_item)

            self.language_model.model.current_index = t_idx
            outputs = self.language_model(
                input_ids=next_token.unsqueeze(0),
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            t_idx += 1

            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)

        append_ids = tokenizer(
            '\n\n' + IMG_START_TOKEN,
            return_tensors='pt',
            add_special_tokens=False,
        )['input_ids'].to(self.device)

        t_idx = self._append_text_tokens_to_cache(past_key_values, t_idx, append_ids)

        think_text = tokenizer.decode(think_token_ids, skip_special_tokens=False)

        return past_key_values, t_idx, think_text

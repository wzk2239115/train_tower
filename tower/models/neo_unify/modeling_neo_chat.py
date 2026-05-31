from typing import List, Optional, Tuple, Union
import math
import os

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import transformers
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_neo_chat import NEOChatConfig, NEOMoELLMConfig
from .conversation import get_conv_template
from .modeling_neo_vit import NEOVisionModel
from .modeling_qwen3 import Qwen3ForCausalLM, create_block_causal_mask
from .modeling_qwen3_moe import Qwen3MoeForCausalLM
from .modeling_fm_modules import PositionEmbedding, TimestepEmbedder, FlowMatchingHead, RMSNorm, NerfEmbedder, SimpleMLPAdaLN, ConvDecoder
from .utils import load_image_native, SYSTEM_MESSAGE_FOR_GEN
from .modeling_omni_decoders import AudioMelDecoder, AudioPatchEncoder, VideoFrameDecoder

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

def prepare_flash_kv_cache(
    past_key_values,
    current_len: int,
    batch_size: int,
):
    """
    Convert prefix cache from [B, H, S, D] to flash-attn friendly [B, S, H, D],
    and preallocate full KV buffer for [prefix + current].

    This is done once before denoising loop.
    """
    if past_key_values is None:
        return

    for layer in past_key_values.layers:
        past_k = layer.keys
        past_v = layer.values

        if past_k is None or past_v is None:
            layer.flash_prefix_len = 0
            layer.flash_total_len = current_len
            layer.flash_k_cache = None
            layer.flash_v_cache = None
            continue

        # original cache layout assumed: [B, H, S, D]
        past_k_flash = past_k.transpose(1, 2).contiguous()  # [B, S, H, D]
        past_v_flash = past_v.transpose(1, 2).contiguous()  # [B, S, H, D]

        prefix_len = past_k_flash.shape[1]
        total_len = prefix_len + current_len

        k_cache = torch.empty(
            (batch_size, total_len, past_k_flash.shape[2], past_k_flash.shape[3]),
            device=past_k_flash.device,
            dtype=past_k_flash.dtype,
        )
        v_cache = torch.empty(
            (batch_size, total_len, past_v_flash.shape[2], past_v_flash.shape[3]),
            device=past_v_flash.device,
            dtype=past_v_flash.dtype,
        )

        k_cache[:, :prefix_len].copy_(past_k_flash)
        v_cache[:, :prefix_len].copy_(past_v_flash)

        layer.flash_prefix_len = prefix_len
        layer.flash_total_len = total_len
        layer.flash_k_cache = k_cache
        layer.flash_v_cache = v_cache

def clear_flash_kv_cache(past_key_values):
    if past_key_values is None:
        return
    for layer in past_key_values.layers:
        if hasattr(layer, "flash_prefix_len"):
            delattr(layer, "flash_prefix_len")
        if hasattr(layer, "flash_total_len"):
            delattr(layer, "flash_total_len")
        if hasattr(layer, "flash_k_cache"):
            delattr(layer, "flash_k_cache")
        if hasattr(layer, "flash_v_cache"):
            delattr(layer, "flash_v_cache")

def optimized_scale(positive_flat, negative_flat):
    # Force the divisor computation to float32 regardless of the surrounding
    # autocast (the squared-norm/division is what we don't want in fp16/bf16).
    # ``device_type`` is taken from the input so this runs equally on CUDA and
    # XPU; ``mps`` is rerouted to ``cpu`` because torch.autocast rejects it.
    device_type = positive_flat.device.type
    if device_type == "mps":
        device_type = "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        positive_flat = positive_flat.float()
        negative_flat = negative_flat.float()

        # Calculate dot production
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

        # Squared norm of uncondition
        squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

        # st_star = v_cond^T * v_uncond / ||v_uncond||^2
        st_star = dot_product / squared_norm

    return st_star

def build_abs_positions_from_grid_hw(grid_hw: torch.Tensor, device=None):
    """
    Compute patch coordinates (x, y)

    Args:
        grid_hw: (B, 2) tensor representing (H, W) per image
    """
    device = grid_hw.device
    B = grid_hw.shape[0]

    # Get the number of patches per image
    H = grid_hw[:, 0]
    W = grid_hw[:, 1]
    N = H * W
    N_total = N.sum()

    # Create the batch index for each patch (B x patch count)
    patch_to_sample = torch.repeat_interleave(torch.arange(B, device=device), N)  # (N_total,)

    # Generate intra-image patch index (row-major order)
    patch_id_within_image = torch.arange(N_total, device=device)
    patch_id_within_image = patch_id_within_image - torch.cumsum(
        torch.cat([torch.tensor([0], device=device), N[:-1]]), dim=0
    )[patch_to_sample]

    # Get H/W for each patch according to its image
    W_per_patch = W[patch_to_sample]
    abs_x = patch_id_within_image % W_per_patch
    abs_y = patch_id_within_image // W_per_patch

    return abs_x, abs_y


class NEOChatModel(PreTrainedModel):
    config_class = NEOChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "NEOVisionModel",
        "Qwen3DecoderLayer",
        "Qwen3MoeDecoderLayer",
    ]

    # support transformers 4.51.+
    _tp_plan = ''

    def __init__(self, config: NEOChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.template = config.template
        self.downsample_ratio = config.downsample_ratio
        config.llm_config._attn_implementation = 'eager'

        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = NEOVisionModel(config.vision_config)
            vision_model_mot_gen = NEOVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            # Pick the right backbone class based on the LLM config: dense
            # Qwen3 (DANCE family) or Qwen3-MoE (A3B family). The two share
            # the same NEO-Unify two-branch attention/norm layout, so the
            # rest of this class works against either.
            if isinstance(config.llm_config, NEOMoELLMConfig):
                self.language_model = Qwen3MoeForCausalLM(config.llm_config)
            else:
                self.language_model = Qwen3ForCausalLM(config.llm_config)

        merge_size = int(1 / self.downsample_ratio)
        output_dim = 3*(patch_size*merge_size)**2
        llm_hidden_size = self.config.llm_config.hidden_size
        self.use_deep_fm_head = self.config.fm_head_layers > 2
        self.use_pixel_head = self.config.use_pixel_head
        if self.use_deep_fm_head:
                fm_head = FlowMatchingHead(llm_hidden_size, output_dim, dim=self.config.fm_head_dim, layers=self.config.fm_head_layers, mlp_ratio=self.config.fm_head_mlp_ratio)
        else:
            fm_head = nn.Sequential(
                    nn.Linear(llm_hidden_size, 4096, bias=True),
                    nn.GELU(),
                    nn.Linear(4096, output_dim, bias=True),
                )

        timestep_embedder = TimestepEmbedder(llm_hidden_size)
        self.fm_modules = nn.ModuleDict(
                    {   
                        "vision_model_mot_gen": vision_model_mot_gen,
                        "timestep_embedder": timestep_embedder,
                        "fm_head": fm_head
                    }
                )

        if self.use_pixel_head:
            self.fm_modules["fm_head"] = ConvDecoder(llm_hidden_size)

        self.concat_time_token_num = config.concat_time_token_num
        self.noise_scale = config.noise_scale
        self.noise_scale_mode = config.noise_scale_mode
        self.noise_scale_base_image_seq_len = config.noise_scale_base_image_seq_len
        self.add_noise_scale_embedding = config.add_noise_scale_embedding
        self.noise_scale_max_value = config.noise_scale_max_value
        self.time_schedule = config.time_schedule
        self.time_shift_type = config.time_shift_type
        self.base_shift = config.base_shift
        self.max_shift = config.max_shift
        self.base_image_seq_len = config.base_image_seq_len
        self.max_image_seq_len = config.max_image_seq_len

        if self.add_noise_scale_embedding:
            noise_scale_embedder = TimestepEmbedder(llm_hidden_size)
            self.fm_modules['noise_scale_embedder'] = noise_scale_embedder


        audio_n_mels = int(getattr(config, 'audio_n_mels', 80))
        audio_patch_t = int(getattr(config, 'audio_patch_t', 4))
        self.audio_patch_encoder = AudioPatchEncoder(n_mels=audio_n_mels, patch_t=audio_patch_t)
        audio_patch_dim = audio_n_mels * audio_patch_t
        self.audio_latent_proj = nn.Sequential(
            nn.Linear(audio_patch_dim, llm_hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(llm_hidden_size, audio_patch_dim, bias=True),
        )
        self.audio_mel_decoder = AudioMelDecoder(n_mels=audio_n_mels)

        video_num_frames = int(getattr(config, 'video_num_frames', 16))
        self.video_frame_decoder = VideoFrameDecoder(
            patch_size=patch_size,
            merge_size=merge_size,
        )
        self.video_num_frames_default = video_num_frames

        self.img_context_token_id = None
        self.img_start_token_id = 151670
        self.audio_context_token_id = None
        self.audio_start_token_id = None
        self.video_context_token_id = None
        self.video_start_token_id = None
        self.last_think_content = ""
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        raise NotImplementedError('forward')
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = min(selected.sum(), vit_embeds.size(0))
            input_embeds[selected][:n_token] = input_embeds[selected][:n_token] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def extract_feature(self, pixel_values, gen_model=False, grid_hw=None):
        if gen_model:
            return self.fm_modules['vision_model_mot_gen'](pixel_values=pixel_values, 
                                 output_hidden_states=False, 
                                 return_dict=True, 
                                 grid_hw=grid_hw).last_hidden_state
        else:
            return self.vision_model(pixel_values=pixel_values, 
                                 output_hidden_states=False, 
                                 return_dict=True, 
                                 grid_hw=grid_hw).last_hidden_state

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        raise NotImplementedError('batch_chat')
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses
    
    def patchify(self, images, patch_size, channel_first=False):
        """
        images: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        h, w = images.shape[2] // patch_size, images.shape[3] // patch_size
        x = images.reshape(shape=(images.shape[0], 3, h, patch_size, w, patch_size))

        if channel_first:
            x = torch.einsum('nchpwq->nhwcpq', x)
        else:
            x = torch.einsum('nchpwq->nhwpqc', x)
        
        x = x.reshape(shape=(images.shape[0], h * w, patch_size**2 * 3))
        return x
    
    def unpatchify(sle, x, patch_size, h=None, w=None):
        """
        x: (N, L, patch_size**2 *3)
        images: (N, 3, H, W)
        """
        if h is None or w is None:
            h = w = int(x.shape[1]**.5)
        else:
            h = h // patch_size
            w = w // patch_size        
        x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        images = x.reshape(shape=(x.shape[0], 3, h * patch_size, w * patch_size))
        return images
    
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

    def _it2i_prefix_forward(self, input_imbeds, indexes, attention_mask, gen_indicators=None):
        out = self.language_model.model(
            inputs_embeds=input_imbeds,
            indexes=indexes,
            attention_mask=attention_mask,
            use_cache=True,
            image_gen_indicators=gen_indicators.view(1, -1) if gen_indicators is not None else None
        )
        return out.past_key_values, out.last_hidden_state

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
        think_end_token_id = tokenizer.convert_tokens_to_ids('</think>')
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
        query_cond = template_cond.get_prompt() + '<think>\n\n</think>\n\n'

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
            query_cond = query_cond + '<think>\n\n</think>\n\n'

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

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False, grid_hw=None, 
             IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        self.img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            print(f'dynamic image size: {grid_hw[0] * self.patch_size}')

        for i in range(grid_hw.shape[0]):
            num_patch_token = int(grid_hw[i, 0] * grid_hw[i, 1] * self.downsample_ratio**2)
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_patch_token + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            grid_hw=grid_hw,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            grid_hw: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        assert input_ids.shape[0] == 1
        assert self.img_context_token_id is not None
        indexes = self.get_thw_indexes(input_ids[0], grid_hw)
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values, grid_hw=grid_hw)
        
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            indexes=indexes,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, value):
        return self.language_model.set_output_embeddings(value)
    
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
                abs_pos_w, abs_pos_h = build_abs_positions_from_grid_hw(
                    grid_hw // int(1 / self.downsample_ratio), device=t_indexes.device)
                h_indexes[selected] = abs_pos_h.to(t_indexes.device, t_indexes.dtype)
                w_indexes[selected] = abs_pos_w.to(t_indexes.device, t_indexes.dtype)
        return torch.stack([t_indexes, h_indexes, w_indexes], dim=0)

    # ------------------------------------------------------------------
    # Super-omni generation: audio / video FM denoising helpers
    # ------------------------------------------------------------------

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
        """Predict velocity for audio latent tokens (same structure as _t2i_predict_v)."""
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
        """Predict velocity for video latent tokens."""
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

    def _append_text_to_cache(self, cache, t_idx, text_ids, device):
        """Append arbitrary text token IDs to KV cache, return new t_index."""
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
        """Append modality context tokens + end token to KV cache.

        Pattern: [context_token_0, ..., context_token_{n-1}, end_token]
        Context tokens share the same t_index; end token gets t_index + 1.
        Returns (next_logits, new_t_idx).
        """
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

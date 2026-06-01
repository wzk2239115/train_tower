from __future__ import annotations

import torch


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
    device = grid_hw.device
    B = grid_hw.shape[0]
    H = grid_hw[:, 0]
    W = grid_hw[:, 1]
    N = H * W
    N_total = N.sum()

    patch_to_sample = torch.repeat_interleave(torch.arange(B, device=device), N)
    patch_id_within_image = torch.arange(N_total, device=device)
    patch_id_within_image = patch_id_within_image - torch.cumsum(
        torch.cat([torch.tensor([0], device=device), N[:-1]]), dim=0
    )[patch_to_sample]

    W_per_patch = W[patch_to_sample]
    abs_x = patch_id_within_image % W_per_patch
    abs_y = patch_id_within_image // W_per_patch

    return abs_x, abs_y

from __future__ import annotations

import math

import torch


def count_image_slots(input_ids: torch.Tensor, img_start_token_id: int) -> int:
    return int((input_ids == img_start_token_id).sum().item())


def trim_pixel_values_to_image_slots(
    input_ids: torch.Tensor,
    pixel_values: list,
    *,
    img_start_token_id: int,
) -> list:
    """Keep only images that still have an ``<img>`` slot in the packed sequence."""
    if not pixel_values:
        return pixel_values
    num_slots = count_image_slots(input_ids, img_start_token_id)
    if num_slots <= 0:
        return []
    if len(pixel_values) > num_slots:
        return pixel_values[:num_slots]
    return pixel_values


def grid_hw_patch_total(grid_hw: torch.Tensor) -> int:
    if grid_hw is None or grid_hw.numel() == 0:
        return 0
    return int((grid_hw[:, 0] * grid_hw[:, 1]).sum().item())


def _factor_patch_grid(num_patches: int, ref_h: int, ref_w: int) -> tuple[int, int]:
    if num_patches <= 0:
        return 0, 0
    if ref_h <= 0 or ref_w <= 0:
        side = max(1, int(round(math.sqrt(num_patches))))
        return side, max(1, num_patches // side)

    ratio = ref_h / ref_w
    best: tuple[int, int] | None = None
    best_score = float("inf")
    for w in range(1, int(math.sqrt(num_patches)) + 1):
        if num_patches % w != 0:
            continue
        h = num_patches // w
        score = abs((h / w) - ratio)
        if score < best_score:
            best_score = score
            best = (h, w)
    if best is not None:
        return best
    side = max(1, int(round(math.sqrt(num_patches))))
    return side, max(1, num_patches // side)


def reconcile_grid_hw_with_patches(
    grid_hw: torch.Tensor,
    num_patches: int,
) -> torch.Tensor:
    """Trim or reshape ``grid_hw`` so patch totals match ``num_patches``."""
    if grid_hw is None or grid_hw.numel() == 0:
        return grid_hw
    if not isinstance(grid_hw, torch.Tensor):
        grid_hw = torch.tensor(grid_hw, dtype=torch.long)
    expected = grid_hw_patch_total(grid_hw)
    if expected == num_patches:
        return grid_hw

    kept: list[list[int]] = []
    total = 0
    for i in range(grid_hw.shape[0]):
        h, w = int(grid_hw[i, 0].item()), int(grid_hw[i, 1].item())
        n = h * w
        if total + n <= num_patches:
            kept.append([h, w])
            total += n
            continue
        remaining = num_patches - total
        if remaining > 0:
            h_part, w_part = _factor_patch_grid(remaining, h, w)
            if h_part * w_part == remaining:
                kept.append([h_part, w_part])
                total += remaining
        break

    if not kept:
        side = max(1, int(round(math.sqrt(max(num_patches, 1)))))
        h, w = _factor_patch_grid(num_patches, side, side)
        kept = [[h, w]]
    return torch.tensor(kept, dtype=grid_hw.dtype, device=grid_hw.device)


def reconcile_vision_inputs(
    pixel_values_flat: torch.Tensor,
    grid_hw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensure flattened pixels and ``grid_hw`` describe the same patch count."""
    num_patches = int(pixel_values_flat.shape[0])
    expected = grid_hw_patch_total(grid_hw)
    if expected == num_patches:
        return pixel_values_flat, grid_hw
    if expected > num_patches:
        grid_hw = reconcile_grid_hw_with_patches(grid_hw, num_patches)
        return pixel_values_flat, grid_hw
    return pixel_values_flat[:expected], grid_hw

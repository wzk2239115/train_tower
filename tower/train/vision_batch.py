from __future__ import annotations

import math

import torch

DEFAULT_VIT_SPATIAL_MERGE = 2


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


def vit_output_patch_total(grid_hw: torch.Tensor, *, spatial_merge: int = DEFAULT_VIT_SPATIAL_MERGE) -> int:
    """Patch count after ViT spatial merge (Conv stride = spatial_merge)."""
    if grid_hw is None or grid_hw.numel() == 0:
        return 0
    merge = max(int(spatial_merge), 1)
    total = 0
    for i in range(grid_hw.shape[0]):
        h = int(grid_hw[i, 0].item())
        w = int(grid_hw[i, 1].item())
        total += (h // merge) * (w // merge)
    return total


def _snap_hw(h: int, w: int, merge: int) -> tuple[int, int]:
    if h <= 0 or w <= 0:
        return 0, 0
    h2 = (h // merge) * merge
    w2 = (w // merge) * merge
    if h2 < merge or w2 < merge:
        return 0, 0
    return h2, w2


def _factor_patch_grid(
    num_patches: int,
    ref_h: int,
    ref_w: int,
    *,
    spatial_merge: int = 1,
) -> tuple[int, int]:
    if num_patches <= 0:
        return 0, 0

    merge = max(int(spatial_merge), 1)
    if merge > 1:
        num_patches = (num_patches // (merge * merge)) * (merge * merge)
        if num_patches <= 0:
            return merge, merge

    if ref_h <= 0 or ref_w <= 0:
        side = max(merge, int(round(math.sqrt(num_patches))))
        side = max(merge, (side // merge) * merge)
        w = side
        h = max(merge, (num_patches // w // merge) * merge)
        if h * w != num_patches:
            h, w = _factor_patch_grid(num_patches, side, side, spatial_merge=merge)
        return h, w

    ratio = ref_h / ref_w
    best: tuple[int, int] | None = None
    best_score = float("inf")
    start_w = merge if merge > 1 else 1
    step_w = merge if merge > 1 else 1
    for w in range(start_w, int(math.sqrt(num_patches)) + 1, step_w):
        if num_patches % w != 0:
            continue
        h = num_patches // w
        if merge > 1 and h % merge != 0:
            continue
        score = abs((h / w) - ratio)
        if score < best_score:
            best_score = score
            best = (h, w)
    if best is not None:
        return best

    side = max(merge, int(round(math.sqrt(num_patches))))
    side = max(merge, (side // merge) * merge)
    return _factor_patch_grid(num_patches, side, side, spatial_merge=merge)


def snap_grid_hw_to_vit_merge(
    pixel_values_flat: torch.Tensor,
    grid_hw: torch.Tensor,
    *,
    spatial_merge: int = DEFAULT_VIT_SPATIAL_MERGE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Snap each image grid to ``spatial_merge`` multiples and trim patch rows."""
    if grid_hw is None or grid_hw.numel() == 0:
        return pixel_values_flat, grid_hw
    if not isinstance(grid_hw, torch.Tensor):
        grid_hw = torch.tensor(grid_hw, dtype=torch.long)

    merge = max(int(spatial_merge), 1)
    kept: list[list[int]] = []
    chunks: list[torch.Tensor] = []
    offset = 0
    feat_dim = int(pixel_values_flat.shape[-1]) if pixel_values_flat.ndim == 2 else 0

    for i in range(grid_hw.shape[0]):
        h = int(grid_hw[i, 0].item())
        w = int(grid_hw[i, 1].item())
        n_old = h * w
        if n_old <= 0 or offset + n_old > pixel_values_flat.shape[0]:
            break

        h2, w2 = _snap_hw(h, w, merge)
        if h2 <= 0 or w2 <= 0:
            offset += n_old
            continue

        chunk = pixel_values_flat[offset : offset + n_old]
        if h2 == h and w2 == w:
            chunks.append(chunk)
        else:
            n_new = h2 * w2
            chunks.append(chunk.view(h, w, feat_dim)[:h2, :w2].reshape(n_new, feat_dim))
        kept.append([h2, w2])
        offset += n_old

    if not kept:
        empty = pixel_values_flat[:0]
        return empty, torch.zeros((0, 2), dtype=grid_hw.dtype, device=grid_hw.device)

    flat = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
    out_grid = torch.tensor(kept, dtype=grid_hw.dtype, device=grid_hw.device)
    return flat, out_grid


def reconcile_grid_hw_with_patches(
    grid_hw: torch.Tensor,
    num_patches: int,
    *,
    spatial_merge: int = DEFAULT_VIT_SPATIAL_MERGE,
) -> torch.Tensor:
    """Trim or reshape ``grid_hw`` so patch totals match ``num_patches``."""
    if grid_hw is None or grid_hw.numel() == 0:
        return grid_hw
    if not isinstance(grid_hw, torch.Tensor):
        grid_hw = torch.tensor(grid_hw, dtype=torch.long)
    expected = grid_hw_patch_total(grid_hw)
    if expected == num_patches:
        return grid_hw

    merge = max(int(spatial_merge), 1)
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
            h_part, w_part = _factor_patch_grid(remaining, h, w, spatial_merge=merge)
            if h_part * w_part == remaining:
                kept.append([h_part, w_part])
                total += remaining
        break

    if not kept:
        h, w = _factor_patch_grid(num_patches, 0, 0, spatial_merge=merge)
        kept = [[h, w]]
    return torch.tensor(kept, dtype=grid_hw.dtype, device=grid_hw.device)


def reconcile_vision_inputs(
    pixel_values_flat: torch.Tensor,
    grid_hw: torch.Tensor,
    *,
    spatial_merge: int = DEFAULT_VIT_SPATIAL_MERGE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensure flattened pixels and ``grid_hw`` match and are ViT-merge aligned."""
    num_patches = int(pixel_values_flat.shape[0])
    expected = grid_hw_patch_total(grid_hw)
    if expected > num_patches:
        grid_hw = reconcile_grid_hw_with_patches(
            grid_hw, num_patches, spatial_merge=spatial_merge
        )
    elif expected < num_patches:
        pixel_values_flat = pixel_values_flat[:expected]

    pixel_values_flat, grid_hw = snap_grid_hw_to_vit_merge(
        pixel_values_flat, grid_hw, spatial_merge=spatial_merge
    )
    return pixel_values_flat, grid_hw

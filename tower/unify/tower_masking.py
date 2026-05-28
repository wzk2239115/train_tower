"""Block-level mask for tower JEPA exit."""

from __future__ import annotations

import random

import torch


def sample_image_token_mask(
    num_tokens: int,
    *,
    min_ratio: float = 0.2,
    max_ratio: float = 0.5,
    min_span: int = 4,
    max_span: int = 32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return bool mask [num_tokens] for JEPA prediction targets."""
    if num_tokens <= 0:
        return torch.zeros(0, dtype=torch.bool, device=device)

    target = random.randint(
        max(1, int(num_tokens * min_ratio)),
        max(1, int(num_tokens * max_ratio)),
    )
    target = min(target, num_tokens)
    pred_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    covered = 0
    attempts = 0
    while covered < target and attempts < 64:
        attempts += 1
        span = random.randint(min_span, min(max_span, num_tokens))
        start = random.randint(0, max(0, num_tokens - span))
        for j in range(start, start + span):
            if not pred_mask[j]:
                pred_mask[j] = True
                covered += 1
            if covered >= target:
                break
    return pred_mask

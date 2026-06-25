"""Length-aware pack sampler for efficient sequence packing.

Instead of blindly packing ``per_device_train_batch_size`` samples into one
sequence (which causes ~80 % truncation waste when the packed length far
exceeds ``max_seq_length``), this sampler pre-groups samples into *packs*
whose estimated token lengths approximately fill ``max_seq_length``.

The sampler acts as a ``BatchSampler`` — each yielded element is a list of
dataset indices that together form one packed sequence.
"""

from __future__ import annotations

import random
from typing import Sequence

from torch.utils.data import BatchSampler


class LengthAwarePackSampler(BatchSampler):
    """BatchSampler that packs samples by estimated token length.

    Parameters
    ----------
    lengths : list[int]
        Estimated token length for each sample (same order as the dataset).
    max_seq_length : int
        Maximum packed sequence length.  Packs are filled to
        ``max_seq_length * pack_efficiency`` (default 0.95).
    num_replicas : int
        Number of distributed training processes.
    rank : int
        Rank of the current process.
    shuffle : bool
        Whether to shuffle packs each epoch.
    seed : int
        Base random seed for shuffling.
    pack_efficiency : float
        Target fraction of ``max_seq_length`` to fill per pack (0.0–1.0).
    epoch : int
        Starting epoch number.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        max_seq_length: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        pack_efficiency: float = 0.95,
        epoch: int = 0,
    ):
        self.lengths = list(lengths)
        self.max_seq_length = max_seq_length
        self.max_pack_length = int(max_seq_length * pack_efficiency)
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch
        self._packs: list[list[int]] = []
        self._build_packs()

    # ------------------------------------------------------------------ #
    #  Pack construction                                                  #
    # ------------------------------------------------------------------ #

    def _build_packs(self) -> None:
        """First-fit-decreasing bin packing.

        Sorts samples by estimated length (descending), then greedily places
        each into the first existing pack with enough remaining capacity.
        Falls back to creating a new pack when nothing fits.
        """
        rng = random.Random(self.seed + self.epoch)

        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)
        indices.sort(key=lambda i: self.lengths[i], reverse=True)

        max_len = self.max_pack_length
        packs: list[list[int]] = []
        pack_remaining: list[int] = []

        for idx in indices:
            sample_len = self.lengths[idx]
            placed = False
            for i in range(len(packs)):
                if pack_remaining[i] >= sample_len:
                    packs[i].append(idx)
                    pack_remaining[i] -= sample_len
                    placed = True
                    break
            if not placed:
                packs.append([idx])
                rem = max_len - sample_len
                pack_remaining.append(rem if rem > 0 else 0)

        if self.shuffle:
            rng.shuffle(packs)

        if self.num_replicas > 1:
            total = len(packs)
            if total % self.num_replicas != 0:
                pad = self.num_replicas - (total % self.num_replicas)
                packs = packs + packs[:pad]
            packs = packs[self.rank :: self.num_replicas]

        self._packs = packs

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def rebuild(self, lengths: Sequence[int], max_seq_length: int | None = None) -> None:
        """Rebuild packs with new lengths (e.g. after curriculum change)."""
        self.lengths = list(lengths)
        if max_seq_length is not None:
            self.max_seq_length = max_seq_length
            self.max_pack_length = int(max_seq_length * 0.95)
        self._build_packs()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._build_packs()

    def __iter__(self):
        for pack in self._packs:
            yield pack

    def __len__(self) -> int:
        return len(self._packs)

    @property
    def stats(self) -> dict:
        """Return packing statistics for logging."""
        if not self._packs:
            return {"num_packs": 0}
        pack_sizes = [len(p) for p in self._packs]
        pack_lens = [sum(self.lengths[i] for i in p) for p in self._packs]
        return {
            "num_packs": len(self._packs),
            "avg_pack_size": sum(pack_sizes) / len(pack_sizes),
            "max_pack_size": max(pack_sizes),
            "min_pack_size": min(pack_sizes),
            "avg_pack_length": sum(pack_lens) / len(pack_lens),
            "max_pack_length": max(pack_lens),
            "efficiency": sum(pack_lens) / (len(pack_lens) * self.max_seq_length),
        }

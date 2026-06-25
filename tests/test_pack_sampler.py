"""Tests for LengthAwarePackSampler and smart packing."""

import pytest

from tower.train.pack_sampler import LengthAwarePackSampler


class TestLengthAwarePackSampler:
    def test_basic_packing(self):
        lengths = [100] * 160
        sampler = LengthAwarePackSampler(
            lengths=lengths,
            max_seq_length=8192,
            num_replicas=1,
            rank=0,
            shuffle=False,
            seed=42,
        )
        packs = list(sampler)
        assert len(packs) > 0
        for pack in packs:
            total = sum(lengths[i] for i in pack)
            assert total <= 8192 * 0.95 + 100  # within one sample of target

    def test_all_samples_included(self):
        lengths = [500, 300, 200, 1000, 50, 100, 800, 150, 600, 400]
        sampler = LengthAwarePackSampler(
            lengths=lengths,
            max_seq_length=2000,
            num_replicas=1,
            rank=0,
            shuffle=False,
            seed=42,
        )
        packs = list(sampler)
        all_indices = set()
        for pack in packs:
            all_indices.update(pack)
        assert all_indices == set(range(len(lengths)))

    def test_no_pack_exceeds_limit(self):
        import random
        rng = random.Random(42)
        lengths = [rng.randint(50, 800) for _ in range(1000)]
        max_seq = 8192
        sampler = LengthAwarePackSampler(
            lengths=lengths,
            max_seq_length=max_seq,
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=42,
        )
        packs = list(sampler)
        for pack in packs:
            total = sum(lengths[i] for i in pack)
            assert total <= max_seq, f"Pack total {total} > {max_seq}"

    def test_oversized_sample_alone(self):
        lengths = [10000, 500, 300, 200]
        sampler = LengthAwarePackSampler(
            lengths=lengths,
            max_seq_length=8192,
            num_replicas=1,
            rank=0,
            shuffle=False,
            seed=42,
        )
        packs = list(sampler)
        assert len(packs) >= 1
        first_pack = packs[0]
        assert len(first_pack) == 1
        assert lengths[first_pack[0]] == 10000

    def test_distributed_split(self):
        lengths = [200] * 100
        num_replicas = 4
        all_indices = set()
        rank_pack_counts = []
        for rank in range(num_replicas):
            sampler = LengthAwarePackSampler(
                lengths=lengths,
                max_seq_length=8192,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                seed=42,
            )
            packs = list(sampler)
            rank_pack_counts.append(len(packs))
            for pack in packs:
                all_indices.update(pack)
        assert all(c == rank_pack_counts[0] for c in rank_pack_counts), rank_pack_counts
        assert all_indices == set(range(100))

    def test_rebuild(self):
        lengths1 = [100] * 160
        sampler = LengthAwarePackSampler(
            lengths=lengths1,
            max_seq_length=8192,
            num_replicas=1,
            rank=0,
            shuffle=False,
            seed=42,
        )
        first_count = len(sampler)
        lengths2 = [500] * 160
        sampler.rebuild(lengths2, max_seq_length=8192)
        second_count = len(sampler)
        assert second_count >= first_count, "Bigger samples should produce more packs"

    def test_stats(self):
        lengths = [500] * 160
        sampler = LengthAwarePackSampler(
            lengths=lengths,
            max_seq_length=8192,
            num_replicas=1,
            rank=0,
            shuffle=False,
            seed=42,
        )
        stats = sampler.stats
        assert stats["num_packs"] > 0
        assert stats["avg_pack_size"] > 0
        assert 0 < stats["efficiency"] <= 1.0

    def test_high_efficiency_packing(self):
        lengths = [500] * 1000
        sampler = LengthAwarePackSampler(
            lengths=lengths,
            max_seq_length=8192,
            num_replicas=1,
            rank=0,
            shuffle=False,
            seed=42,
        )
        stats = sampler.stats
        assert stats["efficiency"] > 0.85, f"Efficiency {stats['efficiency']:.2%} too low"

import torch

from tower.train.config import TrainConfig
from tower.train.packed_batch_monitor import compute_packed_batch_stats, step_peak_vram_score


def test_compute_packed_batch_stats():
    cfg = TrainConfig(
        stage="understanding_warmup",
        per_device_train_batch_size=8,
        max_seq_length=8192,
        max_pixels=4_194_304,
        gradient_checkpointing=True,
    )
    img_id = 999
    l = 128
    input_ids = torch.full((1, l), 1, dtype=torch.long)
    input_ids[0, 10:20] = img_id
    batch = {
        "input_ids": input_ids,
        "seq_boundaries": torch.tensor([0, 64, 128], dtype=torch.int32),
        "image_grid_hw": [torch.tensor([[16, 16], [8, 8]], dtype=torch.long)],
    }
    stats = compute_packed_batch_stats(batch, cfg, img_id)
    assert stats.packed_seq_length == 128
    assert stats.num_packed_samples == 2
    assert stats.num_images == 2
    assert stats.num_vision_tokens == 10
    assert stats.peak_vram_score == step_peak_vram_score(
        packed_seq_length=128,
        num_packed_samples=2,
        cfg=cfg,
    )


def test_step_peak_vram_score_warns_on_long_seq():
    cfg = TrainConfig(stage="understanding_warmup", max_seq_length=8192, max_pixels=4_194_304)
    at_anchor = step_peak_vram_score(packed_seq_length=8192, num_packed_samples=8, cfg=cfg)
    over = step_peak_vram_score(packed_seq_length=8192, num_packed_samples=16, cfg=cfg)
    assert at_anchor <= 1.0 + 1e-9
    assert over > 1.0

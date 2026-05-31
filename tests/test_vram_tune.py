from tower.train.config import TrainConfig
from tower.train.vram_tune import (
    _get_anchor,
    _max_vram_score,
    apply_h800_vram_tune,
    peak_vram_score,
)


def test_max_uw_clamped_to_stable_envelope(monkeypatch):
    monkeypatch.setenv("TOWER_H800_VRAM_TUNE", "1")
    monkeypatch.setenv("H800_PROFILE", "max")
    monkeypatch.setenv("TOWER_TARGET_GLOBAL_BATCH", "160")
    monkeypatch.setenv("WORLD_SIZE", "8")
    cfg = TrainConfig(
        stage="understanding_warmup",
        per_device_train_batch_size=10,
        gradient_accumulation_steps=2,
        max_pixels=6_291_456,
        max_seq_length=8192,
        gradient_checkpointing=True,
    )
    anchor = _get_anchor("understanding_warmup")
    assert anchor is not None
    limit = _max_vram_score()
    assert peak_vram_score(cfg, anchor) > limit

    tuned = apply_h800_vram_tune(cfg)
    assert peak_vram_score(tuned, anchor) <= limit + 1e-9
    # 0.95 headroom: stable anchor pack=8 is tuned down to 7 before pixels/accum rebalance.
    assert tuned.per_device_train_batch_size == 7
    assert tuned.max_pixels == 4_194_304
    assert tuned.gradient_accumulation_steps == 3  # target 160/8 = 20 per GPU → ceil(20/7)


def test_stable_uw_unchanged(monkeypatch):
    monkeypatch.setenv("TOWER_H800_VRAM_TUNE", "1")
    monkeypatch.setenv("H800_PROFILE", "max")
    cfg = TrainConfig(
        stage="understanding_warmup",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        max_pixels=4_194_304,
        max_seq_length=8192,
        gradient_checkpointing=True,
    )
    anchor = _get_anchor("understanding_warmup")
    assert anchor is not None
    limit = _max_vram_score()
    assert peak_vram_score(cfg, anchor) > limit

    tuned = apply_h800_vram_tune(cfg)
    assert peak_vram_score(tuned, anchor) <= limit + 1e-9
    assert tuned.per_device_train_batch_size == 7
    assert tuned.max_pixels == 4_194_304
    assert tuned.gradient_accumulation_steps == 2


def test_extreme_uw_keeps_grad_ckpt_off(monkeypatch):
    monkeypatch.setenv("TOWER_H800_VRAM_TUNE", "1")
    monkeypatch.setenv("H800_PROFILE", "extreme")
    monkeypatch.setenv("TOWER_TARGET_GLOBAL_BATCH", "160")
    monkeypatch.setenv("WORLD_SIZE", "8")
    cfg = TrainConfig(
        stage="understanding_warmup",
        per_device_train_batch_size=10,
        gradient_accumulation_steps=2,
        max_pixels=6_291_456,
        max_seq_length=8192,
        gradient_checkpointing=False,
    )
    anchor = _get_anchor("understanding_warmup")
    assert anchor is not None
    limit = _max_vram_score()
    assert limit == 3.25

    tuned = apply_h800_vram_tune(cfg)
    assert tuned.gradient_checkpointing is False
    assert peak_vram_score(tuned, anchor) <= limit + 1e-9
    assert tuned.per_device_train_batch_size == 10
    assert tuned.max_pixels == 6_291_456
    assert tuned.gradient_accumulation_steps == 2

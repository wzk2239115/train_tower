from tower.train.config import TrainConfig
from tower.train.diagnostics import log_startup_summary, sdpa_block_attn_status


def test_sdpa_block_attn_status_keys():
    status = sdpa_block_attn_status()
    assert "state" in status
    assert status["state"] in ("disabled", "active", "pending")


def test_log_startup_summary_rank0_only(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("H800_PROFILE", "extreme")
    cfg = TrainConfig(
        stage="understanding_warmup",
        per_device_train_batch_size=10,
        gradient_accumulation_steps=2,
        max_seq_length=8192,
        max_pixels=6_291_456,
        gradient_checkpointing=False,
    )
    log_startup_summary(cfg)


def test_log_startup_summary_skips_nonzero_rank(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    cfg = TrainConfig(stage="understanding_warmup")
    log_startup_summary(cfg)

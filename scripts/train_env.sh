# Shared env → CLI flags for torchrun training scripts.
# Source from scripts/train_*.sh (not executed directly).

train_env_extra_args() {
  TRAIN_ENV_EXTRA=()
  if [[ -n "${DATASETS:-}" ]]; then
    TRAIN_ENV_EXTRA+=(--datasets "$DATASETS")
  fi
  if [[ -n "${MAX_STEPS:-}" ]]; then
    TRAIN_ENV_EXTRA+=(--max-steps "$MAX_STEPS")
  fi
  if [[ -n "${OUTPUT_DIR:-}" ]]; then
    TRAIN_ENV_EXTRA+=(--output-dir "$OUTPUT_DIR")
  fi
}

train_env_log_overrides() {
  local parts=()
  [[ -n "${DATASETS:-}" ]] && parts+=("datasets=${DATASETS}")
  [[ -n "${MAX_STEPS:-}" ]] && parts+=("max_steps=${MAX_STEPS}")
  [[ -n "${OUTPUT_DIR:-}" ]] && parts+=("output_dir=${OUTPUT_DIR}")
  if ((${#parts[@]})); then
    echo "[train_env] env overrides: ${parts[*]}"
  fi
}

train_env_setup() {
  export NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-29500}"
  if [[ "${NUM_GPUS}" -le 1 && "${USE_DEEPSPEED:-0}" != "1" ]]; then
    export TOWER_NO_DEEPSPEED=1
  fi
  train_env_extra_args
  train_env_log_overrides
}

train_env_print_training_summary() {
  local mode="$1"   # stage | config
  local value="$2"  # stage name or config path
  python - "$mode" "$value" "${DATASETS:-}" <<'PY'
import json
import re
import sys
from pathlib import Path


def _load_cfg(mode: str, value: str):
    from tower.train.config import load_train_config

    if mode == "stage":
        return load_train_config(stage=value)
    if mode == "config":
        return load_train_config(config_path=Path(value))
    raise ValueError(f"Unsupported mode: {mode}")


def _fmt_samples(n):
    if n is None:
        return "unknown"
    return f"{int(n):,}"


try:
    mode, value, datasets_override = sys.argv[1], sys.argv[2], sys.argv[3]
    cfg = _load_cfg(mode, value)
except Exception as exc:
    print(f"[train_plan] failed to load training config: {exc}")
    sys.exit(0)

datasets_raw = datasets_override.strip() if datasets_override else str(getattr(cfg, "datasets", "")).strip()
datasets = [d.strip() for d in datasets_raw.split(",") if d.strip()]

print(
    f"[train_plan] stage={cfg.stage} init_mode={cfg.init_mode} "
    f"max_steps={cfg.max_steps} output_dir={cfg.output_dir}"
)

manifest_path = Path("data/processed/manifest.json")
manifest = {}
if manifest_path.is_file():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[train_plan] warning: failed to read manifest.json: {exc}")

if not datasets:
    print("[train_plan] datasets: <empty>")
    sys.exit(0)

print("[train_plan] datasets:")
total_samples = 0
known_sample_rows = 0
for key in datasets:
    m = re.match(r"^(.*)_(pt|mt|sft)$", key)
    if not m:
        print(f"  - {key}: samples=unknown (dataset key should end with _pt/_mt/_sft)")
        continue

    dataset_name, stage_name = m.group(1), m.group(2)
    entry = manifest.get(dataset_name) or {}
    samples = entry.get("samples")
    stage_path = (entry.get("stages") or {}).get(stage_name)
    if samples is not None:
        total_samples += int(samples)
        known_sample_rows += 1
    path_hint = stage_path or "missing in manifest"
    print(f"  - {key}: samples={_fmt_samples(samples)} file={path_hint}")

if known_sample_rows:
    print(f"[train_plan] total_samples={total_samples:,} ({known_sample_rows} datasets with known counts)")
else:
    print("[train_plan] total_samples=unknown (manifest missing or no matching entries)")
PY
}

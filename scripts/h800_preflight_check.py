#!/usr/bin/env python3
"""H800/H100 compute preflight: git HEAD, SDPA patch, GPUs, benchmark hint.

Usage:
  python scripts/h800_preflight_check.py
  python scripts/h800_preflight_check.py --run-benchmark
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _git_head_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _gpu_summary() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
        if not lines:
            return "no GPUs"
        first = lines[0]
        return f"{len(lines)}× {first}" if all(ln == first for ln in lines) else f"{len(lines)} GPUs"
    except Exception:
        return "nvidia-smi unavailable"


def _attention_module_ok() -> bool:
    path = os.path.join(ROOT, "tower", "unify", "attention.py")
    try:
        with open(path, encoding="utf-8") as f:
            return "sdpa_block_attention_forward" in f.read()
    except OSError:
        return False


def _sdpa_status() -> dict[str, str | bool]:
    disabled = os.environ.get("TOWER_DISABLE_SDPA_BLOCK_ATTN", "0") == "1"
    compat_path = os.path.join(ROOT, "tower", "unify", "compat.py")
    has_patch = False
    try:
        with open(compat_path, encoding="utf-8") as f:
            has_patch = "_patch_block_causal_attention" in f.read()
    except OSError:
        pass
    attn_ok = _attention_module_ok()
    if disabled:
        state = "disabled"
    elif has_patch and attn_ok:
        state = "pending"  # applied at model build; training log confirms active
    else:
        state = "missing"
    return {
        "state": state,
        "patch_source": has_patch and attn_ok,
        "backends": os.environ.get("TOWER_SDPA_BACKENDS", "efficient,cudnn"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="H800 compute preflight check")
    parser.add_argument(
        "--run-benchmark",
        action="store_true",
        help="Run scripts/benchmark_block_attn.py on CUDA if available",
    )
    args = parser.parse_args()

    print(f"[preflight] git HEAD={_git_head_short()}")
    print(f"[preflight] GPUs: {_gpu_summary()}")

    attn_ok = _attention_module_ok()
    sdpa = _sdpa_status()
    print(
        f"[preflight] SDPA: state={sdpa['state']} patch_source={sdpa['patch_source']} "
        f"backends={sdpa['backends']} attention.py={'ok' if attn_ok else 'MISSING'}"
    )
    if os.environ.get("TOWER_DISABLE_SDPA_BLOCK_ATTN", "0") == "1":
        print("[preflight] WARN: TOWER_DISABLE_SDPA_BLOCK_ATTN=1 (eager path, OOM risk on long seq)")

    profile = os.environ.get("H800_PROFILE", "<unset>")
    print(f"[preflight] H800_PROFILE={profile}")

    bench = os.path.join(ROOT, "scripts", "benchmark_block_attn.py")
    if args.run_benchmark:
        if not os.path.isfile(bench):
            print(f"[preflight] benchmark script missing: {bench}")
            return 1
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
            print("[preflight] torch unavailable; benchmark on cpu")
        print(f"[preflight] running benchmark_block_attn.py --device {device} ...")
        rc = subprocess.call(
            [sys.executable, bench, "--device", device, "--seq-len", "8192", "--heads", "12"],
            cwd=ROOT,
        )
        if rc != 0:
            print(f"[preflight] benchmark exited {rc}")
            return rc
    else:
        print(
            "[preflight] hint: python scripts/benchmark_block_attn.py "
            "--device cuda --seq-len 8192 --heads 12"
        )
        print("[preflight] hint: python scripts/h800_preflight_check.py --run-benchmark")

    if not attn_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

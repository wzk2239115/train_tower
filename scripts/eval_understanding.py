#!/usr/bin/env python3
"""Evaluate text/image understanding from exported train_tower artifacts."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.uint8  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tower.config import PROJECT_ROOT
from tower.models.neo_unify.conversation import get_conv_template
from tower.models.neo_unify.utils import load_image_native
from tower.train.experiment_profile import load_train_config_from_experiment
from tower.train.registry import load_manifest
from tower.unify.build import build_model_and_tokenizer
from tower.unify.train_model import SenseNovaTrainModel


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _load_model(args: argparse.Namespace):
    cfg = load_train_config_from_experiment(args.profile)
    cfg.deepspeed = None
    cfg.gradient_checkpointing = False
    cfg.bf16 = not args.fp32
    model, tokenizer = build_model_and_tokenizer(cfg)
    wrapper = SenseNovaTrainModel(model, cfg)
    model = wrapper.model

    backbone = _resolve(args.checkpoint_dir) / "backbone.pt"
    if not backbone.is_file():
        raise FileNotFoundError(f"Missing backbone artifact: {backbone}")
    state = torch.load(backbone, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded backbone: {backbone}")
    print(f"State dict: missing={len(missing)} unexpected={len(unexpected)}")

    dtype = torch.float32 if args.fp32 else torch.bfloat16
    model.to(device=torch.device(args.device), dtype=dtype)
    model.eval()
    model.config.use_cache = True
    model.language_model.config.use_cache = True
    if hasattr(model.language_model, "gradient_checkpointing_disable"):
        model.language_model.gradient_checkpointing_disable()
    return model, tokenizer


def _dataset_jsonl(reg_key: str) -> Path:
    dataset_key, stage = reg_key.rsplit("_", 1)
    manifest = load_manifest()
    entry = manifest.get(dataset_key)
    if not entry:
        raise KeyError(f"Unknown dataset key in manifest: {dataset_key}")
    rel = entry.get("stages", {}).get(stage)
    if not rel:
        raise KeyError(f"Dataset {dataset_key!r} has no stage {stage!r}")
    return _resolve(rel)


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _first_qa(record: dict[str, Any]) -> tuple[str, str] | None:
    convs = record.get("conversations") or []
    question = None
    for turn in convs:
        role = str(turn.get("from", "")).lower()
        value = str(turn.get("value", ""))
        if role in ("human", "user") and question is None:
            question = value
        elif role in ("gpt", "assistant") and question is not None:
            return question, value
    return None


def _image_paths(record: dict[str, Any], jsonl_path: Path) -> list[Path]:
    raw = record.get("image")
    if not raw:
        return []
    values = raw if isinstance(raw, list) else [raw]
    paths: list[Path] = []
    for item in values:
        if not isinstance(item, str) or not item.strip():
            continue
        p = Path(item)
        candidates = []
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.extend([
                PROJECT_ROOT / item,
                PROJECT_ROOT / "data" / item,
                jsonl_path.parent / item,
                PROJECT_ROOT / "data" / "images" / item,
            ])
        for cand in candidates:
            if cand.is_file():
                paths.append(cand.resolve())
                break
    return paths


def _load_samples(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]]]:
    jsonl_path = _resolve(args.jsonl) if args.jsonl else _dataset_jsonl(args.dataset)
    rows = []
    for rec in _iter_jsonl(jsonl_path):
        qa = _first_qa(rec)
        if qa is None:
            continue
        if args.require_image and not rec.get("image"):
            continue
        rows.append(rec)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    return jsonl_path, rows[: args.limit]


def _normalize_for_f1(text: str) -> list[str]:
    text = text.lower()
    if re.search(r"[\u4e00-\u9fff]", text):
        return [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
    return re.findall(r"[a-z0-9]+", text)


def _f1(prediction: str, reference: str) -> float:
    pred = _normalize_for_f1(prediction)
    ref = _normalize_for_f1(reference)
    if not pred or not ref:
        return 0.0
    common = {}
    for tok in pred:
        common[tok] = common.get(tok, 0) + 1
    overlap = 0
    for tok in ref:
        if common.get(tok, 0) > 0:
            overlap += 1
            common[tok] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def _build_text_prompt(model, question: str) -> str:
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    return template.get_prompt()


@torch.no_grad()
def _predict_text(model, tokenizer, question: str, args: argparse.Namespace) -> str:
    query = _build_text_prompt(model, question)
    input_ids = tokenizer(query, return_tensors="pt")["input_ids"].to(model.device)
    indexes = model.get_thw_indexes(input_ids[0], None)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    generation = model.generate(
        pixel_values=None,
        input_ids=input_ids,
        grid_hw=None,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    del indexes
    text = tokenizer.decode(generation[0], skip_special_tokens=True)
    sep = get_conv_template(model.template).sep.strip()
    return text.split(sep)[0].strip()


@torch.no_grad()
def _predict_image(model, tokenizer, question: str, images: list[Path], args: argparse.Namespace) -> str:
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    model.img_start_token_id = tokenizer.convert_tokens_to_ids("<img>")

    pixel_values = []
    grid_hw = []
    for image in images[: args.max_images]:
        pv, ghw = load_image_native(
            image,
            model.patch_size,
            model.downsample_ratio,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            upscale=False,
        )
        pixel_values.append(pv.to(model.device, dtype=model.dtype))
        grid_hw.append(ghw.to(model.device))

    if not pixel_values:
        return _predict_text(model, tokenizer, question.replace("<image>", "").strip(), args)

    pv_tensor = torch.cat(pixel_values, dim=0)
    ghw_tensor = torch.cat(grid_hw, dim=0)
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
    }
    return model.chat(
        tokenizer,
        pv_tensor,
        question,
        generation_config,
        grid_hw=ghw_tensor,
        verbose=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="500m_continuous")
    parser.add_argument("--checkpoint-dir", default="outputs/pretrain/500m_super_omni/checkpoint")
    parser.add_argument("--dataset", default="textcaps_mt", help="Processed reg key, e.g. textcaps_mt/docvqa_sft")
    parser.add_argument("--jsonl", default=None, help="Explicit processed JSONL path")
    parser.add_argument("--out-dir", default="outputs/eval_understanding/500m_super_omni")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--require-image", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp32", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    jsonl_path, samples = _load_samples(args)
    if not samples:
        raise RuntimeError(f"No evaluable samples from {jsonl_path}")

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset if not args.jsonl else jsonl_path.stem}_predictions.jsonl"

    model, tokenizer = _load_model(args)
    rows = []
    start_all = time.time()
    for i, rec in enumerate(samples):
        qa = _first_qa(rec)
        assert qa is not None
        question, reference = qa
        images = _image_paths(rec, jsonl_path)
        start = time.time()
        try:
            if images:
                prediction = _predict_image(model, tokenizer, question, images, args)
            else:
                prediction = _predict_text(model, tokenizer, question, args)
            error = None
        except Exception as exc:
            prediction = ""
            error = repr(exc)
        score = _f1(prediction, reference)
        row = {
            "index": i,
            "id": rec.get("id", ""),
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "f1_overlap": round(score, 4),
            "images": [str(p) for p in images],
            "error": error,
            "seconds": round(time.time() - start, 2),
        }
        rows.append(row)
        print(f"[{i + 1}/{len(samples)}] f1={score:.3f} id={row['id']} error={error}", flush=True)
        if prediction:
            print("  pred:", prediction[:200].replace("\n", " "), flush=True)

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    valid = [r for r in rows if not r["error"]]
    mean_f1 = sum(float(r["f1_overlap"]) for r in valid) / max(len(valid), 1)
    summary = {
        "dataset": args.dataset,
        "jsonl": str(jsonl_path),
        "samples": len(rows),
        "valid": len(valid),
        "mean_f1_overlap": round(mean_f1, 4),
        "seconds": round(time.time() - start_all, 2),
        "predictions": str(out_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

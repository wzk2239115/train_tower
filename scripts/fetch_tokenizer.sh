#!/usr/bin/env bash
# Download Qwen3 tokenizer files only (no model weights).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/configs/tokenizer/qwen3"
mkdir -p "$OUT"

# Use mirror when HF Hub is unreachable (override: export HF_ENDPOINT=https://huggingface.co)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

SENSENOVA_TOK="${ROOT}/third_party/SenseNova-U1/models/SenseNova-U1-8B-MoT"
if [[ -f "${SENSENOVA_TOK}/vocab.json" ]]; then
  echo "Copying tokenizer from local SenseNova-U1-8B-MoT ..."
  for f in tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json added_tokens.json chat_template.jinja; do
    [[ -f "${SENSENOVA_TOK}/${f}" ]] && cp "${SENSENOVA_TOK}/${f}" "${OUT}/"
  done
else
  echo "Downloading Qwen3-8B tokenizer from ${HF_ENDPOINT} ..."
  hf download Qwen/Qwen3-8B-Base \
    tokenizer.json \
    tokenizer_config.json \
    vocab.json \
    merges.txt \
    special_tokens_map.json \
    added_tokens.json \
    chat_template.jinja \
    --local-dir "$OUT"
fi

if find "$OUT" -name '*.safetensors' -o -name '*.bin' 2>/dev/null | grep -q .; then
  echo "ERROR: weight files found in ${OUT}" >&2
  exit 1
fi
echo "Tokenizer ready at ${OUT}"

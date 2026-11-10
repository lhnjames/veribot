#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source activate.sh

export JUDGE_API_KEY="${JUDGE_API_KEY:-${TEACHER_API_KEY:-}}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.deepseek.com}"
export JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-flash}"

DATA_DIR="${1:-outputs/unified_full}"
OUT_DIR="${2:-outputs/veriroute_lite}"

python scripts/build_cold_start_sft.py \
  --input "${DATA_DIR}/train.jsonl" \
  --output "${OUT_DIR}/cold_start_sft.jsonl" \
  --cache "${OUT_DIR}/teacher_cache"

python -m verifiable_bot.train_sft \
  --data "${OUT_DIR}/cold_start_sft.jsonl" \
  --output "${OUT_DIR}/sft"

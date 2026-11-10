#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source activate.sh

export JUDGE_API_KEY="${JUDGE_API_KEY:-${TEACHER_API_KEY:-}}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.deepseek.com}"
export JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-flash}"

DATA_DIR="${1:-outputs/unified_full}"
OUT_DIR="${2:-outputs/veriroute_lite}"

python -m verifiable_bot.train_grpo \
  --data "${DATA_DIR}/train.jsonl" \
  --sft-merged "${OUT_DIR}/sft/merged" \
  --output "${OUT_DIR}/grpo" \
  --judge-cache "${OUT_DIR}/judge_cache"

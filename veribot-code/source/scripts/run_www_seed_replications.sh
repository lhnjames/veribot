#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"

for dataset in fox8 botsim; do
  if [[ "$dataset" == fox8 ]]; then
    graph="data/processed/fox8_23_qwen_v1/graph.pt"
  else
    graph="data/processed/botsim_24_qwen_v1/graph.pt"
  fi
  for seed in 43 44; do
    output="source/hgt_import/retrain_runs/www_${dataset}_qwen_v1_seed${seed}"
    mkdir -p "$output"
    "$PYTHON" -u source/scripts/train_www_branches.py \
      --graph "$graph" --output "$output" --seed "$seed" \
      > "$output/train.log" 2>&1
    echo "completed dataset=$dataset seed=$seed output=$output" >&2
  done
done

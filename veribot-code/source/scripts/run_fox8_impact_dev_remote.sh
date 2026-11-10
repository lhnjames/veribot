#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/private_data/hanning/VeriBot/.venv/bin/python
for seed in 0 1 2; do
  out="output/target_evidence_fox8/impact_dev_seed${seed}"
  mkdir -p "$out"
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 "$PYTHON" \
    source/scripts/train_target_evidence_fox8_dev.py \
    --device cuda --epochs 40 --patience 8 --seed "$seed" \
    --sparsity 0.1 --counterfactual-weight 0.2 \
    --counterfactual-margin 0.5 --initial-gate-bias 0.0 \
    --intervention-impact-weight 0.5 \
    --intervention-impact-margin 0.5 \
    --intervention-sample-per-class 16 \
    --output "$out" > "$out/train.log" 2>&1
  printf 'completed dev-only impact seed %s\n' "$seed"
done

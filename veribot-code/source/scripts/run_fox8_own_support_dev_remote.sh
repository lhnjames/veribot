#!/usr/bin/env bash
set -euo pipefail

audit_pid="${1:?pass the active class-specific audit PID}"
while kill -0 "$audit_pid" 2>/dev/null; do
  sleep 20
done

python_bin=/root/private_data/hanning/VeriBot/.venv/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=3
for seed in 0 1 2; do
  run="output/target_evidence_fox8/own_support_dev_seed${seed}"
  mkdir -p "$run"
  "$python_bin" source/scripts/train_target_evidence_fox8_dev.py \
    --device cuda --epochs 40 --patience 8 --seed "$seed" \
    --sparsity 0.1 --counterfactual-weight 0.2 \
    --counterfactual-margin 0.5 --initial-gate-bias 0.0 \
    --own-class-support-weight 0.5 --own-class-support-margin 0.5 \
    --output "$run" > "$run/train.log" 2>&1
  echo "completed dev-only own-class support seed ${seed}"
done

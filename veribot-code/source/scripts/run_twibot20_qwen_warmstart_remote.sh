#!/usr/bin/env bash
set -euo pipefail

previous_pipeline_pid="${1:?pass the Qwen HGT pipeline PID}"
while kill -0 "$previous_pipeline_pid" 2>/dev/null; do
  sleep 20
done

python_bin=/root/private_data/hanning/VeriBot/.venv/bin/python
bundle=data/processed/twibot20/native_graph_bundle.npz
features=data/processed/twibot20/qwen_labeled_embeddings.npy
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=3

for seed in 0 1 2; do
  run="output/twibot20_qwen_warmstart/dev_seed${seed}"
  initial="output/twibot20_native_hgt/dev_seed${seed}/model.pt"
  mkdir -p "$run"
  "$python_bin" source/scripts/train_twibot20_native_hgt_dev.py \
    --bundle "$bundle" --text-features "$features" --output "$run" \
    --initialize-from "$initial" --seed "$seed" \
    --epochs 100 --patience 20 --hidden 128 --device cuda \
    > "$run/train.log" 2>&1
  "$python_bin" source/scripts/eval_twibot20_native_hgt_frozen.py \
    --bundle "$bundle" --text-features "$features" --run "$run" --device cuda \
    > "$run/test_eval.log" 2>&1
  echo "completed numeric-initialized Qwen HGT seed ${seed}"
done

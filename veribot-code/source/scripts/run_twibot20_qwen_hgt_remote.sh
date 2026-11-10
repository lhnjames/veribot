#!/usr/bin/env bash
set -euo pipefail

previous_pipeline_pid="${1:?pass the text-hash pipeline PID}"
while kill -0 "$previous_pipeline_pid" 2>/dev/null; do
  sleep 20
done

python_bin=/root/private_data/hanning/VeriBot/.venv/bin/python
bundle=data/processed/twibot20/native_graph_bundle.npz
features=data/processed/twibot20/qwen_labeled_embeddings.npy
model=/root/private_data/hanning/VeriBot/data/models/Qwen3-0.6B
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=3

"$python_bin" source/scripts/build_twibot20_qwen_labeled.py \
  --zip data/raw/twibot20/Twibot-20.zip --bundle "$bundle" \
  --model "$model" --output "$features" --batch-size 32 \
  > data/processed/twibot20/build_qwen_labeled.log 2>&1

for seed in 0 1 2; do
  run="output/twibot20_qwen_hgt/dev_seed${seed}"
  mkdir -p "$run"
  "$python_bin" source/scripts/train_twibot20_native_hgt_dev.py \
    --bundle "$bundle" --text-features "$features" --output "$run" \
    --seed "$seed" --epochs 200 --patience 20 --hidden 128 --device cuda \
    > "$run/train.log" 2>&1
  "$python_bin" source/scripts/eval_twibot20_native_hgt_frozen.py \
    --bundle "$bundle" --text-features "$features" --run "$run" --device cuda \
    > "$run/test_eval.log" 2>&1
  echo "completed frozen-Qwen HGT seed ${seed}"
done

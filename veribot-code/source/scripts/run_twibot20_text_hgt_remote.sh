#!/usr/bin/env bash
set -euo pipefail

# Run only on the remote GPU host after the source-checked text builder exits.
builder_pid="${1:?pass the text-builder PID}"
while kill -0 "$builder_pid" 2>/dev/null; do
  sleep 20
done

bundle=data/processed/twibot20/native_graph_bundle.npz
text=data/processed/twibot20/text_hash256.npz
test -s "$text"
test -s data/processed/twibot20/text_hash256.json
python_bin=/root/private_data/hanning/VeriBot/.venv/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=3

for seed in 0 1 2; do
  run="output/twibot20_text_hgt/dev_seed${seed}"
  mkdir -p "$run"
  "$python_bin" source/scripts/train_twibot20_native_hgt_dev.py \
    --bundle "$bundle" --text-features "$text" --output "$run" \
    --seed "$seed" --epochs 200 --patience 20 --hidden 128 --device cuda \
    > "$run/train.log" 2>&1
  "$python_bin" source/scripts/eval_twibot20_native_hgt_frozen.py \
    --bundle "$bundle" --text-features "$text" --run "$run" --device cuda \
    > "$run/test_eval.log" 2>&1
  echo "completed text-HGT seed ${seed}"
done

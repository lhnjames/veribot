#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${VERIBOT_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
export VERIBOT_TWIBOT20_TRAIN="${VERIBOT_TWIBOT20_TRAIN:-$PROJECT_DIR/data/raw/twibot20/train.json}"
export VERIBOT_QWEN_MODEL="${VERIBOT_QWEN_MODEL:-$PROJECT_DIR/data/models/Qwen3-0.6B}"
export VERIBOT_QWEN_REVISION="${VERIBOT_QWEN_REVISION:-c1899de289a04d12100db370d81485cdf75e47ca}"
RUN_ID="${VERIBOT_HGT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$PROJECT_DIR/source/hgt_import/retrain_runs/$RUN_ID"
export VERIBOT_HGT_DATA_DIR="$RUN_DIR/data"
export VERIBOT_HGT_OUT_DIR="$RUN_DIR/checkpoints"
mkdir -p "$VERIBOT_HGT_DATA_DIR" "$VERIBOT_HGT_OUT_DIR"

test -f "$VERIBOT_TWIBOT20_TRAIN" || { echo "Missing $VERIBOT_TWIBOT20_TRAIN" >&2; exit 2; }
test -f "$VERIBOT_QWEN_MODEL/model.safetensors" || { echo "Missing Qwen model" >&2; exit 2; }
"$PYTHON" -c 'import torch, torch_geometric, transformers, sklearn; assert torch.cuda.is_available(); print("CUDA:", torch.cuda.get_device_name(0))'
sha256sum "$VERIBOT_TWIBOT20_TRAIN" | tee "$RUN_DIR/input_sha256.txt"

"$PYTHON" -u "$PROJECT_DIR/source/hgt_import/build_features_and_graph.py" 2>&1 | tee "$RUN_DIR/build_graph.log"
"$PYTHON" -u "$PROJECT_DIR/source/hgt_import/train_clean_hgt.py" 2>&1 | tee "$RUN_DIR/train_hgt.log"
echo "Completed: $RUN_DIR"

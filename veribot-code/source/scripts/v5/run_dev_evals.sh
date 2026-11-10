#!/usr/bin/env bash
set -u
cd /path/to/project_root
source activate.sh >/dev/null 2>&1
BASE=$(grep -oE "/path/to/data_root/\S+snapshots/\S+" configs/grpo_v4_0.6b.yaml | head -1)
BEST=outputs/v4/e2_sft/run_0.6b/best_lora
OUT=outputs/v4/reports/dev_eval.jsonl
mkdir -p outputs/v4/reports
rm -f "$OUT"
export EVAL_BATCH=24
echo "[run] BASE=$BASE OUT=$OUT $(date)"

# SFT best_lora (E2) — LoRA on base, no merge
PYTHONPATH=. python3 scripts/v5/eval_dev.py --base "$BASE" --checkpoint "$BEST" \
    --tag sft_best_epoch2 --max-new-tokens 256 --out "$OUT"

# E4 GRPO checkpoints — LoRA on the E2-MERGED base (merge best_lora first). Fresh process each = safe.
for c in 500 550 600; do
  PYTHONPATH=. python3 scripts/v5/eval_dev.py --base "$BASE" --merge-lora "$BEST" \
      --checkpoint "outputs/v4/grpo/run_0.6b/checkpoint-$c" --tag "e4_grpo_ckpt$c" \
      --max-new-tokens 256 --out "$OUT"
done

echo "[run] DONE $(date)"
echo "=== SUMMARY ==="
python3 - <<'PY'
import json
for l in open("outputs/v4/reports/dev_eval.jsonl"):
    d=json.loads(l)
    print(f"{d['tag']:20s} acc={d['acc']:.4f} F1_bot={d['f1_bot']:.4f} P_bot={d['precision_bot']:.4f} "
          f"R_bot={d['recall_bot']:.4f} macroF1={d['macro_f1']:.4f} balAcc={d['balanced_acc']:.4f} "
          f"fmt={d['format_valid_rate']:.3f} predBot={d['pred_bot_rate']:.3f}")
PY

#!/bin/bash
# ParaCloud SFT matrix — Qwen3-0.6B, one A800 per array element. DORMANT per current direction
# (experiment doc §12.2: ParaCloud runs E3/E5 GRPO, not the SFT hyperparameter sweep). Kept ready.
# If ever used: submit ONLY after (1) v0 GPU single-batch smoke passes, (2) WIT v7 excluded until its
# weighted-loss smoke passes. Array 0-6 = the 7 non-WIT variants.
#
#SBATCH -J vb_sft_matrix
#SBATCH -p gpu_a800
#SBATCH --qos=gpugpu
#SBATCH -N1 -n1 --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH --array=0-6
#SBATCH -o /path/to/cluster_workspace/veriroute_v4_parallel/logs/%x_%A_%a.out
#SBATCH -e /path/to/cluster_workspace/veriroute_v4_parallel/logs/%x_%A_%a.out

set -euo pipefail
ROOT=/path/to/cluster_workspace/veriroute_v4_parallel
PY=/path/to/cluster_workspace/envs/vb/bin/python

VARIANTS=(v0_baseline_seed0 v1_baseline_seed1 v2_baseline_seed2 \
          v3_lr5e-5 v4_lr1.5e-4 v5_lora_r16_a32 v6_lora_r64_a128)
NAME=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
CFG=$ROOT/code/Verifiable_troll_detection/configs/v5_parallel/$NAME.yaml
OUT=$ROOT/outputs/$NAME
DEV_SFT=$ROOT/data/sft_dev.jsonl        # optional; enables completion-only dev NLL if present

echo "=== $(date) host=$(hostname) task=$SLURM_ARRAY_TASK_ID variant=$NAME ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
grep -q 'Qwen3-0.6B' "$CFG" || { echo "FATAL: $CFG is not Qwen3-0.6B"; exit 3; }
grep -Eq '1\.7B|Qwen3-8B' "$CFG" && { echo "FATAL: $CFG references 1.7B/8B"; exit 3; }

mkdir -p "$OUT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT/code/Verifiable_troll_detection"

DEV_ARG=""
[ -f "$DEV_SFT" ] && DEV_ARG="--sft-dev-data $DEV_SFT"

"$PY" scripts/v5/train_e2_sft_v5.py \
    --config "$CFG" \
    --sft-data "$ROOT/data/sft_train.jsonl" \
    $DEV_ARG \
    --out "$OUT"
echo "=== DONE $NAME $(date) ==="

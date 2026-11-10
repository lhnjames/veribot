# ParaCloud transfer manifest — VeriRoute v4 parallel SFT (Qwen3-0.6B)

Two machines cannot connect directly → relay **A (js02) → local → B (ParaCloud)** with rsync
(`--partial --append-verify --timeout=600`), sha256-verify both ends. **test is sealed** — never
copy `test.jsonl` or `evidence_bank_eval.jsonl` to ParaCloud. Dev subset only for selection.

Target root on B: `~/run/users/<user>/veriroute_v4_parallel/` (only `~/run` persists).

| # | Item | Source (A) | Dest (B) | ~Size | Notes |
|---|---|---|---|---|---|
| 1 | Project code (no .venv/outputs/.git) | `/path/to/project_root` | `.../veriroute_v4_parallel/code/Verifiable_troll_detection` | ~10 MB | `rsync --exclude .venv --exclude outputs --exclude .git` |
| 2 | Render train | `outputs/v4/render_v4/train.jsonl` | `.../data/render_v4/train.jsonl` | ~110 MB | for reference/rebuild |
| 3 | Render dev | `outputs/v4/render_v4/dev.jsonl` | `.../data/render_v4/dev.jsonl` | ~32 MB | selection only |
| 4 | **SFT data (1:1, 7240)** | `outputs/v4/e2_sft/sft_train.jsonl` | `.../data/sft_train.jsonl` | ~small | identical data across all variants (fair comparison) |
| 5 | Evidence bank TRAIN | `outputs/v4/evidence_bank_v1/evidence_bank_train.jsonl` | `.../data/evidence_bank_train.jsonl` | ~50 MB | train only |
| 6 | **Qwen3-0.6B base model** | `/path/to/data_root/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<snap>` | `.../models/Qwen3-0.6B` | ~1.5 GB | OR download on B via `HF_ENDPOINT=https://hf-mirror.com` |
| 7 | dev.json (for gen eval if trainer reads raw) | `data/twibot20_complete/dev.json` | `.../data/dev.json` | ~69 MB | if needed by eval path |

**Explicitly NOT transferred (sealed):** `test.jsonl`, `test.json`, `evidence_bank_eval.jsonl`,
`unified_full/*` (TwiBot-22 mixed), any 1.7B/8B weights, E1 artifacts.

## Post-transfer setup on B (login node, needs internet)
```
mkdir -p ~/run/users/<user>/veriroute_v4_parallel/{code,data,models,outputs,logs}
export PATH=~/run/miniconda3/bin:$PATH   # per doc §11.3
conda create -y -p ~/run/users/<user>/envs/vb python=3.11
~/run/users/<user>/envs/vb/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    torch transformers trl peft accelerate datasets scikit-learn
```
Then set `PC_MODEL` in `scripts/v5/paracloud/gen_variant_configs.py` to `.../models/Qwen3-0.6B`,
regenerate configs, and grep-confirm every config contains `Qwen3-0.6B`.

## Submission order (per user)
1. **Baseline smoke short job** (`v0`, `--smoke`/few steps, `-t 00:20:00`) → confirm env/model/data load, GPU visible, single-batch fwd/bwd finite, completion mask effective.
2. On smoke pass → `sbatch scripts/v5/paracloud/sft_array.sbatch` (array 0-6). Record every JID.
3. **WIT (v7)** only after its weighted-loss completion-mask + single-batch smoke passes; submit as a separate 1-element job — never let a wrong loss occupy a GPU.
4. After matrix: pick best 1-2 by dev (validation-loss trend + task metrics + seed stability), then run GRPO variants on those only — do NOT blindly GRPO all 8.

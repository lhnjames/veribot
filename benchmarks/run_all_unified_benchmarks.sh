#!/bin/bash
set -e
cd /root/private_data/hanning/veribot_workspace

export PYTHONPATH="/root/private_data/hanning/veribot_workspace/code/source:$PYTHONPATH"
PY="/root/private_data/hanning/veribot_workspace/.venv/bin/python"

mkdir -p runs/dual_model/5seeds
LOG="runs/dual_model/5seeds/benchmark_execution.log"

echo "=== STARTING VERIBOT UNIFIED 5-SEED BENCHMARKS ===" | tee -a "$LOG"
date | tee -a "$LOG"

# 1. Fox8-23 (5 Seeds)
echo "=== 1. FOX8-23 (5 SEEDS) ===" | tee -a "$LOG"
$PY -u code/source/benchmarks/run_5seeds.py --dataset fox8 --seeds 0,1,2,3,4 2>&1 | tee -a "$LOG"

# 2. BotSim-24 (5 Seeds)
echo "=== 2. BOTSIM-24 (5 SEEDS) ===" | tee -a "$LOG"
$PY -u code/source/benchmarks/run_5seeds.py --dataset botsim --seeds 0,1,2,3,4 2>&1 | tee -a "$LOG"

# 3. MGTAB (5 Seeds)
echo "=== 3. MGTAB (5 SEEDS) ===" | tee -a "$LOG"
$PY -u code/source/benchmarks/run_5seeds.py --dataset mgtab --seeds 0,1,2,3,4 2>&1 | tee -a "$LOG"

# 4. TwiBot-20 (5 Seeds)
echo "=== 4. TWIBOT-20 (5 SEEDS) ===" | tee -a "$LOG"
$PY -u code/source/benchmarks/run_5seeds.py --dataset twibot20 --seeds 0,1,2,3,4 2>&1 | tee -a "$LOG"

# 5. Camouflage Stress Tests
echo "=== 5. CAMOUFLAGE STRESS TESTS ===" | tee -a "$LOG"
$PY -u code/source/benchmarks/stress_test_camouflage.py \
    --cache code/data/processed/twibot20_source_v5/evidence_group_cache.pt \
    --forward models/forward/twibot20_seed0_unified \
    --split test \
    --out runs/dual_model/5seeds/twibot20_camouflage_stress_test.json 2>&1 | tee -a "$LOG"

$PY -u code/source/benchmarks/stress_test_camouflage.py \
    --cache code/data/processed/mgtab_source_v3/evidence_group_cache.pt \
    --forward models/forward/mgtab_seed0_unified \
    --split test \
    --out runs/dual_model/5seeds/mgtab_camouflage_stress_test.json 2>&1 | tee -a "$LOG"

echo "=== ALL BENCHMARKS COMPLETED SUCCESSFULLY ===" | tee -a "$LOG"
date | tee -a "$LOG"

#!/bin/bash
# Auto-continuation watcher (js02): wait for the FULL train evidence bank (all 8278 accounts) to be
# generated AND the generator process to have exited cleanly, then build the final 1:1 SFT data and
# launch the official E2 0.6B run. Never trains on partial data.
set -u
cd /path/to/project_root
source activate.sh
REPO=/path/to/project_root
OUT=$REPO/outputs/v4
LOG=$OUT/sft_watch.log
mkdir -p $OUT/e2_sft
exec >> "$LOG" 2>&1

log(){ echo "[$(date '+%F %T')] $*"; }
count_cache(){ find "$OUT/evidence_bank_v1/raw_cache" -name '*.json' 2>/dev/null | wc -l; }
train_done(){
  # Require an explicitly complete report and exact full train coverage. A generator process ending
  # after network failures must never release partial data to SFT.
  [ -s "$OUT/evidence_bank_v1/evidence_bank_train.jsonl" ] || return 1
  grep -q '"mode": "full"' "$OUT/evidence_bank_v1/validation_report_full_train.json" 2>/dev/null || return 1
  grep -q '"complete": true' "$OUT/evidence_bank_v1/validation_report_full_train.json" 2>/dev/null || return 1
  [ "$(wc -l < "$OUT/evidence_bank_v1/evidence_bank_train.jsonl")" -eq 8278 ] || return 1
  ! tmux has-session -t evtrain 2>/dev/null
}

log "watcher START pid=$$ ; waiting for FULL train evidence (8278 accounts)"
last=0
while true; do
  if train_done; then
    log "train evidence COMPLETE. bank=$(wc -l < $OUT/evidence_bank_v1/evidence_bank_train.jsonl) lines"
    break
  fi
  # 15-min heartbeat
  now=$(date +%s)
  if [ $(( now - last )) -ge 900 ]; then
    log "progress: raw_cache=$(count_cache) ; $(tail -1 $OUT/evidence_bank_v1/train_run.log 2>/dev/null)"
    last=$now
  fi
  sleep 30
done

# build final 1:1 balanced SFT data (full)
log "building final 1:1 SFT data ..."
PYTHONPATH=. python scripts/build_sft_data.py --split train \
  --out $OUT/e2_sft/sft_train.jsonl >> $OUT/e2_sft/build_final.log 2>&1
BN=$(wc -l < $OUT/e2_sft/sft_train.jsonl 2>/dev/null || echo 0)
log "SFT data built: $BN examples -> $OUT/e2_sft/sft_train.jsonl"
if [ "$BN" -lt 2000 ]; then log "ERROR: SFT data too small ($BN); aborting launch"; exit 1; fi

# launch official E2 0.6B in its own tmux
RUN=$OUT/e2_sft/run_0.6b
mkdir -p $RUN
tmux new-session -d -s e2_06b \
  "cd $REPO && source activate.sh && PYTHONPATH=. python scripts/train_e2_sft.py \
   --config configs/e2_sft_0.6b.yaml --sft-data $OUT/e2_sft/sft_train.jsonl \
   --out $RUN >> $RUN/tmux.log 2>&1"
sleep 4
PID=$(pgrep -f 'train_e2_sft.py --config configs/e2_sft_0.6b.yaml' | head -1)
echo "$PID" > $RUN/pid.txt
log "E2 0.6B LAUNCHED tmux=e2_06b pid=$PID log=$RUN/tmux.log run_log=$RUN/train.log"
log "watcher DONE"

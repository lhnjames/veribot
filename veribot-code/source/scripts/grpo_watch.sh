#!/bin/bash
# GRPO auto-continuation watcher (js02): wait for E2 0.6B SFT to FINISH, then launch the official
# v4 GRPO from E2's best checkpoint. Fully offline reward (train bank only). Idempotent: exits if
# GRPO already running/launched.
set -u
cd /path/to/project_root
source activate.sh
REPO=/path/to/project_root
OUT=$REPO/outputs/v4
LOG=$OUT/grpo_watch.log
E2LOG=$OUT/e2_sft/run_0.6b/train.log
E2LORA=$OUT/e2_sft/run_0.6b/best_lora
BANK=$OUT/evidence_bank_v1/evidence_bank_train.jsonl
RUN=$OUT/grpo/run_0.6b
exec >> "$LOG" 2>&1
log(){ echo "[$(date '+%F %T')] $*"; }

if tmux has-session -t grpo06_v4 2>/dev/null; then log "grpo06_v4 already running; exit"; exit 0; fi
if [ -f "$RUN/.launched" ]; then log "already launched (marker present); exit"; exit 0; fi

log "grpo watcher START pid=$$ ; waiting for E2 0.6B to finish"
last=0
while true; do
  if grep -q '^\[done\]' "$E2LOG" 2>/dev/null && [ -d "$E2LORA" ] && [ -s "$BANK" ]; then
    log "E2 done, best_lora + train bank present -> proceeding to GRPO"
    break
  fi
  now=$(date +%s)
  if [ $(( now - last )) -ge 900 ]; then
    log "waiting: E2_done=$(grep -qc '^\[done\]' $E2LOG 2>/dev/null && echo yes || echo no) lora=$([ -d $E2LORA ] && echo yes || echo no) bank=$([ -s $BANK ] && echo yes || echo no)"
    last=$now
  fi
  sleep 30
done

mkdir -p $RUN
# offline reward check first (quick), then launch full GRPO
log "running GRPO --check-only (offline reward sanity)"
PYTHONPATH=. python scripts/train_grpo_v4.py --config configs/grpo_v4_0.6b.yaml \
  --sft-lora $E2LORA --out $RUN --check-only >> $RUN/check.log 2>&1
if ! grep -q 'CHECK OK' $RUN/check.log; then log "ERROR: GRPO check failed; see $RUN/check.log"; exit 1; fi
log "check OK: $(grep 'CHECK OK' $RUN/check.log | tail -1)"

touch $RUN/.launched
tmux new-session -d -s grpo06_v4 \
  "cd $REPO && source activate.sh && PYTHONPATH=. python scripts/train_grpo_v4.py \
   --config configs/grpo_v4_0.6b.yaml --sft-lora $E2LORA --out $RUN >> $RUN/tmux.log 2>&1"
sleep 5
PID=$(pgrep -f 'train_grpo_v4.py --config configs/grpo_v4_0.6b.yaml' | head -1)
echo "$PID" > $RUN/pid.txt
log "GRPO v4 0.6B LAUNCHED tmux=grpo06_v4 pid=$PID log=$RUN/tmux.log"
log "grpo watcher DONE"

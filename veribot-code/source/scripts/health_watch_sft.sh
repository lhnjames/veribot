#!/usr/bin/env bash
# Lightweight v4 SFT health watcher (js02). Read-only, alert-only, NEVER restarts/overwrites.
# Fixes v1 bug: count captures must emit exactly ONE integer (v1 used `grep -c ... || echo 0`
# which produced "0\n0" and false-fired [done]).
set -u
cd /path/to/project_root || exit 1
RUN=outputs/v4/e2_sft/run_0.6b
TRAIN=$RUN/train.log
TMUX=$RUN/tmux.log
METRICS=$RUN/metrics.jsonl
HLOG=${HLOG:-$RUN/health_watch.log}
INTERVAL=${INTERVAL:-900}

log() { echo "$(date '+%F %T %Z') | $*" >> "$HLOG"; }

# count matching lines -> exactly one integer, robust to missing file / no match
cnt() { local n; n=$(grep -cE "$1" "$2" 2>/dev/null || true); n=${n%%$'\n'*}; echo "${n:-0}"; }

main_pid() { [ -f "$RUN/pid.txt" ] && tr -dc '0-9' < "$RUN/pid.txt"; }

log "sft_health START watcher_pid=$$ interval=${INTERVAL}s (alert-only, no auto-restart)"
alerted=0
while true; do
  MP=$(main_pid)
  if [ -n "$MP" ] && kill -0 "$MP" 2>/dev/null; then
    ALIVE="yes(pid=$MP)"
    ACT=$(ps -o etime=,cputime= -p "$MP" 2>/dev/null | tr -s ' ' | sed 's/^ //')
  else
    ALIVE="NO"; ACT="-"
  fi
  GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null | head -1 | tr -d '\n')
  TR_SZ=$(stat -c '%s' "$TRAIN" 2>/dev/null || echo 0)
  TR_MT=$(stat -c '%y' "$TRAIN" 2>/dev/null | cut -d. -f1)
  TX_SZ=$(stat -c '%s' "$TMUX" 2>/dev/null || echo 0)
  TX_MT=$(stat -c '%y' "$TMUX" 2>/dev/null | cut -d. -f1)
  if [ -f "$METRICS" ]; then M_LINES=$(wc -l < "$METRICS" 2>/dev/null); else M_LINES=0; fi
  M_LINES=${M_LINES:-0}
  OOM=$(cnt 'OOM' "$TMUX")
  DONE=$(cnt '^\[done\]' "$TRAIN")

  log "PID=$ALIVE act[etime,cpu]=$ACT | GPU[util,mem,temp]=$GPU | train.log[sz=$TR_SZ mt=$TR_MT] | metrics.jsonl=$M_LINES | tmux.log[sz=$TX_SZ mt=$TX_MT oom=$OOM] | done=$DONE"

  if [ "$DONE" -gt 0 ] 2>/dev/null; then
    log "SFT completed: line-start [done] present in train.log. grpo_watch takes over. exiting 0."
    exit 0
  fi

  if [ "$ALIVE" = "NO" ]; then
    if [ "$alerted" -eq 0 ]; then
      log "ALERT: main train PID gone and NO [done] in train.log -> unexpected exit. Diagnosing, NOT restarting."
      log "ALERT diag tmux.tail: $(tail -c 500 "$TMUX" 2>/dev/null | tr '\n\r' '  ')"
      log "ALERT diag best_lora: $(ls -d "$RUN"/best_lora 2>/dev/null || echo missing)"
      alerted=1
      sleep "$INTERVAL"
      continue
    else
      log "ALERT persists across two cycles: process still gone. Exiting 2 for human review."
      exit 2
    fi
  fi
  alerted=0
  sleep "$INTERVAL"
done

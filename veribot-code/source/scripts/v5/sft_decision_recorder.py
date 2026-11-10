"""Auditable per-epoch SFT stop-decision recorder (v4 e2_06b run).

Reads metrics.jsonl (one record per completed epoch) and regenerates a human-auditable decision
log applying the authorized rules. It NEVER kills training: the trainer's own patience-2 on dev
acc is the safe epoch-boundary stop (preserves best_lora, emits [done], grpo_watch resumes from
best_lora). This recorder documents whether the richer rules agree, and flags overfit/plateau/
format-degradation so a human sees the reasoning. Idempotent: regenerated from metrics.jsonl.

Rules (train_loss is AUXILIARY ONLY — never 'lower train loss => better'):
  CONTINUE       : train_loss still dropping AND dev improves (d_acc>=+0.003 OR clear F1 gain)
  STOP_OVERFIT   : train_loss dropping BUT dev acc/F1 drop (d_acc<=-0.003 or d_f1<=-0.01)
  STOP_PLATEAU   : train rel-improve <2% AND d_acc<+0.003 AND d_f1<+0.003
  STOP_FORMAT    : format_valid_rate drops >=0.01 or falls <0.95, or pred_bot_rate drifts to extreme
  (max 4 epochs; best_lora = best dev-acc checkpoint, chosen by the trainer)
"""
import json
import os
import time

RUN = "outputs/v4/e2_sft/run_0.6b"
METRICS = f"{RUN}/metrics.jsonl"
TRAIN = f"{RUN}/train.log"
DLOG = f"{RUN}/sft_stop_decision.log"
POLL = int(os.environ.get("DEC_POLL", 120))


def read_epochs():
    out = []
    if not os.path.exists(METRICS):
        return out
    with open(METRICS) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    pass
    out.sort(key=lambda r: r.get("epoch", 0))
    return out


def analyze(prev, cur):
    d = cur["dev"]; p = prev["dev"]
    tl0, tl1 = prev["train_loss"], cur["train_loss"]
    train_rel = (tl0 - tl1) / tl0 if tl0 else 0.0
    d_acc = d["acc"] - p["acc"]
    d_f1 = d["f1"] - p["f1"]
    d_fmt = d["format_valid_rate"] - p["format_valid_rate"]
    flags = []
    dec = None
    # format / prediction-ratio degradation
    if d_fmt <= -0.01 or d["format_valid_rate"] < 0.95:
        flags.append(f"format_down(d_fmt={d_fmt:+.3f}, fmt={d['format_valid_rate']:.3f})")
        dec = "STOP_FORMAT"
    if d["pred_bot_rate"] < 0.15 or d["pred_bot_rate"] > 0.85:
        flags.append(f"pred_bot_rate_extreme({d['pred_bot_rate']:.3f})")
        dec = dec or "STOP_FORMAT"
    # overfitting: train loss down but generalization down
    if dec is None and train_rel > 0 and (d_acc <= -0.003 or d_f1 <= -0.01):
        dec = "STOP_OVERFIT"
        flags.append(f"overfit(train_rel={train_rel:+.3f} but d_acc={d_acc:+.3f} d_f1={d_f1:+.3f})")
    # plateau
    if dec is None and train_rel < 0.02 and d_acc < 0.003 and d_f1 < 0.003:
        dec = "STOP_PLATEAU"
        flags.append(f"plateau(train_rel={train_rel:+.3f} d_acc={d_acc:+.3f} d_f1={d_f1:+.3f})")
    if dec is None:
        dec = "CONTINUE"
        flags.append(f"improving(train_rel={train_rel:+.3f} d_acc={d_acc:+.3f} d_f1={d_f1:+.3f})")
    return dec, train_rel, d_acc, d_f1, d_fmt, flags


def regen():
    eps = read_epochs()
    done = os.path.exists(TRAIN) and any(l.startswith("[done]") for l in open(TRAIN))
    lines = []
    lines.append("# SFT stop-decision log (auditable). Recorder does NOT kill; trainer patience-2 is the safe stop.")
    lines.append("# min_delta=0.003, patience=2, max_epochs=4. best_lora = best dev-acc checkpoint (trainer-chosen).")
    lines.append("")
    for i, cur in enumerate(eps):
        d = cur["dev"]
        base = (f"[epoch {cur['epoch']}] train_loss={cur['train_loss']:.4f} "
                f"dev_acc={d['acc']:.4f} f1={d['f1']:.4f} fmt={d['format_valid_rate']:.3f} "
                f"pred_bot={d['pred_bot_rate']:.3f} conf={d['confusion']} "
                f"best_acc={cur.get('best_acc')} bad_evals={cur.get('bad_evals')}")
        if i == 0:
            lines.append(base + "  -> BASELINE (single point; wait for epoch 1 to judge trend)")
        else:
            dec, tr, da, df, dfmt, flags = analyze(eps[i - 1], cur)
            saved = "best_lora UPDATED" if cur.get("bad_evals", 1) == 0 else "best_lora kept (no dev-acc improve)"
            lines.append(base + f"  -> RECOMMEND={dec} | {saved} | " + "; ".join(flags))
    # trailing status
    if eps:
        last = eps[-1]
        if last.get("bad_evals", 0) >= 2:
            lines.append("")
            lines.append(f"[trainer] patience reached (bad_evals={last['bad_evals']}) -> natural earlystop; best_lora is final.")
    if done:
        lines.append("")
        lines.append("[trainer] [done] present -> SFT finished. GRPO must resume from best_lora only.")
    lines.append("")
    lines.append(f"# regenerated {time.strftime('%F %T %Z')} | epochs_seen={len(eps)} | done={done}")
    tmp = DLOG + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, DLOG)
    return len(eps), done


def main():
    while True:
        try:
            n, done = regen()
        except Exception as e:
            with open(DLOG, "a") as f:
                f.write(f"# recorder error {type(e).__name__}: {e}\n")
            n, done = 0, False
        if done:
            break
        time.sleep(POLL)


if __name__ == "__main__":
    main()

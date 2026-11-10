"""Persistent E4 GRPO health watcher (js02). Read-only, alert-only, never kills.
Parses the latest TRL metric dict from tmux.log and tracks doc §6.3 curves. Logs every INTERVAL to
grpo_health.log and flags danger signs. The lr=2e-6 KL/clip decision is a HUMAN call at ~step 100 —
this watcher only surfaces the evidence, it does not change lr.
"""
import ast
import os
import re
import time

RUN = "outputs/v4/grpo/run_0.6b"
TMUX = f"{RUN}/tmux.log"
HLOG = f"{RUN}/grpo_health.log"
INTERVAL = int(os.environ.get("GRPO_POLL", 900))
DICT_RE = re.compile(r"\{'loss'.*?'epoch':\s*[0-9.]+\}")


def log(msg):
    with open(HLOG, "a") as f:
        f.write(f"{time.strftime('%F %T %Z')} | {msg}\n")


def last_metrics():
    if not os.path.exists(TMUX):
        return None
    txt = open(TMUX, errors="ignore").read()
    m = DICT_RE.findall(txt)
    if not m:
        return None
    try:
        return ast.literal_eval(m[-1])
    except Exception:
        return None


def last_step():
    if not os.path.exists(TMUX):
        return None
    m = re.findall(r"(\d+)/(\d+) \[", open(TMUX, errors="ignore").read())
    return (int(m[-1][0]), int(m[-1][1])) if m else None


def alive():
    return os.popen("pgrep -f 'train_grpo_v4.py' | grep -v check-only | head -1").read().strip()


def main():
    log(f"grpo_health START watcher_pid={os.getpid()} interval={INTERVAL}s (alert-only)")
    init_entropy = None
    alerted = False
    while True:
        pid = alive()
        gpu = os.popen("nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu "
                       "--format=csv,noheader 2>/dev/null | head -1").read().strip()
        d = last_metrics()
        st = last_step()
        step = st[0] if st else None
        total = st[1] if st else None
        done = bool(step is not None and total is not None and step >= total)
        if d:
            if init_entropy is None:
                init_entropy = d.get("entropy")
            reward = d.get("reward"); rstd = d.get("reward_std")
            fzs = d.get("frac_reward_zero_std"); kl = d.get("kl"); ent = d.get("entropy")
            clip = d.get("clip_ratio/region_mean", 0.0)
            clen = d.get("completions/mean_length"); crat = d.get("completions/clipped_ratio")
            gnorm = d.get("grad_norm")
            log(f"pid={pid or 'NONE'} gpu[{gpu}] step={step}/{total} reward={reward:.3f} reward_std={rstd:.3f} "
                f"frac_zero_std={fzs:.3f} kl={kl:.5f} entropy={ent:.3f} clip_region={clip:.3f} "
                f"comp_len={clen:.1f} clipped={crat:.3f} grad_norm={gnorm:.3f}")
            # danger signs (surface only; human decides)
            flags = []
            if step and step >= 100 and kl < 0.001 and clip == 0.0:
                flags.append(f"BARELY_UPDATING (kl={kl:.5f} clip=0 @step{step}) -> consider lr 2e-6->1e-5 per doc")
            if fzs is not None and fzs > 0.5:
                flags.append(f"HIGH_ZERO_STD_GROUPS ({fzs:.2f}) -> sparse reward")
            if init_entropy and ent < 0.4 * init_entropy:
                flags.append(f"ENTROPY_COLLAPSE ({ent:.3f} < 0.4*{init_entropy:.3f})")
            if kl and kl > 50:
                flags.append(f"KL_RUNAWAY ({kl:.2f})")
            if rstd is not None and rstd < 0.1:
                flags.append(f"REWARD_STD_COLLAPSE ({rstd:.3f})")
            for fl in flags:
                log("  ALERT: " + fl)
        else:
            log(f"pid={pid or 'NONE'} gpu[{gpu}] step={step} (no metric dict yet)")

        if done:
            log(f"training reached {step}/{total} (complete) -> exiting.")
            return
        if not pid:
            if not alerted:
                log("ALERT: E4 process gone; diagnosing next cycle, NOT restarting.")
                alerted = True
                time.sleep(INTERVAL)
                continue
            log("ALERT persists: E4 process still gone. Exiting for human review.")
            return
        alerted = False
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()

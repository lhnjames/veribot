"""E3/E5 GRPO runner. Reuses train_grpo_v4's ENTIRE trainer setup unchanged (same model init from
E2 best_lora, same dataset build with the same seed, same GRPOConfig) and swaps ONLY the reward via
reward_modes_v5, selected by cfg['reward_mode']. E4 (scripts/train_grpo_v4.py) is never modified, so
E3/E4/E5 differ in reward alone (doc §12.1). Rollouts -> <out>/rollouts.jsonl for every mode.

  python scripts/v5/train_grpo_v5.py --config configs/grpo_e3_class_only_0.6b.yaml \
      --sft-lora outputs/v4/e2_sft/run_0.6b/best_lora --out outputs/v4/grpo_e3 [--check-only]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import scripts.train_grpo_v4 as g4  # noqa: E402  (imports trl/torch; E4 trainer untouched)
from scripts.v5.reward_modes_v5 import make_reward_func as make_mode_reward  # noqa: E402

# peek --config (reward_mode) and --out (rollouts path) without disturbing g4's own argparse
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config", required=True)
_pre.add_argument("--out", required=True)
_known, _ = _pre.parse_known_args()
_cfg = g4.load_yaml(_known.config)
MODE = _cfg.get("reward_mode", "three_gate")
SEED = _cfg.get("seed", 0)
Path(_known.out).mkdir(parents=True, exist_ok=True)
ROLLOUTS = str(Path(_known.out) / "rollouts.jsonl")


def _patched_make_reward_func(bank, near_miss):
    # signature matches g4.make_reward_func(bank, near_miss) so g4.main() calls this unchanged
    return make_mode_reward(MODE, bank, near_miss, ROLLOUTS,
                            relaxed_getter=lambda: g4._STATE["relaxed"], seed=SEED)


g4.make_reward_func = _patched_make_reward_func


if __name__ == "__main__":
    print(f"[grpo_v5] reward_mode={MODE} seed={SEED} rollouts={ROLLOUTS}")
    g4.main()

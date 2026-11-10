"""Step 4 diagnosis: does the teacher-generated cold-start data for bot-labeled accounts even
pass an independent sufficiency check, using the EXACT same judge config as GRPO reward
computation (same model/K/temperature/system prompt)? If the teacher data itself mostly fails,
RL is optimizing on top of an already-biased starting point, not learning "how to be sufficient"
from scratch.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifiable_bot.llm_client import DeepSeekJudge


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--judge-cache", type=Path, required=True)
    ap.add_argument("--n-bot", type=int, default=60)
    ap.add_argument("--n-human", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bot_rows, human_rows = [], []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row["record"]["label"]
            (bot_rows if label == "bot" else human_rows).append(row)

    rng = random.Random(args.seed)
    rng.shuffle(bot_rows)
    rng.shuffle(human_rows)
    bot_sample = bot_rows[: args.n_bot]
    human_sample = human_rows[: args.n_human]
    print(f"bot pool={len(bot_rows)} sampled={len(bot_sample)}; "
          f"human pool={len(human_rows)} sampled={len(human_sample)}")

    judge = DeepSeekJudge(cache_dir=args.judge_cache, k=5, max_workers=32)

    def score(rows, label_name):
        s_sufs = []
        non_profile_rate = []
        for row in rows:
            target = row["target"]
            claims_text = "\n".join(f"- {c['claim']}" for c in target["claims"])
            s_suf = judge.sufficiency_vote(claims_text, row["record"]["label"])
            s_sufs.append(s_suf)
            cited = [sid for c in target["claims"] for sid in c["source_ids"]]
            non_profile_rate.append(any(not sid.startswith("PROFILE_") for sid in cited))
        n = len(s_sufs)
        pass_rate = sum(1 for s in s_sufs if s >= 2 / 3) / n if n else 0
        mean_suf = sum(s_sufs) / n if n else 0
        np_rate = sum(non_profile_rate) / n if n else 0
        print(f"\n-- {label_name} (n={n}) --")
        print(f"  mean s_suf={mean_suf:.3f}  pass_rate(>=2/3)={pass_rate:.3f}  cites_non_profile_rate={np_rate:.3f}")
        return s_sufs

    score(bot_sample, "cold-start teacher data, label=bot")
    score(human_sample, "cold-start teacher data, label=human")


if __name__ == "__main__":
    main()

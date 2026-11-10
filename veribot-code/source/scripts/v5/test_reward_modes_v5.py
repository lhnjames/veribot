"""CPU-only unit tests for reward_modes_v5 (no GPU, no trl). Uses the real TRAIN evidence bank.
Run: PYTHONPATH=. python3 scripts/v5/test_reward_modes_v5.py
"""
import json
import os
import importlib.util

# import reward_modes_v5 without pulling train_grpo_v4 (which needs trl)
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("reward_modes_v5", os.path.join(HERE, "reward_modes_v5.py"))
rm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm)
from scripts.reward_v4 import EvidenceBank, score  # noqa: E402

BANK = EvidenceBank("outputs/v4/evidence_bank_v1/evidence_bank_train.jsonl")

# pick a real bot + human account_id present in the bank file
_bot = _hum = None
for l in open("outputs/v4/evidence_bank_v1/evidence_bank_train.jsonl"):
    d = json.loads(l)
    if d["label"] == "bot" and _bot is None:
        _bot = d["account_id"]
    if d["label"] == "human" and _hum is None:
        _hum = d["account_id"]
    if _bot and _hum:
        break

GOOD = "<answer>bot</answer>"        # answer-only; format gate would fail but class_only ignores it


def test_class_only_ignores_format_and_evidence():
    rf = rm.make_reward_func("class_only", BANK)
    # bot account: predicting bot -> +2 ; predicting human -> -1 ; no answer -> -1
    out = rf(["p"] * 3,
             ["<answer>bot</answer>", "<answer>human</answer>", "no tags here"],
             [_bot, _bot, _bot], ["bot", "bot", "bot"])
    assert out == [2.0, -1.0, -1.0], out


def test_three_gate_equals_canonical():
    rf = rm.make_reward_func("three_gate", BANK)
    comps = [GOOD, "garbage", "<answer>human</answer>"]
    aids = [_bot, _hum, _bot]
    labs = ["bot", "human", "bot"]
    got = rf(["p"] * 3, comps, aids, labs)
    ref = [float(score(c, a, l, BANK).total) for c, a, l in zip(comps, aids, labs)]
    assert got == ref, (got, ref)


def test_random_preserves_distribution_and_is_seeded():
    comps = ["<answer>bot</answer>", "garbage", "<answer>human</answer>", "x"]
    aids = [_bot, _hum, _bot, _hum]
    labs = ["bot", "human", "bot", "human"]
    ref = sorted(float(score(c, a, l, BANK).total) for c, a, l in zip(comps, aids, labs))
    rf1 = rm.make_reward_func("random", BANK, seed=7)
    out1 = rf1(["p"] * 4, comps, aids, labs)
    assert sorted(out1) == ref, (sorted(out1), ref)          # distribution preserved
    rf2 = rm.make_reward_func("random", BANK, seed=7)
    out2 = rf2(["p"] * 4, comps, aids, labs)
    assert out1 == out2                                       # deterministic given seed


def test_rollouts_written(tmp_path="/tmp/rm_rollouts_test.jsonl"):
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    rf = rm.make_reward_func("class_only", BANK, rollouts_path=tmp_path)
    rf(["p"], ["<answer>bot</answer>"], [_bot], ["bot"])
    lines = [json.loads(l) for l in open(tmp_path)]
    assert len(lines) == 1 and lines[0]["mode"] == "class_only" and lines[0]["reward"] == 2.0
    assert "true_three_gate" in lines[0] and "gate1" in lines[0]
    os.remove(tmp_path)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print("ALL_REWARD_MODE_TESTS_PASSED")

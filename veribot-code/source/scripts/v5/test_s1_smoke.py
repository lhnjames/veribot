"""Small-sample smoke test for v5 S1 WoE miner core functions (no render files needed).
Run: PYTHONPATH=scripts/v5 python3 scripts/v5/test_s1_smoke.py
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s1", os.path.join(HERE, "s1_woe_mining.py"))
s1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s1)


def test_norm_bucket():
    assert s1.norm_bucket("Follower count: very high relative to other accounts in this dataset.") == "very high"
    assert s1.norm_bucket("Verification status: verified.") == "verified"
    # digits normalized, punctuation stripped
    assert "#" in s1.norm_bucket("Handle digit share: 42 percent, high.")


def test_fold_deterministic():
    a = s1.fold_of("1234567890")
    assert a == s1.fold_of("1234567890")   # stable
    assert a in (0, 1)


def test_src_type():
    assert s1.src_type("PROFILE_02") == "PROFILE"
    assert s1.src_type("OBS_01") == "OBS"


def test_woe_sign_and_top2():
    # synthetic accounts: sig X strongly bot, sig Y strongly human, different src types
    rows = []
    for i in range(60):
        rows.append({"account_id": f"b{i}", "label": "bot",
                     "units": [{"unit_id": "OBS_01", "text": "posting rate: high"},
                               {"unit_id": "PROFILE_02", "text": "followers: low"}]})
    for i in range(60):
        rows.append({"account_id": f"h{i}", "label": "human",
                     "units": [{"unit_id": "PROFILE_02", "text": "followers: very high"}]})
    tab = s1.woe_table(rows)
    # bucket = text after first colon; "posting rate: high" -> "OBS_01::high", bot-only -> woe > 0
    assert tab["OBS_01::high"]["woe"] > 0
    # "followers: very high" appears only for humans -> woe < 0
    assert tab["PROFILE_02::very high"]["woe"] < 0
    # top2 diff-src returns 2 distinct source types
    lst = [("OBS_01::high", "OBS"), ("PROFILE_02::low", "PROFILE")]
    g = s1.top2_diff_src(lst, key=lambda x: 1.0)
    assert len(g) == 2 and len(set(g)) == 2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("ALL_SMOKE_TESTS_PASSED")

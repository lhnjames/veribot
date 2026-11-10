"""CPU-only unit/smoke tests for train_e2_sft_v5 enhancements (no model load, no GPU).
Run: PYTHONPATH=. python3 scripts/v5/test_train_e2_sft_v5.py
"""
import importlib.util
import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("t5", os.path.join(HERE, "train_e2_sft_v5.py"))
t5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t5)

torch.manual_seed(0)


def _case():
    # batch=2, seq=6, vocab=11; prompt = first 3 positions, completion = last 3
    B, S, V = 2, 6, 11
    logits = torch.randn(B, S, V)
    input_ids = torch.randint(0, V, (B, S))
    attn = torch.ones(B, S, dtype=torch.long)
    # completion-only labels: -100 on prompt (first 3), real ids on completion (last 3)
    labels = input_ids.clone()
    labels[:, :3] = -100
    # per-token weights: prompt_w on first 3, 1.0 on last 3
    return logits, input_ids, attn, labels


def test_wit_baseline_parity():
    # weighted_lm_loss with prompt weight 0 must equal completion-only mean CE (the -100 path)
    logits, input_ids, attn, labels = _case()
    weights0 = torch.zeros_like(input_ids, dtype=torch.float)
    weights0[:, 3:] = 1.0
    w_loss = t5.weighted_lm_loss(logits, input_ids, weights0, attn)
    nll, ntok = t5.completion_nll_from_logits(logits, labels)
    assert torch.isfinite(w_loss)
    assert abs(float(w_loss) - float(nll)) < 1e-4, (float(w_loss), float(nll))
    assert ntok == 6  # 2 rows x 3 completion tokens


def test_wit_prompt_weight_changes_loss():
    logits, input_ids, attn, labels = _case()
    w0 = torch.zeros_like(input_ids, dtype=torch.float); w0[:, 3:] = 1.0
    w1 = w0.clone(); w1[:, :3] = 0.1                      # WIT: prompt tokens weighted 0.1
    l0 = float(t5.weighted_lm_loss(logits, input_ids, w0, attn))
    l1 = float(t5.weighted_lm_loss(logits, input_ids, w1, attn))
    assert l1 == l1 and l1 != l0                          # finite and different


def test_completion_nll_empty():
    logits, input_ids, attn, labels = _case()
    empty = torch.full_like(labels, -100)
    nll, ntok = t5.completion_nll_from_logits(logits, empty)
    assert ntok == 0 and float(nll) == 0.0


def test_gen_metrics():
    m = t5.gen_metrics([0, 0, 1, 1], [0, 0, 1, 1], fmt_ok=4, n=4)
    assert abs(m["acc"] - 1.0) < 1e-9
    assert abs(m["macro_f1"] - 1.0) < 1e-9
    assert abs(m["balanced_acc"] - 1.0) < 1e-9
    # imperfect case: one bot predicted human -> balanced_acc < 1, keys present
    m2 = t5.gen_metrics([0, 0, 1, 1], [0, 0, 1, 0], fmt_ok=3, n=4)
    for k in ("acc", "f1", "macro_f1", "balanced_acc", "pred_bot_rate", "format_valid_rate", "confusion"):
        assert k in m2
    assert 0.0 <= m2["balanced_acc"] < 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print("ALL_V5_TRAINER_TESTS_PASSED")

"""Generate the ParaCloud parallel SFT optimization matrix (Qwen3-0.6B ONLY).

8 single-A800 variants derived from configs/e2_sft_0.6b.yaml. Fair comparison: identical data,
batch, epoch cap, dev subset, early-stop; only the named knob changes per variant. All configs
carry the SAME Qwen3-0.6B model path (ParaCloud-local, filled after transfer). Writes to
configs/v5_parallel/.  NEVER 1.7B/8B/E1.

Variant 7 (WIT) is emitted but MUST NOT be submitted until the weighted-instruction-loss
implementation passes a completion-mask + single-batch smoke test (see requires_smoke flag).
"""
import os

# ParaCloud-local model path (Qwen3-0.6B), filled once the base model is rsync'd over.
# Kept as a single string so every variant provably contains "Qwen3-0.6B".
PC_MODEL = "/path/to/cluster_workspace/veriroute_v4_parallel/models/Qwen3-0.6B"

BASE = dict(
    model_name=PC_MODEL, max_length=3600, max_new_tokens=320, seed=0,
    lora_rank=32, lora_alpha=64, lora_dropout=0.05, epochs=4,
    batch_size=4, grad_accum=8, lr=1.0e-4, weight_decay=0.01,
    warmup_ratio=0.03, grad_clip=1.0, dev_eval_limit=500, patience=2, min_delta=0.003,
)

VARIANTS = [
    ("v0_baseline_seed0", dict(seed=0)),
    ("v1_baseline_seed1", dict(seed=1)),
    ("v2_baseline_seed2", dict(seed=2)),
    ("v3_lr5e-5",         dict(lr=5.0e-5)),
    ("v4_lr1.5e-4",       dict(lr=1.5e-4)),
    ("v5_lora_r16_a32",   dict(lora_rank=16, lora_alpha=32)),
    ("v6_lora_r64_a128",  dict(lora_rank=64, lora_alpha=128)),
    ("v7_wit_promptw0.1", dict(prompt_loss_weight=0.1, completion_loss_weight=1.0,
                               requires_smoke=True)),
]

OUTDIR = "configs/v5_parallel"


def emit(name, over):
    cfg = dict(BASE)
    cfg.update(over)
    requires_smoke = cfg.pop("requires_smoke", False)
    lines = [
        f"# ParaCloud parallel SFT variant: {name}",
        "# Qwen3-0.6B ONLY. Derived from configs/e2_sft_0.6b.yaml; only this variant's knob differs.",
        "# Fair-comparison invariants (DO NOT change across variants): data, batch_size, grad_accum,",
        "# epochs, dev_eval_limit, patience, min_delta. Metrics logged per epoch: completion-only dev",
        "# NLL (validation loss), acc, bot-F1, macro-F1, balanced-acc, format-valid, confusion.",
    ]
    if requires_smoke:
        lines.append("# !! WIT variant: prompt tokens weighted 0.1 (not -100 masked), completion 1.0.")
        lines.append("# !! DO NOT SUBMIT until weighted-loss impl passes completion-mask + single-batch smoke.")
        lines.append("requires_smoke: true")
    for k, v in cfg.items():
        lines.append(f"{k}: {v}")
    with open(f"{OUTDIR}/{name}.yaml", "w") as f:
        f.write("\n".join(lines) + "\n")
    return f"{OUTDIR}/{name}.yaml"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    written = [emit(n, o) for n, o in VARIANTS]
    print("generated", len(written), "configs:")
    for p in written:
        print("  ", p)
    # self-check: every config has Qwen3-0.6B, none has 1.7B/8B/e1
    bad = []
    for p in written:
        txt = open(p).read()
        if "Qwen3-0.6B" not in txt:
            bad.append((p, "missing Qwen3-0.6B"))
        for forbidden in ("1.7B", "8B", "Qwen3-1.7B", "Qwen3-8B"):
            if forbidden in txt:
                bad.append((p, f"contains {forbidden}"))
    print("SELF-CHECK:", "ALL OK (Qwen3-0.6B, no 1.7B/8B)" if not bad else bad)


if __name__ == "__main__":
    main()

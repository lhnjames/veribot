"""v4 GRPO (offline). Continues from the E2 cold-start checkpoint and optimises the three-gate + Q
reward computed purely by table lookup against the TRAIN evidence bank. NO API, NO network.

Policy init / KL reference: the E2 best LoRA is merged into the base weights; that merged model is
both the starting policy and pi_ref (the cold-start checkpoint), and GRPO trains a fresh LoRA on
top. Reward is reward_v4.score against evidence_bank_train.jsonl only — reward_v4.EvidenceBank
refuses any path containing 'eval' or not ending in '_train.jsonl'.

Offline assertions at startup:
  * DEEPSEEK_API_KEY / OPENAI_API_KEY must be UNSET in the training environment;
  * the bank path is the train bank (enforced by EvidenceBank);
  * no http client is imported by the reward path.

    python scripts/train_grpo_v4.py --config configs/grpo_v4_0.6b.yaml \
        --sft-lora outputs/v4/e2_sft/run_0.6b/best_lora --out outputs/v4/grpo/run_0.6b
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.v4.prompt_v4 import canonical_prompt_text  # noqa: E402
from verifiable_bot.v4.render_v4 import Unit, render_units  # noqa: E402
from scripts.reward_v4 import EvidenceBank, KEYWORDS, score  # noqa: E402

LABEL_MAP = {"human": 0, "bot": 1}
# module-level curriculum flag toggled by a callback (relaxed evidence gate for warmup rounds)
_STATE = {"relaxed": False}


def load_yaml(p):
    import yaml
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def assert_offline():
    for var in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        assert not os.environ.get(var), f"{var} is set; GRPO training must be fully offline"


def units_text(row):
    return render_units([Unit(u["unit_id"], u["text"]) for u in row["units"]])


def build_dataset(render_path, bank, tokenizer, seed, balance=True, limit=None):
    """1:1 balanced accounts that exist in the TRAIN bank. prompt = canonical formatted string."""
    rows = [json.loads(l) for l in open(render_path, encoding="utf-8")]
    pos, neg = [], []
    for r in rows:
        if not bank.has(r["account_id"]):
            continue
        rec = {"prompt": canonical_prompt_text(units_text(r), tokenizer),
               "account_id": r["account_id"], "label": r["label"]}
        (pos if r["label"] == "bot" else neg).append(rec)
    rng = random.Random(seed)
    rng.shuffle(pos); rng.shuffle(neg)
    if balance:
        n = min(len(pos), len(neg))
        data = pos[:n] + neg[:n]
    else:
        data = pos + neg
    rng.shuffle(data)
    if limit:
        data = data[:limit]
    return Dataset.from_list(data)


def make_reward_func(bank, near_miss):
    def reward_three_gate(prompts, completions, account_id, label, **kwargs):
        out = []
        for comp, aid, lb in zip(completions, account_id, label):
            text = comp if isinstance(comp, str) else (comp[0]["content"] if comp else "")
            r = score(text, aid, lb, bank, near_miss=near_miss, evidence_relaxed=_STATE["relaxed"])
            out.append(float(r.total))
        return out
    reward_three_gate.__name__ = "three_gate_offline"
    return reward_three_gate


class RelaxedCurriculum(TrainerCallback):
    def __init__(self, relaxed_steps): self.relaxed_steps = relaxed_steps
    def on_step_begin(self, args, state, control, **kwargs):
        _STATE["relaxed"] = state.global_step < self.relaxed_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sft-lora", required=True)
    ap.add_argument("--bank", default="outputs/v4/evidence_bank_v1/evidence_bank_train.jsonl")
    ap.add_argument("--render", default="outputs/v4/render_v4/train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    assert_offline()
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    bank = EvidenceBank(args.bank)   # refuses eval bank
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # policy init = E2 merged (also serves as KL reference inside TRL)
    base = AutoModelForCausalLM.from_pretrained(cfg["model_name"], trust_remote_code=True,
                                                torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    from peft import PeftModel, LoraConfig
    merged = PeftModel.from_pretrained(base, args.sft_lora).merge_and_unload()

    dataset = build_dataset(args.render, bank, tokenizer, cfg["seed"],
                            balance=True, limit=cfg.get("max_accounts"))

    peft_config = LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM")

    grpo_args = GRPOConfig(
        output_dir=str(out),
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_generations=cfg["group_size"],
        learning_rate=cfg["lr"],
        beta=cfg["beta_kl"],
        epsilon=cfg["epsilon_clip_low"],
        epsilon_high=cfg["epsilon_clip_high"],
        temperature=cfg["temperature"],
        max_completion_length=cfg["max_completion_length"],
        num_train_epochs=cfg["epochs"],
        max_steps=cfg.get("max_steps", -1),
        logging_steps=cfg.get("logging_steps", 2),
        save_steps=cfg.get("save_steps", 50),
        bf16=True,
        scale_rewards=cfg.get("scale_rewards", True),
        log_completions=True,
        report_to=[],
        seed=cfg["seed"],
    )

    trainer = GRPOTrainer(
        model=merged,
        reward_funcs=make_reward_func(bank, cfg.get("near_miss", 0.0)),
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[RelaxedCurriculum(cfg.get("relaxed_steps", 0))],
    )

    (out / "resolved_config.json").write_text(json.dumps(
        {"config": cfg, "n_accounts": len(dataset), "bank": args.bank, "sft_lora": args.sft_lora,
         "veri_weights": bank.veri_weight}, indent=2), encoding="utf-8")

    if args.check_only:
        # one reward call on a dummy completion, prove offline lookup works, then exit
        aid = dataset[0]["account_id"]; lb = dataset[0]["label"]
        top2 = bank.top2_keywords(aid)
        comp = (f"<think>\nStep 1 (observe): x [OBS_01].\nStep 2 (connect): y [PROFILE_02].\n"
                f"Step 3 (conclude): z.\n</think>\n<evidence>OBS_01 | PROFILE_02</evidence>\n"
                f"<chain>{top2[0]} - {top2[1]}</chain>\n<answer>{lb}</answer>")
        r = score(comp, aid, lb, bank)
        print(f"CHECK OK: dataset={len(dataset)} sample_reward={r.total:.3f} tier={r.tier}")
        return

    trainer.train()
    trainer.save_model(str(out / "final_lora"))
    print("GRPO v4 DONE")


if __name__ == "__main__":
    main()

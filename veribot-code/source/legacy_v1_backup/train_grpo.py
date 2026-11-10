"""GRPO entrypoint with verifiability-first reward functions.

TRL calls each reward on grouped completions. Semantic and counterfactual rewards can be
precomputed into the training JSONL or supplied by the remote teacher service; grounding,
label and format are always recomputed from the AccountRecord.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from .prompting import SYSTEM_PROMPT, render_record
from .rewards import score_prediction
from .schema import AccountRecord


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            record = AccountRecord.model_validate(json.loads(line)["record"])
            rows.append({"prompt": SYSTEM_PROMPT + "\n" + render_record(record), "record_json": record.model_dump_json()})
    dataset = Dataset.from_list(rows)

    def reward(completions, record_json, **_):
        values = []
        for completion, raw_record in zip(completions, record_json):
            text = completion[0]["content"] if isinstance(completion, list) else str(completion)
            values.append(score_prediction(AccountRecord.model_validate_json(raw_record), text)["total"])
        return values

    config = GRPOConfig(output_dir=str(args.output), learning_rate=5e-6, bf16=True,
                        per_device_train_batch_size=2, gradient_accumulation_steps=16,
                        num_generations=4, max_prompt_length=4096, max_completion_length=512,
                        save_total_limit=3, report_to="none")
    peft = LoraConfig(r=32, lora_alpha=64, lora_dropout=.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    trainer = GRPOTrainer(model=args.model, reward_funcs=[reward], args=config,
                          train_dataset=dataset, peft_config=peft)
    trainer.train()
    trainer.save_model(str(args.output / "final"))


if __name__ == "__main__": main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

from .model import QwenGraphEvidenceModel
from .prompting import SYSTEM_PROMPT, render_record
from .schema import AccountRecord


class EvidenceCollator:
    def __init__(self, tokenizer, max_length: int): self.tok, self.max_length = tokenizer, max_length

    def __call__(self, rows):
        texts, prompt_lengths, relation_rows = [], [], []
        for row in rows:
            record = AccountRecord.model_validate(row["record"])
            prompt = self.tok.apply_chat_template([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": render_record(record)}], tokenize=False, add_generation_prompt=True)
            answer = json.dumps(row["target"], ensure_ascii=False, separators=(",", ":")) + self.tok.eos_token
            texts.append(prompt + answer)
            prompt_lengths.append(len(self.tok(prompt, add_special_tokens=False)["input_ids"]))
            relation_vocab = {"following": 0, "follower": 1}
            relation_rows.append([relation_vocab.get(e.relation, 2) for e in record.graph_edges[:24]])
        batch = self.tok(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        labels = batch["input_ids"].clone()
        for i, n in enumerate(prompt_lengths): labels[i, :min(n, labels.shape[1])] = -100
        labels[batch["attention_mask"] == 0] = -100
        width = max(1, max(map(len, relation_rows)))
        rel = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros(len(rows), width)
        for i, values in enumerate(relation_rows):
            if values: rel[i, :len(values)] = torch.tensor(values); mask[i, :len(values)] = 1
        batch.update(labels=labels, relation_ids=rel, directions=torch.zeros_like(rel), graph_mask=mask)
        return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    lm = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, device_map="auto",
                                              torch_dtype=torch.bfloat16, trust_remote_code=True)
    lm = prepare_model_for_kbit_training(lm)
    lm = get_peft_model(lm, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model = QwenGraphEvidenceModel(lm)
    dataset = load_dataset("json", data_files=str(args.data), split="train")
    training = TrainingArguments(output_dir=str(args.output), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=16, learning_rate=1e-4,
        bf16=True, gradient_checkpointing=True, logging_steps=10, save_steps=500,
        save_total_limit=3, report_to="none", remove_unused_columns=False, max_steps=args.max_steps)
    trainer = Trainer(model=model, args=training, train_dataset=dataset,
                      data_collator=EvidenceCollator(tokenizer, args.max_length))
    trainer.train()
    (args.output / "final").mkdir(parents=True, exist_ok=True)
    model.language_model.save_pretrained(args.output / "final")
    tokenizer.save_pretrained(args.output / "final")
    torch.save(model.graph_prefix.state_dict(), args.output / "final" / "graph_prefix.pt")


if __name__ == "__main__": main()

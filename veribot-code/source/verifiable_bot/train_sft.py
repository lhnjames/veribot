"""Cold-start LoRA SFT (design spec section 5).

L_cold = L_select + lambda_y * L_agreement + lambda_c * L_claim + lambda_s * L_status.

Because E, y, C, s are all generated in one autoregressive JSON completion, the sum of these
four per-position NLL terms is exactly a per-token-weighted cross-entropy loss over the whole
completion: give the evidence_ids tokens weight 1, agreement tokens weight lambda_y, claims
tokens weight lambda_c, evidence_status tokens weight lambda_s, everything else (prompt tokens,
JSON punctuation/keys) gets label=-100 or weight 1 respectively. See sft_targets.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

from .config import apply_config_defaults, load_config
from .prompting import inference_messages
from .schema import AccountRecord
from .sft_targets import serialize_target_with_spans

CONFIG_MAPPING = {
    "model": "model_name",
    "max_length": "max_length",
    "lora_r": "lora_rank",
    "lora_alpha": "lora_alpha",
    "epochs": "sft.epochs",
    "batch_size": "sft.batch_size",
    "grad_accum": "sft.gradient_accumulation_steps",
    "lr": "sft.learning_rate",
    "lambda_y": "sft.lambda_y",
    "lambda_c": "sft.lambda_c",
    "lambda_s": "sft.lambda_s",
}

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class ColdStartDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


class WeightedEvidenceCollator:
    def __init__(self, tokenizer, max_length: int, lambda_y: float, lambda_c: float, lambda_s: float):
        self.tok = tokenizer
        self.max_length = max_length
        self.lambda_y, self.lambda_c, self.lambda_s = lambda_y, lambda_c, lambda_s

    def _encode_one(self, row: dict) -> dict:
        record = AccountRecord.model_validate(row["record"])
        prompt = self.tok.apply_chat_template(inference_messages(record), tokenize=False, add_generation_prompt=True)
        answer_text, spans = serialize_target_with_spans(row["target"], self.lambda_y, self.lambda_c, self.lambda_s)
        full_text = prompt + answer_text
        enc = self.tok(full_text, add_special_tokens=False, return_offsets_mapping=True,
                       truncation=True, max_length=self.max_length - 1)
        input_ids = enc["input_ids"] + [self.tok.eos_token_id]
        offsets = enc["offset_mapping"] + [(len(full_text), len(full_text))]
        prompt_chars = len(prompt)
        full_spans = [(prompt_chars + s, prompt_chars + e, w) for s, e, w in spans]

        labels, weights = [], []
        for (s, e), tid in zip(offsets, input_ids):
            if s < prompt_chars:
                labels.append(-100)
                weights.append(0.0)
                continue
            labels.append(tid)
            w = 1.0
            for ss, ee, ww in full_spans:
                if ss <= s < ee:
                    w = ww
                    break
            weights.append(w)
        return {"input_ids": input_ids, "labels": labels, "weights": weights}

    def __call__(self, rows: list[dict]) -> dict:
        encoded = [self._encode_one(r) for r in rows]
        max_len = max(len(e["input_ids"]) for e in encoded)
        pad_id = self.tok.pad_token_id
        input_ids, attention_mask, labels, weights = [], [], [], []
        for e in encoded:
            n = len(e["input_ids"])
            pad = max_len - n
            input_ids.append(e["input_ids"] + [pad_id] * pad)
            attention_mask.append([1] * n + [0] * pad)
            labels.append(e["labels"] + [-100] * pad)
            weights.append(e["weights"] + [0.0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "weights": torch.tensor(weights, dtype=torch.float),
        }


class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        weights = inputs.pop("weights")
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = outputs.logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        shift_weights = weights[:, 1:]
        vocab = logits.shape[-1]
        per_token = F.cross_entropy(logits.reshape(-1, vocab), shift_labels.reshape(-1),
                                    ignore_index=-100, reduction="none").view(shift_labels.shape)
        weighted = per_token * shift_weights
        loss = weighted.sum() / shift_weights.sum().clamp_min(1e-6)
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="/path/to/data_root/cache/huggingface/hub/"
                                               "models--Qwen--Qwen3-0.6B/snapshots/"
                                               "c1899de289a04d12100db370d81485cdf75e47ca")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda-y", type=float, default=1.5)
    ap.add_argument("--lambda-c", type=float, default=1.0)
    ap.add_argument("--lambda-s", type=float, default=0.5)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--save-steps", type=int, default=100,
                    help="also controls how often you can eval_checkpoint.py an intermediate checkpoint")
    ap.add_argument("--no-gradient-checkpointing", action="store_true",
                    help="trade activation memory for speed (~20-30%% faster) when VRAM headroom allows it; "
                         "gradient checkpointing recomputes activations during backward instead of storing them")
    ap.add_argument("--config", type=Path, default=None,
                    help="YAML config (default configs/veriroute_lite.yaml); explicit CLI flags override it")
    args, _ = ap.parse_known_args()
    apply_config_defaults(ap, load_config(args.config), CONFIG_MAPPING)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        local_files_only=True,
    )
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                                             task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES))
    if not args.no_gradient_checkpointing:
        model.enable_input_require_grads()  # required so gradient checkpointing backprops into the LoRA adapters
    model.print_trainable_parameters()

    dataset = ColdStartDataset(args.data)
    collator = WeightedEvidenceCollator(tokenizer, args.max_length, args.lambda_y, args.lambda_c, args.lambda_s)

    training = TrainingArguments(
        output_dir=str(args.output), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, bf16=True, gradient_checkpointing=not args.no_gradient_checkpointing,
        logging_steps=10, save_steps=args.save_steps, save_total_limit=3, report_to="none",
        remove_unused_columns=False,
    )
    trainer = WeightedLossTrainer(model=model, args=training, train_dataset=dataset, data_collator=collator)
    trainer.train()

    final_dir = args.output / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Also save a merged full checkpoint: this becomes both pi_ref and the GRPO starting point
    # (spec section 6.3: "pi_ref takes the model at the end of cold start"). GRPO then attaches a
    # *fresh* LoRA adapter on top of this merged model, so TRL's adapter-disabled reference model
    # is exactly this cold-start checkpoint, not the raw pretrained base.
    merged_dir = args.output / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)


if __name__ == "__main__":
    main()

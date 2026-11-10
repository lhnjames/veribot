"""Checkpoint evaluation: JSON validity, reference validity, and classification metrics (accuracy,
precision, recall, F1) on a held-out slice.

Training loss alone can't tell you whether a checkpoint is any good at the actual task -- this
generates from the checkpoint on real held-out accounts and scores exactly what the reward
pipeline checks structurally (format_check.py), plus classification metrics against the dataset's
true labels. Doesn't call the LLM judge (that's rewards.py's job during GRPO); this is a cheap,
fast, judge-free sanity check meant to run after every SFT/GRPO checkpoint.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifiable_bot.format_check import check_format
from verifiable_bot.prompting import inference_messages
from verifiable_bot.schema import AccountRecord
from verifiable_bot.units import build_units, hidden_unit_metadata, unit_map

LABELS = ["bot", "human"]


def load_model(checkpoint: Path, base_model: str | None):
    # Detect adapter-only dirs by the presence of adapter_config.json rather than by probing with
    # try/except: AutoModelForCausalLM.from_pretrained on an adapter-only dir can raise
    # safetensors.SafetensorError ("incomplete metadata, file not fully covered") instead of the
    # OSError/ValueError this used to catch -- that exception type doesn't subclass either, so the
    # probe crashed instead of falling back to --base-model. Checking for the marker file first
    # sidesteps relying on which exception type a given transformers/safetensors version happens
    # to raise for "this isn't a full model".
    is_adapter = (Path(checkpoint) / "adapter_config.json").exists()

    # Intermediate Trainer checkpoints (checkpoint-N/) only contain adapter weights + trainer
    # state, never a tokenizer -- only the final save_pretrained() call writes one. Fall back to
    # --base-model's tokenizer (same vocab/template, since the adapter never touches the tokenizer).
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    except (OSError, ValueError):
        if base_model is None:
            raise SystemExit(f"{checkpoint} has no tokenizer (looks like an intermediate checkpoint); "
                             "pass --base-model")
        tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    if is_adapter:
        if base_model is None:
            raise SystemExit(f"{checkpoint} looks like an adapter-only dir; pass --base-model")
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
        )
        model = PeftModel.from_pretrained(base, str(checkpoint))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint), torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
        )
    return tokenizer, model.eval()


def load_stratified(path: Path, per_stratum: int, seed: int) -> list[AccountRecord]:
    """Balanced draw across (source, label), never a naive file prefix -- the unified validation
    set is heavily skewed (e.g. twibot22/human is ~91%% of it), so a plain head(-limit) slice would
    barely test bot recall or TwiBot-20 at all."""
    buckets: dict[tuple[str, str], list[AccountRecord]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = AccountRecord.model_validate_json(line)
            if record.label is not None:
                buckets[(record.source, record.label)].append(record)
    rng = random.Random(seed)
    records = []
    for group in buckets.values():
        rng.shuffle(group)
        records.extend(group[:per_stratum])
    rng.shuffle(records)
    return records


def classification_report(confusion: dict[tuple[str, str], int], labels: list[str]) -> dict:
    """confusion: (predicted, true) -> count. Returns per-class precision/recall/F1/support,
    overall accuracy, and macro/weighted averages -- the ACC/F1/recall the user asked for."""
    total = sum(confusion.values())
    per_class = {}
    for label in labels:
        tp = confusion.get((label, label), 0)
        fp = sum(v for (pred, true), v in confusion.items() if pred == label and true != label)
        fn = sum(v for (pred, true), v in confusion.items() if true == label and pred != label)
        support = sum(v for (_, true), v in confusion.items() if true == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    accuracy = sum(confusion.get((l, l), 0) for l in labels) / total if total else 0.0
    macro_precision = sum(per_class[l]["precision"] for l in labels) / len(labels)
    macro_recall = sum(per_class[l]["recall"] for l in labels) / len(labels)
    macro_f1 = sum(per_class[l]["f1"] for l in labels) / len(labels)
    weighted_f1 = (sum(per_class[l]["f1"] * per_class[l]["support"] for l in labels) / total) if total else 0.0
    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "support_total": total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True, help="adapter dir or merged full-weight dir")
    ap.add_argument("--base-model", default=None, help="required if --checkpoint is an adapter-only dir")
    ap.add_argument("--data", type=Path, required=True, help="held-out AccountRecord jsonl, one per line")
    ap.add_argument("--per-stratum", type=int, default=125, help="accounts drawn per (source,label) "
                    "bucket, e.g. 125*4=500 total for the default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--predictions-out", type=Path, default=None,
                    help="optional: write one JSON line per account with raw output + verdict")
    args = ap.parse_args()

    tokenizer, model = load_model(args.checkpoint, args.base_model)
    records = load_stratified(args.data, args.per_stratum, args.seed)

    n = 0
    json_valid = 0
    format_valid = 0
    label_correct_overall = 0
    status_counts: dict[str, int] = defaultdict(int)
    confusion: dict[tuple[str, str], int] = defaultdict(int)  # (predicted, true) -> count
    confusion_by_source: dict[str, dict[tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))
    pred_file = open(args.predictions_out, "w", encoding="utf-8") if args.predictions_out else None

    for record in records:
        n += 1
        units = build_units(record)
        valid_ids = unit_map(units)
        prompt = tokenizer.apply_chat_template(inference_messages(record), tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        raw = tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        try:
            json.loads(raw.strip().strip("`"))
            json_valid += 1
        except json.JSONDecodeError:
            pass

        fc = check_format(raw, valid_ids, hidden_unit_metadata(units))
        prediction = None
        if fc.ok:
            prediction = (record.upstream_verdict if fc.output.agreement == "agree"
                          else ("human" if record.upstream_verdict == "bot" else "bot"))
        if fc.ok:
            format_valid += 1
            status_counts[fc.output.evidence_status] += 1
            confusion[(prediction, record.label)] += 1
            confusion_by_source[record.source][(prediction, record.label)] += 1
            if prediction == record.label:
                label_correct_overall += 1
        # format-invalid accounts are excluded from the classification_report confusion matrix
        # (there is no "predicted" label to score), but format_valid_rate below tracks them.

        if pred_file:
            pred_file.write(json.dumps({"account_id": record.account_id, "source": record.source,
                                        "true_label": record.label, "format_ok": fc.ok,
                                        "prediction": prediction, "raw": raw}, ensure_ascii=False) + "\n")

    if pred_file:
        pred_file.close()

    report = classification_report(confusion, LABELS) if format_valid else None
    by_source = {s: classification_report(c, LABELS) for s, c in confusion_by_source.items() if sum(c.values())}

    print(json.dumps({
        "n": n,
        "json_valid_rate": json_valid / n if n else 0.0,
        "format_valid_rate": format_valid / n if n else 0.0,
        "label_accuracy_overall": label_correct_overall / n if n else 0.0,  # format-invalid counted wrong
        "classification_report_among_format_valid": report,
        "evidence_status_among_format_valid": dict(status_counts),
        "by_source": by_source,
    }, indent=2))


if __name__ == "__main__":
    main()

"""Audit named evidence versus published-content coordinates in a frozen model.

This is a diagnostic of the already selected 20-concept MGTAB classifier.
It does not select a new model or threshold from fixed-test outcomes.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "output/v3_mgtab_targeted_20_concept_ab_group_tree"
OUTPUT = ROOT / "research/certificates/mgtab_twenty_named_content_dependence.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(logit: float, threshold: float) -> int:
    threshold_logit = math.log(threshold / (1 - threshold))
    return int(logit >= threshold_logit)


def main() -> None:
    rows = []
    for seed in (0, 1, 2):
        card_path = MODEL_DIR / f"seed{seed}/test_evidence_cards.jsonl.gz"
        counts = {
            "accounts": 0,
            "full_correct": 0,
            "named_only_correct": 0,
            "content_only_correct": 0,
            "named_to_full_class_changes": 0,
            "content_rescues_named": 0,
            "content_harms_named": 0,
            "named_to_full_bot_changes": 0,
            "named_to_full_human_changes": 0,
            "named_nonzero": 0,
            "content_nonzero": 0,
        }
        content_abs, named_abs, content_fraction = [], [], []
        max_additivity_error = 0.0
        with gzip.open(card_path, "rt", encoding="utf-8") as stream:
            for line in stream:
                card = json.loads(line)
                effects = card["group_effects"]
                if len(effects) != 10:
                    raise ValueError("unexpected MGTAB group vocabulary")
                named_groups = [name for name in effects
                                if not name.startswith("published_content_coordinate_")]
                content_groups = [name for name in effects
                                  if name.startswith("published_content_coordinate_")]
                if len(named_groups) != 6 or len(content_groups) != 4:
                    raise ValueError("named/content group partition differs")
                named = math.fsum(effects[name]["exact_zero_mask_logit_difference"]
                                  for name in named_groups)
                content = math.fsum(effects[name]["exact_zero_mask_logit_difference"]
                                    for name in content_groups)
                zero = float(card["zero_evidence_logit"])
                full = float(card["logit"])
                error = abs(zero + named + content - full)
                max_additivity_error = max(max_additivity_error, error)
                if error > 1e-9:
                    raise ValueError("source groups do not reconstruct frozen logit")
                threshold = float(card["threshold"])
                full_class = classify(full, threshold)
                if full_class != int(card["prediction"]):
                    raise ValueError("saved prediction differs from frozen logit")
                named_class = classify(zero + named, threshold)
                content_class = classify(zero + content, threshold)
                label = int(card["label"])
                counts["accounts"] += 1
                counts["full_correct"] += full_class == label
                counts["named_only_correct"] += named_class == label
                counts["content_only_correct"] += content_class == label
                counts["named_nonzero"] += abs(named) > 1e-12
                counts["content_nonzero"] += abs(content) > 1e-12
                if named_class != full_class:
                    counts["named_to_full_class_changes"] += 1
                    counts["named_to_full_bot_changes"] += full_class == 1
                    counts["named_to_full_human_changes"] += full_class == 0
                    counts["content_rescues_named"] += full_class == label
                    counts["content_harms_named"] += named_class == label
                named_abs.append(abs(named))
                content_abs.append(abs(content))
                denominator = abs(named) + abs(content)
                content_fraction.append(abs(content) / denominator
                                        if denominator > 0 else 0.0)
        if counts["accounts"] != 1020:
            raise ValueError("fixed-test population differs")
        rows.append({
            "seed": seed,
            "card_sha256": sha256(card_path),
            **counts,
            "max_logit_group_additivity_error": max_additivity_error,
            "mean_absolute_named_logit_effect": math.fsum(named_abs) / 1020,
            "mean_absolute_content_logit_effect": math.fsum(content_abs) / 1020,
            "mean_content_fraction_of_two_absolute_effects":
                math.fsum(content_fraction) / 1020,
            "max_absolute_content_logit_effect_on_fixed_test": max(content_abs),
        })
        print(rows[-1], flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "scope": "frozen_mgtab_20_concept_named_vs_content_source_dependence",
        "diagnostic_only_no_test_selection": True,
        "definition": {
            "named_only_logit": "zero_evidence_logit + sum(six named group exact deltas)",
            "content_only_logit": "zero_evidence_logit + sum(four published-content exact deltas)",
            "full_logit": "zero_evidence_logit + named effects + content effects",
        },
        "runs": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()

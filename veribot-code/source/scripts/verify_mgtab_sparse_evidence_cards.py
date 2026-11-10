"""Check every exported numeric evidence card against frozen test predictions."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cards-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    exported = json.loads((args.cards_dir / "evidence_cards_manifest.json").read_text(
        encoding="utf-8"))
    runs = []
    for seed in (0, 1, 2):
        directory = args.model_dir / f"seed{seed}"
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        saved = json.loads((directory / "test_predictions.json").read_text(encoding="utf-8"))
        card_path = args.cards_dir / f"seed{seed}_evidence_cards.jsonl.gz"
        export_row = exported["runs"][seed]
        if (sha256_file(card_path) != export_row["cards_sha256"]
                or report["manifest"] != saved["manifest"]):
            raise ValueError("card artifact or model manifest differs")
        groups = report["reasoner"]["groups"]
        names = {name for members in groups.values() for name in members}
        rows = saved["predictions"]
        largest_difference = 0.0
        count = 0
        with gzip.open(card_path, "rt", encoding="utf-8") as stream:
            for line, prediction in zip(stream, rows, strict=True):
                card = json.loads(line)
                if (card["account_id"] != prediction["account_id"]
                        or card["label"] != prediction["label"]
                        or card["model_seed"] != seed
                        or card["source_features_sha256"] !=
                        report["manifest"]["source_features_sha256"]):
                    raise ValueError("card population or source differs")
                terms = card["active_terms"]
                term_values = {term["name"]: float(term["total_logit_contribution"])
                               for term in terms}
                if len(terms) != export_row["active_terms_per_account"] or len(term_values) != len(terms):
                    raise ValueError("card active-term vocabulary differs")
                if not set(term_values) <= names:
                    raise ValueError("card contains term outside frozen groups")
                for term in terms:
                    if abs(term["upper_logit_contribution"] + term["lower_logit_contribution"]
                           - term["total_logit_contribution"]) > 1e-12:
                        raise ValueError("term arithmetic differs")
                logits = float(card["intercept"] + sum(term_values.values()))
                probability = float(expit(logits))
                differences = [abs(logits - card["logit"]),
                               abs(probability - card["probability"]),
                               abs(probability - prediction["probability"])]
                for group, members in groups.items():
                    contribution = sum(term_values.get(name, 0.0) for name in members)
                    differences.extend((abs(contribution - card["group_contributions"][group]),
                                        abs(contribution - prediction["group_contributions"][group])))
                largest_difference = max(largest_difference, *differences)
                if (max(differences) > 1e-12
                        or card["prediction"] != int(probability >= card["threshold"])):
                    raise ValueError("card fails to reconstruct frozen decision")
                count += 1
        if count != len(rows) or count != export_row["test_accounts"]:
            raise ValueError("card count differs from frozen test population")
        runs.append({"seed": seed, "cards": count,
                     "active_terms_per_card": export_row["active_terms_per_account"],
                     "max_probability_or_logit_difference": largest_difference,
                     "status": "verified"})
    result = {"dataset": "mgtab", "scope": "every_fixed_test_numeric_evidence_card",
              "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

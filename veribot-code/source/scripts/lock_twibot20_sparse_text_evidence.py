"""Lock sparse source-addressable unigram reasoner from official dev only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_v3_twibot20 import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    selected = next((run for run in screen["runs"]
                     if run["penalty"] == "l1" and run["C"] == 1.0), None)
    if (screen["scope"] != "official_dev_only_evidence_reasoner_screen"
            or screen["test_predictions_accessed"] is not False
            or screen["text_evidence"] !=
            "source-addressable train-IDF-normalized unigram TF-IDF"
            or selected is None
            or selected["nonzero_lexical_terms"] > 200
            or selected["validation"]["macro_f1"] < 0.84
            or screen["sparse_l1_c1_artifact_sha256"] != sha256(args.artifact)
            or screen["reference_artifact_sha256"] != sha256(args.reference)):
        raise ValueError("sparse unigram model did not pass source or dev gate")
    with np.load(args.artifact, allow_pickle=False) as saved:
        if (len(saved["vocabulary"]) != screen["vocabulary_size"]
                or np.count_nonzero(saved["coefficient"][44:]) !=
                selected["nonzero_lexical_terms"]):
            raise ValueError("sparse artifact differs from dev summary")
    result = {
        "dataset": "twibot20",
        "selected_model": "source_backed_unigram_l1_c1",
        "selection_reason": "source-addressable unigram evidence; at most 200 active lexical terms; dev Macro-F1 >= 0.84",
        "official_dev_macro_f1": selected["validation"]["macro_f1"],
        "active_lexical_terms": selected["nonzero_lexical_terms"],
        "candidate_test_predictions_accessed": False,
        "prior_22_and_128_token_test_predictions_already_observed": True,
        "screen_sha256": sha256(args.screen),
        "model_artifact_sha256": sha256(args.artifact),
        "reference_artifact_sha256": sha256(args.reference),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

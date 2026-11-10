"""Independently replay Fox8 interaction cards with the frozen source model."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime
from verifiable_bot.high_performance_fact_effects import evidence_design


ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "research/certificates/fox8_interaction_cards.json"
READBACK = ROOT / "research/certificates/fox8_interaction_cards_readback.json"
GROUPS = [(0, 23), (23, 31), (31, 43), (43, 74)]
GROUP_NAMES = ["account_and_profile", "sampled_tweet_activity",
               "native_interaction", "native_neighbor_means"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run() -> dict:
    certificate = json.loads(CERT.read_text(encoding="utf-8"))
    rows = []
    for recorded in certificate["runs"]:
        seed = recorded["seed"]
        path = Path(recorded["cards_path"])
        if sha256(path) != recorded["cards_sha256"]:
            raise ValueError("card file changed")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            cards = [json.loads(line) for line in stream]
        if len(cards) != 343 or len({c["account_id"] for c in cards}) != 343:
            raise ValueError("card account population changed")
        runtime = Fox8NeighborAtomicRuntime(ROOT, seed)
        indices = np.array([runtime.index[c["account_id"]] for c in cards])
        matrix = evidence_design(runtime, indices)
        if matrix.shape[1] != 148:
            raise ValueError("frozen model input changed")
        replay = np.empty((len(cards), 16), dtype=float)
        for mask in range(16):
            modified = matrix.copy()
            for j, (start, stop) in enumerate(GROUPS):
                if not mask & (1 << j):
                    modified[:, start:stop] = 0
                    modified[:, start+74:stop+74] = 0
            replay[:, mask] = expit(np.asarray(
                runtime.model.booster_.predict(modified, raw_score=True), dtype=float))
        max_probability_error = 0.0
        max_interaction_error = 0.0
        max_higher_order_error = 0.0
        disagreement_count = 0
        for i, card in enumerate(cards):
            if (card["seed"] != seed or card["model_variant"] != "raw_neighbor_74_lightgbm"
                    or card["source_report_sha256"] != recorded["source_report_sha256"]
                    or card["subset_matrix_sha256"] != recorded["subset_matrix_sha256"]
                    or card["threshold"] != runtime.threshold
                    or card["group_names"] != GROUP_NAMES):
                raise ValueError("card metadata differs from frozen model")
            subset_rows = card["all_16_group_subsets"]
            saved = np.asarray([item["bot_probability"] for item in subset_rows])
            if [item["mask"] for item in subset_rows] != list(range(16)):
                raise ValueError("subset order changed")
            for mask, item in enumerate(subset_rows):
                expected_names = [GROUP_NAMES[j] for j in range(4) if mask & (1 << j)]
                expected_class = "Bot" if replay[i, mask] >= runtime.threshold else "Human"
                if item["included_groups"] != expected_names or item["predicted_class"] != expected_class:
                    raise ValueError("subset groups or class changed")
            max_probability_error = max(max_probability_error,
                                        float(np.max(np.abs(saved-replay[i]))))
            score = logit(replay[i])
            additive = score[0] + sum(score[1 << j]-score[0] for j in range(4))
            residual = float(score[-1]-additive)
            max_interaction_error = max(max_interaction_error,
                                        abs(residual-card["interaction_logit"]),
                                        abs(float(expit(additive))-
                                            card["additive_approximation_probability"]),
                                        abs(float(score[0])-card["zero_input_logit"]),
                                        abs(float(replay[i, 0])-card["zero_input_probability"]),
                                        abs(float(replay[i, -1])-card["full_probability"]))
            if card["full_class"] != ("Bot" if replay[i, -1] >= runtime.threshold else "Human"):
                raise ValueError("full class differs")
            for j, name in enumerate(GROUP_NAMES):
                max_interaction_error = max(max_interaction_error,
                    abs(float(score[1 << j]-score[0])-
                        card["single_group_logit_changes"][name]))
            saved_terms = card["higher_order_interactions"]
            if [term["mask"] for term in saved_terms] != [mask for mask in range(16)
                                                            if mask.bit_count() >= 2]:
                raise ValueError("higher-order interaction masks changed")
            for term in saved_terms:
                mask = term["mask"]
                expected_names = [GROUP_NAMES[j] for j in range(4) if mask & (1 << j)]
                if term["groups"] != expected_names:
                    raise ValueError("higher-order group names changed")
                value = sum((-1) ** (mask.bit_count()-part.bit_count()) * score[part]
                            for part in range(16) if part & mask == part)
                max_higher_order_error = max(max_higher_order_error,
                    abs(float(value)-term["logit_interaction"]))
            disagreement = (expit(additive) >= runtime.threshold) != (
                replay[i, -1] >= runtime.threshold)
            if disagreement != card["additive_approximation_class_disagrees"]:
                raise ValueError("class disagreement claim changed")
            disagreement_count += bool(disagreement)
        if (max_probability_error > 1e-12 or max_interaction_error > 1e-12
                or max_higher_order_error > 1e-12):
            raise ValueError("card values differ from direct frozen inference")
        if disagreement_count != recorded["additive_approximation_class_disagreements"]:
            raise ValueError("class disagreement count changed")
        rows.append({"seed": seed, "accounts": len(cards),
                     "direct_model_subset_probability_max_error": max_probability_error,
                     "direct_model_interaction_logit_max_error": max_interaction_error,
                     "direct_model_higher_order_logit_max_error": max_higher_order_error,
                     "additive_approximation_class_disagreements": disagreement_count,
                     "cards_sha256": recorded["cards_sha256"]})
    result = {"status": "all_fox8_interaction_cards_match_direct_frozen_model_inference",
              "fixed_test_labels_read": False, "card_certificate_sha256": sha256(CERT),
              "runs": rows}
    READBACK.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    for row in run()["runs"]:
        print(row["seed"], row["accounts"],
              row["direct_model_subset_probability_max_error"],
              row["additive_approximation_class_disagreements"])

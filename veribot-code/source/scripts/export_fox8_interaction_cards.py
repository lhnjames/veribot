"""Write per-account Fox8 source-group interaction cards from frozen scores.

The card explains the exact encoded-group subset game of raw_neighbor_74.
It does not treat group masking as a raw-source edit or semantic truth.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit


ROOT = Path(__file__).resolve().parents[2]
SUBSETS = ROOT / "output/v3_minimal_sufficient_source_groups"
REPORTS = ROOT / "output/v3_best_decision_reports"
CARDS = ROOT / "output/v3_fox8_interaction_cards"
CERT = ROOT / "research/certificates/fox8_interaction_cards.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interactions(scores: np.ndarray, names: list[str]) -> list[dict]:
    count = len(names)
    out = []
    for mask in range(1, 1 << count):
        if mask.bit_count() < 2:
            continue
        effect = sum(
            (-1) ** (mask.bit_count() - part.bit_count()) * scores[part]
            for part in range(1 << count) if part & mask == part)
        out.append({"mask": mask,
                    "groups": [names[j] for j in range(count) if mask & (1 << j)],
                    "logit_interaction": float(effect)})
    return out


def run() -> dict:
    subset_cert = json.loads((ROOT / "research/certificates/minimal_sufficient_source_groups.json").read_text(encoding="utf-8"))
    report_cert = json.loads((ROOT / "research/certificates/best_decision_reports_full_test.json").read_text(encoding="utf-8"))
    selected = {(r["dataset"], r["seed"]): r for r in subset_cert["runs"]}
    reports = {(r["dataset"], r["seed"]): r for r in report_cert["runs"]}
    CARDS.mkdir(parents=True, exist_ok=True)
    summary = []
    for seed in (0, 1, 2):
        key = ("fox8", seed)
        subset_info, report_info = selected[key], reports[key]
        matrix_path = SUBSETS / f"fox8_seed{seed}.npz"
        report_path = REPORTS / f"fox8_seed{seed}.jsonl.gz"
        if (sha256(matrix_path) != subset_info["subset_matrix_sha256"] or
                sha256(report_path) != report_info["reports_sha256"]):
            raise ValueError("upstream frozen score or report changed")
        with np.load(matrix_path, allow_pickle=False) as data:
            ids = data["ids"].astype(str).tolist()
            names = data["group_names"].astype(str).tolist()
            probabilities = data["subset_probability"].astype(float)
            threshold = float(data["threshold"])
        if len(names) != 4 or len(ids) != 343:
            raise ValueError("Fox8 group or account population changed")
        with gzip.open(report_path, "rt", encoding="utf-8") as stream:
            report_rows = [json.loads(line) for line in stream]
        if len(report_rows) != len(ids):
            raise ValueError("readable report count changed")
        report_by_id = {str(r["account_id"]): r for r in report_rows}
        if len(report_by_id) != len(ids) or set(report_by_id) != set(ids):
            raise ValueError("readable report account set changed")
        output = CARDS / f"fox8_seed{seed}.jsonl.gz"
        class_disagreements = 0
        max_replay = 0.0
        with output.open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as stream:
                    for i, account_id in enumerate(ids):
                        report = report_by_id[account_id]
                        p = probabilities[i]
                        score = logit(p)
                        base = float(score[0])
                        singleton = [float(score[1 << j] - base) for j in range(4)]
                        additive_logit = base + sum(singleton)
                        residual = float(score[-1] - additive_logit)
                        terms = interactions(score, names)
                        max_replay = max(max_replay, abs(base + sum(singleton) +
                                                 sum(t["logit_interaction"] for t in terms) - score[-1]))
                        additive_p = float(expit(additive_logit))
                        disagreement = (additive_p >= threshold) != (p[-1] >= threshold)
                        class_disagreements += bool(disagreement)
                        if (abs(float(report["probability"]) - p[-1]) > 1e-12 or
                                abs(float(report["all_evidence_zero_probability"]) - p[0]) > 1e-12 or
                                float(report["threshold"]) != threshold or
                                int(report["prediction"]) != int(p[-1] >= threshold)):
                            raise ValueError(f"published report mismatch: {account_id}")
                        card = {
                            "dataset": "fox8", "seed": seed, "account_id": account_id,
                            "model_variant": "raw_neighbor_74_lightgbm",
                            "source_report_sha256": report_info["reports_sha256"],
                            "subset_matrix_sha256": subset_info["subset_matrix_sha256"],
                            "group_names": names, "threshold": threshold,
                            "full_probability": float(p[-1]),
                            "full_class": "Bot" if p[-1] >= threshold else "Human",
                            "zero_input_probability": float(p[0]),
                            "zero_input_logit": base,
                            "single_group_logit_changes": dict(zip(names, singleton)),
                            "additive_approximation_probability": additive_p,
                            "interaction_logit": residual,
                            "additive_approximation_class_disagrees": bool(disagreement),
                            "higher_order_interactions": terms,
                            "all_16_group_subsets": [
                                {"mask": mask,
                                 "included_groups": [names[j] for j in range(4)
                                                     if mask & (1 << j)],
                                 "bot_probability": float(p[mask]),
                                 "predicted_class": "Bot" if p[mask] >= threshold else "Human"}
                                for mask in range(16)],
                            "interpretation": (
                                "These are exact frozen-score responses to zeroing encoded source groups. "
                                "The interaction term belongs to this group partition and zero-input baseline; "
                                "it does not identify a unique raw record or establish behavioral causality."),
                        }
                        stream.write(json.dumps(card, ensure_ascii=False, separators=(",", ":")) + "\n")
        if max_replay > 1e-12:
            raise ValueError("group interaction score reconstruction failed")
        summary.append({
            "seed": seed, "accounts": len(ids),
            "additive_approximation_class_disagreements": class_disagreements,
            "maximum_logit_reconstruction_error": max_replay,
            "source_report_sha256": report_info["reports_sha256"],
            "subset_matrix_sha256": subset_info["subset_matrix_sha256"],
            "cards_path": str(output), "cards_sha256": sha256(output),
        })
    result = {
        "status": "all_fox8_high_performance_decisions_have_exact_interaction_cards",
        "operation": "zero both encoded tails of omitted source groups and run the same frozen model",
        "fixed_test_labels_read": False,
        "runs": summary,
    }
    CERT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    for row in run()["runs"]:
        print(row["seed"], row["accounts"],
              row["additive_approximation_class_disagreements"],
              row["maximum_logit_reconstruction_error"])

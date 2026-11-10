"""Paired accuracy gates must reject a shifted test population or protocol."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_paired_accuracy import compare, summarize


def _run(path, *, predictions, seed=0, split_hash="same", graph_hash="same"):
    manifest = {
        "dataset": "fox8", "seed": seed, "raw_sha256": "raw",
        "split_sha256": split_hash, "feature_sha256": "features",
        "graph_sha256": graph_hash, "threshold": 0.5,
        "model_sha256": path.stem,
    }
    path.write_text(json.dumps({"manifest": manifest, "predictions": predictions}), encoding="utf-8")


class PairedAccuracyTests(unittest.TestCase):
    def test_paired_improvement_counts_exact_account_disagreements(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline, candidate = Path(directory) / "baseline.json", Path(directory) / "candidate.json"
            _run(baseline, predictions=[
                {"account_id": "a", "label": 1, "probability": 0.8, "split": "test"},
                {"account_id": "b", "label": 0, "probability": 0.9, "split": "test"},
                {"account_id": "c", "label": 0, "probability": 0.1, "split": "test"},
            ])
            _run(candidate, predictions=[
                {"account_id": "c", "label": 0, "probability": 0.1, "split": "test"},
                {"account_id": "b", "label": 0, "probability": 0.2, "split": "test"},
                {"account_id": "a", "label": 1, "probability": 0.8, "split": "test"},
            ])
            row = compare(baseline, candidate)
            self.assertEqual((row["baseline_acc"], row["candidate_acc"]), (2 / 3, 1.0))
            self.assertEqual(row["candidate_correct_baseline_wrong"], 1)
            self.assertEqual(row["baseline_correct_candidate_wrong"], 0)
            self.assertFalse(summarize([row], bootstraps=20)["three_seed_gate"])

    def test_rejects_nonpaired_comparisons(self):
        for change in ("split", "graph", "account", "label", "duplicate"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                baseline, candidate = Path(directory) / "baseline.json", Path(directory) / "candidate.json"
                rows = [
                    {"account_id": "a", "label": 1, "probability": 0.8, "split": "test"},
                    {"account_id": "b", "label": 0, "probability": 0.1, "split": "test"},
                ]
                other = [dict(row) for row in rows]
                if change == "account":
                    other[1]["account_id"] = "c"
                elif change == "label":
                    other[1]["label"] = 1
                elif change == "duplicate":
                    other[1]["account_id"] = "a"
                _run(baseline, predictions=rows)
                _run(candidate, predictions=other,
                     split_hash="different" if change == "split" else "same",
                     graph_hash="different" if change == "graph" else "same")
                with self.assertRaises(ValueError):
                    compare(baseline, candidate)


if __name__ == "__main__":
    unittest.main()

"""Check source-to-both-branches reachability and arithmetic in evidence DAGs."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from source.verifiable_bot.aaai_evidence_dag import (
    build_evidence_dag, validate_evidence_dag,
)

ROOT = Path(__file__).resolve().parents[2]
CERTIFICATE = ROOT / "research/certificates/aaai_veribot_fox8_profile_class_flip.json"
NEIGHBOR_CERTIFICATE = ROOT / "research/certificates/aaai_veribot_fox8_neighbor_context.json"


class EvidenceDagTests(unittest.TestCase):
    def setUp(self):
        self.report = json.loads(CERTIFICATE.read_text(encoding="utf-8"))

    def test_raw_source_reaches_both_frozen_branch_outputs(self):
        graph = build_evidence_dag(self.report)
        self.assertEqual(graph["account_id"], "3328298343")
        self.assertEqual(graph["nodes"]["fusion:before"]["kind"], "mean_logit_fusion")
        self.assertEqual(graph["nodes"]["fusion:after"]["kind"], "mean_logit_fusion")

    def test_invalid_fusion_and_cycle_are_rejected(self):
        original = build_evidence_dag(self.report)
        wrong_fusion = deepcopy(original)
        wrong_fusion["nodes"]["fusion:after"]["value"] += 0.5
        with self.assertRaisesRegex(ValueError, "fusion"):
            validate_evidence_dag(wrong_fusion)
        cycle = deepcopy(original)
        cycle["edges"].append({"from": "comparison", "to": "claim",
                               "relation": "invalid_back_edge"})
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_evidence_dag(cycle)

    def test_neighbor_source_and_raw_bridge_both_reach_model_input(self):
        report = json.loads(NEIGHBOR_CERTIFICATE.read_text(encoding="utf-8"))
        graph = build_evidence_dag(report)
        candidate = report["certificate"]["claim"]["source_ids"][0]
        self.assertEqual(graph["nodes"][f"bridge:{candidate}"]["kind"],
                         "graph_bridge_source")
        self.assertEqual(graph["nodes"][f"named_input:{candidate}:0"]["input_path"],
                         "user[25073877].statuses_count")
        self.assertTrue(any(edge["from"] == f"bridge:{candidate}" and
                            edge["to"] == "input:after"
                            for edge in graph["edges"]))

    def test_dev_threshold_is_part_of_decision_path(self):
        report = deepcopy(self.report)
        report["decision_policy"] = {"logit_threshold": 0.48280709981918335,
                                     "sha256": "test-policy"}
        graph = build_evidence_dag(report)
        self.assertEqual(graph["nodes"]["decision:before"]["label"], "Human")
        self.assertEqual(graph["nodes"]["decision:after"]["label"], "Human")
        changed = deepcopy(graph)
        changed["nodes"]["decision:after"]["label"] = "Bot"
        with self.assertRaisesRegex(ValueError, "threshold decision"):
            validate_evidence_dag(changed)


if __name__ == "__main__":
    unittest.main()

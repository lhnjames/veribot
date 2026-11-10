"""Evidence checks must reject true-looking claims without source/model links."""

import unittest

from source.verifiable_bot.evidence_v2 import (
    AtomicClaim, BuiltInputs, RebuildingAdapter, SourceRecord,
    audit_candidate, digest,
)


class EvidenceV2Tests(unittest.TestCase):
    def setUp(self):
        self.sources = {
            "post:a": SourceRecord("post:a", "u", "post", {"score": 2}, "raw.json#post:a"),
            "post:b": SourceRecord("post:b", "u", "post", {"score": 3}, "raw.json#post:b"),
            "post:c": SourceRecord("post:c", "u", "post", {"score": -4}, "raw.json#post:c"),
        }

        def build(records):
            # Full rebuild: account count, aggregate score, and similarity edges
            # are derived together from the current raw-source collection.
            account = [r for r in records.values() if r.account_id == "u"]
            values = {"count": len(account), "sum": sum(r.payload["score"] for r in account),
                      "similarity_edges": sorted((a.source_id, b.source_id)
                                                 for a in account for b in account
                                                 if a.source_id != b.source_id)}
            return BuiltInputs(values,
                               {sid: ("features.count", "features.sum", "graph.similarity_edges")
                                for sid in records}, digest(values))

        def predict(_account_id, value):
            return [0.0, float(value["sum"])]

        self.adapter = RebuildingAdapter(
            model_name="toy", model_sha256="weights", preprocessor_sha256="preprocess",
            sources=self.sources, build_inputs=build, predict=predict)

    def test_recomputes_claim_and_rebuilds_feature_and_graph(self):
        report = audit_candidate("u", AtomicClaim(("post:a", "post:b"), "sum", 5, "score"),
                                 self.sources, self.adapter, ("post:c",))
        self.assertEqual(report["status"], "checked")
        self.assertEqual(report["recomputed_value"], 5)
        self.assertEqual(report["delta_margin"], 5.0)
        self.assertEqual(report["control_probes"][0]["delta_margin"], -4.0)
        self.assertNotEqual(report["input_sha256_before"], report["input_sha256_after"])
        types = {edge["type"] for edge in report["evidence_graph"]["edges"]}
        self.assertEqual(types, {"compute", "ground", "map-to-input", "probe"})

    def test_rejects_incorrect_value_and_missing_source(self):
        wrong = audit_candidate("u", AtomicClaim(("post:a",), "field", 42, "score"),
                                self.sources, self.adapter)
        missing = audit_candidate("u", AtomicClaim(("post:missing",), "count", 1),
                                  self.sources, self.adapter)
        self.assertEqual(wrong["status"], "rejected")
        self.assertEqual(missing["status"], "rejected")

    def test_preserves_negative_response(self):
        report = audit_candidate("u", AtomicClaim(("post:c",), "field", -4, "score"),
                                 self.sources, self.adapter)
        self.assertEqual(report["status"], "checked")
        self.assertEqual(report["delta_margin"], -4.0)


if __name__ == "__main__":
    unittest.main()

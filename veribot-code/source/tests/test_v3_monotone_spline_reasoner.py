import unittest

import numpy as np

from source.verifiable_bot.v3_evidence_reasoner import EvidenceBatch, LabelBlindReference
from source.verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner
from source.tests.test_v3_evidence_reasoner import take


class MonotoneSplineReasonerTests(unittest.TestCase):
    def test_nonnegative_interval_slopes_preserve_fixed_direction(self):
        model = MonotoneSplineReasoner({"behavior": ["activity"]})
        model.names = ("activity",)
        model.reference_version = "fixed"
        model.tail = ("upper",)
        model.knots = {"activity": (0.3, 0.8)}
        model.coef = np.array([0.2, 1.0, 0.1])
        model.intercept = 0.0
        score = np.linspace(0.0, 5.0, 30)[:, None]
        batch = EvidenceBatch(score, np.zeros_like(score), np.ones_like(score),
                              np.ones_like(score, dtype=bool), "fixed", ("activity",))
        model.sign = np.array([1.0])
        self.assertTrue(np.all(np.diff(model.contributions(batch)["behavior"]) >= -1e-12))
        model.sign = np.array([-1.0])
        self.assertTrue(np.all(np.diff(model.contributions(batch)["behavior"]) <= 1e-12))

    def test_monotonic_contributions_and_round_trip(self):
        raw = np.arange(16, dtype=float).reshape(-1, 1)
        labels = (raw[:, 0] >= 8).astype(int)
        reference = LabelBlindReference(min_context_n=2).fit(raw, ["activity"])
        evidence = reference.transform(raw)
        model = MonotoneSplineReasoner({"behavior": ["activity"]})
        model.fit(take(evidence, [0, 2, 4, 6, 8, 10, 12, 14]),
                  labels[[0, 2, 4, 6, 8, 10, 12, 14]],
                  take(evidence, [1, 3, 5, 7, 9, 11, 13, 15]),
                  labels[[1, 3, 5, 7, 9, 11, 13, 15]])
        self.assertTrue(np.all(model.coef >= 0))
        probability = model.predict_proba(evidence)
        self.assertLess(probability[0], probability[-1])
        np.testing.assert_allclose(model.logits(evidence),
                                   model.intercept + model.contributions(evidence)["behavior"])
        restored = MonotoneSplineReasoner.from_manifest(model.manifest())
        np.testing.assert_allclose(restored.predict_proba(evidence), probability)


if __name__ == "__main__":
    unittest.main()

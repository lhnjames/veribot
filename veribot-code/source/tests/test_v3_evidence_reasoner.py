import unittest
import tempfile
from pathlib import Path

import numpy as np

from source.verifiable_bot.v3_evidence_reasoner import (
    EvidenceBatch, LabelBlindReference, SignConstrainedReasoner,
)


def take(batch: EvidenceBatch, index) -> EvidenceBatch:
    return EvidenceBatch(batch.upper[index], batch.lower[index],
                         batch.reliability[index], batch.available[index],
                         batch.reference_version, batch.names)


class EvidenceReasonerTests(unittest.TestCase):
    def test_reference_is_train_only_and_missing_values_have_zero_reliability(self):
        train = np.array([[0.0], [1.0], [2.0], [np.nan]])
        reference = LabelBlindReference(min_context_n=2).fit(train, ["activity"])
        first_version = reference.version
        evidence = reference.transform(np.array([[100.0], [np.nan]]))
        self.assertEqual(reference.version, first_version)
        self.assertFalse(evidence.available[1, 0])
        self.assertEqual(evidence.reliability[1, 0], 0.0)
        self.assertEqual(evidence.upper[1, 0], 0.0)
        self.assertGreater(evidence.upper[0, 0], 0.0)
        # Test observations do not change the fitted reference/version.
        reference.transform(np.array([[-1000.0]]))
        self.assertEqual(reference.version, first_version)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json.gz"
            reference.save(path)
            restored = LabelBlindReference.load(path)
            self.assertEqual(restored.version, reference.version)
            np.testing.assert_allclose(restored.transform(np.array([[100.0]])).upper,
                                       evidence.upper[:1])

    def test_additive_predictions_equal_reported_group_contributions(self):
        x = np.arange(12, dtype=float).reshape(-1, 1)
        y = (x[:, 0] >= 6).astype(int)
        reference = LabelBlindReference(min_context_n=2).fit(x, ["activity"])
        batch = reference.transform(x)
        model = SignConstrainedReasoner({"behavior": ["activity"]}, l2=0.01)
        model.fit(take(batch, [0, 2, 6, 8]), y[[0, 2, 6, 8]],
                  take(batch, [1, 3, 7, 9]), y[[1, 3, 7, 9]])
        self.assertTrue(np.all(model.coef >= 0))
        self.assertEqual(model.manifest()["directions"]["activity"]["sign"], 1)
        pred = model.predict_proba(batch)
        self.assertLess(pred[0], pred[-1])
        group = model.contributions(batch)
        np.testing.assert_allclose(model.logits(batch), model.intercept + group["behavior"])
        loaded = SignConstrainedReasoner.from_manifest(model.manifest())
        np.testing.assert_allclose(loaded.predict_proba(batch), pred)

    def test_missing_concept_cannot_make_direct_contribution(self):
        x = np.array([[0., 10.], [1., 9.], [2., 8.], [3., 7.],
                      [7., 3.], [8., 2.], [9., 1.], [10., 0.]])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        reference = LabelBlindReference(min_context_n=2).fit(x, ["a", "b"])
        batch = reference.transform(x)
        model = SignConstrainedReasoner({"first": ["a"], "second": ["b"]})
        model.fit(take(batch, [0, 2, 4, 6]), y[[0, 2, 4, 6]],
                  take(batch, [1, 3, 5, 7]), y[[1, 3, 5, 7]])
        altered = reference.transform(np.array([[10.0, np.nan]]))
        self.assertEqual(float(model.contributions(altered)["second"][0]), 0.0)
        with self.assertRaises(ValueError):
            model.predict_proba(EvidenceBatch(batch.upper, batch.lower,
                                              batch.reliability, batch.available,
                                              "different-reference", batch.names))


if __name__ == "__main__":
    unittest.main()

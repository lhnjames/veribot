import unittest

import numpy as np

from source.verifiable_bot.pairwise_evidence_reasoner import PairwiseEvidenceReasoner
from source.verifiable_bot.v3_evidence_reasoner import EvidenceBatch


class PairwiseEvidenceReasonerTests(unittest.TestCase):
    def test_pair_is_same_direction_monotone_and_contributions_exact(self):
        names = ("a", "b")
        batch = EvidenceBatch(
            upper=np.array([[1.0, 2.0], [2.0, 2.0]]),
            lower=np.zeros((2, 2)), reliability=np.ones((2, 2)),
            available=np.ones((2, 2), dtype=bool),
            reference_version="v", names=names)
        model = PairwiseEvidenceReasoner({"g": names}, [("a", "b")])
        model.names = names
        model.reference_version = "v"
        model.tail = ("upper", "upper")
        model.sign = np.array([1.0, 1.0])
        model._bind_pairs(names)
        model.coef = np.array([0.1, 0.2, 0.3])
        model.intercept = -0.5
        design = model._design(batch)
        self.assertGreater(design[1, 2], design[0, 2])
        np.testing.assert_allclose(model.logits(batch),
                                   model.intercept + design @ model.coef)
        pair = model.pair_contributions(batch)
        np.testing.assert_allclose(pair["pair_00"], design[:, 2] * 0.3)
        restored = PairwiseEvidenceReasoner.from_manifest(model.manifest())
        np.testing.assert_allclose(restored.logits(batch), model.logits(batch))
        model.sign = np.array([1.0, -1.0])
        np.testing.assert_array_equal(model._design(batch)[:, 2], 0)


if __name__ == "__main__":
    unittest.main()

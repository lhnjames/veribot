import unittest

import numpy as np

from source.verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner
from source.verifiable_bot.v3_evidence_reasoner import EvidenceBatch, LabelBlindReference
from source.verifiable_bot.mgtab_neighbor_facts import evidence_with_neighbor_support


def take(batch: EvidenceBatch, index: list[int]) -> EvidenceBatch:
    return EvidenceBatch(batch.upper[index], batch.lower[index],
                         batch.reliability[index], batch.available[index],
                         batch.reference_version, batch.names)


class DualTailReasonerTests(unittest.TestCase):
    def test_neighbor_support_scales_only_neighbor_evidence(self):
        values = np.tile(np.arange(3, dtype=float).reshape(-1, 1), (1, 86))
        names = [f"fact_{j}" for j in range(86)]
        reference = LabelBlindReference(min_context_n=2).fit(values, names)
        support = np.ones((3, 24), dtype=np.int32)
        evidence = evidence_with_neighbor_support(reference, values, support)
        plain = reference.transform(values)
        np.testing.assert_array_equal(evidence.reliability[:, :62],
                                      plain.reliability[:, :62])
        np.testing.assert_allclose(evidence.reliability[:, 62:],
                                   plain.reliability[:, 62:] / 6)

    def test_both_extremes_can_support_bot_and_manifest_replays(self):
        values = np.arange(15, dtype=float).reshape(-1, 1)
        labels = np.asarray([1] * 4 + [0] * 7 + [1] * 4)
        reference = LabelBlindReference(min_context_n=2).fit(values, ["activity"])
        evidence = reference.transform(values)
        model = DualTailEvidenceReasoner({"activity": ["activity"]}, l2=1e-4)
        model.fit(take(evidence, [0, 1, 4, 5, 8, 9, 13, 14]),
                  labels[[0, 1, 4, 5, 8, 9, 13, 14]],
                  take(evidence, [2, 3, 6, 7, 10, 11, 12]),
                  labels[[2, 3, 6, 7, 10, 11, 12]])
        self.assertEqual(model.manifest()["directions"]["activity"],
                         {"upper": 1, "lower": 1})
        self.assertTrue(np.all(model.coef >= 0))
        probability = model.predict_proba(evidence)
        self.assertGreater(probability[0], probability[7])
        self.assertGreater(probability[-1], probability[7])
        contribution = model.contributions(evidence)["activity"]
        np.testing.assert_allclose(model.logits(evidence), model.intercept + contribution)
        replay = DualTailEvidenceReasoner.from_manifest(model.manifest())
        np.testing.assert_allclose(replay.predict_proba(evidence), probability)
        missing = reference.transform(np.array([[np.nan]]))
        self.assertEqual(float(model.contributions(missing)["activity"][0]), 0.0)
        self.assertAlmostEqual(float(model.logits(missing)[0]), model.intercept)


if __name__ == "__main__":
    unittest.main()

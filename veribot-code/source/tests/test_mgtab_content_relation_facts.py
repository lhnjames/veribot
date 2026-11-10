import unittest

import numpy as np

from source.verifiable_bot.mgtab_content_relation_facts import extract


class MGTABContentRelationFactsTests(unittest.TestCase):
    def test_unique_pairs_and_zero_vectors(self):
        features = np.zeros((3, 788), dtype=np.float32)
        features[0, 20] = 1.0
        features[1, 20] = 1.0
        triples = [(0, 1, 0), (0, 1, 0), (0, 2, 0), (1, 0, 0)]
        triples += [(0, 1, kind) for kind in range(1, 7)]
        edge = np.asarray([(a, b) for a, b, _ in triples], dtype=np.int64).T
        kind = np.asarray([k for _, _, k in triples], dtype=np.int64)
        values, supports, names, summary = extract(features, edge, kind,
                                                    chunk_size=1)
        self.assertEqual(values.shape, (3, 6))
        self.assertEqual(supports[0, 0], 1)
        self.assertAlmostEqual(float(values[0, 0]), 1.0)
        self.assertEqual(supports[2, 0], 0)
        self.assertTrue(np.isnan(values[2, 0]))
        self.assertEqual(summary["followers"]["unique_nonself_directed_pairs"], 3)
        self.assertEqual(summary["followers"]["pairs_with_two_nonzero_vectors"], 2)
        self.assertEqual(names[0],
                         "published_followers_out_neighbor_tweet_embedding_cosine_mean")


if __name__ == "__main__":
    unittest.main()

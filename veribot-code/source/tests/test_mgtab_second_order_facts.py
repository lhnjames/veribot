import unittest

import numpy as np

from source.verifiable_bot.mgtab_second_order_facts import extract


class MGTABSecondOrderFactsTests(unittest.TestCase):
    def test_unique_neighbor_mean_uses_target_relation_row_degree(self):
        triples = [(0, 1, 0), (0, 1, 0), (1, 0, 0),
                   (1, 2, 3), (1, 2, 3), (2, 1, 3)]
        triples += [(0, 2, kind) for kind in (1, 2, 4, 5, 6)]
        edge = np.asarray([(a, b) for a, b, _ in triples], dtype=np.int64).T
        kind = np.asarray([kind for _, _, kind in triples], dtype=np.int64)
        values, supports, names, _ = extract(edge, kind, 3)
        self.assertEqual(values.shape, (3, 24))
        self.assertEqual(supports[0, 0], 1)
        self.assertAlmostEqual(float(values[0, 0]), float(np.log1p(1)))
        self.assertEqual(supports[2, 0], 0)
        self.assertTrue(np.isnan(values[2, 0]))
        self.assertEqual(names[0],
                         "published_followers_out_neighbor_followers_out_degree_mean")
        self.assertEqual(supports[1, 16], 1)
        self.assertAlmostEqual(float(values[1, 16]), float(np.log1p(0)))


if __name__ == "__main__":
    unittest.main()

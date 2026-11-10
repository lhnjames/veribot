import unittest

import numpy as np

from source.verifiable_bot.mgtab_source_certificate import relation_source


class MGTABSourceCertificateTests(unittest.TestCase):
    def test_relation_direction_counts_published_rows(self):
        edge = np.array([[0, 0, 1, 2], [1, 1, 0, 0]])
        kinds = np.array([3, 3, 3, 2])
        outgoing = relation_source(edge, kinds, 0, 3, "out")
        incoming = relation_source(edge, kinds, 0, 3, "in")
        self.assertEqual(outgoing["count"], 2)
        self.assertEqual(incoming["count"], 1)
        self.assertAlmostEqual(outgoing["value"], np.log1p(2), places=6)
        self.assertEqual(outgoing["edge_row_indices_first_10"], [0, 1])


if __name__ == "__main__":
    unittest.main()

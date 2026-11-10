import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from source.verifiable_bot.mgtab_neighbor_facts import extract, build
from source.verifiable_bot.mgtab_neighbor_certificate import certificate


class MGTABNeighborFactsTests(unittest.TestCase):
    def test_unique_nonself_neighbors_missing_and_direction(self):
        features = np.zeros((3, 20), dtype=np.float32)
        features[:, 2] = [1, 0, 1]  # verified
        features[:, 14] = [0.2, 0.8, 0.6]  # normalized ratio
        pairs = [(0, 1), (0, 1), (0, 2), (1, 0), (0, 0)]
        types = [0] * len(pairs)
        for relation in range(1, 7):
            pairs.append((2, 2))
            types.append(relation)
        values, supports, names, summary = extract(
            features, np.asarray(pairs, dtype=np.int64).T,
            np.asarray(types, dtype=np.int64))
        self.assertEqual(values.shape, (3, 24))
        self.assertEqual(names[1],
                         "published_followers_out_neighbor_verified_mean")
        self.assertEqual(supports[0, 1], 2)
        self.assertAlmostEqual(values[0, 1], 0.5)
        self.assertEqual(supports[0, 5], 1)
        self.assertAlmostEqual(values[0, 5], 0.0)
        self.assertTrue(np.isnan(values[2, 1]))
        self.assertAlmostEqual(values[2, 5], 1.0)
        self.assertEqual(summary["followers"]["unique_nonself_directed_pairs"], 3)

    def test_certificate_checks_neighbor_mean_and_missingness(self):
        features = np.zeros((3, 20), dtype=np.float32)
        features[:, 2] = [1, 0, 1]
        pairs = [(0, 1), (0, 1), (0, 2), (1, 0)]
        types = [0] * len(pairs)
        for relation in range(1, 7):
            pairs.append((2, 2))
            types.append(relation)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.save(torch.tensor(features), root / "features.pt")
            torch.save(torch.tensor(pairs).T, root / "edge_index.pt")
            torch.save(torch.tensor(types), root / "edge_type.pt")
            facts = root / "neighbor_facts.npz"
            build(root, facts)
            name = "published_followers_out_neighbor_verified_mean"
            result = certificate(root, facts, 0, name)
            self.assertEqual(result["unique_nonself_neighbors"], 2)
            self.assertAlmostEqual(result["value"], 0.5)
            missing = certificate(root, facts, 2, name)
            self.assertEqual(missing["unique_nonself_neighbors"], 0)
            self.assertIsNone(missing["value"])


if __name__ == "__main__":
    unittest.main()

import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from source.verifiable_bot.mgtab_structural_facts import extract
from source.verifiable_bot.mgtab_structural_facts import build
from source.verifiable_bot.mgtab_structural_certificate import certificate


class MGTABStructuralFactsTests(unittest.TestCase):
    def test_nonself_unique_reciprocal_and_self_rows_are_distinct(self):
        # Type 0 has two copies of 0->1, one reciprocal edge and one self row.
        pairs = [(0, 1), (0, 1), (1, 0), (0, 0)]
        types = [0] * len(pairs)
        for relation in range(1, 7):
            pairs.append((2, 2))
            types.append(relation)
        edge = np.asarray(pairs, dtype=np.int64).T
        values, names, summary = extract(edge, np.asarray(types), 3)
        count = np.expm1(values)
        self.assertEqual(values.shape, (3, 28))
        self.assertEqual(names[0], "published_followers_nonself_out_unique_log1p")
        np.testing.assert_allclose(count[0, :4], [1, 1, 1, 1], atol=1e-6)
        np.testing.assert_allclose(count[1, :4], [1, 1, 1, 0], atol=1e-6)
        self.assertEqual(summary["followers"]["edge_rows"], 4)
        self.assertEqual(summary["followers"]["unique_nonself_directed_pairs"], 2)
        self.assertEqual(summary["followers"]["unique_reciprocal_directed_pairs"], 2)

    def test_certificate_requeries_published_edges(self):
        pairs = [(0, 1), (0, 1), (1, 0), (0, 0)]
        types = [0] * len(pairs)
        for relation in range(1, 7):
            pairs.append((2, 2))
            types.append(relation)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.save(torch.zeros((3, 20)), root / "features.pt")
            torch.save(torch.tensor(pairs, dtype=torch.long).T,
                       root / "edge_index.pt")
            torch.save(torch.tensor(types, dtype=torch.long), root / "edge_type.pt")
            cache = root / "structural_facts.npz"
            build(root, cache)
            for suffix, count in (("nonself_out_unique_log1p", 1),
                                  ("nonself_in_unique_log1p", 1),
                                  ("reciprocal_unique_log1p", 1),
                                  ("self_edge_rows_log1p", 1)):
                result = certificate(root, cache, 0,
                                     f"published_followers_{suffix}")
                self.assertEqual(result["raw_count"], count)
                self.assertEqual(result["status"],
                                 "verified_against_published_typed_edges")


if __name__ == "__main__":
    unittest.main()

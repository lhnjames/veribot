import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from source.scripts.gate0_shallow_tree import relation_degrees
from source.verifiable_bot.mgtab_full_source_audit import audit, recompute
from source.verifiable_bot.mgtab_neighbor_facts import build as build_neighbor, extract as extract_neighbor
from source.verifiable_bot.mgtab_structural_facts import build as build_structure, extract as extract_structure


class MGTABFullSourceAuditTests(unittest.TestCase):
    def test_recomputes_all_relation_facts_with_duplicate_and_self_edges(self):
        features = np.zeros((3, 20), dtype=np.float32)
        features[:, [1, 2, 13, 14]] = np.asarray(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8],
             [0.9, 1.0, 0.0, 0.1]], dtype=np.float32)
        triples = [(0, 1, 0), (0, 1, 0), (1, 0, 0), (0, 0, 0)]
        triples += [(1, 2, kind) for kind in range(1, 7)]
        edge = np.asarray([(src, dst) for src, dst, _ in triples], dtype=np.int64).T
        kind = np.asarray([relation for _, _, relation in triples], dtype=np.int64)

        base, structure, neighbor, supports, _ = recompute(features, edge, kind)
        production_base, _ = relation_degrees(edge, kind, len(features))
        production_structure, _, _ = extract_structure(edge, kind, len(features))
        production_neighbor, production_supports, _, _ = extract_neighbor(
            features, edge, kind)
        np.testing.assert_array_equal(base, production_base)
        np.testing.assert_array_equal(structure, production_structure)
        np.testing.assert_allclose(neighbor, production_neighbor, atol=1e-7,
                                   rtol=0, equal_nan=True)
        np.testing.assert_array_equal(supports, production_supports)
        self.assertEqual(np.expm1(structure[0, 0]), 1)
        self.assertEqual(np.expm1(structure[0, 2]), 1)
        self.assertEqual(supports[0, 0], 1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "raw"
            data.mkdir()
            torch.save(torch.from_numpy(features), data / "features.pt")
            torch.save(torch.from_numpy(edge), data / "edge_index.pt")
            torch.save(torch.from_numpy(kind), data / "edge_type.pt")
            structural_path = root / "structural_facts.npz"
            neighbor_path = root / "neighbor_facts.npz"
            build_structure(data, structural_path)
            build_neighbor(data, neighbor_path)
            result = audit(data, structural_path, neighbor_path)
            self.assertEqual(result["account_count"], 3)
            self.assertEqual(result["total_checked_cells_excluding_support"], 198)
            self.assertEqual(result["label_fields_read"], False)


if __name__ == "__main__":
    unittest.main()

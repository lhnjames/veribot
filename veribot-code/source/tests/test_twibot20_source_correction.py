import unittest

import numpy as np

from source.scripts.stress_twibot20_source_correction import select_indices


class SourceCorrectionTests(unittest.TestCase):
    def test_selection_is_deterministic_and_ignores_missing_values(self):
        ids = np.array([f"account-{i}" for i in range(20)])
        values = np.array([i % 2 for i in range(20)], dtype=float)
        values[0] = np.nan
        first = select_indices(ids, values)
        second = select_indices(ids.copy(), values.copy())
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), round(0.2 * 19))
        self.assertNotIn(0, first)


if __name__ == "__main__":
    unittest.main()

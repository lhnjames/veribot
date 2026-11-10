import unittest

import numpy as np

from source.verifiable_bot.twibot20_full_source_audit import direct_facts, declared_edges


class TwiBot20FullSourceAuditTests(unittest.TestCase):
    def test_direct_facts_and_declared_edge_direction(self):
        row = {
            "ID": "u", "profile": {"followers_count": "12", "verified": "true",
                                   "description": "abc"},
            "tweet": ["RT @a hi https://x #tag", "Hello @b"],
            "neighbor": {"following": ["v"], "follower": ["w"]},
        }
        values = direct_facts(row)
        self.assertEqual(len(values), 19)
        self.assertEqual(values[0], 12)
        self.assertTrue(np.isnan(values[1]))
        self.assertEqual(values[5], 1)
        self.assertEqual(values[9], 2)
        self.assertEqual(values[10], 0.5)
        self.assertEqual(values[11], 0.5)
        self.assertEqual(values[16], 3)
        self.assertEqual(declared_edges(row), {("u", "v"), ("w", "u")})


if __name__ == "__main__":
    unittest.main()

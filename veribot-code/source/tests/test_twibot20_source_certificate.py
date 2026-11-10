import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from source.verifiable_bot.twibot20_facts import FACT_NAMES, account_facts
from source.verifiable_bot.twibot20_source_certificate import direct_query, graph_for_account, row_at


class TwiBot20SourceCertificateTests(unittest.TestCase):
    def test_native_direction_and_deduplication_from_zip_rows(self):
        rows = {
            "train": [{"ID": "a", "profile": {}, "tweet": [],
                       "neighbor": {"following": ["b", "b"], "follower": ["c"]}}],
            "dev": [{"ID": "b", "profile": {}, "tweet": [],
                     "neighbor": {"following": ["a"], "follower": ["a"]}}],
            "test": [{"ID": "c", "profile": {}, "tweet": [],
                      "neighbor": {"following": ["a"], "follower": []}}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny.zip"
            with zipfile.ZipFile(path, "w") as zipped:
                for part, content in rows.items():
                    zipped.writestr(f"Twibot-20/{part}.json", json.dumps(content))
            with zipfile.ZipFile(path) as zipped:
                self.assertEqual(row_at(zipped, "train", 0, "a")["ID"], "a")
                values, query = graph_for_account(zipped, "a")
                with self.assertRaises(ValueError):
                    row_at(zipped, "train", 0, "wrong")
        self.assertEqual(values, {"graph_out_degree": 1.0,
                                  "graph_in_degree": 2.0,
                                  "graph_reciprocal_degree": 1.0})
        self.assertEqual(query["source_declarations"], 6)

    def test_fact_names_match_cache_contract(self):
        values, _, _ = account_facts({"ID": "a", "profile": {}, "tweet": [],
                                      "neighbor": None})
        self.assertEqual(len(values), len(FACT_NAMES) - 3)
        self.assertTrue(np.isnan(values[FACT_NAMES.index("retweet_fraction")]))
        query = direct_query("test", 2, "retweet_fraction", {"tweet": ["RT @x"]})
        self.assertEqual(query["query"], "Twibot-20/test.json[2].tweet[*]")
        self.assertIn("RT @", query["operation"])


if __name__ == "__main__":
    unittest.main()

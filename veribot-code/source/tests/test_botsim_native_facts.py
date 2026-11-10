import unittest

import numpy as np

from source.verifiable_bot.botsim_native_facts import FACT_NAMES, extract


class BotSimNativeFactsTests(unittest.TestCase):
    def test_parent_resolution_direction_and_label_independence(self):
        users = [{"user_id": "a", "name": "Alice", "label": 1},
                 {"user_id": "b", "name": "Bob", "label": 0}]
        activity = {
            "a": {"posts": [{"submission_id": "p1", "author_name": "Alice"}],
                  "comment_1": [{"comment_id": "c1", "comment_author_name": "Alice",
                                 "parent_id": "t3_p2"}], "comment_2": []},
            "b": {"posts": [{"submission_id": "p2", "author_name": "Bob"}],
                  "comment_1": [{"comment_id": "c2", "comment_author_name": "Bob",
                                 "parent_id": "t3_p1"}], "comment_2": []},
        }
        ids, values, trace, counts = extract(users, activity)
        users[0]["label"] = 0
        activity["a"]["character_setting"] = "bot"
        _, changed, _, _ = extract(users, activity)
        np.testing.assert_array_equal(values, changed)
        self.assertEqual(ids, ["a", "b"])
        self.assertEqual([row["source_account_id"] for row in trace], ["a", "b"])
        self.assertEqual([row["target_account_id"] for row in trace], ["b", "a"])
        self.assertEqual(counts["unique_directed_reply_edges"], 2)
        self.assertEqual(values.shape, (2, len(FACT_NAMES)))
        self.assertAlmostEqual(values[0, FACT_NAMES.index("native_reply_mutual_unique_log1p")], np.log(2))


if __name__ == "__main__":
    unittest.main()

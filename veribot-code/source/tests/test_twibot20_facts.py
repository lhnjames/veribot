import unittest

import numpy as np

from source.verifiable_bot.twibot20_facts import FACT_NAMES, account_facts


class TwiBot20FactsTests(unittest.TestCase):
    def test_fact_extraction_ignores_label_and_preserves_follow_direction(self):
        row = {
            "ID": "u1", "label": 0, "domain": "sports",
            "profile": {"followers_count": " 10 ", "friends_count": 2,
                        "statuses_count": 5, "listed_count": 0,
                        "favourites_count": 1, "verified": "False",
                        "protected": False, "default_profile": True,
                        "default_profile_image": False, "description": "abc"},
            "tweet": ["hello #topic", "RT @u2 https://example.org"],
            "neighbor": {"following": ["u2"], "follower": ["u3"]},
        }
        first, edges, context = account_facts(row)
        row["label"] = 1
        second, _, _ = account_facts(row)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(edges, [("u1", "u2"), ("u3", "u1")])
        self.assertEqual(context, "sports")
        self.assertEqual(len(first), len(FACT_NAMES) - 3)
        self.assertEqual(first[FACT_NAMES.index("retweet_fraction")], 0.5)

    def test_absent_tweets_and_neighbors_are_missing_not_zero(self):
        row = {"ID": "u1", "profile": {}, "tweet": [], "neighbor": None}
        values, edges, _ = account_facts(row)
        self.assertEqual(edges, [])
        self.assertTrue(np.isnan(values[FACT_NAMES.index("retweet_fraction")]))
        self.assertTrue(np.isnan(values[FACT_NAMES.index("sampled_following_count")]))


if __name__ == "__main__":
    unittest.main()

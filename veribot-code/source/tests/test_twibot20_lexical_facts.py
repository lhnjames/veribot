import unittest

import numpy as np

from source.verifiable_bot.twibot20_lexical_facts import FACT_NAMES, extract
from source.verifiable_bot.twibot20_lexical_audit import recompute


class TwiBot20LexicalFactsTests(unittest.TestCase):
    def test_template_and_target_concentration(self):
        tweets = [
            "RT @Alice Visit https://a.example/1 #News",
            "RT @Bob Visit https://b.example/2 #news",
            "Hello @Alice #Other",
        ]
        facts = dict(zip(FACT_NAMES, extract(tweets)))
        np.testing.assert_allclose(extract(tweets), recompute(tweets),
                                   rtol=0, atol=1e-12, equal_nan=True)
        self.assertAlmostEqual(facts["normalized_template_repeat_fraction"], 1 / 3)
        self.assertAlmostEqual(facts["top_retweet_target_fraction"], 1 / 2)
        self.assertAlmostEqual(facts["top_mentioned_account_fraction"], 2 / 3)
        self.assertAlmostEqual(facts["top_hashtag_fraction"], 2 / 3)
        self.assertAlmostEqual(facts["mean_hashtags_per_tweet"], 1.0)
        self.assertAlmostEqual(facts["mean_mentions_per_tweet"], 1.0)

    def test_empty_tweet_sample_is_missing(self):
        self.assertTrue(np.isnan(extract([])).all())


if __name__ == "__main__":
    unittest.main()

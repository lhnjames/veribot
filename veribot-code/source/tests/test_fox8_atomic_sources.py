import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from source.verifiable_bot.aaai_fox8_feature_schema import FEATURE_FIELDS
from source.verifiable_bot.aaai_fox8_raw_features import account_features, profile_features, tweet_features
from source.verifiable_bot.fox8_atomic_sources import (
    RelationAtom, graph_from_relation_atoms, load_account_atoms,
)


class Fox8AtomicSourceTests(unittest.TestCase):
    def test_relation_selection_rebuilds_both_edge_directions(self):
        relations = (
            RelationAtom("r1", "a", "b", "reply", ((1, 0, "in_reply_to_user_id"),)),
            RelationAtom("r2", "b", "a", "retweet", ((2, 3, "retweeted_status.user.id"),)),
        )
        graph = graph_from_relation_atoms(relations, ["a", "b"], ["r1"])
        self.assertEqual(graph["reply"].tolist(), [[0], [1]])
        self.assertEqual(graph["rev_reply"].tolist(), [[1], [0]])
        self.assertEqual(graph["retweet"].shape, (2, 0))
        with self.assertRaisesRegex(ValueError, "absent"):
            graph_from_relation_atoms(relations, ["a", "b"], ["missing"])

    def test_full_and_selected_views_recompute_from_original_sources(self):
        profile = {"id_str": "42", "followers_count": 8, "friends_count": 3,
                   "description": "example", "default_profile": True}
        base = {"user": profile, "text": "RT hello", "favorite_count": 2,
                "retweet_count": 1, "lang": "en", "in_reply_to_status_id": None,
                "created_at": "Mon Jan 01 00:00:00 +0000 2024",
                "entities": {"user_mentions": [], "hashtags": [{"text": "a"}], "urls": []}}
        second = {**base, "text": "different", "favorite_count": 4,
                  "created_at": "Mon Jan 01 01:00:00 +0000 2024",
                  "entities": {"user_mentions": [{"id": 9}], "hashtags": [], "urls": []}}
        third = {**base, "created_at": "Mon Jan 01 03:00:00 +0000 2024"}
        row = {"user_id": "42", "user_tweets": [base, second, third]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fox8.ndjson"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            atoms = load_account_atoms(path, ["42"])["42"]
        expected = np.asarray([account_features(row)[name] for name in FEATURE_FIELDS],
                              dtype=np.float32)
        np.testing.assert_array_equal(atoms.features(), expected)
        kept = atoms.features(profile_fields={"followers_count"}, tweet_ordinals={1, 2})
        profile_expected = profile_features({"id_str": "42", "followers_count": 8})
        tweet_expected = tweet_features([second, third])
        np.testing.assert_array_equal(kept, np.asarray(
            [profile_expected[name] for name in tuple(profile_expected)] +
            [tweet_expected[name] for name in tuple(tweet_expected)],
            dtype=np.float32))
        extra = atoms.extended_features()
        self.assertEqual(extra["observed_span_hours"], 3.0)
        self.assertAlmostEqual(extra["exact_text_duplicate_ratio"], 1 / 3)
        self.assertAlmostEqual(extra["mention_rate"], 1 / 3)
        self.assertTrue(atoms.tweets[0].record_sha256)
        self.assertEqual(atoms.source_id("tweet", 1), "fox8:42:tweet:1:1")


if __name__ == "__main__":
    unittest.main()

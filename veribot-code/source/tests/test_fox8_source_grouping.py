import unittest
from types import SimpleNamespace

from verifiable_bot.fox8_source_grouping import fine_grained_target_groups


class TinyAccount:
    profile_field_ids = ("followers_count", "name", "verified")
    tweets = (SimpleNamespace(ordinal=0, is_retweet=0, is_reply=1, is_quote=0),
              SimpleNamespace(ordinal=1, is_retweet=1, is_reply=0, is_quote=0),
              SimpleNamespace(ordinal=2, is_retweet=0, is_reply=0, is_quote=0))

    def source_id(self, kind, item):
        return f"{kind}:{item}"


class SourceGroupingTests(unittest.TestCase):
    def test_fine_groups_partition_selected_sources(self):
        account = TinyAccount()
        relations = (SimpleNamespace(source_id="r:0", relation="reply"),
                     SimpleNamespace(source_id="r:1", relation="mention"))
        result = fine_grained_target_groups(
            account=account,
            target_profile={"profile:followers_count", "profile:name",
                            "profile:verified"},
            target_tweets={"tweet:0", "tweet:1", "tweet:2"},
            incident_relations=relations, selected_relations={"r:0", "r:1"},
            neighbor_profile={"neighbor:profile"},
            neighbor_tweets={"neighbor:tweet"},
            hgt_remainder=frozenset(("context:source",)))
        self.assertEqual(result["target_profile_counts"],
                         frozenset(("profile:followers_count",)))
        self.assertEqual(result["target_profile_text"],
                         frozenset(("profile:name",)))
        self.assertEqual(result["target_profile_flags"],
                         frozenset(("profile:verified",)))
        self.assertEqual(result["target_tweets_reply"],
                         frozenset(("tweet:0",)))
        self.assertEqual(result["target_tweets_retweet"],
                         frozenset(("tweet:1",)))
        self.assertEqual(result["target_tweets_original"],
                         frozenset(("tweet:2",)))
        self.assertEqual(result["incident_relation_reply"],
                         frozenset(("r:0",)))
        self.assertEqual(len(result), 11)

    def test_unknown_selected_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "do not partition"):
            fine_grained_target_groups(
                account=TinyAccount(), target_profile={"profile:missing"},
                target_tweets=set(), incident_relations=(),
                selected_relations=set(), neighbor_profile=set(),
                neighbor_tweets=set(), hgt_remainder=frozenset())


if __name__ == "__main__":
    unittest.main()

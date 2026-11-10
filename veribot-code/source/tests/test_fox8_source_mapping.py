"""Profile-field edits must preserve unrelated raw fields and their features."""

import unittest

from source.verifiable_bot.evidence_v2 import SourceRecord
from source.verifiable_bot.fox8_adapter import FEATURE_FIELDS, raw_feature_matrix


class Fox8MappingTests(unittest.TestCase):
    def test_one_profile_field_is_one_source_unit(self):
        sources = {
            "followers": SourceRecord("followers", "u", "profile_field",
                                      {"followers_count": 100}, "raw#followers_count"),
            "friends": SourceRecord("friends", "u", "profile_field",
                                    {"friends_count": 25}, "raw#friends_count"),
        }
        original = raw_feature_matrix(sources, ["u"])[0]
        edited = raw_feature_matrix({"friends": sources["friends"]}, ["u"])[0]
        self.assertEqual(original[FEATURE_FIELDS.index("friends_count")], 25)
        self.assertEqual(edited[FEATURE_FIELDS.index("friends_count")], 25)
        self.assertEqual(edited[FEATURE_FIELDS.index("followers_count")], 0)
        self.assertLess(edited[FEATURE_FIELDS.index("ff_ratio")],
                        original[FEATURE_FIELDS.index("ff_ratio")])


if __name__ == "__main__":
    unittest.main()

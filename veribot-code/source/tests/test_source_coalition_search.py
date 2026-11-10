import unittest

from source.verifiable_bot.source_coalition_search import (
    check_forward_contexts, find_budgeted_flip_groups,
    find_minimal_flip_groups,
)


class SourceCoalitionSearchTests(unittest.TestCase):
    def test_budgeted_search_finds_joint_cause_without_prescribing_group_size(self):
        def score(removed):
            return 1.0 - 0.6 * len({"a", "b"} & removed)

        result = find_budgeted_flip_groups(["a", "b", "c", "d"], score, 0.0,
                                           max_evaluations=15)
        self.assertTrue(result["all_singles_checked"])
        self.assertIn(["a", "b"], [row["sources"] for row in result["solutions"]])
        self.assertTrue(next(row["locally_minimal_checked"] for row in
                             result["solutions"] if row["sources"] == ["a", "b"]))
        self.assertFalse(result["global_minimum_proven"])

    def test_bot_search_finds_all_minimum_interacting_groups(self):
        def score(removed):
            return -1.0 if {"a", "b"} <= removed or {"a", "c"} <= removed else 1.0

        result = find_minimal_flip_groups(["a", "b", "c", "d"], score, 0.0)
        self.assertTrue(result["baseline_bot"])
        self.assertEqual(result["minimum_cardinality"], 2)
        self.assertTrue(result["minimum_cardinality_proven"])
        self.assertTrue(result["all_minimum_groups_found"])
        self.assertEqual(
            {tuple(group["sources"]) for group in result["minimal_groups"]},
            {("a", "b"), ("a", "c")})
        forward = check_forward_contexts(["a", "b"], ["c", "d"], score, 0.0,
                                         target_bot=True)
        self.assertEqual(forward["class_change_count"], 2)

    def test_human_search_and_forward_check_reverse_inequalities(self):
        def score(removed):
            return 1.0 if {"a", "b"} <= removed else -1.0

        result = find_minimal_flip_groups(["a", "b", "c"], score, 0.0)
        self.assertFalse(result["baseline_bot"])
        self.assertEqual(result["minimum_cardinality"], 2)
        self.assertEqual([g["sources"] for g in result["minimal_groups"]],
                         [["a", "b"]])
        forward = check_forward_contexts(["a", "b"], ["c"], score, 0.0,
                                         target_bot=False)
        self.assertEqual(forward["class_change_count"], 1)
        self.assertGreater(forward["contexts"][0]["signed_addition_effect_toward_target"], 0)

    def test_budget_does_not_look_like_a_minimality_proof(self):
        result = find_minimal_flip_groups(
            ["a", "b", "c"], lambda removed: 1.0, 0.0,
            max_evaluations=2)
        self.assertTrue(result["budget_exhausted"])
        self.assertFalse(result["minimum_cardinality_proven"])
        self.assertFalse(result["all_minimum_groups_found"])


if __name__ == "__main__":
    unittest.main()

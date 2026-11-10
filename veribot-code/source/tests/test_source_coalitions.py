import unittest

from verifiable_bot.source_coalitions import (
    enumerate_source_coalitions, greedy_bot_certificate,
    search_source_coalitions,
)


class ToyOracle:
    def __init__(self):
        self.lookup = {name: ("source", i)
                       for i, name in enumerate(("a", "b", "c"))}

    def predict(self, account_id, removed=frozenset()):
        present = self.lookup.keys() - removed
        margin = 1.0 if (("a" in present and "b" in present) or
                         "c" in present) else -1.0
        return {"margin": margin}


class SourceCoalitionTests(unittest.TestCase):
    def test_greedy_bot_certificate_keeps_joint_cause(self):
        class JointOracle:
            lookup = {name: None for name in ("a", "b", "c")}

            def predict(self, account_id, removed=frozenset()):
                visible = self.lookup.keys() - removed
                margin = (0.8 if {"a", "b"} <= visible else 0.0)
                margin += 0.1 if "c" in visible else 0.0
                return {"margin": margin}

        report = greedy_bot_certificate(
            account_id="target",
            groups={name: frozenset((name,)) for name in ("a", "b", "c")},
            candidate_sources=frozenset(("a", "b", "c")),
            threshold=0.5, margin_buffer=0.05,
            oracle=JointOracle(), query_budget=20)
        self.assertEqual(report["retained_groups"], ["a", "b"])
        self.assertTrue(report["passes_bidirectional"])
        self.assertLessEqual(report["query_count"], 20)

    def test_joint_and_alternative_sufficient_sets(self):
        groups = {name: frozenset((name,)) for name in ("a", "b", "c")}
        report = enumerate_source_coalitions(
            account_id="target", groups=groups,
            full_dependency=frozenset(("a", "b", "c")),
            threshold=0.0, oracle=ToyOracle())
        self.assertEqual({tuple(x) for x in report["minimal_sufficient_group_sets"]},
                         {("a", "b"), ("c",)})
        self.assertFalse(report["empty_input_preserves_original_class"])
        ab = next(x for x in report["pair_interactions"]
                  if x["first"] == "a" and x["second"] == "b")
        self.assertEqual(ab["signed_pair_interaction"], 2.0)

    def test_budgeted_search_ranks_and_finds_joint_source(self):
        groups = {name: frozenset((name,))
                  for name in ("a", "b", "c")}
        report = search_source_coalitions(
            account_id="target", groups=groups,
            full_dependency=frozenset(groups), threshold=0.0,
            oracle=ToyOracle(), query_budget=9, beam_width=2)
        found = {tuple(row["groups"])
                 for row in report["observed_nondominated_sufficient_group_sets"]}
        self.assertIn(("a", "b"), found)
        self.assertIn(("c",), found)
        self.assertNotIn(("c",), {tuple(row["groups"]) for row in
                                   report["searched_bidirectional_group_sets"]})
        self.assertEqual(report["search_method"],
                         "budgeted_beam_with_singleton_and_full_context_ranking")
        self.assertLessEqual(report["oracle_query_count"], 9)

    def test_budgeted_search_does_not_explore_every_subset(self):
        names = tuple("abcdef")
        class SixOracle:
            lookup = {name: ("source", i) for i, name in enumerate(names)}
            def predict(self, account_id, removed=frozenset()):
                present = self.lookup.keys() - removed
                return {"margin": 1.0 if {"a", "b"} <= present else -1.0}
        report = search_source_coalitions(
            account_id="target",
            groups={name: frozenset((name,)) for name in names},
            full_dependency=frozenset(names), threshold=0.0,
            oracle=SixOracle(), query_budget=24, beam_width=2)
        self.assertLessEqual(report["oracle_query_count"], 24)
        self.assertLess(len(report["coalitions"]), 2 ** len(names))
        self.assertIn(("a", "b"),
                      {tuple(row["groups"]) for row in
                       report["searched_bidirectional_group_sets"]})

    def test_empty_input_prevents_positive_evidence_claim(self):
        class DefaultHumanOracle:
            lookup = {"a": None, "b": None, "c": None}
            def predict(self, account_id, removed=frozenset()):
                return {"margin": -1.0}
        groups = {name: frozenset((name,)) for name in ("a", "b", "c")}
        report = search_source_coalitions(
            account_id="target", groups=groups,
            full_dependency=frozenset(groups), threshold=0.0,
            oracle=DefaultHumanOracle(), query_budget=9)
        self.assertEqual(report["observed_nondominated_sufficient_group_sets"],
                         [{"groups": [], "source_count": 0,
                           "retained_margin": -1.0}])
        self.assertEqual(report["searched_bidirectional_group_sets"], [])
        self.assertFalse(report["empty_input_fails_original_class"])
        self.assertEqual(report["full_signed_gain_over_empty"], 0.0)

    def test_human_can_have_directional_support_without_label_flip(self):
        class HumanSupportOracle:
            lookup = {"a": None, "b": None, "c": None}
            def predict(self, account_id, removed=frozenset()):
                return {"margin": -2.0 if "a" not in removed else -1.0}
        groups = {name: frozenset((name,)) for name in ("a", "b", "c")}
        report = search_source_coalitions(
            account_id="target", groups=groups,
            full_dependency=frozenset(groups), threshold=0.0,
            oracle=HumanSupportOracle(), query_budget=9)
        self.assertFalse(report["empty_input_fails_original_class"])
        self.assertEqual(report["searched_bidirectional_group_sets"], [])
        self.assertEqual(report["full_signed_gain_over_empty"], 1.0)
        self.assertIn(("a",), {tuple(row["groups"]) for row in
                               report["directional_support_group_sets"]})

    def test_full_context_rank_rescues_weak_singletons(self):
        names = tuple("abcdef")
        class SynergyOracle:
            lookup = {name: None for name in names}
            def predict(self, account_id, removed=frozenset()):
                present = self.lookup.keys() - removed
                if {"a", "b"} <= present:
                    return {"margin": 1.0}
                for name, margin in (("c", -0.1), ("d", -0.2),
                                     ("e", -0.3)):
                    if name in present:
                        return {"margin": margin}
                return {"margin": -1.0}
        report = search_source_coalitions(
            account_id="target",
            groups={name: frozenset((name,)) for name in names},
            full_dependency=frozenset(names), threshold=0.0,
            oracle=SynergyOracle(), query_budget=24, beam_width=3)
        self.assertEqual(report["ranked_by_singleton_gain"][:3],
                         ["c", "d", "e"])
        self.assertEqual(report["ranked_by_full_context_removal_loss"][:2],
                         ["a", "b"])
        self.assertIn(("a", "b"),
                      {tuple(row["groups"]) for row in
                       report["searched_bidirectional_group_sets"]})


if __name__ == "__main__":
    unittest.main()

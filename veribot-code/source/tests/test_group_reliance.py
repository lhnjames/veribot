import unittest

import numpy as np
from scipy.special import expit

from source.verifiable_bot.group_reliance import evaluate_group_reliance


class GroupRelianceTests(unittest.TestCase):
    def test_removal_changes_only_decisions_that_depend_on_group(self):
        columns = {"profile": np.array([2.0, -2.0]),
                   "graph": np.array([-3.0, 0.0])}
        saved = expit(-0.1 + columns["profile"] + columns["graph"])
        result = evaluate_group_reliance(np.array([1, 0]), saved, columns, -0.1, 0.5)
        self.assertEqual(result["groups"]["graph"]["changed_decisions"], 1)
        self.assertEqual(result["groups"]["graph"]["corrected_errors"], 1)
        self.assertEqual(result["groups"]["profile"]["new_errors"], 0)

    def test_saved_probabilities_must_match_contributions(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            evaluate_group_reliance(np.array([0]), np.array([0.9]),
                                    {"graph": np.array([0.0])}, 0.0, 0.5)

    def test_follow_neighbor_profile_is_counted_as_graph_sourced(self):
        columns = {"profile": np.array([0.2]),
                   "native_follow_graph": np.array([0.3]),
                   "follow_neighbor_profile": np.array([0.4])}
        result = evaluate_group_reliance(
            np.array([1]), expit(0.1 + sum(columns.values())),
            columns, 0.1, 0.5)
        self.assertEqual(result["combined_graph_group_names"],
                         ["native_follow_graph", "follow_neighbor_profile"])

    def test_native_interaction_is_counted_as_graph_sourced(self):
        columns = {"profile": np.array([0.2]),
                   "native_interaction": np.array([0.4])}
        result = evaluate_group_reliance(
            np.array([1]), expit(0.1 + sum(columns.values())),
            columns, 0.1, 0.5)
        self.assertEqual(result["combined_graph_group_names"],
                         ["native_interaction"])


if __name__ == "__main__":
    unittest.main()

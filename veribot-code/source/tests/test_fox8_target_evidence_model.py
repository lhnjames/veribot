import unittest

import torch

from source.hgt_import.code.relnet_reimplementation import GatedRelNet
from source.verifiable_bot.fox8_atomic_sources import (
    AccountAtoms, RelationAtom, TweetAtom, graph_from_relation_atoms,
)
from source.verifiable_bot.fox8_target_evidence_model import (
    HGTTargetRelationSelector, TargetEvidenceFox8,
)
from source.verifiable_bot.fox8_two_stage_inputs import ALL_FACTS, Fox8TwoStageInputs


class ContextSelector(torch.nn.Module):
    def forward(self, context, batch):
        profile = context[batch.profile_account, :1].expand(-1, 2)
        tweet = context[batch.tweet_account, :1].expand(-1, 2)
        return {"profile": profile, "tweet": tweet}


class ContextRelationSelector(ContextSelector):
    def forward(self, context, batch):
        result = super().forward(context, batch)
        result["relation_forward"] = context[
            batch.relation_target, :1].expand(-1, 2)
        result["relation_reverse"] = context[
            batch.relation_source, :1].expand(-1, 2)
        return result


class TargetEvidenceModelTests(unittest.TestCase):
    def test_full_selection_matches_one_layer_relnet_and_neighbor_gate_is_local(self):
        def account(uid, line):
            tweet = TweetAtom(0, line, uid, 0, 0, 5, 1, 0, "en",
                              1704067200, 0, 0, 0, 0, "text", "record")
            return AccountAtoms(uid, line,
                                {"id_str": uid, "followers_count": line},
                                (tweet,))

        ids = ["a", "b"]
        relation = RelationAtom("r", "a", "b", "reply", ((1, 0, "reply"),))
        bank = Fox8TwoStageInputs.build(
            {uid: account(uid, i + 1) for i, uid in enumerate(ids)},
            (relation,), ids)
        relnet = GatedRelNet(len(ALL_FACTS), hidden_channels=12,
                             relation_names=tuple(bank.full_graph),
                             num_layers=1, dropout=0.0, out_channels=2)
        model = TargetEvidenceFox8(
            ContextSelector(), relnet, torch.zeros(len(ALL_FACTS)),
            torch.ones(len(ALL_FACTS))).eval()
        context = torch.tensor([[20.0], [20.0]])
        with torch.inference_mode():
            result = model(context, bank)["logits"]
            expected = relnet(torch.from_numpy(bank.full_facts).clamp(-3, 3),
                              bank.full_graph)
            changed = model(torch.tensor([[20.0], [-20.0]]), bank)["logits"]
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))
        self.assertTrue(torch.allclose(result[0], changed[0], atol=1e-6))

    def test_directional_relation_gate_blocks_unselected_neighbor(self):
        def account(uid, line, followers):
            tweet = TweetAtom(0, line, uid, 0, 0, 5, 1, 0, "en",
                              1704067200, 0, 0, 0, 0, "text", "record")
            return AccountAtoms(uid, line,
                                {"id_str": uid, "followers_count": followers},
                                (tweet,))

        ids = ["a", "b"]
        relation = RelationAtom("r", "a", "b", "reply", ((1, 0, "reply"),))
        first = Fox8TwoStageInputs.build(
            {"a": account("a", 1, 1), "b": account("b", 2, 2)},
            (relation,), ids)
        changed_source = Fox8TwoStageInputs.build(
            {"a": account("a", 1, 100), "b": account("b", 2, 2)},
            (relation,), ids)
        relnet = GatedRelNet(len(ALL_FACTS), hidden_channels=12,
                             relation_names=tuple(first.full_graph),
                             num_layers=1, dropout=0.0, out_channels=2)
        model = TargetEvidenceFox8(
            ContextRelationSelector(), relnet,
            torch.zeros(len(ALL_FACTS)), torch.ones(len(ALL_FACTS))).eval()
        with torch.inference_mode():
            full = model(torch.tensor([[20.0], [20.0]]), first)["logits"]
            expected = relnet(torch.from_numpy(first.full_facts).clamp(-3, 3),
                              first.full_graph)
            gated = model(torch.tensor([[20.0], [-20.0]]), first)["logits"]
            gated_changed = model(torch.tensor([[20.0], [-20.0]]),
                                  changed_source)["logits"]
        self.assertTrue(torch.allclose(full, expected, atol=1e-6))
        self.assertTrue(torch.allclose(gated[1], gated_changed[1], atol=1e-6))

    def test_centered_class_scores_vanish_without_sources(self):
        def account(uid, line):
            tweet = TweetAtom(0, line, uid, 0, 0, 5, 1, 0, "en",
                              1704067200, 0, 0, 0, 0, "text", "record")
            return AccountAtoms(uid, line,
                                {"id_str": uid, "followers_count": line},
                                (tweet,))

        ids = ["a", "b"]
        relation = RelationAtom("r", "a", "b", "reply", ((1, 0, "reply"),))
        bank = Fox8TwoStageInputs.build(
            {uid: account(uid, i + 1) for i, uid in enumerate(ids)},
            (relation,), ids)
        relnet = GatedRelNet(len(ALL_FACTS), hidden_channels=12,
                             relation_names=tuple(bank.full_graph),
                             num_layers=1, dropout=0.0, out_channels=2)
        model = TargetEvidenceFox8(
            ContextRelationSelector(), relnet,
            torch.zeros(len(ALL_FACTS)), torch.ones(len(ALL_FACTS)),
            centered_evidence=True).eval()
        availability = {
            "profile": torch.zeros(len(bank.profile_account), dtype=torch.bool),
            "tweet": torch.zeros(len(bank.tweet_account), dtype=torch.bool),
            "relation": torch.zeros(len(bank.relations), dtype=torch.bool),
        }
        empty_graph = graph_from_relation_atoms(bank.relations, ids, ())
        with torch.inference_mode():
            result = model(torch.tensor([[20.0], [20.0]]), bank)
            empty = model(torch.tensor([[20.0], [20.0]]), bank,
                          availability=availability,
                          graph_override=empty_graph)
        self.assertEqual(result["logits"].shape, (2, 2))
        self.assertTrue(torch.allclose(empty["logits"],
                                       torch.zeros_like(empty["logits"]),
                                       atol=1e-6))
        model.sample_training_gates = False
        model.train()
        first = model(torch.tensor([[20.0], [20.0]]), bank)
        second = model(torch.tensor([[20.0], [20.0]]), bank)
        for name in first["hard_selection"]:
            self.assertTrue(torch.equal(first["hard_selection"][name],
                                        second["hard_selection"][name]))
        bot_model = TargetEvidenceFox8(
            ContextRelationSelector(), relnet,
            torch.zeros(len(ALL_FACTS)), torch.ones(len(ALL_FACTS)),
            centered_evidence=True, bot_only_support=True).eval()
        with torch.inference_mode():
            bot_result = bot_model(torch.tensor([[20.0], [20.0]]), bank)
            bot_empty = bot_model(torch.tensor([[20.0], [20.0]]), bank,
                                  availability=availability,
                                  graph_override=empty_graph)
        self.assertTrue(torch.equal(bot_result["logits"][:, 0],
                                    torch.zeros(2)))
        self.assertTrue(torch.allclose(bot_empty["logits"],
                                       torch.zeros_like(bot_empty["logits"]),
                                       atol=1e-6))
        bot_probabilities = list(bot_result["gate_probabilities"].values())
        expected_sparsity = torch.stack([
            probabilities[:, 1].mean() for probabilities in bot_probabilities
        ]).mean()
        self.assertTrue(torch.allclose(
            bot_result["mean_source_selection_probability"],
            expected_sparsity))

    def test_centered_support_backpropagates_to_source_selector(self):
        torch.manual_seed(7)
        tweet = TweetAtom(0, 1, "a", 0, 1, 12, 1, 0, "en",
                          1704067200, 0, 0, 1, 0, "text", "record")
        account = AccountAtoms("a", 1,
                               {"id_str": "a", "followers_count": 12},
                               (tweet,))
        bank = Fox8TwoStageInputs.build({"a": account}, (), ["a"])
        selector = HGTTargetRelationSelector(context_dim=1, hidden=8)
        relnet = GatedRelNet(len(ALL_FACTS), hidden_channels=12,
                             relation_names=tuple(bank.full_graph),
                             num_layers=1, dropout=0.0, out_channels=2)
        model = TargetEvidenceFox8(
            selector, relnet, torch.zeros(len(ALL_FACTS)),
            torch.ones(len(ALL_FACTS)), centered_evidence=True).train()
        output = model(torch.tensor([[0.5]]), bank)
        output["logits"][0, 0].backward()
        gradients = [parameter.grad for parameter in selector.parameters()
                     if parameter.grad is not None]
        self.assertTrue(any(torch.any(gradient.abs() > 0)
                            for gradient in gradients))

    def test_selected_neighbor_sources_block_unselected_neighbor_facts(self):
        def account(uid, line, followers):
            tweet = TweetAtom(0, line, uid, 0, 0, 5, 1, 0, "en",
                              1704067200, 0, 0, 0, 0, "text", "record")
            return AccountAtoms(uid, line,
                                {"id_str": uid, "followers_count": followers},
                                (tweet,))

        ids = ["a", "b"]
        relation = RelationAtom("r", "a", "b", "reply", ((1, 0, "reply"),))
        first = Fox8TwoStageInputs.build(
            {"a": account("a", 1, 1), "b": account("b", 2, 2)},
            (relation,), ids)
        changed = Fox8TwoStageInputs.build(
            {"a": account("a", 1, 100), "b": account("b", 2, 2)},
            (relation,), ids)
        relnet = GatedRelNet(len(ALL_FACTS), hidden_channels=12,
                             relation_names=tuple(first.full_graph),
                             num_layers=1, dropout=0.0, out_channels=2)
        model = TargetEvidenceFox8(
            ContextRelationSelector(), relnet,
            torch.zeros(len(ALL_FACTS)), torch.ones(len(ALL_FACTS)),
            centered_evidence=True, bot_only_support=True,
            selected_neighbor_sources=True).eval()
        # Account a withholds its facts, while b still reads the a->b edge.
        context = torch.tensor([[-20.0], [20.0]])
        with torch.inference_mode():
            emitted = model(context, first)
            original = emitted["logits"]
            altered = model(context, changed)["logits"]
            verified = model.verify_bot_evidence(
                first, emitted["hard_selection"])
            verified_after_change = model.verify_bot_evidence(
                changed, emitted["hard_selection"])
            none_available = {
                "profile": torch.zeros(len(first.profile_account), dtype=torch.bool),
                "tweet": torch.zeros(len(first.tweet_account), dtype=torch.bool),
                "relation": torch.zeros(len(first.relations), dtype=torch.bool),
            }
            empty_graph = graph_from_relation_atoms(first.relations, ids, ())
            verified_empty = model.verify_bot_evidence(
                first, emitted["hard_selection"], none_available, empty_graph)
        self.assertTrue(torch.allclose(original[1], altered[1], atol=1e-6))
        self.assertTrue(torch.allclose(original[:, 1], verified, atol=1e-6))
        self.assertTrue(torch.allclose(verified[1],
                                       verified_after_change[1], atol=1e-6))
        self.assertTrue(torch.allclose(verified_empty,
                                       torch.zeros_like(verified_empty),
                                       atol=1e-6))


if __name__ == "__main__":
    unittest.main()

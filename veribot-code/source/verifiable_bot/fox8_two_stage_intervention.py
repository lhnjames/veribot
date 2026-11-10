"""Recompute two-stage Fox8 judgments after withholding named input sources."""

from __future__ import annotations

import numpy as np
import torch

from .fox8_atomic_sources import graph_from_relation_atoms
from .fox8_two_stage_inputs import Fox8TwoStageInputs
from .fox8_two_stage_model import TwoStageFox8


class TwoStageIntervention:
    def __init__(self, *, hgt: torch.nn.Module, model: TwoStageFox8,
                 batch: Fox8TwoStageInputs, base_mean: np.ndarray,
                 base_scale: np.ndarray):
        self.hgt = hgt.eval()
        self.model = model.eval()
        self.batch = batch
        self.base_mean = np.asarray(base_mean)
        self.base_scale = np.asarray(base_scale)
        self.account_index = {uid: i for i, uid in enumerate(batch.ids)}
        self.lookup: dict[str, tuple[str, int]] = {}
        for i, account in enumerate(batch.accounts):
            p0, p1 = batch.profile_offsets[i:i + 2]
            t0, t1 = batch.tweet_offsets[i:i + 2]
            for j, field in enumerate(account.profile_field_ids, p0):
                self.lookup[account.source_id("profile", field)] = ("profile", j)
            for j, ordinal in enumerate(account.tweet_ordinals, t0):
                self.lookup[account.source_id("tweet", ordinal)] = ("tweet", j)
        for j, relation in enumerate(batch.relations):
            self.lookup[relation.source_id] = ("relation", j)
        expected = (len(batch.profile_account) + len(batch.tweet_account) +
                    len(batch.relations))
        if len(self.lookup) != expected:
            raise ValueError("source IDs are not unique across channels")
        self.cache: dict[frozenset[str], np.ndarray] = {}

    def target_source_ids(self, account_id: str) -> tuple[str, ...]:
        account = self.batch.accounts[self.account_index[account_id]]
        own = [account.source_id("profile", field) for field in account.profile_field_ids]
        own.extend(account.source_id("tweet", ordinal)
                   for ordinal in account.tweet_ordinals)
        own.extend(item.source_id for item in self.batch.relations
                   if item.source_account_id == account_id or
                   item.target_account_id == account_id)
        return tuple(own)

    def _availability(self, removed: frozenset[str]) -> dict[str, torch.Tensor]:
        if removed - self.lookup.keys():
            raise ValueError("withheld source is absent from input universe")
        available = {
            "profile": torch.ones(len(self.batch.profile_account)),
            "tweet": torch.ones(len(self.batch.tweet_account)),
            "relation": torch.ones(len(self.batch.relations)),
        }
        for source_id in removed:
            kind, position = self.lookup[source_id]
            available[kind][position] = 0
        return available

    def predict(self, account_id: str,
                removed: frozenset[str] = frozenset()) -> dict:
        if removed not in self.cache:
            hgt_parameter = next(iter(self.hgt.parameters()), None)
            device = (hgt_parameter.device if hgt_parameter is not None else
                      next(self.model.parameters()).device)
            available = self._availability(removed)
            facts = self.batch.exact_view(available["profile"], available["tweet"])
            x = torch.from_numpy(np.clip(
                (facts[:, :31] - self.base_mean) / self.base_scale,
                -3, 3).astype(np.float32)).to(device)
            selected_relations = [item.source_id for item, keep in
                                  zip(self.batch.relations,
                                      available["relation"].tolist()) if keep]
            graph = graph_from_relation_atoms(
                self.batch.relations, self.batch.ids, selected_relations)
            with torch.inference_mode():
                context = self.hgt({"user": x}, {
                    ("user", name, "user"): edges.to(device)
                    for name, edges in graph.items()
                })["user_emb"]
                result = self.model(context, self.batch, availability=available)
            self.cache[removed] = result["logits"].detach().cpu().numpy()
        scores = self.cache[removed][self.account_index[account_id]]
        return {"human_score": float(scores[0]), "bot_score": float(scores[1]),
                "margin": float(scores[1] - scores[0])}

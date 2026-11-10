"""Re-run the complete target-conditioned HGT→RelNet decision after source edits."""

from __future__ import annotations

import numpy as np
import torch

from .fox8_atomic_sources import graph_from_relation_atoms
from .fox8_two_stage_intervention import TwoStageIntervention


class TargetEvidenceIntervention(TwoStageIntervention):
    def predict(self, account_id: str,
                removed: frozenset[str] = frozenset()) -> dict:
        if removed not in self.cache:
            device = next(self.model.parameters()).device
            available = self._availability(removed)
            facts = self.batch.exact_view(available["profile"],
                                          available["tweet"])
            x = torch.from_numpy(np.clip(
                (facts[:, :31] - self.base_mean) / self.base_scale,
                -3, 3).astype(np.float32)).to(device)
            relation_ids = [atom.source_id for atom, keep in
                            zip(self.batch.relations,
                                available["relation"].tolist()) if keep]
            graph = graph_from_relation_atoms(
                self.batch.relations, self.batch.ids, relation_ids)
            with torch.inference_mode():
                context = self.hgt({"user": x}, {
                    ("user", name, "user"): edges.to(device)
                    for name, edges in graph.items()
                })["user_emb"]
                result = self.model(context, self.batch,
                                    availability=available,
                                    graph_override=graph)
            self.cache[removed] = result["logits"].detach().cpu().numpy()
        scores = self.cache[removed][self.account_index[account_id]]
        return {"human_score": float(scores[0]),
                "bot_score": float(scores[1]),
                "margin": float(scores[1] - scores[0])}


class FixedBotEvidenceIntervention:
    """Intervene on raw sources while holding the Bot proposal fixed."""

    def __init__(self, full_path: TargetEvidenceIntervention,
                 hard_selection: dict[str, torch.Tensor]):
        if not (full_path.model.bot_only_support and
                full_path.model.selected_neighbor_sources):
            raise ValueError("Fixed Bot certificate requires selected neighbors")
        self.full_path = full_path
        self.hard_selection = hard_selection
        self.lookup = full_path.lookup
        self.cache: dict[frozenset[str], np.ndarray] = {}

    def predict(self, account_id: str,
                removed: frozenset[str] = frozenset()) -> dict:
        if removed not in self.cache:
            bank = self.full_path.batch
            available = self.full_path._availability(removed)
            relation_ids = [atom.source_id for atom, keep in
                            zip(bank.relations,
                                available["relation"].tolist()) if keep]
            graph = graph_from_relation_atoms(
                bank.relations, bank.ids, relation_ids)
            with torch.inference_mode():
                scores = self.full_path.model.verify_bot_evidence(
                    bank, self.hard_selection, available, graph)
            self.cache[removed] = scores.detach().cpu().numpy()
        score = float(self.cache[removed][
            self.full_path.account_index[account_id]])
        return {"human_score": 0.0, "bot_score": score, "margin": score}

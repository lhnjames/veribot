"""HGT-guided source selection followed by evidence-view RelNet judgment."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .aaai_fox8_raw_features import RAW_PROFILE_FIELDS
from .fox8_two_stage_inputs import Fox8TwoStageInputs


class HGTSourceSelector(nn.Module):
    """Choose Bot and Human source views before the final RelNet decision."""

    def __init__(self, context_dim: int = 128, hidden: int = 48,
                 initial_gate_bias: float = 0.0):
        super().__init__()
        self.field_embedding = nn.Embedding(len(RAW_PROFILE_FIELDS), 12)
        self.relation_embedding = nn.Embedding(2, 8)
        self.profile_head = self._head(context_dim + 12 + 3, hidden)
        self.tweet_head = self._head(context_dim + 11, hidden)
        self.relation_head = self._head(context_dim * 2 + 8 + 1, hidden)
        for head in (self.profile_head, self.tweet_head, self.relation_head):
            nn.init.constant_(head[-1].bias, initial_gate_bias)

    @staticmethod
    def _head(input_dim: int, hidden: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(),
                             nn.Linear(hidden, hidden), nn.GELU(),
                             nn.Linear(hidden, 2))

    def forward(self, context: torch.Tensor,
                batch: Fox8TwoStageInputs) -> dict[str, torch.Tensor]:
        device = context.device
        p = self.profile_head(torch.cat((
            context[batch.profile_account.to(device)],
            self.field_embedding(batch.profile_field.to(device)),
            batch.profile_descriptor.to(device)), dim=1))
        t = self.tweet_head(torch.cat((
            context[batch.tweet_account.to(device)],
            batch.tweet_descriptor.to(device)), dim=1))
        r = self.relation_head(torch.cat((
            context[batch.relation_source.to(device)],
            context[batch.relation_target.to(device)],
            self.relation_embedding(batch.relation_kind.to(device)),
            batch.relation_descriptor.to(device)), dim=1))
        return {"profile": p, "tweet": t, "relation": r}


def straight_through_gates(logits: torch.Tensor, *, training: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probability = torch.sigmoid(logits)
    hard = ((torch.rand_like(probability) < probability) if training else
            (probability >= 0.5)).to(probability.dtype)
    gate = hard + probability - probability.detach()
    return gate, hard, probability


class TwoStageFox8(nn.Module):
    """RelNet scores each class on its own HGT-selected source view.

    The straight-through surrogate affects gradients only. Every forward
    classification uses exact features reconstructed from hard source IDs.
    """

    def __init__(self, selector: HGTSourceSelector, relnet: nn.Module,
                 scaler_mean: torch.Tensor, scaler_scale: torch.Tensor):
        super().__init__()
        self.selector = selector
        self.relnet = relnet
        self.register_buffer("scaler_mean", scaler_mean.float())
        self.register_buffer("scaler_scale", scaler_scale.float())

    def _view(self, batch: Fox8TwoStageInputs, profile_gate: torch.Tensor,
              tweet_gate: torch.Tensor, relation_gate: torch.Tensor,
              profile_hard: torch.Tensor, tweet_hard: torch.Tensor) -> torch.Tensor:
        device = profile_gate.device
        hard = torch.from_numpy(batch.exact_view(profile_hard, tweet_hard)).to(device)
        surrogate = torch.from_numpy(batch.full_facts).to(device).clone()
        profile_delta = (profile_gate - 1)[:, None] * batch.profile_surrogate.to(device)
        tweet_delta = (tweet_gate - 1)[:, None] * batch.tweet_surrogate.to(device)
        surrogate.index_add_(0, batch.profile_account.to(device), profile_delta)
        surrogate.index_add_(0, batch.tweet_account.to(device), tweet_delta)
        facts = hard + surrogate - surrogate.detach()
        return torch.clamp((facts - self.scaler_mean) / self.scaler_scale, -3, 3)

    def forward(self, context: torch.Tensor,
                batch: Fox8TwoStageInputs,
                availability: dict[str, torch.Tensor] | None = None,
                compute_complement: bool = False) -> dict[str, object]:
        if availability is not None and compute_complement:
            raise ValueError("complement view requires the full source universe")
        proposals = self.selector(context, batch)
        if availability is not None:
            for name, logits in proposals.items():
                allowed = availability[name].to(device=logits.device, dtype=torch.bool)
                if allowed.ndim != 1 or allowed.numel() != logits.shape[0]:
                    raise ValueError(f"source availability differs for {name}")
                proposals[name] = logits.masked_fill(~allowed[:, None], -1e9)
        selected = {name: straight_through_gates(value, training=self.training)
                    for name, value in proposals.items()}
        graph = {name: edges.to(context.device)
                 for name, edges in batch.full_graph.items()}
        views = []
        complement_views = []
        for class_index in (0, 1):  # 0 = Human hypothesis; 1 = Bot hypothesis
            pg, ph, _ = selected["profile"]
            tg, th, _ = selected["tweet"]
            rg, _, _ = selected["relation"]
            x = self._view(batch, pg[:, class_index], tg[:, class_index],
                           rg[:, class_index], ph[:, class_index], th[:, class_index])
            edge_weights = batch.relation_weights(rg[:, class_index])
            scores = self.relnet(x, graph, edge_weights)
            views.append(scores[:, class_index])
            if compute_complement:
                complement_x = self._view(
                    batch, 1 - pg[:, class_index], 1 - tg[:, class_index],
                    1 - rg[:, class_index], 1 - ph[:, class_index],
                    1 - th[:, class_index])
                complement_edges = batch.relation_weights(1 - rg[:, class_index])
                complement_scores = self.relnet(
                    complement_x, graph, complement_edges)
                complement_views.append(complement_scores[:, class_index])
        logits = torch.stack(views, dim=1)
        mean_selected = torch.stack([
            selected[name][2].mean(dim=0) for name in ("profile", "tweet", "relation")
        ]).mean()
        result = {"logits": logits, "gate_probabilities": {
                    name: outcome[2] for name, outcome in selected.items()},
                "hard_selection": {name: outcome[1].detach()
                                   for name, outcome in selected.items()},
                "mean_source_selection_probability": mean_selected}
        if compute_complement:
            result["complement_logits"] = torch.stack(complement_views, dim=1)
        return result


def initialize_relnet_from_binary(relnet: nn.Module, binary_state: dict) -> None:
    """Preserve the 31-feature binary model as a two-logit starting point."""
    state = relnet.state_dict()
    old_input = binary_state["input.weight"]
    if (state["input.weight"].shape[0] != old_input.shape[0] or
            state["input.weight"].shape[1] < old_input.shape[1] or
            state["classifier.weight"].shape[0] != 2):
        raise ValueError("two-stage RelNet dimensions differ from the binary initializer")
    for name, tensor in binary_state.items():
        if name in ("input.weight", "classifier.weight", "classifier.bias"):
            continue
        if name not in state or state[name].shape != tensor.shape:
            raise ValueError(f"RelNet layer differs: {name}")
        state[name] = tensor.clone()
    state["input.weight"] = torch.zeros_like(state["input.weight"])
    state["input.weight"][:, :old_input.shape[1]] = old_input
    state["classifier.weight"] = torch.cat((
        -binary_state["classifier.weight"] / 2,
        binary_state["classifier.weight"] / 2), dim=0)
    state["classifier.bias"] = torch.cat((
        -binary_state["classifier.bias"] / 2,
        binary_state["classifier.bias"] / 2), dim=0)
    relnet.load_state_dict(state)

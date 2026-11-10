"""Target-conditioned HGT source selection and one-hop evidence-view RelNet.

Each account score uses HGT context at that account only. RelNet reads the
selected raw sources of the target and unselected raw sources of its direct
neighbors. This prevents an account's score from depending on HGT contexts at
all neighbors selected by a global source gate.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .aaai_fox8_raw_features import RAW_PROFILE_FIELDS
from .fox8_two_stage_inputs import Fox8TwoStageInputs
from .fox8_two_stage_model import straight_through_gates


class HGTTargetFactSelector(nn.Module):
    def __init__(self, context_dim: int = 128, hidden: int = 48,
                 initial_gate_bias: float = 0.0):
        super().__init__()
        self.field_embedding = nn.Embedding(len(RAW_PROFILE_FIELDS), 12)
        self.profile_head = self._head(context_dim + 12 + 3, hidden)
        self.tweet_head = self._head(context_dim + 11, hidden)
        for head in (self.profile_head, self.tweet_head):
            nn.init.constant_(head[-1].bias, initial_gate_bias)

    @staticmethod
    def _head(input_dim: int, hidden: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(),
                             nn.Linear(hidden, hidden), nn.GELU(),
                             nn.Linear(hidden, 2))

    def forward(self, context: torch.Tensor,
                batch: Fox8TwoStageInputs) -> dict[str, torch.Tensor]:
        device = context.device
        profile = self.profile_head(torch.cat((
            context[batch.profile_account.to(device)],
            self.field_embedding(batch.profile_field.to(device)),
            batch.profile_descriptor.to(device)), dim=1))
        tweet = self.tweet_head(torch.cat((
            context[batch.tweet_account.to(device)],
            batch.tweet_descriptor.to(device)), dim=1))
        return {"profile": profile, "tweet": tweet}


class HGTTargetRelationSelector(HGTTargetFactSelector):
    """Select each native relation for its receiving target and class view."""

    def __init__(self, context_dim: int = 128, hidden: int = 48,
                 initial_gate_bias: float = 0.0):
        super().__init__(context_dim, hidden, initial_gate_bias)
        self.relation_embedding = nn.Embedding(2, 8)
        self.relation_head = self._head(context_dim + 8 + 1, hidden)
        nn.init.constant_(self.relation_head[-1].bias, initial_gate_bias)

    def forward(self, context: torch.Tensor,
                batch: Fox8TwoStageInputs) -> dict[str, torch.Tensor]:
        proposals = super().forward(context, batch)
        device = context.device
        descriptor = torch.cat((
            self.relation_embedding(batch.relation_kind.to(device)),
            batch.relation_descriptor.to(device)), dim=1)
        # The forward edge is received at relation_target; the reverse edge
        # is received at relation_source. Each target uses only its own HGT
        # context to decide which adjacent native events to read.
        proposals["relation_forward"] = self.relation_head(torch.cat((
            context[batch.relation_target.to(device)], descriptor), dim=1))
        proposals["relation_reverse"] = self.relation_head(torch.cat((
            context[batch.relation_source.to(device)], descriptor), dim=1))
        return proposals


class TargetEvidenceFox8(nn.Module):
    """Produce Bot/Human logits from separate target-source hypotheses.

    In centered mode each class score is the change from that class's
    no-target-source, no-relation anchor in the same graph context. Thus an
    empty evidence view has zero support in both channels before calibration.
    """

    def __init__(self, selector: HGTTargetFactSelector, relnet: nn.Module,
                 scaler_mean: torch.Tensor, scaler_scale: torch.Tensor,
                 centered_evidence: bool = False,
                 sample_training_gates: bool = True,
                 bot_only_support: bool = False,
                 selected_neighbor_sources: bool = False):
        super().__init__()
        if len(relnet.layers) != 1:
            raise ValueError("target evidence RelNet must have one relation layer")
        if bot_only_support and not centered_evidence:
            raise ValueError("Bot-only support requires a neutral anchor")
        if selected_neighbor_sources and not bot_only_support:
            raise ValueError("Selected neighbor sources require Bot-only support")
        self.selector = selector
        self.relnet = relnet
        self.centered_evidence = centered_evidence
        self.sample_training_gates = sample_training_gates
        self.bot_only_support = bot_only_support
        self.selected_neighbor_sources = selected_neighbor_sources
        self.register_buffer("scaler_mean", scaler_mean.float())
        self.register_buffer("scaler_scale", scaler_scale.float())

    def _facts(self, batch: Fox8TwoStageInputs,
               profile_gate: torch.Tensor, tweet_gate: torch.Tensor,
               profile_hard: torch.Tensor, tweet_hard: torch.Tensor) -> torch.Tensor:
        device = profile_gate.device
        hard = torch.from_numpy(batch.exact_view(profile_hard, tweet_hard)).to(device)
        surrogate = torch.from_numpy(batch.full_facts).to(device).clone()
        surrogate.index_add_(
            0, batch.profile_account.to(device),
            (profile_gate - 1)[:, None] * batch.profile_surrogate.to(device))
        surrogate.index_add_(
            0, batch.tweet_account.to(device),
            (tweet_gate - 1)[:, None] * batch.tweet_surrogate.to(device))
        facts = hard + surrogate - surrogate.detach()
        return torch.clamp((facts - self.scaler_mean) / self.scaler_scale, -3, 3)

    def _target_layer(self, target_h: torch.Tensor, source_h: torch.Tensor,
                      graph: dict[str, torch.Tensor],
                      edge_weights: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        layer = self.relnet.layers[0]
        aggregates = []
        for name in layer.relation_names:
            src, dst = graph[name]
            # A native self-edge reads the target's selected sources too.
            sender = torch.where((src == dst)[:, None],
                                 target_h[src], source_h[src])
            messages = layer.messages[name](sender)
            weight = None if edge_weights is None else edge_weights[name]
            if weight is not None:
                if weight.numel() != len(dst):
                    raise ValueError(f"relation gate differs from {name} graph")
                messages = messages * weight[:, None]
            summed = torch.zeros_like(target_h)
            summed.index_add_(0, dst, messages)
            degree = torch.zeros((len(target_h), 1),
                                 device=target_h.device, dtype=target_h.dtype)
            degree.index_add_(0, dst, (
                torch.ones((len(dst), 1), device=target_h.device,
                           dtype=target_h.dtype) if weight is None
                else weight[:, None]))
            aggregates.append(summed / degree.clamp(min=1))
        aggregate = torch.stack(aggregates).mean(dim=0)
        gated = torch.sigmoid(layer.gate(torch.cat((target_h, aggregate), dim=-1)))
        updated = layer.norm(target_h + gated * F.gelu(aggregate))
        return F.dropout(updated, p=layer.dropout, training=self.training)

    def forward(self, context: torch.Tensor, batch: Fox8TwoStageInputs,
                availability: dict[str, torch.Tensor] | None = None,
                graph_override: dict[str, torch.Tensor] | None = None,
                compute_complement: bool = False) -> dict[str, object]:
        if availability is not None and compute_complement:
            raise ValueError("complement view requires the full source universe")
        proposals = self.selector(context, batch)
        if availability is not None:
            for name, logits in proposals.items():
                channel = "relation" if name.startswith("relation_") else name
                allowed = availability[channel].to(device=logits.device,
                                                   dtype=torch.bool)
                proposals[name] = logits.masked_fill(~allowed[:, None], -1e9)
        selected = {name: straight_through_gates(
            value, training=self.training and self.sample_training_gates)
                    for name, value in proposals.items()}
        if self.centered_evidence and not {
                "relation_forward", "relation_reverse"}.issubset(selected):
            raise ValueError("centered evidence requires class-specific relation gates")
        graph = {name: edges.to(context.device) for name, edges in
                 (graph_override or batch.full_graph).items()}
        source_h = None
        if not self.selected_neighbor_sources:
            if availability is None:
                full_facts = torch.from_numpy(batch.full_facts).to(context.device)
            else:
                full_facts = torch.from_numpy(batch.exact_view(
                    availability["profile"], availability["tweet"])).to(context.device)
            full_x = torch.clamp((full_facts - self.scaler_mean) / self.scaler_scale,
                                 -3, 3)
            source_h = F.gelu(self.relnet.input(full_x))
        views = []
        complements = []
        pg, ph, _ = selected["profile"]
        tg, th, _ = selected["tweet"]
        neutral_logits = None
        if self.centered_evidence:
            neutral_x = self._facts(
                batch, torch.zeros_like(pg[:, 0]), torch.zeros_like(tg[:, 0]),
                torch.zeros_like(ph[:, 0]), torch.zeros_like(th[:, 0]))
            neutral_h = F.gelu(self.relnet.input(neutral_x))
            no_relation = {name: torch.zeros(
                graph[name].shape[1], device=context.device,
                dtype=neutral_h.dtype) for name in
                self.relnet.layers[0].relation_names}
            neutral_result = self._target_layer(
                neutral_h, neutral_h if source_h is None else source_h,
                graph, no_relation)
            neutral_logits = self.relnet.classifier(neutral_result)
        for cls in (0, 1):
            if self.bot_only_support and cls == 0:
                views.append(torch.zeros_like(neutral_logits[:, 0]))
                if compute_complement:
                    complements.append(torch.zeros_like(neutral_logits[:, 0]))
                continue
            relation_weights = None
            complement_weights = None
            if "relation_forward" in selected:
                relation_weights = {}
                complement_weights = {}
                for name in self.relnet.layers[0].relation_names:
                    channel = ("relation_reverse" if name.startswith("rev_")
                               else "relation_forward")
                    order = batch.relation_order[name].to(context.device)
                    gates = selected[channel][0][:, cls][order]
                    if availability is not None:
                        present = availability["relation"].to(
                            context.device, dtype=torch.bool)[order]
                        gates = gates[present]
                    if len(gates) != graph[name].shape[1]:
                        raise ValueError(f"selected relation order differs for {name}")
                    relation_weights[name] = gates
                    if compute_complement:
                        complement_weights[name] = 1 - gates
            target_x = self._facts(batch, pg[:, cls], tg[:, cls],
                                   ph[:, cls], th[:, cls])
            target_h = F.gelu(self.relnet.input(target_x))
            result_h = self._target_layer(target_h,
                                          target_h if source_h is None else source_h,
                                          graph,
                                          relation_weights)
            class_score = self.relnet.classifier(result_h)[:, cls]
            if neutral_logits is not None:
                class_score = class_score - neutral_logits[:, cls]
            views.append(class_score)
            if compute_complement:
                complement_x = self._facts(
                    batch, 1 - pg[:, cls], 1 - tg[:, cls],
                    1 - ph[:, cls], 1 - th[:, cls])
                complement_h = F.gelu(self.relnet.input(complement_x))
                complement_result = self._target_layer(
                    complement_h,
                    complement_h if source_h is None else source_h,
                    graph, complement_weights)
                complement_score = self.relnet.classifier(complement_result)[:, cls]
                if neutral_logits is not None:
                    complement_score = complement_score - neutral_logits[:, cls]
                complements.append(complement_score)
        regularized_probabilities = [
            item[2][:, 1] if self.bot_only_support else item[2]
            for item in selected.values()]
        output = {
            "logits": torch.stack(views, dim=1),
            "hard_selection": {name: item[1].detach() for name, item in selected.items()},
            "gate_probabilities": {name: item[2] for name, item in selected.items()},
            "mean_source_selection_probability": torch.stack(
                [probabilities.mean() for probabilities in
                 regularized_probabilities]).mean(),
            "mean_gate_uncertainty": torch.stack([
                (4 * probabilities * (1 - probabilities)).mean()
                for probabilities in regularized_probabilities]).mean(),
            "expected_bot_source_count_per_account": (
                sum(item[2][:, 1].sum() for item in selected.values()) /
                len(batch.ids)),
        }
        if compute_complement:
            output["complement_logits"] = torch.stack(complements, dim=1)
        if neutral_logits is not None:
            output["neutral_anchor_logits"] = neutral_logits
        return output

    def verify_bot_evidence(
            self, batch: Fox8TwoStageInputs,
            hard_selection: dict[str, torch.Tensor],
            availability: dict[str, torch.Tensor] | None = None,
            graph_override: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Score a fixed Bot candidate using only its selected raw sources.

        Candidate generation is finished before this call. No HGT context or
        selector output is recomputed; availability can withhold raw sources
        to measure their effect on this evidence-only score.
        """
        if not (self.bot_only_support and self.selected_neighbor_sources):
            raise ValueError("Evidence-only verification needs selected Bot neighbors")
        if self.training:
            raise ValueError("Evidence-only verification requires evaluation mode")
        device = self.scaler_mean.device

        def selected(name: str) -> torch.Tensor:
            values = hard_selection[name].to(device)
            if values.ndim != 2 or values.shape[1] != 2:
                raise ValueError(f"{name} must contain two class columns")
            mask = values[:, 1].bool()
            if availability is not None:
                channel = "relation" if name.startswith("relation_") else name
                mask = mask & availability[channel].to(device).bool()
            return mask

        profile = selected("profile")
        tweet = selected("tweet")
        target_x = self._facts(batch, profile.float(), tweet.float(),
                               profile, tweet)
        target_h = F.gelu(self.relnet.input(target_x))
        graph = {name: edges.to(device) for name, edges in
                 (graph_override or batch.full_graph).items()}
        weights = {}
        for name in self.relnet.layers[0].relation_names:
            channel = ("relation_reverse" if name.startswith("rev_")
                       else "relation_forward")
            order = batch.relation_order[name].to(device)
            chosen = selected(channel)[order].float()
            if availability is not None:
                chosen = chosen[availability["relation"].to(device).bool()[order]]
            if len(chosen) != graph[name].shape[1]:
                raise ValueError(f"relation order differs for {name}")
            weights[name] = chosen
        selected_h = self._target_layer(target_h, target_h, graph, weights)
        neutral_x = self._facts(batch, torch.zeros_like(profile, dtype=torch.float),
                                torch.zeros_like(tweet, dtype=torch.float),
                                torch.zeros_like(profile), torch.zeros_like(tweet))
        neutral_h = F.gelu(self.relnet.input(neutral_x))
        no_relation = {name: torch.zeros(
            graph[name].shape[1], device=device, dtype=neutral_h.dtype)
            for name in self.relnet.layers[0].relation_names}
        neutral_result = self._target_layer(
            neutral_h, neutral_h, graph, no_relation)
        return (self.relnet.classifier(selected_h)[:, 1] -
                self.relnet.classifier(neutral_result)[:, 1])


def initialize_target_relnet(relnet: nn.Module, binary_state: dict) -> None:
    """Warm-start the one-hop classifier from the original binary RelNet."""
    state = relnet.state_dict()
    for name in state:
        if name in ("input.weight", "classifier.weight", "classifier.bias"):
            continue
        if name not in binary_state or state[name].shape != binary_state[name].shape:
            raise ValueError(f"one-hop RelNet initializer differs: {name}")
        state[name] = binary_state[name].clone()
    old_input = binary_state["input.weight"]
    if state["input.weight"].shape[1] < old_input.shape[1]:
        raise ValueError("one-hop RelNet input has fewer facts than initializer")
    state["input.weight"] = torch.zeros_like(state["input.weight"])
    state["input.weight"][:, :old_input.shape[1]] = old_input
    state["classifier.weight"] = torch.cat((
        -binary_state["classifier.weight"] / 2,
        binary_state["classifier.weight"] / 2), dim=0)
    state["classifier.bias"] = torch.cat((
        -binary_state["classifier.bias"] / 2,
        binary_state["classifier.bias"] / 2), dim=0)
    relnet.load_state_dict(state)

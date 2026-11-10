from __future__ import annotations

import torch
from torch import nn


class RelationGraphPrefix(nn.Module):
    """Encodes sampled typed ego edges into soft prefix tokens."""
    def __init__(self, lm_hidden: int, graph_hidden: int = 256, n_relations: int = 16, prefix_tokens: int = 8):
        super().__init__()
        self.prefix_tokens = prefix_tokens
        self.relation = nn.Embedding(n_relations, graph_hidden)
        self.direction = nn.Embedding(2, graph_hidden)
        self.project = nn.Sequential(nn.Linear(graph_hidden, graph_hidden), nn.GELU(),
                                     nn.Linear(graph_hidden, lm_hidden * prefix_tokens))

    def forward(self, relation_ids: torch.Tensor, directions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.relation(relation_ids) + self.direction(directions)
        denom = mask.sum(1, keepdim=True).clamp_min(1)
        pooled = (x * mask.unsqueeze(-1)).sum(1) / denom
        return self.project(pooled).view(x.shape[0], self.prefix_tokens, -1)


class QwenGraphEvidenceModel(nn.Module):
    def __init__(self, language_model: nn.Module, graph_hidden: int = 256, prefix_tokens: int = 8):
        super().__init__()
        self.language_model = language_model
        hidden = language_model.config.hidden_size
        self.graph_prefix = RelationGraphPrefix(hidden, graph_hidden, prefix_tokens=prefix_tokens)

    @property
    def config(self): return self.language_model.config

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        return self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def enable_input_require_grads(self): return self.language_model.enable_input_require_grads()

    def forward(self, input_ids, attention_mask, labels=None, relation_ids=None, directions=None, graph_mask=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        if relation_ids is None:
            return self.language_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        token_embeds = self.language_model.get_input_embeddings()(input_ids)
        prefix = self.graph_prefix(relation_ids, directions, graph_mask)
        prefix = prefix.to(device=token_embeds.device, dtype=token_embeds.dtype)
        embeds = torch.cat([prefix, token_embeds], dim=1)
        pmask = torch.ones(prefix.shape[:2], device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([pmask, attention_mask], dim=1)
        if labels is not None:
            ignore = torch.full(prefix.shape[:2], -100, device=labels.device, dtype=labels.dtype)
            labels = torch.cat([ignore, labels], dim=1)
        return self.language_model(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels, **kwargs)

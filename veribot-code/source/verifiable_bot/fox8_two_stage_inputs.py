"""Atomic-source tensors and exact selected views for a two-stage Fox8 model."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, log1p, pi, sin

import numpy as np
import torch

from .aaai_fox8_feature_schema import FEATURE_FIELDS, PROFILE_FIELDS
from .aaai_fox8_raw_features import RAW_PROFILE_FIELDS, profile_features
from .fox8_atomic_sources import (
    EXTENDED_TWEET_FIELDS, AccountAtoms, RelationAtom,
    graph_from_relation_atoms,
)


ALL_FACTS = FEATURE_FIELDS + EXTENDED_TWEET_FIELDS


@dataclass
class Fox8TwoStageInputs:
    ids: tuple[str, ...]
    accounts: tuple[AccountAtoms, ...]
    relations: tuple[RelationAtom, ...]
    full_facts: np.ndarray
    profile_account: torch.Tensor
    profile_field: torch.Tensor
    profile_descriptor: torch.Tensor
    profile_surrogate: torch.Tensor
    profile_offsets: tuple[int, ...]
    tweet_account: torch.Tensor
    tweet_descriptor: torch.Tensor
    tweet_surrogate: torch.Tensor
    tweet_offsets: tuple[int, ...]
    relation_source: torch.Tensor
    relation_target: torch.Tensor
    relation_kind: torch.Tensor
    relation_descriptor: torch.Tensor
    relation_order: dict[str, torch.Tensor]
    full_graph: dict[str, torch.Tensor]

    @classmethod
    def build(cls, accounts_by_id: dict[str, AccountAtoms],
              relations: tuple[RelationAtom, ...], ids: list[str]) -> "Fox8TwoStageInputs":
        accounts = tuple(accounts_by_id[uid] for uid in ids)
        index = {uid: i for i, uid in enumerate(ids)}
        full = np.asarray([
            np.concatenate((account.features(),
                            np.asarray(list(account.extended_features().values()),
                                       dtype=np.float32)))
            for account in accounts], dtype=np.float32)
        profile_account, profile_field, profile_descriptor, profile_surrogate = [], [], [], []
        tweet_account, tweet_descriptor, tweet_surrogate = [], [], []
        profile_offsets, tweet_offsets = [0], [0]
        for i, account in enumerate(accounts):
            full_profile = account.features()[:len(PROFILE_FIELDS)]
            for field in account.profile_field_ids:
                field_index = RAW_PROFILE_FIELDS.index(field)
                value = account.profile[field]
                if isinstance(value, bool):
                    numeric, string_len, present = float(value), 0.0, float(value)
                elif isinstance(value, (int, float)):
                    numeric = float(np.sign(value) * log1p(abs(value)))
                    string_len, present = 0.0, float(value != 0)
                else:
                    numeric = 0.0
                    string_len = log1p(len(str(value or "")))
                    present = float(bool(value))
                profile_account.append(i)
                profile_field.append(field_index)
                profile_descriptor.append([numeric, string_len, present])
                reduced_profile = profile_features({
                    key: val for key, val in account.profile.items() if key != field})
                reduced = np.asarray([reduced_profile[name] for name in PROFILE_FIELDS],
                                     dtype=np.float32)
                surrogate = np.zeros(len(ALL_FACTS), dtype=np.float32)
                surrogate[:len(PROFILE_FIELDS)] = full_profile - reduced
                profile_surrogate.append(surrogate)
            profile_offsets.append(len(profile_account))
            n = len(account.tweets)
            for atom in account.tweets:
                hour = ((atom.created_at_unix or 0) // 3600) % 24
                tweet_account.append(i)
                tweet_descriptor.append([
                    float(atom.is_retweet), float(atom.is_reply),
                    log1p(atom.text_length), log1p(atom.favorites),
                    log1p(atom.retweets), log1p(atom.mention_count),
                    log1p(atom.hashtag_count), log1p(atom.url_count),
                    float(atom.is_quote), sin(2 * pi * hour / 24),
                    cos(2 * pi * hour / 24),
                ])
                surrogate = np.zeros(len(ALL_FACTS), dtype=np.float32)
                start = len(PROFILE_FIELDS)
                surrogate[start:start + 7] = [
                    1, atom.is_retweet / n, atom.is_reply / n,
                    atom.text_length / n, atom.favorites / n,
                    atom.retweets / n, (atom.favorites + atom.retweets) / n,
                ]
                extra = len(FEATURE_FIELDS)
                surrogate[extra + 3:extra + 7] = [
                    atom.mention_count / n, atom.hashtag_count / n,
                    atom.url_count / n, atom.is_quote / n,
                ]
                tweet_surrogate.append(surrogate)
            tweet_offsets.append(len(tweet_account))

        relation_source, relation_target, relation_kind, relation_descriptor = [], [], [], []
        by_relation = {name: [] for name in ("reply", "rev_reply", "retweet", "rev_retweet")}
        for j, atom in enumerate(relations):
            src = index[atom.source_account_id]
            dst = index[atom.target_account_id]
            relation_source.append(src)
            relation_target.append(dst)
            relation_kind.append(int(atom.relation == "retweet"))
            relation_descriptor.append([log1p(len(atom.raw_locators))])
            by_relation[atom.relation].append((src, dst, j))
            by_relation[f"rev_{atom.relation}"].append((dst, src, j))
        relation_order = {name: torch.tensor(
            [item[2] for item in sorted(entries)], dtype=torch.long)
            for name, entries in by_relation.items()}
        graph = graph_from_relation_atoms(relations, ids)
        return cls(
            tuple(ids), accounts, relations, full,
            torch.tensor(profile_account, dtype=torch.long),
            torch.tensor(profile_field, dtype=torch.long),
            torch.tensor(np.asarray(profile_descriptor, dtype=np.float32).reshape(-1, 3)),
            torch.tensor(np.asarray(profile_surrogate), dtype=torch.float32),
            tuple(profile_offsets),
            torch.tensor(tweet_account, dtype=torch.long),
            torch.tensor(np.asarray(tweet_descriptor, dtype=np.float32).reshape(-1, 11)),
            torch.tensor(np.asarray(tweet_surrogate), dtype=torch.float32),
            tuple(tweet_offsets),
            torch.tensor(relation_source, dtype=torch.long),
            torch.tensor(relation_target, dtype=torch.long),
            torch.tensor(relation_kind, dtype=torch.long),
            torch.tensor(np.asarray(relation_descriptor, dtype=np.float32).reshape(-1, 1)),
            relation_order, graph,
        )

    def exact_view(self, profile_gate: torch.Tensor, tweet_gate: torch.Tensor) -> np.ndarray:
        """Rebuild all account facts from one class's hard source decisions."""
        profile = profile_gate.detach().cpu().numpy().astype(bool)
        tweet = tweet_gate.detach().cpu().numpy().astype(bool)
        if len(profile) != len(self.profile_account) or len(tweet) != len(self.tweet_account):
            raise ValueError("source gate size differs")
        rows = []
        for i, account in enumerate(self.accounts):
            p0, p1 = self.profile_offsets[i:i + 2]
            t0, t1 = self.tweet_offsets[i:i + 2]
            pfields = [field for field, selected in
                       zip(account.profile_field_ids, profile[p0:p1]) if selected]
            ordinals = [ordinal for ordinal, selected in
                        zip(account.tweet_ordinals, tweet[t0:t1]) if selected]
            rows.append(np.concatenate((
                account.features(profile_fields=pfields, tweet_ordinals=ordinals),
                np.asarray(list(account.extended_features(ordinals).values()),
                           dtype=np.float32))))
        return np.asarray(rows, dtype=np.float32)

    def relation_weights(self, relation_gate: torch.Tensor) -> dict[str, torch.Tensor]:
        if len(relation_gate) != len(self.relations):
            raise ValueError("relation gate size differs")
        return {name: relation_gate[order.to(relation_gate.device)]
                for name, order in self.relation_order.items()}

"""Exact two-layer HGT context dependencies for source-selection nodes."""

from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import torch

from .fox8_two_stage_inputs import Fox8TwoStageInputs


def hgt_context_closure(
    batch: Fox8TwoStageInputs,
    edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    seed_account_ids: list[str],
) -> tuple[dict, dict[tuple[str, str, str], torch.Tensor], set[int]]:
    """Keep precisely the incoming edges used by two HGT layers at seeds.

    Layer two needs layer-one states at seeds and their direct predecessors.
    Layer one therefore needs original inputs at those nodes and their own
    direct predecessors. Residual additions retain each node's own input.
    """
    index = {uid: i for i, uid in enumerate(batch.ids)}
    seeds = {index[uid] for uid in seed_account_ids}
    if not seeds:
        raise ValueError("HGT context needs at least one account")
    predecessor = set()
    for edges in edge_index_dict.values():
        predecessor.update(int(source) for source, destination in edges.T.tolist()
                           if destination in seeds)
    first_layer_targets = seeds | predecessor
    closure_nodes = set(first_layer_targets)
    restricted_graph = {}
    directed_edges = set()
    for key, edges in edge_index_dict.items():
        mask = torch.tensor([int(destination) in first_layer_targets
                             for destination in edges[1].tolist()],
                            dtype=torch.bool)
        kept = edges[:, mask].contiguous()
        restricted_graph[key] = kept
        for source, destination in kept.T.tolist():
            closure_nodes.add(int(source))
            directed_edges.add((key[1], int(source), int(destination)))
    relation_ids = set()
    source_lookup = {uid: i for i, uid in enumerate(batch.ids)}
    for atom in batch.relations:
        source = source_lookup[atom.source_account_id]
        target = source_lookup[atom.target_account_id]
        if ((atom.relation, source, target) in directed_edges or
                (f"rev_{atom.relation}", target, source) in directed_edges):
            relation_ids.add(atom.source_id)
    represented = set()
    for atom in batch.relations:
        if atom.source_id not in relation_ids:
            continue
        source = source_lookup[atom.source_account_id]
        target = source_lookup[atom.target_account_id]
        represented.add((atom.relation, source, target))
        represented.add((f"rev_{atom.relation}", target, source))
    if directed_edges - represented:
        raise ValueError("HGT closure edge has no native relation source")
    # A deterministic digest binds every raw profile value, tweet record,
    # and native relation locator in the closed context to this inventory.
    digest = sha256()
    profile_sources = 0
    tweet_sources = 0
    for i in sorted(closure_nodes):
        account = batch.accounts[i]
        digest.update(f"account:{account.account_id}:{account.line_number}\n".encode())
        for field in account.profile_field_ids:
            value = json.dumps(account.profile[field], ensure_ascii=False,
                               sort_keys=True, separators=(",", ":"))
            digest.update(f"{account.source_id('profile', field)}:{value}\n".encode())
            profile_sources += 1
        for atom in account.tweets:
            digest.update(f"{account.source_id('tweet', atom.ordinal)}:"
                          f"{atom.record_sha256}\n".encode())
            tweet_sources += 1
    for atom in sorted(batch.relations, key=lambda item: item.source_id):
        if atom.source_id in relation_ids:
            digest.update(f"{atom.source_id}:{atom.raw_locators}\n".encode())
    report = {
        "hgt_layers": 2,
        "seed_accounts": sorted(seed_account_ids),
        "direct_predecessor_account_count": len(predecessor),
        "context_account_count": len(closure_nodes),
        "context_account_ids": [batch.ids[i] for i in sorted(closure_nodes)],
        "raw_profile_field_source_count": profile_sources,
        "raw_tweet_source_count": tweet_sources,
        "native_relation_source_count": len(relation_ids),
        "native_relation_source_ids": sorted(relation_ids),
        "directed_hgt_edge_count": len(directed_edges),
        "raw_source_inventory_sha256": digest.hexdigest(),
    }
    return report, restricted_graph, closure_nodes


def verify_hgt_context_locality(
    hgt: torch.nn.Module, x: torch.Tensor,
    full_graph: dict[tuple[str, str, str], torch.Tensor],
    restricted_graph: dict[tuple[str, str, str], torch.Tensor],
    closure_nodes: set[int], seed_account_ids: list[str],
    account_ids: tuple[str, ...],
) -> float:
    """Re-run frozen HGT with every outside node and edge removed."""
    seed_indices = torch.tensor([account_ids.index(uid) for uid in seed_account_ids],
                                dtype=torch.long, device=x.device)
    local_x = torch.zeros_like(x)
    keep = torch.tensor(sorted(closure_nodes), dtype=torch.long, device=x.device)
    local_x[keep] = x[keep]
    full_graph = {key: edges.to(x.device) for key, edges in full_graph.items()}
    restricted_graph = {key: edges.to(x.device)
                        for key, edges in restricted_graph.items()}
    hgt.eval()
    with torch.inference_mode():
        reference = hgt({"user": x}, full_graph)["user_emb"][seed_indices]
        isolated = hgt({"user": local_x}, restricted_graph)[
            "user_emb"][seed_indices]
    difference = float((reference - isolated).abs().max())
    if not np.isfinite(difference) or difference > 1e-5:
        raise ValueError(f"HGT context depends on an omitted source: {difference}")
    return difference

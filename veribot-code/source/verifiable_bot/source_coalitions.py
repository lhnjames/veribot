"""Budgeted group search and legacy exact queries for a frozen oracle."""

from __future__ import annotations

from itertools import combinations


def greedy_bot_certificate(*, account_id: str,
                           groups: dict[str, frozenset[str]],
                           candidate_sources: frozenset[str],
                           threshold: float, margin_buffer: float,
                           oracle, query_budget: int = 60) -> dict:
    """Remove source groups while a fixed Bot verifier keeps both responses.

    A retained coalition must score above the Bot threshold, and removing
    that same coalition from the original candidate must score below it.
    Each step considers one group deletion from the current coalition and
    keeps the largest valid source reduction. This is local, budgeted search.
    """
    if margin_buffer < 0 or query_budget < 3 or not groups:
        raise ValueError("Invalid Bot certificate search settings")
    names = tuple(groups)
    listed = [source for group in groups.values() for source in group]
    if len(listed) != len(set(listed)) or frozenset(listed) != candidate_sources:
        raise ValueError("Bot certificate groups must partition candidates")
    universe = frozenset(oracle.lookup)
    if not candidate_sources <= universe:
        raise ValueError("Candidate source is absent from the raw universe")
    cache: dict[tuple[str, frozenset[str]], float] = {}
    queries = []

    def score(removed: frozenset[str], purpose: str) -> float | None:
        key = (account_id, removed)
        if key not in cache:
            if len(cache) >= query_budget:
                return None
            cache[key] = float(oracle.predict(account_id, removed)["margin"])
            queries.append({"purpose": purpose, "removed_count": len(removed),
                            "margin": cache[key]})
        return cache[key]

    baseline = score(frozenset(), "full_candidate")
    empty = score(candidate_sources, "no_candidate")
    candidate_only = score(universe - candidate_sources, "candidate_only")
    assert baseline is not None and empty is not None and candidate_only is not None
    if abs(candidate_only - baseline) > 1e-4:
        raise ValueError("Fixed verifier reads raw sources outside the candidates")
    current = frozenset(names)
    history = []
    exhausted = False
    if baseline >= threshold + margin_buffer and empty < threshold - margin_buffer:
        while len(current) > 1:
            options = []
            for name in sorted(current):
                trial = current - {name}
                retained = frozenset().union(*(groups[item] for item in trial))
                keep = score(candidate_sources - retained,
                             "retain_without_one_group")
                remove = score(retained, "remove_retained_coalition")
                if keep is None or remove is None:
                    exhausted = True
                    break
                if (keep >= threshold + margin_buffer and
                        remove < threshold - margin_buffer):
                    options.append((len(retained), -keep, name, trial,
                                    keep, remove))
            if not options:
                break
            _, _, deleted, current, keep, remove = min(options)
            history.append({"deleted_group": deleted,
                            "retained_source_count": sum(
                                len(groups[item]) for item in current),
                            "keep_margin": keep,
                            "remove_margin": remove})
            if exhausted:
                break
    retained = frozenset().union(*(groups[item] for item in current))
    final_keep = score(candidate_sources - retained, "final_keep")
    final_remove = score(retained, "final_remove")
    return {
        "account_id": account_id, "method": "budgeted_backward_group_greedy",
        "threshold": threshold, "margin_buffer": margin_buffer,
        "baseline_margin": baseline, "empty_margin": empty,
        "candidate_source_count": len(candidate_sources),
        "retained_groups": sorted(current),
        "retained_source_ids": sorted(retained),
        "retained_source_count": len(retained),
        "retained_only_margin": final_keep,
        "full_without_retained_margin": final_remove,
        "passes_bidirectional": (final_keep is not None and
                                 final_remove is not None and
                                 final_keep >= threshold + margin_buffer and
                                 final_remove < threshold - margin_buffer),
        "query_budget": query_budget,
        "query_count": len(cache), "budget_exhausted": exhausted,
        "deletion_history": history,
        "query_log": queries,
    }


def search_source_coalitions(*, account_id: str,
                             groups: dict[str, frozenset[str]],
                             full_dependency: frozenset[str],
                             threshold: float, oracle,
                             query_budget: int = 40,
                             beam_width: int = 3) -> dict:
    """Rank source groups and search coalitions with a strict oracle budget.

    This routine never claims exhaustive minimality. It measures singleton
    retention, leave-one-group-out loss, and interactions only for pairs the
    beam visits. It distinguishes class-flipping coalitions from directional
    score support, since a shared empty input has only one default class.
    """
    names = tuple(groups)
    if not names or beam_width < 1 or query_budget < 3 + 2 * len(names):
        raise ValueError("Need nonempty groups and a budget for ranking queries")
    seen: set[str] = set()
    for name in names:
        if seen.intersection(groups[name]):
            raise ValueError("source groups overlap")
        seen.update(groups[name])
    if seen != full_dependency:
        raise ValueError("source groups do not partition the dependency closure")

    all_mask = (1 << len(names)) - 1
    cache: dict[int, dict] = {}
    order: list[int] = []

    def ask(mask: int, purpose: str) -> dict | None:
        if mask in cache:
            return cache[mask]
        if len(order) >= query_budget - 1:  # reserve a full-only closure check
            return None
        retained = frozenset().union(*(
            groups[name] for j, name in enumerate(names) if mask & (1 << j)))
        margin = oracle.predict(account_id, full_dependency - retained)["margin"]
        row = {"mask": mask,
               "groups": [name for j, name in enumerate(names) if mask & (1 << j)],
               "source_count": len(retained), "margin": margin,
               "query_purpose": purpose}
        cache[mask] = row
        order.append(mask)
        return row

    full = ask(all_mask, "full_input")
    assert full is not None
    baseline = full["margin"]
    original_bot = baseline >= threshold
    direction = 1 if original_bot else -1

    def signed(mask: int) -> float:
        return direction * (cache[mask]["margin"] - threshold)

    empty = ask(0, "empty_input")
    assert empty is not None
    for j in range(len(names)):
        if ask(1 << j, "singleton_retention") is None:
            raise ValueError("Query budget exhausted before singleton ranking")
        if ask(all_mask ^ (1 << j), "leave_one_group_out") is None:
            raise ValueError("Query budget exhausted before full-context ranking")
    full_only = oracle.predict(
        account_id, frozenset(oracle.lookup) - full_dependency)["margin"]
    if abs(full_only - baseline) > 1e-4:
        raise ValueError("source graph is not a complete prediction dependency")

    ranked_groups = []
    for j, name in enumerate(names):
        bit = 1 << j
        ranked_groups.append({
            "group": name,
            "source_count": len(groups[name]),
            "singleton_margin": cache[bit]["margin"],
            "signed_gain_over_empty": signed(bit) - signed(0),
            "full_without_group_margin": cache[all_mask ^ bit]["margin"],
            "signed_loss_when_removed_from_full":
                signed(all_mask) - signed(all_mask ^ bit),
        })
    by_singleton = sorted(ranked_groups,
                          key=lambda x: (-x["signed_gain_over_empty"],
                                         x["source_count"], x["group"]))
    by_full_loss = sorted(ranked_groups,
                          key=lambda x: (-x["signed_loss_when_removed_from_full"],
                                         x["source_count"], x["group"]))

    sufficient_candidates: dict[int, dict] = {}
    if signed(0) >= 0:
        sufficient_candidates[0] = {"groups": [], "source_count": 0,
                                    "retained_margin": cache[0]["margin"]}
    successes: dict[int, dict] = {}
    conditional_full_context_sets: dict[int, dict] = {}
    directional_support_sets: dict[int, dict] = {}

    def consider(mask: int) -> None:
        if signed(mask) < 0 or mask == 0:
            return
        sufficient_candidates[mask] = {
            "groups": cache[mask]["groups"],
            "source_count": cache[mask]["source_count"],
            "retained_margin": cache[mask]["margin"],
        }
        removed_row = ask(all_mask ^ mask, "remove_candidate_from_full")
        if removed_row is None:
            return
        gain_over_empty = signed(mask) - signed(0)
        loss_when_removed = signed(all_mask) - signed(all_mask ^ mask)
        if gain_over_empty > 1e-6 and loss_when_removed > 1e-6:
            directional_support_sets[mask] = {
                "groups": cache[mask]["groups"],
                "source_count": cache[mask]["source_count"],
                "retained_margin": cache[mask]["margin"],
                "full_without_candidate_margin": removed_row["margin"],
                "signed_gain_over_empty": gain_over_empty,
                "signed_loss_when_removed_from_full": loss_when_removed,
            }
        if signed(all_mask ^ mask) < 0:
            hit = {"groups": cache[mask]["groups"],
                   "source_count": cache[mask]["source_count"],
                   "retained_margin": cache[mask]["margin"],
                   "full_without_candidate_margin": removed_row["margin"]}
            conditional_full_context_sets[mask] = hit
            if signed(0) < 0:
                successes[mask] = hit

    for j in range(len(names)):
        consider(1 << j)
    # A singleton can look weak in sparse input but have a large effect when
    # removed from the full input. Seed the search from both rankings so that
    # either context can nominate a group for interaction tests.
    singleton_seeds = [1 << names.index(row["group"]) for row in by_singleton
                       if (1 << names.index(row["group"])) not in successes]
    full_context_seeds = [1 << names.index(row["group"]) for row in by_full_loss
                          if (1 << names.index(row["group"])) not in successes]
    if not singleton_seeds:
        singleton_seeds = [1 << j for j in range(len(names))]
    frontier = []
    for mask in singleton_seeds[:max(1, beam_width // 2)]:
        if mask not in frontier:
            frontier.append(mask)
    for ranked in (full_context_seeds, singleton_seeds):
        for mask in ranked:
            if len(frontier) >= beam_width:
                break
            if mask not in frontier:
                frontier.append(mask)
    for depth in range(2, len(names) + 1):
        next_masks = set()
        stopped = False
        for mask in frontier:
            for j in range(len(names)):
                bit = 1 << j
                if mask & bit:
                    continue
                candidate = mask | bit
                row = ask(candidate, "beam_expand")
                if row is None:
                    stopped = True
                    break
                next_masks.add(candidate)
                consider(candidate)
            if stopped:
                break
        if not next_masks or stopped:
            break
        frontier = sorted(next_masks,
                          key=lambda mask: (-signed(mask),
                                            cache[mask]["source_count"], mask))[:beam_width]

    pairs = []
    for i, j in combinations(range(len(names)), 2):
        pair = (1 << i) | (1 << j)
        if pair in cache:
            pairs.append({"first": names[i], "second": names[j],
                          "signed_pair_interaction":
                              signed(pair) - signed(1 << i) -
                              signed(1 << j) + signed(0),
                          "joint_preserves_original_class": signed(pair) >= 0})
    solutions = sorted(successes.values(),
                       key=lambda x: (x["source_count"], len(x["groups"]),
                                      -direction * x["retained_margin"],
                                      x["groups"]))
    observed_sufficient = [
        value for mask, value in sufficient_candidates.items()
        if not any(other != mask and (other & mask) == other
                   for other in sufficient_candidates)]
    observed_sufficient.sort(key=lambda x: (x["source_count"],
                                             len(x["groups"]), x["groups"]))
    for row in cache.values():
        row["signed_margin_from_threshold"] = direction * (row["margin"] - threshold)
        row["preserves_original_class"] = (
            (row["margin"] >= threshold) == original_bot)
    return {
        "account_id": account_id, "baseline_margin": baseline,
        "full_only_margin": full_only, "threshold": threshold,
        "predicted_bot": original_bot, "group_names": names,
        "group_source_ids": {name: sorted(groups[name]) for name in names},
        "source_universe_count": len(oracle.lookup),
        "full_dependency_source_count": len(full_dependency),
        "search_method": "budgeted_beam_with_singleton_and_full_context_ranking",
        "query_budget": query_budget, "oracle_query_count": len(order) + 1,
        "beam_width": beam_width, "query_budget_exhausted":
            len(order) + 1 >= query_budget,
        "ranked_by_singleton_gain": [row["group"] for row in by_singleton],
        "ranked_by_full_context_removal_loss":
            [row["group"] for row in by_full_loss],
        "ranked_group_effects": ranked_groups,
        "coalitions": [cache[mask] for mask in order],
        "observed_nondominated_sufficient_group_sets": observed_sufficient,
        "searched_bidirectional_group_sets": solutions,
        "conditional_full_context_group_sets": sorted(
            conditional_full_context_sets.values(),
            key=lambda x: (x["source_count"], len(x["groups"]), x["groups"])),
        "directional_support_group_sets": sorted(
            directional_support_sets.values(),
            key=lambda x: (x["source_count"], len(x["groups"]),
                           -x["signed_gain_over_empty"], x["groups"])),
        "pair_interactions": pairs,
        "empty_input_preserves_original_class": signed(0) >= 0,
        "full_signed_gain_over_empty": signed(all_mask) - signed(0),
        "empty_input_fails_original_class": signed(0) < 0,
    }


def enumerate_source_coalitions(*, account_id: str,
                                groups: dict[str, frozenset[str]],
                                full_dependency: frozenset[str],
                                threshold: float, oracle) -> dict:
    """Evaluate every subset of a partition of the account's source graph.

    Each query withholds the complement of the retained groups and reruns the
    full detector. Sources outside the verified dependency closure stay present;
    their irrelevance is checked by comparing the all-group and full-only scores.
    """
    names = tuple(groups)
    if not names or len(names) > 20:
        raise ValueError("exact coalition enumeration needs 1..20 groups")
    seen: set[str] = set()
    for name in names:
        if seen.intersection(groups[name]):
            raise ValueError("source groups overlap")
        seen.update(groups[name])
    if seen != full_dependency:
        raise ValueError("source groups do not partition the dependency closure")
    baseline = oracle.predict(account_id)["margin"]
    original_bot = baseline >= threshold
    direction = 1 if original_bot else -1
    all_mask = (1 << len(names)) - 1
    rows = []
    signed = {}
    for mask in range(all_mask + 1):
        retained = frozenset().union(*(
            groups[name] for j, name in enumerate(names) if mask & (1 << j)))
        # Outside the complete dependency closure remains available. A
        # separate full-only query below checks that this does not matter.
        margin = oracle.predict(account_id, full_dependency - retained)["margin"]
        signed_margin = direction * (margin - threshold)
        signed[mask] = signed_margin
        rows.append({
            "mask": mask,
            "groups": [name for j, name in enumerate(names) if mask & (1 << j)],
            "source_count": len(retained),
            "margin": margin,
            "signed_margin_from_threshold": signed_margin,
            "preserves_original_class": (margin >= threshold) == original_bot,
        })
    full_only = oracle.predict(
        account_id, frozenset(oracle.lookup) - full_dependency)["margin"]
    if abs(full_only - baseline) > 1e-4 or abs(signed[all_mask] -
                                               direction * (baseline - threshold)) > 1e-4:
        raise ValueError("source graph is not a complete prediction dependency")
    sufficient = {row["mask"] for row in rows if row["preserves_original_class"]}
    minimal = [mask for mask in sufficient
               if not any((other & mask) == other and other != mask
                          for other in sufficient)]
    pair_interactions = []
    for i, j in combinations(range(len(names)), 2):
        a, b = 1 << i, 1 << j
        pair_interactions.append({
            "first": names[i], "second": names[j],
            "signed_pair_interaction":
                signed[a | b] - signed[a] - signed[b] + signed[0],
            "joint_preserves_original_class": (a | b) in sufficient,
        })
    return {
        "account_id": account_id,
        "baseline_margin": baseline,
        "full_only_margin": full_only,
        "threshold": threshold,
        "predicted_bot": original_bot,
        "group_names": names,
        "group_source_ids": {name: sorted(groups[name]) for name in names},
        "source_universe_count": len(oracle.lookup),
        "full_dependency_source_count": len(full_dependency),
        "coalitions": rows,
        "minimal_sufficient_group_sets": [
            [names[j] for j in range(len(names)) if mask & (1 << j)]
            for mask in minimal],
        "pair_interactions": pair_interactions,
        "empty_input_preserves_original_class": 0 in sufficient,
    }

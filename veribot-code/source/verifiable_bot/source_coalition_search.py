"""Find source coalitions that change a frozen binary model's decision.

The caller supplies source-consistent reconstruction and prediction. This
module never substitutes feature masking for editing the original records.
"""

from __future__ import annotations

from itertools import combinations
import heapq
from math import comb
from typing import Callable, Hashable, Sequence


Source = Hashable
Score = Callable[[frozenset[Source]], float]


def find_minimal_flip_groups(
    source_ids: Sequence[Source],
    score_after_removal: Score,
    threshold: float,
    *,
    initial_removed: frozenset[Source] = frozenset(),
    max_evaluations: int = 10000,
) -> dict:
    """Search all source groups in increasing size until a class flip.

    If the evaluation budget expires, completeness flags state exactly what
    has and has not been established. ``source_ids`` is the declared search
    universe; a minimality claim never extends beyond that universe.
    """
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be positive")
    ids = tuple(source_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("source IDs must be unique")
    if any(source in initial_removed for source in ids):
        raise ValueError("search IDs must be present in the initial context")

    cache: dict[frozenset[Source], float] = {}

    def evaluate(removed: frozenset[Source]) -> float:
        if removed not in cache:
            if len(cache) >= max_evaluations:
                raise _BudgetReached
            cache[removed] = float(score_after_removal(removed))
        return cache[removed]

    baseline = evaluate(initial_removed)
    baseline_bot = baseline >= threshold
    tested_by_size: list[dict] = []
    minimal_groups: list[dict] = []
    minimum_cardinality: int | None = None
    minimum_cardinality_proven = False
    all_minimum_groups_found = False
    budget_exhausted = False

    for size in range(1, len(ids) + 1):
        tested = 0
        total = comb(len(ids), size)
        for group in combinations(ids, size):
            try:
                logit = evaluate(initial_removed | frozenset(group))
            except _BudgetReached:
                budget_exhausted = True
                break
            tested += 1
            if (logit >= threshold) != baseline_bot:
                minimal_groups.append({
                    "sources": list(group),
                    "after_removal_logit": logit,
                    "after_removal_bot": logit >= threshold,
                    "signed_effect_toward_original_class": (
                        (baseline - logit) if baseline_bot else (logit - baseline)
                    ),
                })
        tested_by_size.append({"size": size, "tested": tested, "total": total})
        if minimal_groups:
            minimum_cardinality = size
            minimum_cardinality_proven = True
            all_minimum_groups_found = tested == total
            break
        if budget_exhausted:
            break

    return {
        "search_universe": list(ids),
        "initial_removed": list(initial_removed),
        "baseline_logit": baseline,
        "baseline_bot": baseline_bot,
        "threshold": threshold,
        "minimum_cardinality": minimum_cardinality,
        "minimum_cardinality_proven": minimum_cardinality_proven,
        "all_minimum_groups_found": all_minimum_groups_found,
        "no_flip_in_search_universe_proven": (
            minimum_cardinality is None and not budget_exhausted),
        "budget_exhausted": budget_exhausted,
        "tested_by_size": tested_by_size,
        "minimal_groups": minimal_groups,
        "unique_evaluations": len(cache),
    }


def check_forward_contexts(
    group: Sequence[Source],
    other_sources: Sequence[Source],
    score_after_removal: Score,
    threshold: float,
    *,
    target_bot: bool,
    initial_removed: frozenset[Source] = frozenset(),
) -> dict:
    """Add a group back in each predeclared one-source neighboring context.

    Every comparison uses two distinct reconstructed inputs. The context is
    ``R \\ (group ∪ {other})``; the paired input adds ``group`` back.
    """
    chosen = frozenset(group)
    if not chosen:
        raise ValueError("group must not be empty")
    if len(set(other_sources)) != len(other_sources):
        raise ValueError("context source IDs must be unique")
    if any(source in chosen or source in initial_removed for source in other_sources):
        raise ValueError("context sources must be outside group and initial removals")

    contexts = []
    for other in other_sources:
        without = float(score_after_removal(initial_removed | chosen | {other}))
        with_group = float(score_after_removal(initial_removed | {other}))
        without_bot = without >= threshold
        with_bot = with_group >= threshold
        contexts.append({
            "other_removed_source": other,
            "without_group_logit": without,
            "with_group_logit": with_group,
            "without_group_bot": without_bot,
            "with_group_bot": with_bot,
            "signed_addition_effect_toward_target": (
                (with_group - without) if target_bot else (without - with_group)
            ),
            "addition_changes_to_target": with_bot == target_bot and without_bot != target_bot,
        })
    return {
        "group": list(group),
        "target_bot": target_bot,
        "context_rule": "remove one other source; compare without group to adding group back",
        "contexts": contexts,
        "class_change_count": sum(c["addition_changes_to_target"] for c in contexts),
        "context_count": len(contexts),
    }


class _BudgetReached(Exception):
    pass


def find_budgeted_flip_groups(
    source_ids: Sequence[Source], score_after_removal: Score,
    threshold: float, *, max_evaluations: int = 200,
) -> dict:
    """Best-first source-group search with exact oracle checks and honest scope.

    This search does not claim a global minimum unless the complete smaller
    subset space was checked. It is suited to a large source pool where exact
    enumeration would exceed the declared forward-pass budget.
    """
    ids = tuple(source_ids)
    if len(set(ids)) != len(ids) or max_evaluations < 1:
        raise ValueError("unique sources and a positive evaluation budget are required")
    cache: dict[frozenset[Source], float] = {}

    def evaluate(group: frozenset[Source]) -> float:
        if group not in cache:
            if len(cache) >= max_evaluations:
                raise _BudgetReached
            cache[group] = float(score_after_removal(group))
        return cache[group]

    baseline = evaluate(frozenset())
    baseline_bot = baseline >= threshold
    heap = []
    serial = 0
    seen = set()
    single_effects = []
    solutions = []
    budget_exhausted = False
    all_singles_checked = True

    def flipped(value: float) -> bool:
        return (value >= threshold) != baseline_bot

    def push(group: frozenset[Source], value: float) -> None:
        nonlocal serial
        serial += 1
        priority = value if baseline_bot else -value
        heapq.heappush(heap, (priority, len(group), serial, group))

    for source in ids:
        group = frozenset({source})
        try:
            value = evaluate(group)
        except _BudgetReached:
            all_singles_checked = False
            budget_exhausted = True
            break
        seen.add(group)
        single_effects.append({"source": source, "after_removal_logit": value,
                               "delta_logit": value - baseline,
                               "flips_class": flipped(value)})
        push(group, value)

    while heap and not budget_exhausted:
        _, _, _, group = heapq.heappop(heap)
        value = cache[group]
        if any(solution["sources_set"] <= group for solution in solutions):
            continue
        if flipped(value):
            reduced = group
            local_minimality_checked = True
            changed = True
            while changed:
                changed = False
                for source in tuple(reduced):
                    trial = reduced - {source}
                    try:
                        trial_value = evaluate(trial)
                    except _BudgetReached:
                        local_minimality_checked = False
                        budget_exhausted = True
                        break
                    if flipped(trial_value):
                        reduced = trial
                        changed = True
                        break
                if budget_exhausted:
                    break
            if not any(item["sources_set"] == reduced for item in solutions):
                solutions.append({"sources_set": reduced,
                                  "after_removal_logit": cache[reduced],
                                  "locally_minimal_checked": local_minimality_checked})
            continue
        for source in ids:
            if source in group:
                continue
            expanded = group | {source}
            if expanded in seen or any(item["sources_set"] <= expanded
                                       for item in solutions):
                continue
            seen.add(expanded)
            try:
                expanded_value = evaluate(expanded)
            except _BudgetReached:
                budget_exhausted = True
                break
            push(expanded, expanded_value)

    return {
        "search_universe": list(ids), "baseline_logit": baseline,
        "baseline_bot": baseline_bot, "threshold": threshold,
        "single_effects": single_effects,
        "all_singles_checked": all_singles_checked,
        "solutions": [{"sources": [item for item in ids if item in row["sources_set"]],
                       "after_removal_logit": row["after_removal_logit"],
                       "locally_minimal_checked": row["locally_minimal_checked"]}
                      for row in solutions],
        "global_minimum_proven": (all_singles_checked and
                                  any(len(row["sources_set"]) == 1
                                      for row in solutions)),
        "budget_exhausted": budget_exhausted,
        "unique_evaluations": len(cache),
    }

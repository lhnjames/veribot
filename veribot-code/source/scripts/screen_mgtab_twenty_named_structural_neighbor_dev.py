"""Twenty named MGTAB concepts that separate event structure from neighbors."""

from __future__ import annotations

from pathlib import Path

from screen_mgtab_twenty_named_bounded_linear_residual_dev import ROOT, main
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


OUT = ROOT / "output/v3_mgtab_twenty_named_structural_neighbor/dev_only.json"


def fixed_groups(runtime: MGTABFullAtomicRuntime) -> dict[str, list[str]]:
    names = runtime.names
    groups = {
        "profile_audience": [names[j] for j in (3, 5, 7, 14)],
        "profile_activity_presentation": [names[j] for j in range(20)
                                          if j not in (3, 5, 7, 14)],
    }
    relations = [runtime.group_fact_indices[f"published_relation_{j}"]
                 for j in range(7)]
    for label, j in (("followers", 0), ("friends", 1), ("reply", 3)):
        structural = [i for i in relations[j] if "_neighbor_" not in names[i]]
        neighbor = [i for i in relations[j] if "_neighbor_" in names[i]]
        groups[f"{label}_event_structure"] = [names[i] for i in structural]
        groups[f"{label}_neighbor_profile"] = [names[i] for i in neighbor]
    groups["mention_quoted_events"] = [names[i] for j in (2, 4)
                                       for i in relations[j]]
    groups["url_hashtag_events"] = [names[i] for j in (5, 6)
                                    for i in relations[j]]
    used = [name for members in groups.values() for name in members]
    if (len(groups) != 10 or any(not members for members in groups.values())
            or len(used) != 86 or len(set(used)) != 86
            or set(used) != set(names[:86])):
        raise ValueError("Structural-neighbor named groups do not partition facts")
    return groups


if __name__ == "__main__":
    main(fixed_groups, OUT, "profile_two_structural_neighbor_six_other_events_two")

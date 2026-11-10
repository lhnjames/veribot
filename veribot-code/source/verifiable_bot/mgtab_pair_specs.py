"""Fixed, interpretable within-group evidence conjunctions for MGTAB."""

from __future__ import annotations

from .mgtab_source_certificate import RELATIONS


PROFILE_DIMENSION_PAIRS = (
    (1, 4),    # default profile + default image
    (1, 15),   # default profile + default background color
    (15, 16),  # background and sidebar fill defaults
    (16, 17),  # sidebar fill and border defaults
    (2, 18),   # verified + profile URL
    (3, 5),    # follower count + list count
    (3, 14),   # follower count + follower/friend ratio
    (6, 10),   # status count + creation time
    (11, 12),  # screen name and name lengths
    (13, 18),  # description length + profile URL
    (0, 19),   # background image use + image URL
)


def fixed_pairs(names: list[str]) -> tuple[tuple[str, str], ...]:
    if len(names) != 62:
        raise ValueError("MGTAB pair specification requires 62 fixed facts")
    result = [(f"published_profile_coordinate_{a:02d}",
               f"published_profile_coordinate_{b:02d}")
              for a, b in PROFILE_DIMENSION_PAIRS]
    for relation in RELATIONS:
        result.append((f"published_{relation}_out_degree_log1p",
                       f"published_{relation}_in_degree_log1p"))
        result.append((f"published_{relation}_nonself_out_unique_log1p",
                       f"published_{relation}_reciprocal_unique_log1p"))
        result.append((f"published_{relation}_nonself_in_unique_log1p",
                       f"published_{relation}_reciprocal_unique_log1p"))
    if any(a not in names or b not in names for a, b in result):
        raise ValueError("fixed pair feature missing from MGTAB evidence vocabulary")
    return tuple(result)

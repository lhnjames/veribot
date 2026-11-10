"""MGTAB published profile dimensions from the original paper's Table A5.

The table gives 1-based dimensions. The released features.pt uses zero-based
columns here. Values remain released normalized features, not raw records.
"""

from __future__ import annotations


PAPER_URL = "https://arxiv.org/pdf/2301.01123"
STANDARDIZATION_URL = (
    "https://github.com/GraphDetec/MGTAB/blob/main/Standardization/"
    "Minimum%20and%20Maximum%20value")

PROFILE_DIMENSIONS = (
    ("profile_uses_background_image", "boolean"),
    ("default_profile", "boolean"),
    ("verified", "boolean"),
    ("followers_count", "normalized_numerical"),
    ("default_profile_image", "boolean"),
    ("listed_count", "normalized_numerical"),
    ("statuses_count", "normalized_numerical"),
    ("friends_count", "normalized_numerical"),
    ("geo_enabled", "boolean"),
    ("favourites_count", "normalized_numerical"),
    ("created_at", "normalized_numerical"),
    ("screen_name_length", "normalized_numerical"),
    ("name_length", "normalized_numerical"),
    ("description_length", "normalized_numerical"),
    ("followers_friends_ratio", "normalized_numerical"),
    ("default_profile_background_color", "boolean"),
    ("default_profile_sidebar_fill_color", "boolean"),
    ("default_profile_sidebar_border_color", "boolean"),
    ("has_url", "boolean"),
    ("profile_background_image_url", "boolean"),
)


def published_profile_description(fact_name: str) -> dict | None:
    prefix = "published_profile_coordinate_"
    if not fact_name.startswith(prefix):
        return None
    suffix = fact_name[len(prefix):]
    if not suffix.isdigit() or int(suffix) >= len(PROFILE_DIMENSIONS):
        raise ValueError("unknown MGTAB published profile coordinate")
    index = int(suffix)
    name, value_type = PROFILE_DIMENSIONS[index]
    return {
        "published_zero_based_column": index,
        "paper_table_a5_one_based_dimension": index + 1,
        "paper_feature_name": name,
        "released_value_type": value_type,
        "source_level": "published_normalized_profile_feature",
        "feature_dictionary_reference": PAPER_URL,
    }

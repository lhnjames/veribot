"""Check released profile column types against MGTAB Table A5's schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.mgtab_profile_dictionary import (
    PAPER_URL, PROFILE_DIMENSIONS, STANDARDIZATION_URL)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB/features.pt")
    audit_path = root / "research/certificates/mgtab_full_source_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    digest = sha256_file(source)
    if audit["source_sha256"]["features.pt"] != digest:
        raise ValueError("released feature tensor differs from source audit")
    tensor = torch.load(source, map_location="cpu", weights_only=True)
    values = np.asarray(tensor)
    if values.shape != (10199, 788) or len(PROFILE_DIMENSIONS) != 20:
        raise ValueError("released MGTAB dimension differs from Table A5")
    columns = []
    for index, (name, value_type) in enumerate(PROFILE_DIMENSIONS):
        column = values[:, index]
        finite = column[np.isfinite(column)]
        unique = np.unique(finite)
        if not len(finite) or (value_type == "boolean" and
                              not np.isin(unique, [0.0, 1.0]).all()) or (
                                  value_type != "boolean" and len(unique) <= 2):
            raise ValueError(f"published column {index} disagrees with paper type")
        columns.append({"zero_based_column": index,
                        "table_a5_dimension": index + 1,
                        "paper_feature_name": name,
                        "paper_type": value_type,
                        "released_unique_value_count": len(unique),
                        "released_min": float(unique.min()),
                        "released_max": float(unique.max())})
    integer_normalization = []
    for index, expected_name, minimum, maximum in (
            (11, "screen_name_length", 3, 15),
            (12, "name_length", 1, 50),
            (13, "description_length", 0, 204)):
        if PROFILE_DIMENSIONS[index][0] != expected_name:
            raise ValueError("published profile dictionary order differs")
        inverted = values[:, index] * (maximum - minimum) + minimum
        maximum_integer_difference = float(np.max(np.abs(inverted - np.rint(inverted))))
        if maximum_integer_difference > 1e-4:
            raise ValueError("released column disagrees with publisher normalization grid")
        integer_normalization.append({
            "zero_based_column": index, "paper_feature_name": expected_name,
            "publisher_minimum": minimum, "publisher_maximum": maximum,
            "max_distance_to_integer_after_inverse_scaling": maximum_integer_difference,
        })
    output = {
        "status": "published_profile_dimensions_match_table_a5_types",
        "source_level": "released_normalized_feature_tensor",
        "paper_table_a5": PAPER_URL,
        "publisher_normalization_metadata": STANDARDIZATION_URL,
        "features_sha256": digest,
        "prior_independent_source_audit_sha256": sha256_file(audit_path),
        "account_count": len(values), "total_feature_width": values.shape[1],
        "raw_profile_records_available": False,
        "claims_from_this_audit": "paper_dimension_names_released_column_types_and_three_normalization_grids_not_original_raw_records",
        "integer_normalization_checks": integer_normalization,
        "columns": columns,
    }
    target = root / "research/certificates/mgtab_published_profile_dictionary_audit.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"status": output["status"],
                      "columns": len(columns),
                      "followers_friends_ratio": columns[14]}, indent=2))


if __name__ == "__main__":
    main()

"""Attach Table A5 meanings to all frozen MGTAB test cards without modifying them."""

from __future__ import annotations

import gzip
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.mgtab_profile_dictionary import published_profile_description


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    original_dir = root / "output/v3_mgtab_ab_spline_content/direct_selected_20"
    overlay_dir = root / "output/v3_mgtab_published_profile_semantics"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    dictionary_path = root / "research/certificates/mgtab_published_profile_dictionary_audit.json"
    dictionary = json.loads(dictionary_path.read_text(encoding="utf-8"))
    source = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB/features.pt")
    if (dictionary["features_sha256"] != sha256_file(source)
            or dictionary["status"] != "published_profile_dimensions_match_table_a5_types"):
        raise ValueError("released tensor and profile dictionary differ")
    values = np.asarray(torch.load(source, map_location="cpu", weights_only=True))
    runs = []
    for seed in (0, 1, 2):
        source_dir = original_dir / f"seed{seed}"
        report = json.loads((source_dir / "report.json").read_text(encoding="utf-8"))
        cards = source_dir / "evidence_cards.jsonl.gz"
        if report["cards_sha256"] != sha256_file(cards):
            raise ValueError("frozen card hash differs")
        target = overlay_dir / f"seed{seed}.jsonl.gz"
        count = 0
        maximum_difference = 0.0
        with gzip.open(cards, "rt", encoding="utf-8") as source_stream:
            with target.open("wb") as binary:
                with gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as zipped:
                    with io.TextIOWrapper(zipped, encoding="utf-8") as stream:
                        for line in source_stream:
                            card = json.loads(line)
                            profile = []
                            for term in card["terms"]:
                                semantics = published_profile_description(term["name"])
                                if semantics is None:
                                    continue
                                locator = term["source_locator"]
                                if (locator["file"] != "features.pt"
                                        or locator["row"] != int(card["account_id"])
                                        or locator["column"] != semantics[
                                            "published_zero_based_column"]):
                                    raise ValueError("profile card tensor location differs")
                                actual = float(values[locator["row"], locator["column"]])
                                difference = abs(actual - float(term["source_value"]))
                                if difference > 1e-7:
                                    raise ValueError("profile card value differs from released tensor")
                                maximum_difference = max(maximum_difference, difference)
                                profile.append({
                                    "frozen_fact_name": term["name"],
                                    "published_semantics": semantics,
                                    "published_normalized_value": actual,
                                    "source_locator": locator,
                                    "logit_contribution": term["logit_contribution"],
                                })
                            if len(profile) != 1 or profile[0]["frozen_fact_name"] != \
                                    "published_profile_coordinate_14":
                                raise ValueError("selected profile evidence differs")
                            stream.write(json.dumps({
                                "dataset": "mgtab", "seed": seed,
                                "account_id": card["account_id"],
                                "profile_evidence": profile,
                            }, ensure_ascii=False, separators=(",", ":")) + "\n")
                            count += 1
        if count != report["test_accounts"]:
            raise ValueError("profile overlay account count differs")
        runs.append({"seed": seed, "accounts": count,
                     "original_cards_sha256": sha256_file(cards),
                     "semantic_overlay_sha256": sha256_file(target),
                     "maximum_tensor_value_difference": maximum_difference,
                     "path": str(target)})
    output = {
        "status": "all_frozen_mgtab_test_profile_terms_mapped_to_table_a5",
        "original_cards_unchanged": True,
        "classification_labels_read": False,
        "dictionary_audit_sha256": sha256_file(dictionary_path),
        "released_features_sha256": sha256_file(source),
        "runs": runs,
    }
    target = root / "research/mgtab_profile_semantic_overlay_2026-11-03.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"status": output["status"],
                      "accounts": sum(run["accounts"] for run in runs),
                      "max_difference": max(run["maximum_tensor_value_difference"]
                                            for run in runs)}, indent=2))


if __name__ == "__main__":
    main()

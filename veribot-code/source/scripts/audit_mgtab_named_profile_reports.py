"""Check every MGTAB fixed statement against Table A5 mapping and released tensor."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.compact_verified_report import render_verified_card
from verifiable_bot.mgtab_profile_dictionary import published_profile_description


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    summary = json.loads((root / "research/three_dataset_immutable_evidence_reports_2026-11-03.json")
                         .read_text(encoding="utf-8"))
    mgtab = next(row for row in summary["datasets"] if row["dataset"] == "mgtab")
    dictionary_path = root / "research/certificates/mgtab_published_profile_dictionary_audit.json"
    dictionary = json.loads(dictionary_path.read_text(encoding="utf-8"))
    source = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB/features.pt")
    if (mgtab["published_profile_dictionary_audit_sha256"] !=
            sha256_file(dictionary_path)
            or dictionary["features_sha256"] != sha256_file(source)):
        raise ValueError("published profile dictionary or source differs")
    tensor = np.asarray(torch.load(source, map_location="cpu", weights_only=True))
    expected_semantics = published_profile_description("published_profile_coordinate_14")
    results = []
    for run in mgtab["runs"]:
        seed = run["seed"]
        source_cards = (root / "output/v3_mgtab_ab_spline_content/direct_selected_20" /
                        f"seed{seed}" / "evidence_cards.jsonl.gz")
        rendered_path = root / run["output"]
        if (sha256_file(source_cards) != run["model_cards_sha256"]
                or sha256_file(rendered_path) != run["export_sha256"]):
            raise ValueError("card or rendered report hash differs")
        count = 0
        with gzip.open(source_cards, "rt", encoding="utf-8") as cards, \
                gzip.open(rendered_path, "rt", encoding="utf-8") as rendered:
            for card_line, report_line in zip(cards, rendered, strict=True):
                card = json.loads(card_line)
                report = json.loads(report_line)
                if card["account_id"] != report["account_id"] or len(report["all_items"]) != 20:
                    raise ValueError("profile report account or width differs")
                source_term = next(term for term in card["terms"] if term["name"] ==
                                   "published_profile_coordinate_14")
                item = next(term for term in report["all_items"] if term["fact_name"] ==
                            "published_profile_coordinate_14")
                row = int(card["account_id"])
                value = float(tensor[row, 14])
                expected_claim = (
                    "Published normalized followers-to-friends ratio "
                    f"(MGTAB Table A5, dimension 15) is {value:.12g} in features.pt. "
                    f"Its contribution to the bot log-odds is "
                    f"{float(source_term['logit_contribution']):+.12g}.")
                if (item.get("published_profile_semantics") != expected_semantics
                        or item["source_locator"] != source_term["source_locator"]
                        or abs(item["source_value"] - value) > 1e-12
                        or item["claim"] != expected_claim):
                    raise ValueError("Table A5 report claim differs from published tensor")
                if any("published_profile_semantics" in other
                       for other in report["all_items"] if other is not item):
                    raise ValueError("projection or relation term received profile semantics")
                count += 1
        if count != 1020 or count != run["accounts"]:
            raise ValueError("MGTAB fixed report population differs")
        results.append({"seed": seed, "accounts": count,
                        "rendered_sha256": sha256_file(rendered_path),
                        "published_profile_14_tensor_values_equal": True})
    sample = json.loads(next(gzip.open(
        root / "output/v3_mgtab_ab_spline_content/direct_selected_20/seed0/evidence_cards.jsonl.gz",
        "rt", encoding="utf-8")))
    altered = json.loads(json.dumps(sample))
    altered["terms"][0]["source_locator"]["column"] = 15
    rejected = False
    try:
        render_verified_card(altered)
    except ValueError as error:
        rejected = "profile statement differs" in str(error)
    if not rejected:
        raise ValueError("wrong MGTAB dimension was accepted as a semantic claim")
    output = {"status": "all_mgtab_published_profile_claims_checked",
              "accounts": sum(row["accounts"] for row in results),
              "altered_dimension_rejected": rejected,
              "dictionary_audit_sha256": sha256_file(dictionary_path),
              "runs": results}
    target = root / "research/mgtab_named_profile_report_audit_2026-11-03.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"status": output["status"],
                      "accounts": output["accounts"],
                      "altered_dimension_rejected": rejected}))


if __name__ == "__main__":
    main()

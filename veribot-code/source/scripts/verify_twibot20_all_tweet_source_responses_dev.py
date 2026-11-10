"""Read back all tweet-deletion deltas and verify selected raw source records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import ijson
import numpy as np
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path(r"C:\Users\Lenovo\Downloads\Twibot-20.zip")
CERT = ROOT / "research/certificates/twibot20_split_visible_all_tweet_source_responses_dev.json"
DELTAS = ROOT / "research/certificates/twibot20_split_visible_all_tweet_source_response_deltas_dev.npz"
CARDS = ROOT / "research/certificates/twibot20_split_visible_graph_ab_dev_decisions.json"
OUT = ROOT / "research/certificates/twibot20_split_visible_all_tweet_source_responses_readback.json"
EPS = 1e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    report = json.loads(CERT.read_text(encoding="utf-8"))
    cards = json.loads(CARDS.read_text(encoding="utf-8"))
    if (report["source_archive_sha256"] != sha256(ARCHIVE)
            or report["cards_sha256"] != sha256(CARDS)
            or report["full_deltas_npz_sha256"] != sha256(DELTAS)):
        raise ValueError("Source, frozen cards or complete response array differs")
    with np.load(DELTAS, allow_pickle=False) as saved:
        offsets = saved["offsets"].astype(np.int64)
        delta = saved["deletion_logit_delta"].astype(float)
        ids = saved["dev_ids"].astype(str).tolist()
        if (str(saved["cards_sha256"]) != sha256(CARDS)
                or str(saved["source_archive_sha256"]) != sha256(ARCHIVE)
                or str(saved["model_sha256"]) != report["model_sha256"]):
            raise ValueError("Response array metadata differs")
    rows = report["rows"]
    card_rows = cards["decision_cards"]
    if (len(ids) != len(rows) or len(ids) != len(card_rows)
            or len(offsets) != len(ids) + 1 or offsets[0] != 0
            or offsets[-1] != len(delta) or np.any(np.diff(offsets) < 0)
            or ids != [c["account_id"] for c in card_rows]
            or ids != [r["account_id"] for r in rows]):
        raise ValueError("Per-account response indexing differs")
    accounts_nonzero = accounts_flip = 0
    accounts_supportive = accounts_opposing = accounts_both = 0
    accounts_nonunique_strongest = accounts_uniform_nonzero_response = 0
    largest_strongest_tie = 0
    total_nonzero = total_flip = 0
    source_hashes_checked = 0
    with zipfile.ZipFile(ARCHIVE) as zipped, zipped.open("Twibot-20/dev.json") as stream:
        for k, source in enumerate(ijson.items(stream, "item")):
            if k >= len(ids) or str(source["ID"]) != ids[k]:
                raise ValueError("Original dev source order differs")
            tweets = list(source.get("tweet") or [])
            d = delta[offsets[k]:offsets[k + 1]]
            card = card_rows[k]
            row = rows[k]
            if (len(tweets) != len(d) or row["sampled_tweet_count"] != len(d)
                    or card["tweet_sample_support_count"] != len(d)
                    or row["original_prediction"] != card["prediction"]
                    or abs(row["tweet_group_all_zero_logit_effect"] -
                           card["group_logit_effects"]["sampled_tweet_stats"]) > EPS):
                raise ValueError(f"Source/card count or group effect differs: {ids[k]}")
            nz = int(np.count_nonzero(np.abs(d) > EPS))
            flips = int(np.count_nonzero(
                (expit(float(card["full_logit"]) + d) >= card["threshold"])
                != bool(card["prediction"])))
            if (row["nonzero_single_tweet_responses"] != nz
                    or row["single_tweet_deletions_changing_class"] != flips):
                raise ValueError(f"Response summary differs: {ids[k]}")
            positive = int(np.argmax(d)) if len(d) and np.max(d) > EPS else None
            negative = int(np.argmin(d)) if len(d) and np.min(d) < -EPS else None
            strongest = int(np.argmax(np.abs(d))) if nz else None
            if strongest is not None:
                tie_count = int(np.count_nonzero(np.isclose(
                    np.abs(d), np.abs(d[strongest]), rtol=0, atol=EPS)))
                accounts_nonunique_strongest += tie_count > 1
                largest_strongest_tie = max(largest_strongest_tie, tie_count)
                accounts_uniform_nonzero_response += bool(
                    len(d) >= 2 and np.max(d) - np.min(d) <= EPS)
            else:
                tie_count = 0
            if row["strongest_absolute_tie_count"] != tie_count:
                raise ValueError(f"Strongest-source tie count differs: {ids[k]}")
            for key, slot in (("top_positive_deletion_delta", positive),
                              ("top_negative_deletion_delta", negative),
                              ("strongest_absolute_deletion_delta", strongest)):
                choice = row[key]
                if slot is None:
                    if choice is not None:
                        raise ValueError(f"Spurious source selection: {ids[k]} {key}")
                    continue
                expected_locator = f"Twibot-20/dev.json[{k}].tweet[{slot}]"
                expected_hash = hashlib.sha256(str(tweets[slot] or "").encode()).hexdigest()
                if (choice is None or choice["source_locator"] != expected_locator
                        or choice["tweet_sha256"] != expected_hash
                        or abs(choice["deletion_logit_delta"] - d[slot]) > EPS):
                    raise ValueError(f"Selected raw source differs: {ids[k]} {key}")
                source_hashes_checked += 1
            direction = 1 if card["prediction"] == 1 else -1
            supportive = bool(np.any(d * direction < -EPS))
            opposing = bool(np.any(d * direction > EPS))
            accounts_supportive += supportive
            accounts_opposing += opposing
            accounts_both += supportive and opposing
            accounts_nonzero += nz > 0
            accounts_flip += flips > 0
            total_nonzero += nz
            total_flip += flips
    if k + 1 != len(ids):
        raise ValueError("Original dev source population differs")
    if (report["source_deletions"] != len(delta)
            or report["nonzero_source_deletions"] != total_nonzero
            or report["class_changing_source_deletions"] != total_flip
            or report["accounts_with_any_nonzero_single_tweet_response"] != accounts_nonzero
            or report["accounts_with_any_single_tweet_class_flip"] != accounts_flip):
        raise ValueError("Full-population aggregates differ")
    result = {
        "scope": "independent source locator/hash and full-delta readback",
        "full_response_certificate_sha256": sha256(CERT),
        "full_deltas_npz_sha256": sha256(DELTAS),
        "official_source_archive_sha256": sha256(ARCHIVE),
        "dev_accounts": len(ids),
        "source_deletions": len(delta),
        "source_hashes_checked": source_hashes_checked,
        "accounts_with_nonzero_source_response": accounts_nonzero,
        "accounts_with_class_changing_source_deletion": accounts_flip,
        "accounts_with_tweet_supporting_current_prediction": accounts_supportive,
        "accounts_with_tweet_opposing_current_prediction": accounts_opposing,
        "accounts_with_both_supporting_and_opposing_tweets": accounts_both,
        "accounts_with_nonunique_strongest_absolute_tweet": accounts_nonunique_strongest,
        "largest_number_of_tied_strongest_tweets": largest_strongest_tie,
        "accounts_with_uniform_nonzero_deletion_response_across_tweets": accounts_uniform_nonzero_response,
        "accounts_with_no_single_tweet_response": len(ids) - accounts_nonzero,
        "nonzero_source_deletions": total_nonzero,
        "class_changing_source_deletions": total_flip,
        "all_selected_source_locators_and_hashes_match_raw_zip": True,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

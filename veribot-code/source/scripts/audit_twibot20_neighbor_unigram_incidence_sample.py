"""Check neighbor word cache against an independent sklearn analyzer on ZIP rows."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import ijson
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from train_v3_twibot20 import sha256


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cache_path = root / "data/processed/twibot20_v3/neighbor_active_unigram_incidence.npz"
    manifest = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
    raw_manifest = json.loads((root / "data/raw/twibot20/manifest.json")
                              .read_text(encoding="utf-8"))
    if (manifest["fact_cache_sha256"] != sha256(cache_path)
            or manifest["source_archive_sha256"] != raw_manifest["archive_sha256"]):
        raise ValueError("source cache or official ZIP binding differs")
    with np.load(cache_path, allow_pickle=False) as cache:
        ids = cache["ids"].astype(str)
        parts = cache["parts"].astype(int)
        positions = cache["positions"].astype(int)
        counts = cache["tweet_counts"].astype(int)
        words = cache["vocabulary"].astype(str)
        incidence = cache["incidence"].astype(np.uint8)
    if (len(ids) != manifest["matched_source_accounts"]
            or incidence.shape != (len(ids), 109)
            or len(set(ids)) != len(ids)):
        raise ValueError("neighbor token cache population differs")
    lookup = {account_id: index for index, account_id in enumerate(ids)}
    analyzer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", ngram_range=(1, 1),
        vocabulary=list(words)).build_analyzer()
    checked = {}
    with zipfile.ZipFile(raw_manifest["archive_path"]) as zipped:
        for part_code, part in enumerate(("train", "dev", "test", "support")):
            count = 0
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    account_id = str(row["ID"])
                    index = lookup.get(account_id)
                    if index is None:
                        continue
                    if parts[index] != part_code or positions[index] != position:
                        raise ValueError("source member or row position differs")
                    tweets = row.get("tweet") or []
                    if counts[index] != len(tweets):
                        raise ValueError("sample tweet count differs")
                    observed = set(analyzer(" ".join(str(item or "")
                                                     for item in tweets)))
                    expected = np.fromiter((word in observed for word in words),
                                           dtype=np.uint8, count=len(words))
                    if not np.array_equal(expected, incidence[index]):
                        differing = np.flatnonzero(expected != incidence[index])
                        raise ValueError(
                            f"neighbor source token incidence differs at {part} "
                            f"{position} {account_id}: "
                            f"{[(words[i], int(expected[i]), int(incidence[index, i])) for i in differing[:15]]}")
                    count += 1
                    if count == 100:
                        break
            checked[part] = count
            if count != 100:
                raise ValueError("insufficient source rows for tokenizer audit")
    result = {
        "dataset": "twibot20",
        "status": "400_source_rows_match_independent_train_vocabulary_analyzer",
        "rows_checked_by_member": checked,
        "fact_cells_checked": 400 * len(words),
        "source_archive_sha256": raw_manifest["archive_sha256"],
        "cache_sha256": sha256(cache_path),
        "cache_manifest_sha256": sha256(cache_path.with_suffix(".json")),
        "independent_operation": "sklearn TfidfVectorizer.build_analyzer token set on raw ZIP tweet strings",
    }
    destination = root / "research/certificates/twibot20_neighbor_active_unigram_sample_audit.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["fact_cells_checked"], flush=True)


if __name__ == "__main__":
    main()

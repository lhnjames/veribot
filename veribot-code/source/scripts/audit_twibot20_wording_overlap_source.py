"""Independent full-population readback of first-40 literal overlap facts."""

from __future__ import annotations

import difflib
import gzip
import hashlib
import itertools
import json
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

import ijson
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import sha256


def normalize_independently(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    words = re.findall(r"[#@]?\w+", text)
    return " ".join(words)


def recount(tweets: list[str | None]) -> tuple[float, int, set[tuple[int, int]], int]:
    eligible = []
    for index, tweet in enumerate(tweets[:40]):
        original = str(tweet or "")
        if original.lstrip().casefold().startswith("rt @"):
            continue
        normalized = normalize_independently(original)
        words = normalized.split()
        if len(normalized) >= 25 and len(words) >= 4:
            eligible.append((index, normalized, set(words)))
    if not eligible:
        return float("nan"), 0, set(), 0
    matched = set()
    pairs = set()
    for left, right in itertools.combinations(eligible, 2):
        i, a, aw = left
        j, b, bw = right
        if len(aw & bw) / len(aw | bw) < 0.7:
            continue
        if difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() < 0.88:
            continue
        matched.update((i, j))
        pairs.add((i, j))
    return len(matched) / len(eligible), len(pairs), pairs, len(eligible)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    original = json.loads((root / "data/raw/twibot20/manifest.json")
                          .read_text(encoding="utf-8"))
    archive = Path(original["archive_path"])
    feature = root / "data/processed/twibot20_v3/wording_overlap_first40.npz"
    traces = feature.with_name("wording_overlap_first40_pairs.jsonl.gz")
    manifest = json.loads(feature.with_suffix(".json").read_text(encoding="utf-8"))
    if (sha256(archive) != original["archive_sha256"] or
            sha256(feature) != manifest["source_fact_cache_sha256"] or
            sha256(traces) != manifest["source_trace_sha256"]):
        raise ValueError("source or fact artifact hash differs")
    with np.load(feature, allow_pickle=False) as cache:
        ids = cache["ids"].astype(str)
        splits = cache["split"].astype(str)
        positions = cache["positions"].astype(int)
        saved = cache["values"].astype(float)
    with gzip.open(traces, "rt", encoding="utf-8") as stream:
        locators = {row["account_id"]: row for row in map(json.loads, stream)}
    if len(locators) != manifest["source_trace_account_count"]:
        raise ValueError("duplicated trace accounts")
    account_count = 0
    checked_pairs = 0
    positive = 0
    with zipfile.ZipFile(archive) as zipped:
        for split in ("train", "dev", "test"):
            with zipped.open(f"Twibot-20/{split}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    uid = str(row["ID"])
                    if (ids[account_count] != uid or splits[account_count] != split
                            or positions[account_count] != position):
                        raise ValueError("account source row misjoined")
                    fraction, count, pairs, eligible = recount(row.get("tweet") or [])
                    if not np.allclose(saved[account_count],
                            (fraction, np.log1p(count) if eligible else float("nan")),
                            rtol=0, atol=1e-14,
                            equal_nan=True):
                        raise ValueError(f"computed literal overlap differs: {uid}")
                    located = locators.get(uid)
                    if bool(count) != bool(located):
                        raise ValueError("positive account lacks source locator")
                    if located:
                        positive += 1
                        if (located["source_member"] != f"Twibot-20/{split}.json"
                                or located["source_position"] != position
                                or located["eligible_non_retweet_posts"] != eligible
                                or located["qualifying_pair_count"] != count):
                            raise ValueError("trace source or count differs")
                        for example in located["example_pairs"]:
                            i, j = example["tweet_indices"]
                            if (i, j) not in pairs:
                                raise ValueError("cited tweet pair fails predicate")
                            normalized = [normalize_independently(
                                str(row["tweet"][k] or "")) for k in (i, j)]
                            expected = [hashlib.sha256(v.encode()).hexdigest()
                                        for v in normalized]
                            if expected != example["normalized_sha256"]:
                                raise ValueError("cited original tweet hash differs")
                            checked_pairs += 1
                    account_count += 1
    if (account_count != len(ids) or positive != len(locators)
            or account_count != 11826):
        raise ValueError("full population differs")
    result = {"status": "full_source_to_literal_wording_overlap_values_and_pairs_match",
        "source_archive_sha256": original["archive_sha256"],
        "feature_sha256": sha256(feature), "trace_sha256": sha256(traces),
        "accounts": account_count, "positive_accounts": positive,
        "cited_pairs_rechecked": checked_pairs,
        "class_labels_read": False}
    out = root / "research/certificates/twibot20_wording_overlap_full_source_audit.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

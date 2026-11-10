"""Build source-linked first-40-post wording-overlap facts from official ZIP."""

from __future__ import annotations

import gzip
import json
import sys
import zipfile
from pathlib import Path

import ijson
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import sha256
from verifiable_bot.wording_overlap import FACT_NAMES, MAX_POSTS, analyze


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = json.loads((root / "data/raw/twibot20/manifest.json")
                        .read_text(encoding="utf-8"))
    archive = Path(source["archive_path"])
    if sha256(archive) != source["archive_sha256"]:
        raise ValueError("official TwiBot-20 ZIP differs")
    ids, splits, positions, values, locators = [], [], [], [], []
    counts = {"train": 0, "dev": 0, "test": 0}
    with zipfile.ZipFile(archive) as zipped:
        for split in ("train", "dev", "test"):
            with zipped.open(f"Twibot-20/{split}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    uid = str(row["ID"])
                    scores, pairs, eligible = analyze(row.get("tweet") or [])
                    ids.append(uid)
                    splits.append(split)
                    positions.append(position)
                    values.append(scores)
                    counts[split] += 1
                    if pairs:
                        locators.append({"account_id": uid, "split": split,
                            "source_member": f"Twibot-20/{split}.json",
                            "source_position": position,
                            "first_posts_considered": MAX_POSTS,
                            "eligible_non_retweet_posts": eligible,
                            "qualifying_pair_count": int(round(np.expm1(scores[1]))),
                            "example_pairs": pairs})
            print({"split": split, "accounts": counts[split]}, flush=True)
    if counts != {"train": 8278, "dev": 2365, "test": 1183}:
        raise ValueError("official split population differs")
    destination = root / "data/processed/twibot20_v3/wording_overlap_first40.npz"
    np.savez_compressed(destination, ids=np.asarray(ids), split=np.asarray(splits),
                        positions=np.asarray(positions), names=np.asarray(FACT_NAMES),
                        values=np.asarray(values, dtype=np.float64))
    traces = destination.with_name("wording_overlap_first40_pairs.jsonl.gz")
    with traces.open("wb") as file, gzip.GzipFile(filename="", mode="wb",
            fileobj=file, mtime=0) as compressed:
        for row in locators:
            compressed.write((json.dumps(row, ensure_ascii=False) + "\n")
                             .encode("utf-8"))
    summary = {"dataset": "twibot20", "scope": "literal_high_wording_overlap_in_first_40_sampled_non_retweets",
        "source_archive_sha256": source["archive_sha256"],
        "class_labels_used": False, "split_account_counts": counts,
        "fact_names": FACT_NAMES, "qualifying_accounts": len(locators),
        "source_fact_cache_sha256": sha256(destination),
        "source_trace_sha256": sha256(traces),
        "source_trace_account_count": len(locators),
        "rule": "NFKC/casefold; remove URLs/digits; no RT @; first 40 sampled tweets; >=25 chars and >=4 tokens; token Jaccard >=0.70 and SequenceMatcher ratio >=0.88"}
    destination.with_suffix(".json").write_text(json.dumps(summary, indent=2)
        + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

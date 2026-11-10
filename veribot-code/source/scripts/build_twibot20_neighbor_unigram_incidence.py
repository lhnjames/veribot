"""Stream the official ZIP to cache source-local words of native-follow neighbors."""

from __future__ import annotations

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


PARTS = ("train", "dev", "test", "support")


def strip_accents(text: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(char))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    raw_manifest = json.loads((root / "data/raw/twibot20/manifest.json")
                              .read_text(encoding="utf-8"))
    archive_path = Path(raw_manifest["archive_path"])
    selected_path = root / "output/v3_twibot20_sparse_text_evidence/sparse_l1_c1_model.npz"
    with np.load(selected_path, allow_pickle=False) as selected:
        vocabulary = selected["vocabulary"].astype(str)
        active = vocabulary[np.flatnonzero(selected["coefficient"][44:] != 0)]
    if len(active) != 109 or any(not re.fullmatch(r"[a-zA-Z]{3,}", word)
                                 for word in active):
        raise ValueError("selected source-local unigram vocabulary differs")
    word_index = {word: index for index, word in enumerate(active)}
    pattern = re.compile(r"(?u)\b(?:" + "|".join(map(re.escape, active)) + r")\b")
    if sha256(archive_path) != raw_manifest["archive_sha256"]:
        raise ValueError("official TwiBot ZIP differs from audited manifest")
    target_ids: set[str] = set()
    owner_count = 0
    with zipfile.ZipFile(archive_path) as zipped:
        for part in PARTS[:3]:
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for row in ijson.items(stream, "item"):
                    owner_count += 1
                    target_ids.add(str(row["ID"]))
                    neighbor = row.get("neighbor") or {}
                    for direction in ("following", "follower"):
                        target_ids.update(map(str, neighbor.get(direction) or ()))
        ids, part_codes, positions, tweet_counts, incidences = [], [], [], [], []
        found = set()
        for part_code, part in enumerate(PARTS):
            member = f"Twibot-20/{part}.json"
            with zipped.open(member) as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    if part == "support" and position and position % 20000 == 0:
                        print("support_progress", position, len(found), flush=True)
                    account_id = str(row["ID"])
                    if account_id not in target_ids:
                        continue
                    if account_id in found:
                        raise ValueError("official account ID appears in multiple members")
                    found.add(account_id)
                    tweets = row.get("tweet") or []
                    # Match sklearn's analyzer order: lowercase before Unicode
                    # accent stripping. Mathematical bold uppercase letters may
                    # normalize back to uppercase after the lowercase step.
                    text = strip_accents(" ".join(
                        str(item or "") for item in tweets).lower())
                    observed = set(pattern.findall(text))
                    indicator = np.zeros(len(active), dtype=np.uint8)
                    for word in observed:
                        indicator[word_index[word]] = 1
                    ids.append(account_id)
                    part_codes.append(part_code)
                    positions.append(position)
                    tweet_counts.append(len(tweets))
                    incidences.append(indicator)
            print(part, len(found), flush=True)
    destination = root / "data/processed/twibot20_v3/neighbor_active_unigram_incidence.npz"
    np.savez_compressed(
        destination, ids=np.asarray(ids), parts=np.asarray(part_codes, dtype=np.uint8),
        positions=np.asarray(positions, dtype=np.int32),
        tweet_counts=np.asarray(tweet_counts, dtype=np.int16),
        vocabulary=active, incidence=np.stack(incidences))
    manifest = {
        "dataset": "twibot20", "scope": "official_zip_native_neighbor_source_token_presence",
        "source_archive_sha256": raw_manifest["archive_sha256"],
        "selected_sparse_artifact_sha256": sha256(selected_path),
        "fact_cache_sha256": sha256(destination),
        "owner_count": owner_count, "target_neighbor_ids": len(target_ids),
        "matched_source_accounts": len(ids),
        "unmatched_neighbor_ids": len(target_ids - found),
        "vocabulary_size": len(active),
        "operation": "per published account, exact train-vocabulary unigram presence in its raw sampled tweet strings after lowercase and Unicode accent stripping",
        "source_locator": "ZIP member, row position, account ID; each word can be checked at raw tweet indices",
        "classification_label_field_used": False,
        "test_labels_used_for_feature_definition": False,
        "part_codes": {part: code for code, part in enumerate(PARTS)},
    }
    destination.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"matched": len(ids), "unmatched": len(target_ids - found),
                      "archive_sha256": manifest["source_archive_sha256"]}), flush=True)


if __name__ == "__main__":
    main()

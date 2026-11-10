"""Build a transductive native-follow graph with source-addressable accounts.

The official labeled train/dev/test rows retain their split labels. Official
support rows enter only as unlabeled graph nodes. Account facts never read the
label field. Every row retains its archive member and array ordinal so later
source edits can return to the exact raw record.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import ijson
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.twibot20_facts import FACT_NAMES, account_facts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--reference-facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive_sha = sha256(args.zip)
    ids, splits, labels, members, ordinals, contexts = [], [], [], [], [], []
    base_facts = []
    edges: set[tuple[str, str]] = set()
    with zipfile.ZipFile(args.zip) as zipped:
        for split in ("train", "dev", "test", "support"):
            member = f"Twibot-20/{split}.json"
            with zipped.open(member) as stream:
                for ordinal, row in enumerate(ijson.items(stream, "item")):
                    values, relations, context = account_facts(row)
                    ids.append(str(row["ID"]))
                    splits.append(split)
                    labels.append(int(row["label"]) if split != "support" else -1)
                    members.append(member)
                    ordinals.append(ordinal)
                    contexts.append(context)
                    base_facts.append(values)
                    edges.update(relations)
            print(json.dumps({"split": split, "rows_seen": len(ids),
                              "unique_edges": len(edges)}), flush=True)
    if len(ids) != len(set(ids)) or len(ids) != 11826 + 217754:
        raise ValueError("official account population differs")
    index = {uid: i for i, uid in enumerate(ids)}
    if any(src not in index or dst not in index for src, dst in edges):
        raise ValueError("native follow edge has no source record")
    indegree = Counter(dst for _, dst in edges)
    outdegree = Counter(src for src, _ in edges)
    reciprocal = Counter(src for src, dst in edges if (dst, src) in edges)
    structural = np.asarray([
        [float(outdegree[uid]), float(indegree[uid]),
         float(reciprocal[uid])] for uid in ids], dtype=np.float32)
    values = np.column_stack((np.asarray(base_facts, dtype=np.float32),
                              structural))
    directed = np.asarray(sorted((index[src], index[dst])
                                 for src, dst in edges), dtype=np.int64).T
    if values.shape != (len(ids), len(FACT_NAMES)) or directed.shape[0] != 2:
        raise ValueError("native graph bundle shape differs")
    with np.load(args.reference_facts, allow_pickle=False) as reference:
        n = len(reference["ids"])
        if (n != 11826 or not np.array_equal(np.asarray(ids[:n]), reference["ids"])
                or not np.array_equal(np.asarray(splits[:n]), reference["split"])
                or not np.allclose(values[:n], reference["values"],
                                   equal_nan=True, atol=0)):
            raise ValueError("labeled facts differ from source-indexed reference")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, values=values, edge_index=directed,
        labels=np.asarray(labels, dtype=np.int8),
        ids=np.asarray(ids), split=np.asarray(splits),
        members=np.asarray(members), ordinals=np.asarray(ordinals, dtype=np.int32),
        contexts=np.asarray(contexts), names=np.asarray(FACT_NAMES))
    manifest = {
        "dataset": "twibot20", "graph_scope": "all official sampled follow declarations",
        "node_population": "official train/dev/test plus unlabeled support",
        "archive_sha256": archive_sha,
        "reference_facts_sha256": sha256(args.reference_facts),
        "n_nodes": len(ids), "n_directed_edges": directed.shape[1],
        "split_counts": dict(Counter(splits)),
        "feature_names": list(FACT_NAMES),
        "raw_locator": "archive member + array ordinal + account ID",
        "label_access": "account_facts reads no label; support labels omitted",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "manifest": manifest}), flush=True)


if __name__ == "__main__":
    main()

"""Audit what the official TwiBot-20 ZIP actually supplies as graph neighbors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import ijson
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import sha256


FIELDS = ("sampled_following_count", "sampled_follower_count",
          "graph_out_degree", "graph_in_degree", "graph_reciprocal_degree")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    fact_manifest = json.loads(args.facts.with_suffix(".json").read_text(encoding="utf-8"))
    archive_hash = sha256(args.zip)
    if (archive_hash != audit["source_archive_sha256"]
            or archive_hash != fact_manifest["archive_sha256"]
            or sha256(args.facts) != audit["fact_cache_sha256"]):
        raise ValueError("Official ZIP, fact cache or full source audit differs")
    with zipfile.ZipFile(args.zip) as zipped:
        readme_bytes = zipped.read("Twibot-20/README.txt")
        readme = readme_bytes.decode("utf-8", errors="strict")
        if "random 20 followers and followings" not in readme:
            raise ValueError("README neighbor sampling statement differs")
        support_info = zipped.getinfo("Twibot-20/support.json")
        count = 0
        nonnull = 0
        source_edge_declarations = 0
        examples = []
        started = time.perf_counter()
        with zipped.open("Twibot-20/support.json") as stream:
            for neighbor in ijson.items(stream, "item.neighbor"):
                count += 1
                if neighbor is not None:
                    nonnull += 1
                    source_edge_declarations += (
                        len(neighbor.get("following") or []) +
                        len(neighbor.get("follower") or []))
                    if len(examples) < 3:
                        examples.append({
                            "following_count": len(neighbor.get("following") or []),
                            "follower_count": len(neighbor.get("follower") or []),
                        })
                if count % 50000 == 0:
                    print(json.dumps({"support_rows_scanned": count,
                                      "neighbor_nonnull": nonnull,
                                      "seconds": round(time.perf_counter() - started, 1)}),
                          flush=True)
    if count != fact_manifest["support_ids"]:
        raise ValueError("Support row count differs from fact manifest")
    with np.load(args.facts, allow_pickle=False) as saved:
        split = saved["split"].astype(str)
        values = saved["values"].astype(float)
        names = saved["names"].astype(str).tolist()
    distribution = {}
    for part in ("train", "dev", "test"):
        mask = split == part
        distribution[part] = {}
        for name in FIELDS:
            row = values[mask, names.index(name)]
            available = np.isfinite(row)
            distribution[part][name] = {
                "accounts": len(row),
                "available": int(np.count_nonzero(available)),
                "missing": int(np.count_nonzero(~available)),
                "exactly_10": int(np.count_nonzero(row == 10)),
                "positive": int(np.count_nonzero(row > 0)),
                "zero": int(np.count_nonzero(row == 0)),
                "maximum": float(np.nanmax(row)),
            }
    report = {
        "scope": "official TwiBot-20 ZIP README, all support neighbor fields, audited labeled-account fact cache",
        "archive_sha256": archive_hash,
        "facts_sha256": sha256(args.facts),
        "full_source_audit_sha256": sha256(args.source_audit),
        "readme_sha256": hashlib.sha256(readme_bytes).hexdigest(),
        "readme_neighbor_statement": "Each user sample contains the random 20 followers and followings of this user.",
        "support_member_compressed_bytes": support_info.compress_size,
        "support_member_uncompressed_bytes": support_info.file_size,
        "support_rows_scanned": count,
        "support_neighbor_nonnull_rows": nonnull,
        "support_neighbor_edge_declarations": source_edge_declarations,
        "support_nonnull_examples": examples,
        "labeled_account_graph_fact_distribution": distribution,
        "graph_definition": "deduplicated union of directed neighbor declarations available in the official labeled train/dev/test rows; support IDs are endpoints but support row neighbor declarations, if any, must be audited separately",
        "interpretation": "These are observed sample-graph degrees, not complete Twitter follower/following degrees. A self-neighbor list of length 10 is at the observed bound and must not be described as having exactly 10 real-world neighbors. Whether support neighbors were omitted from the builder depends on the full scan result above.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "labeled_account_graph_fact_distribution"}, indent=2))


if __name__ == "__main__":
    main()

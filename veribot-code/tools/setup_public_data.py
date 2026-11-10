"""Download and validate the two public evaluation datasets for VeriBot.

Run from any directory: python setup_public_data.py --all
TwiBot-20/22 are deliberately excluded: their maintainers require permission.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "data" / "raw"
FOX_URL = "https://zenodo.org/records/8035290/files/fox8_23_dataset.ndjson.gz?download=1"
FOX_MD5 = "15fe7f6fbd610084283587f97984792b"
BOT_BASE = "https://raw.githubusercontent.com/QQQQQQBY/BotSim/main/BotSim-24-Dataset"


def download(url: str, target: Path, expected_md5: str | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (not expected_md5 or digest(target, "md5") == expected_md5):
        print(f"Using existing {target}")
        return
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "VeriBot-data-setup/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if expected_md5 and digest(partial, "md5") != expected_md5:
        raise ValueError(f"MD5 mismatch for {target}")
    partial.replace(target)
    print(f"Downloaded {target} ({target.stat().st_size:,} bytes)")


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fox8() -> dict:
    folder = ROOT / "fox8_23"
    archive = folder / "fox8_23_dataset.ndjson.gz"
    raw = folder / "fox8_23_dataset.ndjson"
    download(FOX_URL, archive, FOX_MD5)
    if not raw.exists():
        with gzip.open(archive, "rb") as source, raw.open("wb") as output:
            shutil.copyfileobj(source, output)
    counts = {"bot": 0, "human": 0}
    ids = set()
    with raw.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            counts[record["label"]] += 1
            ids.add(str(record["user_id"]))
    if counts != {"bot": 1140, "human": 1140}:
        raise ValueError(f"Unexpected Fox8 population: {counts}")
    return {"source": FOX_URL, "counts": counts, "unique_ids": len(ids),
            "duplicate_account_records": sum(counts.values()) - len(ids),
            "archive_md5": digest(archive, "md5"), "raw_sha256": digest(raw)}


def botsim() -> dict:
    folder = ROOT / "botsim_24"
    for name in ("Users.csv", "user_post_comment.json", "Readme.md"):
        download(f"{BOT_BASE}/{name}", folder / name)
    with (folder / "Users.csv").open(encoding="utf-8-sig", newline="") as source:
        users = list(csv.DictReader(source))
    with (folder / "user_post_comment.json").open(encoding="utf-8") as source:
        posts = json.load(source)
    if len(users) != 2907 or len({u["user_id"] for u in users}) != 2907:
        raise ValueError("Unexpected BotSim Users.csv population or duplicate IDs")
    return {"source": BOT_BASE, "users": len(users), "post_records": len(posts),
            "labels_from_official_order": {"human_first": 1907, "bot_last": 1000},
            "sha256": {name: digest(folder / name) for name in ("Users.csv", "user_post_comment.json")}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--fox8", action="store_true")
    parser.add_argument("--botsim", action="store_true")
    args = parser.parse_args()
    if not (args.all or args.fox8 or args.botsim):
        parser.error("select --all, --fox8, or --botsim")
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if args.all or args.fox8:
        manifest["fox8_23"] = fox8()
    if args.all or args.botsim:
        manifest["botsim_24"] = botsim()
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Validated {', '.join(manifest)}; manifest: {manifest_path}")


if __name__ == "__main__":
    main()

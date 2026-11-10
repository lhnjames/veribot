"""Extract only the labelled TwiBot-20 splits from the official archive.

The 5.3 GB support set stays compressed because the current train-only HGT graph
does not use it. Each extracted file is hashed and atomically named after its
zip member passes CRC verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile


EXPECTED_TRAIN_SHA256 = "18bef76a8b848eef5d1a9845bfae7a525c779bd3cf67051ea71e9d5fedfe3df8"
SPLITS = ("train.json", "dev.json", "test.json")


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    archive_sha = digest_file(args.archive)
    manifest: dict = {"archive": str(args.archive), "archive_sha256": archive_sha, "splits": {}}
    with zipfile.ZipFile(args.archive) as zf:
        for name in SPLITS:
            matches = [m for m in zf.infolist() if Path(m.filename).name.lower() == name]
            if len(matches) != 1:
                raise ValueError(f"expected exactly one {name} in archive, found {len(matches)}")
            member = matches[0]
            if not 0 < member.file_size < 1_000_000_000:
                raise ValueError(f"unexpected {name} size: {member.file_size}")
            target = args.output / name
            partial = args.output / (name + ".part")
            h = hashlib.sha256()
            n = 0
            with zf.open(member) as source, partial.open("wb") as dest:
                while block := source.read(8 * 1024 * 1024):
                    dest.write(block)
                    h.update(block)
                    n += len(block)
            if n != member.file_size:
                raise ValueError(f"truncated {name}: got {n}, expected {member.file_size}")
            sha = h.hexdigest()
            if name == "train.json" and sha != EXPECTED_TRAIN_SHA256:
                raise ValueError(f"train.json differs from old HGT input: {sha}")
            os.replace(partial, target)
            manifest["splits"][name] = {"zip_member": member.filename, "bytes": n, "sha256": sha}
            print(f"{name}: {n} bytes sha256={sha}", flush=True)

    with (args.output / "extraction_manifest.json").open("w", encoding="utf-8") as dest:
        json.dump(manifest, dest, indent=2)
    print(f"Archive SHA256: {archive_sha}", flush=True)


if __name__ == "__main__":
    main()

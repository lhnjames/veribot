"""Check blind packet separation and, when complete, independent rater agreement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.annotation_audit import (
    read_jsonl, sha256, validate_packet, validate_responses,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--responses", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.directory / "manifest.json").read_text(encoding="utf-8"))
    packet_path = args.directory / "rater_materials" / "blind_packet.jsonl"
    provenance_path = args.directory / "private" / "provenance.jsonl"
    html_path = args.directory / "rater_materials" / "blind_packet.html"
    if sha256(packet_path) != manifest["blind_packet_sha256"] or \
            sha256(provenance_path) != manifest["private_provenance_sha256"] or \
            sha256(html_path) != manifest["blind_html_sha256"]:
        raise ValueError("packet or source mapping differs from manifest")
    packet = read_jsonl(packet_path)
    provenance = read_jsonl(provenance_path)
    result = {"packet": validate_packet(packet, provenance)}
    if result["packet"]["concept_counts"] != manifest["counts"]:
        raise ValueError("blind packet concept counts differ from manifest")
    html = html_path.read_text(encoding="utf-8")
    if html.count("<section>") != len(packet) or \
            any(row["blind_id"] not in html for row in packet) or \
            any(row["account_id"] in html for row in provenance):
        raise ValueError("blind HTML is incomplete or contains a source account ID")
    result["packet"]["blind_html_checked"] = True
    for language, filename in (("en", "README.md"), ("zh", "README_zh.md")):
        if sha256(args.directory / "rater_materials" / filename) != \
                manifest["rater_guide_sha256"][language]:
            raise ValueError("rater instructions differ from manifest")
    for rater in ("A", "B"):
        if sha256(args.directory / "rater_materials" / f"annotation_rater_{rater}.csv") != \
                manifest["annotation_template_sha256"][rater]:
            raise ValueError("rater template differs from manifest")
    rater_zip = args.directory / "rater_materials_only.zip"
    if sha256(rater_zip) != manifest["rater_materials_only_zip_sha256"]:
        raise ValueError("rater-only ZIP differs from manifest")
    with zipfile.ZipFile(rater_zip) as zipped:
        expected = {path.name for path in (args.directory / "rater_materials").iterdir()
                    if path.is_file()}
        if set(zipped.namelist()) != expected or any("private" in name.lower()
                                                     for name in zipped.namelist()):
            raise ValueError("rater ZIP has missing or private files")
    result["packet"]["rater_only_zip_checked"] = True
    if args.responses:
        result["responses"] = validate_responses(
            packet, args.directory / "rater_materials" / "annotation_rater_A.csv",
            args.directory / "rater_materials" / "annotation_rater_B.csv")
    output = args.directory / "packet_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

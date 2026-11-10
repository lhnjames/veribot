"""Define ten label-blind MGTAB source groups for a capacity screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.selected_report.read_text(encoding="utf-8"))
    old = report["reasoner"]["groups"]
    names = list(report["reasoner"]["directions"])
    relation = [fact for group, members in old.items()
                if group.startswith("published_relation_") for fact in members]
    content = old["published_tweet_projections"]
    if (len(names) != 150 or len(old["published_profile_coordinates"]) != 20
            or len(relation) != 66 or len(content) != 64
            or any(not name.endswith(f"_{j:03d}")
                   for j, name in enumerate(content))):
        raise ValueError("expected 150-fact MGTAB vocabulary differs")
    groups = {"published_profile_coordinates": old["published_profile_coordinates"],
              "published_typed_relations": relation}
    for block in range(8):
        groups[f"published_tweet_projection_block_{block:02d}"] = content[
            8 * block:8 * (block + 1)]
    listed = [name for members in groups.values() for name in members]
    if len(groups) != 10 or len(listed) != 150 or set(listed) != set(names):
        raise ValueError("source groups do not partition all 150 facts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(groups, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(groups)} groups with {len(listed)} facts to {args.output}")


if __name__ == "__main__":
    main()

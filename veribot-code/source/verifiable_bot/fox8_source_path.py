"""Locate the published Fox8 feature table without duplicating it."""

from __future__ import annotations

import json
import os
from pathlib import Path


def resolve_fox8_source_dir(root: Path) -> Path:
    configured = os.environ.get("VERIBOT_FOX8_SOURCE_DIR")
    if not configured:
        pointer = Path(root) / "data/raw/fox8_source.json"
        configured = json.loads(pointer.read_text(encoding="utf-8"))["path"]
    source = Path(configured).expanduser().resolve()
    if not (source / "features.csv").is_file():
        raise FileNotFoundError(f"Fox8 source table is absent: {source / 'features.csv'}")
    return source

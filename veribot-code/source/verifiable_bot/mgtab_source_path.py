"""Resolve the published MGTAB tensors without copying them into the project."""

from __future__ import annotations

import json
import os
from pathlib import Path


def resolve_mgtab_source_dir(root: Path) -> Path:
    """Use an explicit environment override or the workspace source pointer."""
    configured = os.environ.get("VERIBOT_MGTAB_SOURCE_DIR")
    if not configured:
        pointer = Path(root) / "data/raw/mgtab_source.json"
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        configured = payload["path"]
    source = Path(configured).expanduser().resolve()
    required = ("features.pt", "edge_index.pt", "edge_type.pt")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"MGTAB source directory {source} lacks {', '.join(missing)}")
    return source

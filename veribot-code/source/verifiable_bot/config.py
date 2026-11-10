"""Loads configs/veriroute_lite.yaml (or an override) and applies it as argparse defaults.

Without this, the YAML was pure documentation: the training scripts only ever read their own
hardcoded argparse defaults, so editing the YAML silently did nothing. `apply_config_defaults`
closes that gap -- CLI flags still take precedence (argparse defaults are just a floor), so
existing invocations that pass explicit flags are unaffected.
"""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "veriroute_lite.yaml"


def load_config(path: Path | None) -> dict:
    path = path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get(config: dict, dotted_key: str):
    node = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def apply_config_defaults(parser, config: dict, mapping: dict[str, str]) -> None:
    """mapping: {argparse dest name -> dotted path into the YAML}. Missing keys are left alone,
    so a partial or absent config file never breaks argument parsing."""
    defaults = {}
    for dest, dotted_key in mapping.items():
        value = _get(config, dotted_key)
        if value is not None:
            defaults[dest] = value
    parser.set_defaults(**defaults)

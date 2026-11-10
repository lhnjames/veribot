"""Bind a source-response certificate to the code used for its computation."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

SCHEMA_VERSION = "veribot-aaai-source-response-v1"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_manifest(root: Path, paths: tuple[str, ...]) -> dict[str, str]:
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("code manifest needs distinct paths")
    result = {}
    for relative in paths:
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("code manifest path leaves project root")
        result[relative] = file_sha256(path)
    return result


def verify_code_manifest(root: Path, recorded: dict[str, str]) -> None:
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("certificate has no computation-code manifest")
    actual = code_manifest(root, tuple(recorded))
    if actual != recorded:
        differences = [name for name in recorded if recorded[name] != actual.get(name)]
        raise ValueError(f"certificate computation code differs: {differences}")

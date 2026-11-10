"""Inventory the official MGTAB tensor package used by the runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
OUT = ROOT / "research/certificates/mgtab_release_source_scope.json"
EXPECTED = {"edge_index.pt", "edge_type.pt", "edge_weight.pt",
            "features.pt", "labels_bot.pt", "labels_stance.pt"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = sorted(p for p in SOURCE.iterdir() if p.is_file())
    names = {p.name for p in files}
    if names != EXPECTED:
        raise ValueError(f"Released source package inventory changed: {names}")
    shapes = {}
    dtypes = {}
    for path in files:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Expected released tensor: {path}")
        shapes[path.name] = list(value.shape)
        dtypes[path.name] = str(value.dtype)
    if shapes["features.pt"][0] != shapes["labels_bot.pt"][0]:
        raise ValueError("Published features and labels differ in account count")
    if shapes["edge_index.pt"][0] != 2:
        raise ValueError("Published edge index must have source and target")
    if not torch.is_floating_point(torch.load(
            SOURCE / "features.pt", map_location="cpu", weights_only=True)):
        raise ValueError("Published features are not numeric floating point")
    result = {
        "scope": "local_official_MGTAB_release_package_used_for_models",
        "source_directory": str(SOURCE),
        "account_count": shapes["features.pt"][0],
        "file_inventory": [{"name": p.name, "bytes": p.stat().st_size,
                            "sha256": sha256(p), "tensor_shape": shapes[p.name],
                            "tensor_dtype": dtypes[p.name]}
                           for p in files],
        "raw_tweet_text_or_tweet_ids_in_this_package": False,
        "published_feature_tensor_recomputable": True,
        "published_relation_tensors_recomputable": True,
        "per_tweet_semantic_statement_from_this_package": False,
        "external_semantic_truth_from_this_package": False,
        "official_repository": "https://github.com/GraphDetec/MGTAB",
        "official_paper": "https://arxiv.org/abs/2301.01123",
        "public_raw_tweet_request": "https://github.com/GraphDetec/MGTAB/issues/10",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "account_count", "raw_tweet_text_or_tweet_ids_in_this_package",
        "published_feature_tensor_recomputable")}, indent=2))


if __name__ == "__main__":
    main()

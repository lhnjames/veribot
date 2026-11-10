"""Exercise frozen MGTAB source hash guards without modifying source files."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot import compact_source_population as population


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    run_dir = root / "output/v3_mgtab_ab_spline_content/selected/seed0"
    structural = root / "data/processed/mgtab_v3/structural_facts.npz"
    neighbor = root / "data/processed/mgtab_v3/neighbor_facts.npz"
    originals = {
        "typed_edges": data_dir / "edge_index.pt",
        "typed_edge_kinds": data_dir / "edge_type.pt",
        "structural_cache": structural,
        "neighbor_cache": neighbor,
        "train_only_projection": run_dir / "projection.npz",
    }
    original_load = torch.load

    def reject_labels(path, *args, **kwargs):
        if Path(path).name.lower() in {"labels_bot.pt", "labels.pt", "label.pt"}:
            raise AssertionError("source loader attempted to read class labels")
        return original_load(path, *args, **kwargs)

    def load() -> tuple:
        return population.load_source_population(
            "mgtab", data_dir, run_dir,
            structural_facts=structural, neighbor_facts=neighbor)

    with patch.object(torch, "load", reject_labels):
        facts, names, ids = load()
        if facts.shape != (10199, 150) or len(names) != 150 or len(ids) != 10199:
            raise ValueError("clean MGTAB source load differs")
        original_hash = population._sha256
        rows = []
        for name, target in originals.items():
            target = target.resolve()

            def corrupt_one(path: Path, *, _target=target) -> str:
                return ("0" * 64 if Path(path).resolve() == _target
                        else original_hash(path))

            with patch.object(population, "_sha256", corrupt_one):
                try:
                    load()
                except ValueError as exc:
                    message = str(exc)
                    if "differs" not in message:
                        raise ValueError(
                            f"{name} rejected for the wrong reason: {message}") from exc
                else:
                    raise AssertionError(f"{name} hash corruption was accepted")
            rows.append({
                "source": name,
                "path": str(target),
                "hash_guard_rejected_corruption": True,
                "source_file_modified": False,
            })
            print(name, "rejected", flush=True)
    certificate = {
        "scope": "mgtab_source_loader_hash_guard_fault_injection",
        "clean_source_shape": list(facts.shape),
        "label_file_read_attempts": 0,
        "source_files_modified": False,
        "guards": rows,
    }
    output = root / "research/certificates/mgtab_source_integrity_guards.json"
    output.write_text(json.dumps(certificate, indent=2) + "\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()

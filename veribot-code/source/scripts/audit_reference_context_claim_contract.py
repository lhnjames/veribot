"""Check which frozen evidence models actually use context-specific references."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SELECTED = {
    "botsim": "output/v3_computable/botsim_native_graph",
    "fox8": "output/v3_fox8_native_spline_l2/selected",
    "mgtab": "output/v3_mgtab_ab_spline_content/selected",
}
SCHEMA_V2_LOCKED = {
    "botsim": "output/evidence_schema_v2_group_additive_candidates/botsim",
    "fox8": "output/evidence_schema_v2_group_additive_candidates/fox8",
}


def inspect(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        saved = json.load(stream)
    payload = saved["payload"]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    if digest != saved["version"]:
        raise ValueError(f"reference payload checksum mismatch: {path}")
    contexts = sorted({key.rsplit(":", 1)[0] for key in payload["reference"]})
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reference_version": saved["version"],
        "contexts": contexts,
        "reference_arrays": len(payload["reference"]),
    }


def main() -> None:
    schema_path = ROOT / "research/evidence_schema_v2.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))["schema"]
    selected = {name: [inspect(ROOT / directory / f"seed{seed}/reference.json.gz")
                       for seed in range(3)] for name, directory in SELECTED.items()}
    if any(row["contexts"] != ["all"] for rows in selected.values() for row in rows):
        raise AssertionError("selected-model context contract changed")
    candidates = {name: [inspect(ROOT / directory / f"seed{seed}/reference.json.gz")
                         for seed in range(3)]
                  for name, directory in SCHEMA_V2_LOCKED.items()}
    if any(row["contexts"] != ["activity_0", "activity_1", "activity_2", "all"]
           for row in candidates["botsim"]):
        raise AssertionError("BotSim schema-v2 candidate context contract changed")
    if any(row["contexts"] != ["all"] for row in candidates["fox8"]):
        raise AssertionError("Fox8 schema-v2 candidate context contract changed")
    twibot = inspect(ROOT / "output/twibot20_compact20_ab_group_tree_dev/"
                     "reference_split_visible_train_dev.json.gz")
    runtime = ROOT / "source/verifiable_bot/compact_evidence_runtime.py"
    runtime_text = runtime.read_text(encoding="utf-8")
    if "atomic = self.reference.transform(values)" not in runtime_text:
        raise AssertionError("selected runtime call signature changed; inspect manually")
    result = {
        "question": "Do the frozen high-performance evidence models use account-specific context strata?",
        "selected_high_performance_references": selected,
        "selected_runtime": {
            "path": str(runtime.relative_to(ROOT)).replace("\\", "/"),
            "sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
            "reference_call": "self.reference.transform(values)",
        },
        "schema_v2_intended_rules": {
            name: spec["reference_context"]["rule"]
            for name, spec in schema.items()
        },
        "schema_v2_locked_candidate_references": candidates,
        "twibot20_ab_group_tree_dev_reference": twibot,
        "conclusion": (
            "All nine selected high-performance references use only the pooled 'all' "
            "population, and their inference runtime passes no account context. "
            "BotSim schema-v2 candidates and the TwiBot-20 AB group-tree dev model "
            "have context-specific references; they are different model artifacts. "
            "The selected-model results establish label-free pooled comparisons, "
            "not a gain from context conditioning."
        ),
    }
    target = ROOT / "research/certificates/reference_context_claim_contract.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()

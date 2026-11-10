"""Predict BotSim account IDs using the independently checked literal-zero model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.botsim_literal_zero_runtime import BotSimLiteralZeroRuntime


def predict(root: Path, dataset: str, seed: int, account_ids: list[str]) -> dict:
    if dataset != "botsim":
        raise ValueError("literal-zero runtime is specific to BotSim")
    root = Path(root).resolve()
    certificate_path = root / "research/certificates/botsim_literal_zero_source_to_decision.json"
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    if (certificate["status"] !=
            "literal_zero_raw_source_and_all_fixed_test_decisions_verified" or
            len(certificate["runs"]) != 3 or
            certificate["runs"][seed]["seed"] != seed):
        raise ValueError("independent literal-zero source audit is absent")
    runtime = BotSimLiteralZeroRuntime(root, seed)
    if (certificate["raw_users_sha256"] != runtime.source_binding["raw_users_sha256"] or
            certificate["raw_activity_sha256"] !=
            runtime.source_binding["raw_activity_sha256"]):
        raise ValueError("raw source changed since independent audit")
    decisions = runtime.predict_many(account_ids)
    return {
        "dataset": "botsim", "seed": seed,
        "source_level": "raw_user_post_comment_and_native_reply_graph",
        "source_binding": {
            **runtime.source_binding,
            "independent_decision_audit_sha256": sha256_file(certificate_path),
        },
        "classification_labels_accessed": False,
        "caller_supplied_fact_matrix_accepted": False,
        "decisions": decisions,
    }

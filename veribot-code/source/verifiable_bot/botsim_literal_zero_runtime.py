"""Source-bound BotSim classifier trained with literal post upvote zeros."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from .compact_evidence_runtime import _sha256
from .v3_evidence_reasoner import LabelBlindReference
from .v3_group_tail_bottleneck import collapse_group_tails
from .v3_monotone_spline_reasoner import MonotoneSplineReasoner


BASE_NAMES = (
    "submission_num", "comment_num", "comment_num_1", "comment_num_2",
    "has_description", "description_len", "n_subreddits", "n_posts",
    "n_comments_received", "n_comments_made", "avg_post_score",
    "avg_upvote_ratio", "avg_post_len", "avg_comment_score",
    "avg_comment_len", "n_unique_subs", "is_active",
)
NATIVE_NAMES = (
    "native_reply_out_unique_log1p", "native_reply_in_unique_log1p",
    "native_reply_mutual_unique_log1p", "native_reply_out_comments_log1p",
    "native_reply_in_comments_log1p",
)


def _number(value: object) -> float:
    return 0.0 if value is None or str(value).strip().lower() in {"", "nan"} else float(value)


def _account_facts(row: dict, record: dict, *, observed_means_only: bool = False
                   ) -> list[float]:
    posts = record.get("posts", [])
    comments1 = record.get("comment_1", [])
    comments2 = record.get("comment_2", [])
    description = row.get("description", "")
    if description is None or str(description).lower() == "nan":
        description = ""
    subreddits = row.get("subreddit", "")
    try:
        parsed = ast.literal_eval(subreddits) if subreddits else []
    except (SyntaxError, ValueError):
        parsed = []
    scores = [post.get("score", 0) or 0 for post in posts]
    received = [post.get("num_comments", 0) or 0 for post in posts]
    upvotes = [(.5 if post.get("upvote_ratio") is None else
                float(post["upvote_ratio"])) for post in posts]
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in upvotes):
        raise ValueError("raw post upvote ratio outside [0,1]")
    post_lengths = [len(post.get("posts", "") or "") for post in posts]
    comment_scores = [item.get("comment_score", 0) or 0 for item in comments1]
    comment_lengths = [len(item.get("comment_body", "") or "") for item in comments1]
    values = (
        _number(row.get("submission_num")), _number(row.get("comment_num")),
        _number(row.get("comment_num_1")), _number(row.get("comment_num_2")),
        float(bool(description)), float(len(description)), float(len(parsed)),
        float(len(posts)), float(int(sum(received))) if received else 0.0,
        float(len(comments1) + len(comments2)),
        float(sum(scores)/len(scores)) if scores else 0.0,
        float(sum(upvotes)/len(upvotes)) if upvotes else .5,
        float(sum(post_lengths)/len(post_lengths)) if post_lengths else 0.0,
        float(sum(comment_scores)/len(comment_scores)) if comment_scores else 0.0,
        float(sum(comment_lengths)/len(comment_lengths)) if comment_lengths else 0.0,
        float(len({post.get("subreddit", "") for post in posts})),
        float(bool(posts or comments1 or comments2)),
    )
    result = [float(np.float32(value)) for value in values]
    if observed_means_only:
        if not posts:
            for index in (10, 11, 12):
                result[index] = float("nan")
        if not comments1:
            for index in (13, 14):
                result[index] = float("nan")
    return result


class BotSimLiteralZeroRuntime:
    """Account-ID inference; only raw records and five verified graph facts enter."""

    def __init__(self, root: Path, seed: int, *, observed_means_only: bool = False,
                 shortcut_ablated: bool = False,
                 engagement_ablated: bool = False):
        if seed not in (0, 1, 2):
            raise ValueError("unsupported frozen seed")
        if shortcut_ablated and engagement_ablated:
            raise ValueError("BotSim ablation variants are mutually exclusive")
        if (shortcut_ablated or engagement_ablated) and not observed_means_only:
            raise ValueError("shortcut ablation requires defined observed means")
        self.root = Path(root).resolve()
        self.seed = seed
        self.observed_means_only = observed_means_only
        folder = self.root / ("output/v3_botsim_engagement_shortcuts/selected"
                              if engagement_ablated else
                              "output/v3_botsim_shortcut_ablated_10" if shortcut_ablated else
                              "output/v3_botsim_observed_means_12" if observed_means_only else
                              "output/v3_botsim_literal_upvote_zero_12")
        lock_path = folder / "dev_only.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        run = lock["runs"][seed]
        expected_preprocessing = (
            "literal zero preserved; post means missing without posts; "
            "comment means missing without first-party comments"
            if observed_means_only else
            "upvote_ratio explicit zero preserved; .5 only for null/missing or no posts")
        allowed_dev_errors = 2 if shortcut_ablated or engagement_ablated else 0
        if (run["seed"] != seed or run["validation"]["errors"] > allowed_dev_errors or
                lock["test_predictions_or_metrics_accessed"] is not False or
                lock["preprocessing"] != expected_preprocessing):
            raise ValueError("literal-zero development lock differs")
        user_path = self.root / "data/raw/botsim_24/Users.csv"
        activity_path = self.root / "data/raw/botsim_24/user_post_comment.json"
        native_path = self.root / "data/processed/botsim_v3/native_graph_facts.npz"
        trace_path = self.root / "data/processed/botsim_v3/reply_sources.jsonl.gz"
        source_cert_path = self.root / "research/certificates/botsim24_full_source_audit.json"
        source_cert = json.loads(source_cert_path.read_text(encoding="utf-8"))
        if (source_cert["status"] != "verified_full_population_against_raw_comments" or
                _sha256(user_path) != run["users_sha256"] or
                _sha256(user_path) != source_cert["input_sha256"]["Users.csv"] or
                _sha256(activity_path) != run["activity_sha256"] or
                _sha256(activity_path) != source_cert["input_sha256"]["user_post_comment.json"] or
                _sha256(native_path) != run["native_graph_facts_sha256"] or
                _sha256(native_path) != source_cert["cache_sha256"] or
                _sha256(trace_path) != source_cert["trace_sha256"]):
            raise ValueError("raw BotSim source differs from audited training source")
        with user_path.open("r", encoding="utf-8-sig", newline="") as stream:
            users = list(csv.DictReader(stream))
        self.ids = [row["user_id"] for row in users]
        activity = json.loads(activity_path.read_text(encoding="utf-8"))
        if (len(self.ids) != 2907 or len(set(self.ids)) != len(self.ids) or
                set(activity) != set(self.ids)):
            raise ValueError("BotSim raw account population differs")
        base = np.asarray([_account_facts(
            row, activity[row["user_id"]], observed_means_only=observed_means_only)
                           for row in users], dtype=np.float32)
        with np.load(native_path, allow_pickle=False) as native:
            if (native["ids"].astype(str).tolist() != self.ids or
                    native["names"].astype(str).tolist() != list(NATIVE_NAMES)):
                raise ValueError("BotSim native graph account or fact order differs")
            graph = native["values"].copy()
        full_facts = np.column_stack((base, graph))
        full_names = BASE_NAMES + NATIVE_NAMES
        retained_names = (tuple(lock["retained_source_facts"])
                          if shortcut_ablated or engagement_ablated else full_names)
        if not set(retained_names) <= set(full_names) or len(set(retained_names)) != len(retained_names):
            raise ValueError("invalid retained BotSim source fact vocabulary")
        self.facts = full_facts[:, [full_names.index(name) for name in retained_names]]
        self.index = {uid: i for i, uid in enumerate(self.ids)}
        model_dir = folder / f"seed{seed}"
        reference_path = model_dir / "reference.json.gz"
        reasoner_path = model_dir / "reasoner.json"
        if (_sha256(reference_path) != run["reference_artifact_sha256"] or
                _sha256(reasoner_path) != run["reasoner_artifact_sha256"]):
            raise ValueError("dev-locked model artifact differs")
        self.reference = LabelBlindReference.load(reference_path)
        self.reasoner = MonotoneSplineReasoner.from_manifest(json.loads(
            reasoner_path.read_text(encoding="utf-8")))
        self.atomic_groups = lock["atomic_groups"]
        self.threshold = float(run["threshold"])
        if (self.reference.version != run["reference_version"] or
                self.reference.names != retained_names or
                not 0 < self.threshold < 1 or
                len(self.reasoner.names) != (10 if shortcut_ablated else 12)):
            raise ValueError("BotSim frozen evidence model contract differs")
        self.source_binding = {
            "raw_users_sha256": _sha256(user_path),
            "raw_activity_sha256": _sha256(activity_path),
            "native_graph_facts_sha256": _sha256(native_path),
            "native_reply_trace_sha256": _sha256(trace_path),
            "independent_native_graph_audit_sha256": _sha256(source_cert_path),
            "dev_lock_sha256": _sha256(lock_path),
            "reference_sha256": _sha256(reference_path),
            "reasoner_sha256": _sha256(reasoner_path),
        }

    def predict_many(self, account_ids: list[str]) -> list[dict]:
        ids = [str(uid) for uid in account_ids]
        if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(self.index):
            raise ValueError("supply distinct IDs from the audited BotSim population")
        facts = self.facts[[self.index[uid] for uid in ids]]
        atomic = self.reference.transform(facts)
        concepts, winners = collapse_group_tails(atomic, self.atomic_groups)
        if (concepts.reference_version != self.reasoner.reference_version or
                concepts.names != self.reasoner.names):
            raise ValueError("frozen concept definition differs")
        probability = self.reasoner.predict_proba(concepts)
        design = self.reasoner._design(concepts)
        effects = np.column_stack([
            design[:, self.reasoner._slices[name]] @
            self.reasoner.coef[self.reasoner._slices[name]]
            for name in concepts.names
        ])
        logit = self.reasoner.intercept + effects.sum(axis=1)
        if np.max(np.abs(expit(logit) - probability)) > 1e-12:
            raise ValueError("evidence contributions do not reconstruct classifier")
        out = []
        for i, uid in enumerate(ids):
            terms = []
            for j, name in enumerate(concepts.names):
                source_index = int(winners[name][i])
                fact_name = self.reference.names[source_index] if source_index >= 0 else None
                tail = self.reasoner.tail[j]
                selected = concepts.upper[i, j] if tail == "upper" else concepts.lower[i, j]
                terms.append({
                    "name": name,
                    "source_fact_name": fact_name,
                    "source_fact_value": float(facts[i, source_index]) if source_index >= 0 else None,
                    "source_locator": (
                        f"{self.root / 'data/raw/botsim_24/Users.csv'}#user_id={uid};"
                        f"{self.root / 'data/raw/botsim_24/user_post_comment.json'}#account_id={uid}"
                        if fact_name and not fact_name.startswith("native_reply_") else
                        f"{self.root / 'data/processed/botsim_v3/reply_sources.jsonl.gz'}#incident_account={uid}"
                        if fact_name else None),
                    "available": bool(concepts.available[i, j]),
                    "evidence_upper": float(concepts.upper[i, j]),
                    "evidence_lower": float(concepts.lower[i, j]),
                    "evidence_reliability": float(concepts.reliability[i, j]),
                    "selected_tail": tail,
                    "model_input_magnitude": float(np.log1p(selected) * concepts.reliability[i, j]),
                    "logit_contribution": float(effects[i, j]),
                })
            out.append({
                "dataset": "botsim", "seed": self.seed, "account_id": uid,
                "reference_version": concepts.reference_version,
                "atomic_reference_version": self.reference.version,
                "intercept": float(self.reasoner.intercept), "terms": terms,
                "logit": float(logit[i]), "probability": float(probability[i]),
                "threshold": self.threshold,
                "prediction": int(probability[i] >= self.threshold),
            })
        return out


class BotSimObservedMeanRuntime(BotSimLiteralZeroRuntime):
    """BotSim inference with empty-denominator means marked unavailable."""

    def __init__(self, root: Path, seed: int):
        super().__init__(root, seed, observed_means_only=True)


class BotSimShortcutAblatedRuntime(BotSimLiteralZeroRuntime):
    """Dev-selected ten-concept model without five simulator-linked cues."""

    def __init__(self, root: Path, seed: int):
        super().__init__(root, seed, observed_means_only=True, shortcut_ablated=True)


class BotSimTwoEngagementCuesRemovedRuntime(BotSimLiteralZeroRuntime):
    """Dev-locked model with upvote ratio and comment score excluded."""

    def __init__(self, root: Path, seed: int):
        super().__init__(root, seed, observed_means_only=True,
                         engagement_ablated=True)

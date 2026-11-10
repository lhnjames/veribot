"""Screen A/B signed dual-tail reasoning and tweet-count support on TwiBot-20 dev only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_twibot20_compact20_capacity import GROUPS
from train_v3_twibot20 import metric, sha256, threshold_on_dev
from verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch, LabelBlindReference


TWEET_FACTS = set(GROUPS["sampled_tweet_stats"])
SUPPORT_PRIOR = 20.0


def evidence_subset(full: EvidenceBatch, keep: list[int], names: list[str],
                    sample_count: np.ndarray, support_aware: bool) -> EvidenceBatch:
    reliability = full.reliability[:, keep].copy()
    if support_aware:
        factor = sample_count / (sample_count + SUPPORT_PRIOR)
        for j, name in enumerate(names):
            if name in TWEET_FACTS:
                reliability[:, j] *= factor
    rule = {"base_reference": full.reference_version, "facts": names,
            "tweet_support_prior": SUPPORT_PRIOR if support_aware else 0.0}
    version = hashlib.sha256(json.dumps(rule, sort_keys=True).encode()).hexdigest()
    return EvidenceBatch(full.upper[:, keep], full.lower[:, keep], reliability,
                         full.available[:, keep], version, tuple(names))


def take(batch: EvidenceBatch, indices: np.ndarray) -> EvidenceBatch:
    return EvidenceBatch(batch.upper[indices], batch.lower[indices],
                         batch.reliability[indices], batch.available[indices],
                         batch.reference_version, batch.names)


def unconstrained_b_only(batch_b: EvidenceBatch, y_b: np.ndarray,
                         batch_dev: EvidenceBatch, l2: float) -> np.ndarray:
    """Same B accounts, tails, and mean-loss L2 penalty, without A signs."""
    def design(batch: EvidenceBatch) -> np.ndarray:
        return np.column_stack((np.log1p(batch.upper) * batch.reliability,
                                np.log1p(batch.lower) * batch.reliability))

    x_b = design(batch_b)
    x_dev = design(batch_dev)
    prior = float(np.clip(np.mean(y_b), 1e-5, 1 - 1e-5))
    initial = np.r_[np.log(prior / (1 - prior)), np.zeros(x_b.shape[1])]

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        intercept, coef = params[0], params[1:]
        logit = intercept + x_b @ coef
        loss = float(np.mean(np.logaddexp(0.0, logit) - y_b * logit) +
                     0.5 * l2 * np.sum(coef ** 2))
        residual = expit(logit) - y_b
        gradient = np.r_[np.mean(residual), x_b.T @ residual / len(y_b) + l2 * coef]
        return loss, gradient

    result = minimize(objective, initial, method="L-BFGS-B", jac=True,
                      options={"maxiter": 1000, "ftol": 1e-11})
    if not result.success:
        raise RuntimeError(f"matched B-only optimization failed: {result.message}")
    return expit(result.x[0] + x_dev @ result.x[1:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    if sha256(args.facts) != compact["base_facts_sha256"] or \
            sha256(args.reference) != compact["reference_artifact_sha256"]:
        raise ValueError("Frozen facts or reference differ from compact model")
    with np.load(args.facts, allow_pickle=False) as saved:
        split = saved["split"].astype(str)
        labels = saved["labels"].astype(int)
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        all_names = saved["names"].astype(str).tolist()
    train = np.flatnonzero(split == "train")
    dev = np.flatnonzero(split == "dev")
    names = compact["retained_facts"]
    if len(names) != 20 or set(names) != set().union(*map(set, GROUPS.values())):
        raise ValueError("Compact 20-fact vocabulary differs")
    sample_count = values[:, all_names.index("tweet_sample_count")]
    if not np.isfinite(sample_count[train]).all() or not np.isfinite(sample_count[dev]).all():
        raise ValueError("Train/dev tweet support is incomplete")
    reference = LabelBlindReference.load(args.reference)
    full = reference.transform(values, contexts=contexts)
    keep = [all_names.index(name) for name in names]
    rows = []
    for support_aware in (False, True):
        batch = evidence_subset(full, keep, names, sample_count, support_aware)
        for seed in (0, 1, 2):
            a, b = train_test_split(train, test_size=.5, stratify=labels[train],
                                    random_state=1000 + seed)
            model = DualTailEvidenceReasoner(GROUPS, l2=.01).fit(
                take(batch, a), labels[a], take(batch, b), labels[b])
            dev_batch = take(batch, dev)
            p = model.predict_proba(dev_batch)
            threshold = threshold_on_dev(p, labels[dev])
            b_only_p = unconstrained_b_only(take(batch, b), labels[b], dev_batch, .01)
            b_only_threshold = threshold_on_dev(b_only_p, labels[dev])
            effects = model.contributions(dev_batch)
            reconstruction = float(np.max(np.abs(model.logits(dev_batch) -
                                                 (model.intercept + sum(effects.values())))))
            if reconstruction > 1e-10 or np.any(model.coef < 0):
                raise ValueError("Signed group decision did not replay")
            rows.append({"support_aware": support_aware, "ab_seed": seed,
                         "a_n": len(a), "b_n": len(b), "dev_n": len(dev),
                         "threshold": threshold,
                         "dev": metric(labels[dev], p, threshold),
                         "matched_b_only_unconstrained_dev": metric(
                             labels[dev], b_only_p, b_only_threshold),
                         "nonzero_tail_coefficients": int(np.count_nonzero(model.coef > 1e-9)),
                         "max_group_logit_reconstruction_error": reconstruction,
                         "manifest": model.manifest()})
    report = {
        "scope": "dev-only TwiBot-20 compact 20-fact feasibility screen; no candidate test evaluated",
        "facts_sha256": sha256(args.facts),
        "reference_sha256": sha256(args.reference),
        "compact_sha256": sha256(args.compact),
        "method": "A fixes both tail signs; disjoint B fits nonnegative coefficients; groups partition all 20 facts; linear within each group; 3 prespecified split seeds",
        "support_rule": "Only six sampled-tweet facts receive n/(n+20), n=observed tweet list length; all other reliability unchanged",
        "comparison_note": "The B-only unconstrained control uses identical B accounts, tails and mean-loss L2 penalty; the saved L1 model instead used all train and L1, so that comparison is exploratory",
        "original_l1_dev": compact["dev"],
        "candidate_test_evaluated": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"original_l1_dev": report["original_l1_dev"],
                      "rows": [{key: value for key, value in row.items() if key != "manifest"}
                               for row in rows]}, indent=2))


if __name__ == "__main__":
    main()

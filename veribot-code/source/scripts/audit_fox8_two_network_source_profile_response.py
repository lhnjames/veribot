"""Compare one raw profile-field edit across two frozen Fox8 neural models.

The same original profile followers_count is set to zero for each of twelve
prediction-blind, hash-selected test accounts. Both models' shared 31-feature
preprocessor and full kNN graph are rebuilt. This is a source sensitivity
diagnostic, not a correction of a known erroneous profile field.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/scripts"))
sys.path.insert(0, str(ROOT / "source"))
from audit_hypbot_fox8_raw_edge_response import DATA, RAW, REPO  # noqa: E402
from train_v2_detector import file_sha256, load_data  # noqa: E402
from verifiable_bot.fox8_adapter import (  # noqa: E402
    FEATURE_FIELDS, profile_features, tweet_features,
)
from verifiable_bot.v2_detector import (  # noqa: E402
    EvidenceGatedDetector, knn_edges, relation_means,
)

HYP_RUN = ROOT / "output/hypbot_fox8_paired/seed0"
GATED_RUN = ROOT / "output/v2_fox8_full/seed0"
OUT = ROOT / "research/certificates/fox8_two_network_profile_source_response.json"
SEED = "fox8-two-network-profile-source-2026-11-04"
N_SELECTED = 12


def rank(uid: str) -> str:
    return hashlib.sha256(f"{SEED}|{uid}".encode()).hexdigest()


def raw_row_features(row: dict, profile: dict | None = None) -> np.ndarray:
    tweets = row["user_tweets"]
    actual_profile = profile if profile is not None else tweets[0]["user"]
    fields = {**profile_features(actual_profile), **tweet_features(tweets)}
    return np.asarray([fields[name] for name in FEATURE_FIELDS], dtype=np.float32)


def main() -> None:
    sys.path.insert(0, str(REPO.resolve()))
    from hypbot.data import knn_graph
    from hypbot.model import HypBot
    from train import PRESETS

    torch.set_num_threads(4)
    args = argparse.Namespace(dataset="fox8", data_dir=DATA, account_file=None,
                              raw_source=[DATA / "features.csv"], knn_k=10)
    x, labels, split, ids, edge, edge_type, hashes, scaler = load_data(args, 0)
    index = {uid: i for i, uid in enumerate(ids)}
    frame = pd.read_csv(DATA / "features.csv")
    if list(frame.columns) != ["user_id", *FEATURE_FIELDS]:
        raise ValueError("Saved feature columns differ from source formula")
    raw_x = frame[list(FEATURE_FIELDS)].fillna(0.0).to_numpy(dtype=np.float32)
    test = np.asarray(split["test"], dtype=np.int64)
    eligible = [ids[int(i)] for i in test if raw_x[int(i), 0] > 0]
    selected = sorted(eligible, key=rank)[:N_SELECTED]
    if len(selected) != N_SELECTED:
        raise ValueError("Insufficient source-edit eligible test accounts")
    raw_records = {}
    raw_locators = {}
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            uid = str(row["user_id"])
            if uid not in selected or uid in raw_records:
                continue
            raw_records[uid] = row
            raw_locators[uid] = {"raw_gzip_line": line_number,
                                 "field": "user_tweets[0].user.followers_count"}
            if len(raw_records) == len(selected):
                break
    if len(raw_records) != N_SELECTED:
        raise ValueError("Selected raw accounts absent")
    for uid in selected:
        row = raw_records[uid]
        if not row["user_tweets"]:
            raise ValueError("Selected account has no raw tweets")
        if not np.allclose(raw_row_features(row), raw_x[index[uid]], rtol=1e-5,
                           atol=1e-5):
            raise ValueError(f"Raw source does not rebuild saved features: {uid}")
        raw_locators[uid]["raw_followers_count"] = int(
            row["user_tweets"][0]["user"]["followers_count"])
    if not torch.equal(knn_graph(x, k=10), edge):
        raise ValueError("Shared graph differs from HypBot's kNN construction")
    if not torch.equal(knn_edges(x, 10)[0], edge):
        raise ValueError("Shared graph differs from gated model kNN construction")
    hyp_saved = json.loads((HYP_RUN / "test_predictions.json").read_text(encoding="utf-8"))
    gated_saved = json.loads((GATED_RUN / "test_predictions.json").read_text(encoding="utf-8"))
    hyp_checkpoint_path = HYP_RUN / "model.pt"
    gated_checkpoint_path = GATED_RUN / "model.pt"
    hyp_checkpoint = torch.load(hyp_checkpoint_path, map_location="cpu", weights_only=True)
    gated_checkpoint = torch.load(gated_checkpoint_path, map_location="cpu", weights_only=False)
    for run, saved, checkpoint_path, checkpoint in (
            ("hypbot", hyp_saved, hyp_checkpoint_path, hyp_checkpoint),
            ("gated", gated_saved, gated_checkpoint_path, gated_checkpoint)):
        manifest = saved["manifest"]
        if (file_sha256(checkpoint_path) != manifest["model_sha256"]
                or checkpoint["threshold"] != manifest["threshold"]
                or any(checkpoint["hashes"][key] != hashes[key]
                       or manifest[key] != hashes[key] for key in (
                           "feature_sha256", "graph_sha256", "split_sha256"))
                or [ids[int(i)] for i in test] !=
                [r["account_id"] for r in saved["predictions"]]):
            raise ValueError(f"{run} frozen checkpoint or test population differs")
    hyp_model = HypBot(x.shape[1], K=hyp_checkpoint["num_sectors"], gated=False,
                       **PRESETS["fox8"]["model"])
    hyp_model.load_state_dict(hyp_checkpoint["model_state"])
    hyp_model.eval()
    if (gated_checkpoint["projection"] is not None or
            not np.array_equal(gated_checkpoint["scaler_mean"], scaler.mean_) or
            not np.array_equal(gated_checkpoint["scaler_scale"], scaler.scale_)):
        raise ValueError("Gated model preprocessing differs from shared features")
    gated_model = EvidenceGatedDetector(
        gated_checkpoint["feature_dim"], gated_checkpoint["graph_dim"],
        gated_checkpoint["num_relations"], gated_checkpoint["hidden"],
        gated_checkpoint["dropout"])
    gated_model.load_state_dict(gated_checkpoint["state_dict"])
    gated_model.eval()
    with torch.no_grad():
        hyp_baseline_logit = hyp_model(x, edge).cpu().numpy()
        hyp_baseline = torch.sigmoid(torch.from_numpy(hyp_baseline_logit)).numpy()
        means, degree = relation_means(x, edge, edge_type, 1)
        gated_baseline_logit = gated_model(x, means, degree).cpu().numpy()
        gated_baseline = torch.sigmoid(torch.from_numpy(gated_baseline_logit)).numpy()
    saved_replay = {
        "hypbot": float(np.max(np.abs(hyp_baseline[test] -
            [r["probability"] for r in hyp_saved["predictions"]]))),
        "gated": float(np.max(np.abs(gated_baseline[test] -
            [r["probability"] for r in gated_saved["predictions"]]))),
    }
    if max(saved_replay.values()) > 1e-6:
        raise ValueError("A frozen neural model does not replay saved predictions")
    rows = []
    for uid in selected:
        j = index[uid]
        original_profile = raw_records[uid]["user_tweets"][0]["user"]
        edited_profile = dict(original_profile)
        edited_profile["followers_count"] = 0
        edited_row = raw_row_features(raw_records[uid], edited_profile)
        if edited_row[0] != 0 or edited_row[FEATURE_FIELDS.index("ff_ratio")] != 0:
            raise ValueError("Profile edit failed to recompute derived ratio")
        changed = [FEATURE_FIELDS[k] for k in range(len(FEATURE_FIELDS))
                   if edited_row[k] != raw_x[j, k]]
        if changed != ["followers_count", "ff_ratio"]:
            raise ValueError(f"Unexpected feature changes for {uid}: {changed}")
        edited_raw = raw_x.copy()
        edited_raw[j] = edited_row
        edited_x = torch.from_numpy(np.clip(
            scaler.transform(edited_raw), -3, 3).astype(np.float32))
        edited_hyp_edge = knn_graph(edited_x, k=10)
        edited_gated_edge, edited_relation = knn_edges(edited_x, 10)
        if not torch.equal(edited_hyp_edge, edited_gated_edge):
            raise ValueError("Neural models disagree on edited kNN graph")
        with torch.no_grad():
            hyp_edited_logit = hyp_model(edited_x, edited_hyp_edge).cpu().numpy()
            hyp_edited = torch.sigmoid(torch.from_numpy(hyp_edited_logit)).numpy()
            edited_means, edited_degree = relation_means(
                edited_x, edited_gated_edge, edited_relation, 1)
            gated_edited_logit = gated_model(
                edited_x, edited_means, edited_degree).cpu().numpy()
            gated_edited = torch.sigmoid(torch.from_numpy(gated_edited_logit)).numpy()
        row = {"account_id": uid, **raw_locators[uid],
               "edited_followers_count": 0,
               "linked_feature_changes": changed,
               "changed_scaled_feature_coordinates": int(torch.count_nonzero(
                   edited_x[j] != x[j]).item()),
               "knn_edge_slots_changed": int(torch.count_nonzero(
                   edited_hyp_edge != edge).item()),
               "models": {}}
        for name, original, edited, original_logit, edited_logit, saved in (
                ("hypbot", hyp_baseline, hyp_edited, hyp_baseline_logit,
                 hyp_edited_logit, hyp_saved),
                ("gated", gated_baseline, gated_edited, gated_baseline_logit,
                 gated_edited_logit, gated_saved)):
            threshold = float(saved["manifest"]["threshold"])
            row["models"][name] = {
                "original_logit": float(original_logit[j]),
                "edited_logit": float(edited_logit[j]),
                "logit_delta": float(edited_logit[j] - original_logit[j]),
                "original_probability": float(original[j]),
                "edited_probability": float(edited[j]),
                "probability_delta": float(edited[j] - original[j]),
                "original_class": int(original[j] >= threshold),
                "edited_class": int(edited[j] >= threshold),
                "test_accounts_with_probability_change_gt_1e_6": int(
                    np.count_nonzero(np.abs(edited[test] - original[test]) > 1e-6)),
            }
        rows.append(row)
    report = {
        "scope": "same prediction-blind raw profile-field edit on two frozen Fox8 seed0 neural architectures",
        "source_operation": "set the first raw tweet's user.followers_count to zero, recompute followers_count and ff_ratio, full scaled 31-feature tensor and cosine kNN graph",
        "not_an_observed_source_correction": True,
        "selection_seed": SEED,
        "eligible_test_accounts_with_positive_followers_count": len(eligible),
        "selected_test_accounts": len(rows),
        "raw_source_sha256": file_sha256(RAW),
        "feature_csv_sha256": file_sha256(DATA / "features.csv"),
        "shared_feature_sha256": hashes["feature_sha256"],
        "shared_knn_graph_sha256": hashes["graph_sha256"],
        "hypbot_model_sha256": file_sha256(hyp_checkpoint_path),
        "gated_model_sha256": file_sha256(gated_checkpoint_path),
        "saved_prediction_replay_max_abs_error": saved_replay,
        "hypbot_selected_target_probability_responses_gt_1e_6": sum(
            abs(r["models"]["hypbot"]["probability_delta"]) > 1e-6 for r in rows),
        "gated_selected_target_probability_responses_gt_1e_6": sum(
            abs(r["models"]["gated"]["probability_delta"]) > 1e-6 for r in rows),
        "hypbot_selected_target_logit_responses_gt_1e_6": sum(
            abs(r["models"]["hypbot"]["logit_delta"]) > 1e-6 for r in rows),
        "gated_selected_target_logit_responses_gt_1e_6": sum(
            abs(r["models"]["gated"]["logit_delta"]) > 1e-6 for r in rows),
        "hypbot_selected_target_class_changes": sum(
            r["models"]["hypbot"]["original_class"] !=
            r["models"]["hypbot"]["edited_class"] for r in rows),
        "gated_selected_target_class_changes": sum(
            r["models"]["gated"]["original_class"] !=
            r["models"]["gated"]["edited_class"] for r in rows),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

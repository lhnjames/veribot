"""Choose a decision threshold on dev only, then report fixed-test results."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "source/scripts"))
sys.path.insert(0, str(ROOT / "source/hgt_import/code"))

from hgt_model import VerifiableHGT
from relnet_reimplementation import GatedRelNet
from train_aaai_veribot_fox8 import load_inputs as fox8_inputs
from train_aaai_veribot_botsim import load_inputs as botsim_inputs
from verifiable_bot.aaai_dense_relnet import DenseGatedRelNet


def choose_threshold(logits: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    ordered = np.sort(np.unique(logits))
    candidates = np.r_[-np.inf, 0.0,
                       (ordered[:-1] + ordered[1:]) / 2,
                       np.inf]
    scores = np.asarray([np.count_nonzero((logits >= t) == labels)
                         for t in candidates])
    best = int(scores.max())
    eligible = candidates[scores == best]
    # The fixed threshold zero wins ties; otherwise choose the smallest shift.
    chosen = float(min(eligible, key=lambda value: abs(value)))
    return chosen, best


def evaluate(dataset: str) -> dict:
    if dataset == "fox8":
        x, y, split, ids, graph, pack = fox8_inputs()
        inputs = {"user": x}
        relnet_edges = pack["relnet_edges"]
        metadata = pack["metadata"]
        relnet_type = GatedRelNet
        output = ROOT / "output/aaai_veribot_fox8/seed0"
    else:
        inputs, y, split, ids, graph, relnet_edges, metadata, _ = botsim_inputs()
        relnet_type = DenseGatedRelNet
        output = ROOT / "output/aaai_veribot_botsim/seed0"
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=False)
    hgt = VerifiableHGT(metadata,
                        {name: value.shape[1] for name, value in inputs.items()},
                        hidden_channels=128, out_channels=1,
                        num_layers=2, heads=4, dropout=0.3)
    relnet = relnet_type(inputs["user"].shape[1], hidden_channels=128,
                         relation_names=tuple(relnet_edges),
                         num_layers=2, dropout=0.3)
    hgt.load_state_dict(checkpoint["model_state"]["hgt"])
    relnet.load_state_dict(checkpoint["model_state"]["relnet"])
    hgt.eval()
    relnet.eval()
    with torch.inference_mode():
        hgt_logit = hgt(inputs, graph)["bot_logit"].numpy()
        relnet_logit = relnet(inputs["user"], relnet_edges).numpy()
    outputs = {"hgt": hgt_logit, "relnet": relnet_logit,
               "mean_logit": 0.5 * (hgt_logit + relnet_logit)}
    saved = json.loads((output / "test_predictions.json").read_text(encoding="utf-8"))
    by_id = {row["account_id"]: row for row in saved["predictions"]}
    maximum = max(abs(float(values[int(i)]) - by_id[ids[int(i)]][f"{name}_logit"])
                  for i in split["test"]
                  for name, values in (("hgt", hgt_logit),
                                       ("relnet", relnet_logit),
                                       ("fused", outputs["mean_logit"])))
    if maximum > 1e-5:
        raise ValueError(f"{dataset} checkpoint replay differs from saved test outputs")
    truth = y.numpy().astype(int)
    result = {"dataset": dataset, "checkpoint": str(output / "model.pt"),
              "checkpoint_sha256": sha256((output / "model.pt").read_bytes()).hexdigest(),
              "saved_predictions_sha256": sha256((output / "test_predictions.json").read_bytes()).hexdigest(),
              "selection_population": "dev only",
              "saved_test_replay_max_difference": maximum,
              "split_sizes": {name: len(indices) for name, indices in split.items()},
              "variants": {}}
    for name, values in outputs.items():
        dev, test = split["dev"], split["test"]
        threshold, best_dev_correct = choose_threshold(values[dev], truth[dev])
        variant = {"threshold_logit": threshold,
                   "dev_correct_at_zero": int(np.count_nonzero((values[dev] >= 0) == truth[dev])),
                   "dev_correct_at_selected": best_dev_correct,
                   "dev_n": len(dev),
                   "test_correct_at_zero": int(np.count_nonzero((values[test] >= 0) == truth[test])),
                   "test_correct_at_selected": int(np.count_nonzero((values[test] >= threshold) == truth[test])),
                   "test_n": len(test),
                   "test_accuracy_at_selected": float(accuracy_score(
                       truth[test], values[test] >= threshold)),
                   "test_macro_f1_at_selected": float(f1_score(
                       truth[test], values[test] >= threshold, average="macro"))}
        result["variants"][name] = variant
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research/certificates/aaai_veribot_dev_threshold_audit.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    results = [evaluate(name) for name in ("fox8", "botsim")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results}, indent=2) + "\n",
                           encoding="utf-8")
    audit_sha256 = sha256(args.output.read_bytes()).hexdigest()
    for result in results:
        variant = result["variants"]["mean_logit"]
        policy = {
            "schema_version": "veribot-mean-logit-dev-threshold-v1",
            "dataset": result["dataset"],
            "branch_combination": "mean_logit_hgt_relnet",
            "checkpoint_sha256": result["checkpoint_sha256"],
            "saved_predictions_sha256": result["saved_predictions_sha256"],
            "threshold_selection_audit_sha256": audit_sha256,
            "threshold_selection": "maximize dev accuracy over distinct-logit midpoints and zero; ties choose the smallest absolute threshold",
            "selection_population": "dev only",
            "logit_threshold": variant["threshold_logit"],
            "probability_threshold": float(1 / (1 + np.exp(-variant["threshold_logit"]))),
            "dev_n": variant["dev_n"],
            "dev_correct": variant["dev_correct_at_selected"],
            "test_n": variant["test_n"],
            "test_correct": variant["test_correct_at_selected"],
            "test_accuracy": variant["test_accuracy_at_selected"],
            "test_macro_f1": variant["test_macro_f1_at_selected"],
        }
        destination = ROOT / f"output/aaai_veribot_{result['dataset']}/seed0/decision_policy.json"
        destination.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "results": [
        {"dataset": result["dataset"],
         "selected_mean_logit": result["variants"]["mean_logit"]}
        for result in results]}), flush=True)


if __name__ == "__main__":
    main()

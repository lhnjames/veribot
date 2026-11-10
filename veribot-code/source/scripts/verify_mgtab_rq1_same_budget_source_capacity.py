"""Read back six same-budget MGTAB dev models against released source facts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import lightgbm
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_mgtab_rq1_same_budget_source_capacity import matrix
from gate0_shallow_tree import evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_content_projection_evidence import transform_with_support
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research/certificates/mgtab_rq1_same_budget_source_capacity.json"
OUTPUT = ROOT / "research/certificates/mgtab_rq1_same_budget_source_capacity_readback.json"


def run() -> dict:
    audit = json.loads(SOURCE.read_text(encoding="utf-8"))
    if audit["fixed_test_predictions_or_metrics_computed"] is not False:
        raise ValueError("capacity audit unexpectedly used fixed test")
    args = SimpleNamespace(
        dataset="mgtab",
        data_dir=Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB"),
        account_file=None, native_graph_facts=None,
        structural_facts=ROOT / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=ROOT / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=ROOT / "output/v3_mgtab_ab_spline_content/selected")
    runs = []
    for row in audit["runs"]:
        seed = row["seed"]
        source_report = args.selected_model_dir / f"seed{seed}/report.json"
        ref_path = args.selected_model_dir / f"seed{seed}/reference.json.gz"
        pred_path = ROOT / row["prediction_path"]
        if (sha256_file(source_report) != row["source_report_sha256"] or
                sha256_file(ref_path) != row["reference_sha256"] or
                sha256_file(pred_path) != row["prediction_sha256"]):
            raise ValueError("source report, reference or predictions changed")
        original = json.loads(source_report.read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, original)
        train, dev = split["train"], split["val"]
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=y[train], random_state=seed)
        reference = LabelBlindReference.load(ref_path)
        if reference.version != original["manifest"]["reference_version"]:
            raise ValueError("reference version changed")
        evidence = transform_with_support(
            reference, x[dev], split["supports"][dev])
        encoded = matrix(evidence)
        with np.load(pred_path, allow_pickle=False) as saved:
            if (not np.array_equal(saved["dev_indices"], dev) or
                    not np.array_equal(saved["train_A_indices"], a) or
                    not np.array_equal(saved["train_B_indices"], b)):
                raise ValueError("saved A/B and dev partitions differ")
            max_error = 0.0
            model_count = 0
            for key, item in row["results"].items():
                model_path = ROOT / item["model_path"]
                if sha256_file(model_path) != item["model_sha256"]:
                    raise ValueError("saved candidate model differs")
                booster = lightgbm.Booster(model_file=str(model_path))
                if item["selected_fact_names"] is None:
                    if key == "all_named86":
                        selection = list(range(86))
                    elif key == "all_named_relation66":
                        selection = list(range(20, 86))
                    elif key == "all_named_and_content150":
                        selection = list(range(150))
                    else:
                        raise ValueError("unrecognized full fact set")
                else:
                    selection = [names.index(fact)
                                 for fact in item["selected_fact_names"]]
                if len(selection) != item["fact_count"]:
                    raise ValueError("model fact count differs")
                columns = np.r_[selection, np.asarray(selection)+150]
                probability = np.asarray(booster.predict(encoded[:, columns]), dtype=float)
                error = float(np.max(np.abs(probability-saved[key])))
                max_error = max(max_error, error)
                if error > 1e-12:
                    raise ValueError("source-to-model dev probabilities differ")
                metrics = evaluate(y[dev], probability, 0.5)
                if (metrics["errors"] != item["dev"]["errors"] or
                        abs(metrics["macro_f1"]-item["dev"]["macro_f1"]) > 1e-12):
                    raise ValueError("saved fixed-threshold dev metrics differ")
                model_count += 1
        runs.append({"seed": seed, "models_verified": model_count,
                     "dev_accounts": len(dev),
                     "maximum_probability_replay_error": max_error})
        print(json.dumps(runs[-1]), flush=True)
    result = {
        "status": "all_mgtab_same_budget_source_capacity_models_replayed_from_published_facts",
        "source_certificate_sha256": sha256_file(SOURCE),
        "fixed_test_predictions_or_metrics_computed": False,
        "runs": runs,
    }
    OUTPUT.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    run()

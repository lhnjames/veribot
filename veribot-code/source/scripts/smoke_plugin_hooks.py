"""Remote-only smoke test for a shared intermediate-state plugin interface.

This checks capture, identity replay, and a synthetic patch. The synthetic
patch is deliberately not interpreted as input evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "hgt_import" / "code"))
from hgt_model import VerifiableHGT  # noqa: E402
from relnet_reimplementation import GatedRelNet  # noqa: E402


def select_validation_account(data) -> int:
    for index, account_id in enumerate(data["user"].account_id):
        bucket = int(hashlib.sha256(f"hgt_val_v1:{account_id}".encode()).hexdigest()[:8], 16) % 20
        if bucket == 0 and not bool(data["user"].holdout_mask[index]):
            return index
    raise RuntimeError("no validation account found")


def audit_hook(model, layer, forward, target: int, kind: str) -> dict:
    state = {}

    def get_tensor(output):
        return output["user"] if isinstance(output, dict) else output

    def capture(_module, _inputs, output):
        state["captured"] = get_tensor(output).detach().clone()

    handle = layer.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            baseline = forward()
    finally:
        handle.remove()

    def replace_with(saved):
        def hook(_module, _inputs, output):
            tensor = get_tensor(output).clone()
            tensor[target] = saved
            if isinstance(output, dict):
                return {**output, "user": tensor}
            return tensor
        return hook

    h = layer.register_forward_hook(replace_with(state["captured"][target]))
    try:
        with torch.inference_mode():
            identity = forward()
    finally:
        h.remove()

    synthetic = state["captured"][target] + 0.1
    h = layer.register_forward_hook(replace_with(synthetic))
    try:
        with torch.inference_mode():
            changed = forward()
    finally:
        h.remove()

    max_identity_delta = float((identity - baseline).abs().max().item())
    if max_identity_delta > 1e-5:
        raise RuntimeError(f"{kind}: identity patch changed predictions by {max_identity_delta}")
    return {
        "model": kind,
        "site": "first message-passing layer output",
        "target_index": target,
        "activation_shape": list(state["captured"].shape),
        "baseline_target_logit": float(baseline[target].item()),
        "identity_max_abs_logit_delta_all_accounts": max_identity_delta,
        "synthetic_target_logit_delta": float((changed[target] - baseline[target]).item()),
        "synthetic_patch_note": "adds 0.1 to one hidden vector; interface smoke test only, not source evidence",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("plugin smoke test must run on remote CUDA server")

    graph = torch.load(args.run / "data" / "clean_train_only_graph.pt", map_location="cpu", weights_only=False)
    target = select_validation_account(graph)
    x_dict = {name: x.cuda() for name, x in graph.x_dict.items()}
    edge_dict = {name: e.cuda() for name, e in graph.edge_index_dict.items()}

    hgt_ckpt = torch.load(args.run / "checkpoints" / "clean_hgt_best.pt", map_location="cpu", weights_only=False)
    hgt = VerifiableHGT(
        metadata=hgt_ckpt["metadata"], in_channels=hgt_ckpt["in_channels"],
        hidden_channels=hgt_ckpt["config"]["hidden_channels"], out_channels=1,
        num_layers=hgt_ckpt["config"]["num_layers"], heads=hgt_ckpt["config"]["heads"],
        dropout=hgt_ckpt["config"]["dropout"],
    ).cuda().eval()
    hgt.load_state_dict(hgt_ckpt["model_state"])
    hgt_result = audit_hook(
        hgt, hgt.convs[0], lambda: hgt(x_dict, edge_dict)["bot_logit"], target, "HGT",
    )
    del hgt

    rel_ckpt = torch.load(args.run / "relnet_reimplementation" / "relnet_reimplementation_best.pt", map_location="cpu", weights_only=False)
    relnet = GatedRelNet(
        input_channels=rel_ckpt["input_channels"],
        hidden_channels=rel_ckpt["hidden_channels"],
        relation_names=tuple(rel_ckpt["relation_names"]),
        num_layers=rel_ckpt["num_layers"], dropout=rel_ckpt["dropout"],
    ).cuda().eval()
    relnet.load_state_dict(rel_ckpt["model_state"])
    edges = {"follows": edge_dict[("user", "follows", "user")],
             "rev_follows": edge_dict[("user", "rev_follows", "user")]}
    rel_result = audit_hook(
        relnet, relnet.layers[0], lambda: relnet(x_dict["user"], edges), target, "RelNet reimplementation",
    )

    report = {
        "status": "interface smoke test only; no attribution or evidence claim",
        "graph": str(args.run / "data" / "clean_train_only_graph.pt"),
        "account_id": str(graph["user"].account_id[target]),
        "models": [hgt_result, rel_result],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

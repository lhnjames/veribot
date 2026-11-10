"""Train a clean (leakage-free) VerifiableHGT on the graph built by build_features_and_graph.py.

Three-way split, strictly separated by purpose (fixes a model-selection leakage bug found in an
earlier version of this script, which used the project's own internal 5% holdout for BOTH
early-stopping/checkpoint-selection AND final reporting -- "not in the loss" is not enough when the
same set also drives which epoch/threshold gets picked):

  1. hgt_train      (~90% of the 8278 official-train accounts): the ONLY labels ever used in the
                     loss / backward pass.
  2. hgt_val        (~5%, carved from the 95% non-project-holdout pool with an INDEPENDENT salted
                     deterministic rule, disjoint from verifiable_bot/data.py's internal_split):
                     drives early stopping (best-checkpoint selection by AUC) AND classification
                     threshold selection. Never touches the loss.
  3. project_holdout (~5%, verifiable_bot/data.py's internal_split() verbatim -- the SAME holdout
                     the rest of the project uses): touched exactly ONCE, after the model and the
                     classification threshold are both already frozen from hgt_val. Pure reporting.

Run with `python -u` (unbuffered) so progress is visible in the log while running, not only after
the process exits.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
from hgt_model import VerifiableHGT  # noqa: E402

DATA_DIR = Path(os.environ.get("VERIBOT_HGT_DATA_DIR", HERE / "data"))
OUT_DIR = Path(os.environ.get("VERIBOT_HGT_OUT_DIR", HERE / "checkpoints"))
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HGT_VAL_SALT = "hgt_val_v1"  # independent from verifiable_bot/data.py's internal_split rule


def hgt_val_split(uid: str) -> str:
    """Deterministic, salted differently from internal_split() so hgt_val and project_holdout are
    guaranteed to be computed by unrelated hash inputs (not just 'the same rule with bad luck')."""
    key = f"{HGT_VAL_SALT}:{uid}"
    return "hgt_val" if int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 20 == 0 else "hgt_train"


class Config:
    lr = 5e-4
    epochs = 300
    patience = 40  # real early stopping -- was bugged to equal `epochs` (=never triggers) before
    hidden_channels = 128
    heads = 4
    num_layers = 2
    dropout = 0.3
    seed = 42


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def focal_loss(logits, targets, alpha_pos: float, gamma: float = 2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha_pos * targets + (1 - alpha_pos) * (1 - targets)
    return (a_t * (1 - p_t) ** gamma * ce).mean()


def compute_metrics(probs, y_true, thr: float) -> dict:
    """thr is passed in, never re-searched on this function's own data -- the caller is
    responsible for choosing thr on hgt_val ONLY and reusing it everywhere else."""
    pred = (probs >= thr).astype(int)
    return dict(
        acc=accuracy_score(y_true, pred), auc=roc_auc_score(y_true, probs), thr=thr,
        f1=f1_score(y_true, pred, average="macro"),
        f1_bot=f1_score(y_true, pred, pos_label=1),
        f1_hum=f1_score(y_true, pred, pos_label=0),
    )


def best_threshold_on(probs, y_true) -> float:
    best_acc, best_thr = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 100):
        acc = accuracy_score(y_true, (probs >= t).astype(int))
        if acc > best_acc:
            best_acc, best_thr = acc, t
    return best_thr


def main():
    cfg = Config()
    set_seed(cfg.seed)
    print(f"Device: {DEVICE}", flush=True)

    data = torch.load(DATA_DIR / "clean_train_only_graph.pt", map_location="cpu", weights_only=False)
    print(f"Loaded clean graph: {data}", flush=True)

    labels = data["user"].y
    account_ids = data["user"].account_id
    project_holdout_mask = data["user"].holdout_mask  # verifiable_bot/data.py internal_split()
    non_project_holdout = ~project_holdout_mask

    hgt_val_flags = torch.tensor([hgt_val_split(uid) == "hgt_val" for uid in account_ids])
    hgt_val_mask = hgt_val_flags & non_project_holdout
    hgt_train_mask = (~hgt_val_flags) & non_project_holdout

    n_train, n_val, n_holdout = (hgt_train_mask.sum().item(), hgt_val_mask.sum().item(),
                                  project_holdout_mask.sum().item())
    assert n_train + n_val + n_holdout == labels.shape[0]
    print(f"[split] hgt_train={n_train} ({n_train/labels.shape[0]*100:.1f}%), "
          f"hgt_val={n_val} ({n_val/labels.shape[0]*100:.1f}%), "
          f"project_holdout={n_holdout} ({n_holdout/labels.shape[0]*100:.1f}%) "
          f"-- rules: hgt_train/hgt_val split by sha256('{HGT_VAL_SALT}:'+account_id)[:8] mod 20; "
          f"project_holdout by verifiable_bot/data.py internal_split() verbatim (sha256(account_id)"
          f"[:8] mod 20)", flush=True)

    # ── class-balanced focal alpha, computed from hgt_train labels only (never hardcoded, never
    # from val/holdout) -- alpha_pos = fraction of hgt_train that is NEGATIVE (human), so the
    # minority class in hgt_train gets upweighted regardless of which direction the imbalance runs.
    train_labels_np = labels[hgt_train_mask].numpy()
    n_pos_train = int((train_labels_np == 1).sum())
    n_neg_train = int((train_labels_np == 0).sum())
    alpha_pos = n_neg_train / (n_pos_train + n_neg_train)
    print(f"[class balance] hgt_train bot={n_pos_train} human={n_neg_train} "
          f"-> focal alpha_pos(bot)={alpha_pos:.4f} (class-balanced, not hardcoded)", flush=True)

    x_dict = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}
    labels_d = labels.to(DEVICE)
    hgt_train_mask_d = hgt_train_mask.to(DEVICE)

    in_channels = {nt: x_dict[nt].shape[1] for nt in x_dict}
    model = VerifiableHGT(
        metadata=data.metadata(), in_channels=in_channels, hidden_channels=cfg.hidden_channels,
        out_channels=1, num_layers=cfg.num_layers, heads=cfg.heads, dropout=cfg.dropout,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.02)

    best_val_auc, best_state, patience_ctr = 0.0, None, 0
    labels_np = labels.numpy()
    val_np = hgt_val_mask.numpy()

    print("=" * 60, flush=True)
    print("  Training clean VerifiableHGT (hgt_train loss only; hgt_val drives early stop + "
          "threshold; project_holdout untouched until the very end)", flush=True)
    print("=" * 60, flush=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x_dict, edge_index_dict)
        loss = focal_loss(out["bot_logit"][hgt_train_mask_d], labels_d[hgt_train_mask_d], alpha_pos)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            out_v = model(x_dict, edge_index_dict)
            probs_v = torch.sigmoid(out_v["bot_logit"]).cpu().numpy()
        val_auc = roc_auc_score(labels_np[val_np], probs_v[val_np])

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                  f"hgt_val AUC: {val_auc*100:.2f}% | Best: {best_val_auc*100:.2f}% | "
                  f"patience {patience_ctr}/{cfg.patience}", flush=True)
        if patience_ctr >= cfg.patience:
            print(f"  Early stopping at epoch {epoch} (no hgt_val AUC improvement for "
                  f"{cfg.patience} epochs)", flush=True)
            break

    model.load_state_dict(best_state)

    # ── threshold chosen on hgt_val ONLY, then frozen ───────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        out_final = model(x_dict, edge_index_dict)
        probs_final = torch.sigmoid(out_final["bot_logit"]).cpu().numpy()
    frozen_thr = best_threshold_on(probs_final[val_np], labels_np[val_np])
    print(f"[threshold] chosen on hgt_val only: {frozen_thr:.3f}", flush=True)

    m_val = compute_metrics(probs_final[val_np], labels_np[val_np], frozen_thr)
    print(f"\nhgt_val metrics (model-selection set, NOT independent reporting): {m_val}", flush=True)

    # ── project_holdout touched exactly once, here, model+threshold already frozen ─────────────
    holdout_np = project_holdout_mask.numpy()
    m_holdout = compute_metrics(probs_final[holdout_np], labels_np[holdout_np], frozen_thr)
    print(f"\nproject_holdout metrics (independent, touched exactly once, model+threshold frozen "
          f"beforehand): {m_holdout}", flush=True)

    config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith("_")}
    metadata_plain = (list(data.metadata()[0]), [tuple(t) for t in data.metadata()[1]])
    ckpt_path = OUT_DIR / "clean_hgt_best.pt"
    torch.save({
        "model_state": best_state, "best_hgt_val_auc": best_val_auc, "frozen_threshold": frozen_thr,
        "in_channels": in_channels, "metadata": metadata_plain, "config": config_dict,
        "alpha_pos": alpha_pos, "n_hgt_train": n_train, "n_hgt_val": n_val,
        "n_project_holdout": n_holdout,
    }, ckpt_path)
    print(f"Checkpoint saved -> {ckpt_path}", flush=True)

    with open(OUT_DIR / "clean_hgt_training_report.json", "w") as f:
        json.dump({
            "split_sizes": {"hgt_train": n_train, "hgt_val": n_val, "project_holdout": n_holdout},
            "split_rules": {
                "hgt_train_vs_hgt_val": f"sha256('{HGT_VAL_SALT}:'+account_id)[:8] as int mod 20 == 0 -> hgt_val",
                "project_holdout": "verifiable_bot/data.py internal_split(): sha256(account_id)[:8] as int mod 20 == 0 -> holdout",
            },
            "class_balance": {"alpha_pos_bot": alpha_pos, "n_pos_train": n_pos_train, "n_neg_train": n_neg_train},
            "best_hgt_val_auc": best_val_auc,
            "frozen_threshold": frozen_thr,
            "hgt_val_metrics": m_val,
            "project_holdout_metrics": m_holdout,
        }, f, indent=2)
    print("Report written.", flush=True)


if __name__ == "__main__":
    main()

"""Unified clean trainer for CleanCoalitionNet (Forward A) & CleanChallengerNet (Reverse B).

Strictly Dev-selected and Dev-calibrated; never evaluates Test.
Zero mask-invariance penalty hacks needed: payload-vs-type separation mathematically guarantees
zero_all & zero_payload outputs equal an exact constant scalar across all masks.
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from veribot.forward_miner import ForwardEvidenceMiner, CleanCoalitionNet
from veribot.heterogeneous_challenger import HeterogeneousChallenger, CleanChallengerNet
from veribot.primitives import resolve_payload_cols_auto as resolve_payload_cols


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def group_family_id(name: str) -> int:
    prefix = name.split(":", 1)[0]
    if prefix in ("profile", "user"):
        return 0
    if prefix in ("relation", "reply", "neighbor", "neighbor_profile"):
        return 1
    if prefix in ("content", "tweet", "posts", "comment_1", "comment_2"):
        return 2
    return 3


def group_slot_key(name: str) -> str:
    part = name.split(":")
    if part[0] == "profile" or part[0] == "content" or part[0] == "relation":
        return name
    if len(part) >= 2 and part[0] in ("neighbor_profile", "tweet", "neighbor", "reply"):
        return f"{part[0]}:{part[1]}"
    return part[0]


def precompute_fams_and_slots(cache: dict) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
    fams = [torch.tensor([group_family_id(n) for n in ns], dtype=torch.long) for ns in cache["group_names"]]
    unique_exact = {n for ns in cache["group_names"] for n in ns}
    use_exact = len(unique_exact) <= 256
    vocab: dict[str, int] = {}
    slots: list[torch.Tensor] = []
    for ns in cache["group_names"]:
        ids = []
        for n in ns:
            k = n if use_exact else group_slot_key(n)
            if k not in vocab:
                vocab[k] = len(vocab)
            ids.append(vocab[k])
        slots.append(torch.tensor(ids, dtype=torch.long))
    return fams, slots, len(vocab)


def resolve_payload_cols(cache: dict, train_positions: list[int], spec: str) -> np.ndarray:
    dim = int(cache["features"][0].shape[1])
    if spec and spec.lower() != "auto":
        cols: list[int] = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                s, e = part.split(":", 1)
                cols.extend(range(int(s), min(int(e), dim)))
            else:
                idx = int(part)
                if 0 <= idx < dim:
                    cols.append(idx)
        return np.array(sorted(set(cols)), dtype=np.int64)
    if "payload_slice" in cache:
        s, e = int(cache["payload_slice"][0]), int(cache["payload_slice"][1])
        return np.arange(s, min(e, dim), dtype=np.int64)
    by_name: dict[str, list[np.ndarray]] = {}
    for pos in train_positions[:800]:
        x_np = cache["features"][int(pos)].numpy()
        for nm, row in zip(cache["group_names"][int(pos)], x_np):
            lst = by_name.setdefault(nm, [])
            if len(lst) < 60:
                lst.append(row)
    const_within = np.ones(dim, dtype=bool)
    used_groups = 0
    for lst in by_name.values():
        if len(lst) >= 5:
            used_groups += 1
            const_within &= np.stack(lst, axis=0).std(axis=0) < 1e-4
    if used_groups == 0:
        return np.arange(dim, dtype=np.int64)
    return np.where(~const_within)[0].astype(np.int64)


def batchify(
    cache: dict,
    fams_all: list[torch.Tensor],
    slots_all: list[torch.Tensor],
    positions: list[int],
    device: torch.device,
):
    x = [cache["features"][int(i)] for i in positions]
    length = max(len(t) for t in x)
    dim = x[0].shape[1]
    padded = torch.zeros(len(x), length, dim, dtype=torch.float32)
    valid = torch.zeros(len(x), length, dtype=torch.bool)
    fams = torch.zeros(len(x), length, dtype=torch.long)
    slots = torch.zeros(len(x), length, dtype=torch.long)
    for j, i in enumerate(positions):
        idx = int(i)
        t = x[j]
        L = len(t)
        padded[j, :L] = t
        valid[j, :L] = True
        fams[j, :L] = fams_all[idx]
        slots[j, :L] = slots_all[idx]
    labels = cache["labels"][list(positions)].float()
    return padded.to(device), valid.to(device), fams.to(device), slots.to(device), labels.to(device)


def sample_subset_mask(mask: torch.Tensor, min_keep: int = 2, keep_lo: float = 0.55, keep_hi: float = 0.85) -> torch.Tensor:
    b, m = mask.shape
    device = mask.device
    probs = torch.empty((b, 1), device=device).uniform_(keep_lo, keep_hi)
    sub = (torch.rand((b, m), device=device) < probs) & mask
    counts = mask.sum(dim=1)
    need = (sub.sum(dim=1) < torch.clamp(counts, max=min_keep)) & (counts > 0)
    if need.any():
        noise = torch.rand((b, m), device=device).masked_fill(~mask, -1.0)
        topk = torch.topk(noise, k=min(min_keep, m), dim=1).indices
        for row_i in torch.nonzero(need, as_tuple=False).squeeze(-1):
            k_req = int(min(min_keep, int(counts[row_i].item())))
            sub[row_i, topk[row_i, :k_req]] = True
    full_eq = (sub.sum(dim=1) == counts) & (counts >= 3)
    if full_eq.any():
        drop_scores = torch.rand((b, m), device=device).masked_fill(~sub, -1.0)
        drop_idx = drop_scores.argmax(dim=1)
        for row_i in torch.nonzero(full_eq, as_tuple=False).squeeze(-1):
            sub[row_i, drop_idx[row_i]] = False
    return sub & mask


def sample_compact_witness_mask(mask: torch.Tensor, fams: torch.Tensor, min_k: int = 2, max_k: int = 5) -> torch.Tensor:
    """Sample compact k-token witness subsets (k in [min_k, max_k]) balanced across active families."""
    b, m = mask.shape
    device = mask.device
    sub = torch.zeros_like(mask)
    noise = torch.rand((b, m), device=device).masked_fill(~mask, -1.0)
    # First pick at most 1 random group from each active family (up to 4)
    for fid in range(4):
        fm = mask & (fams == fid)
        has_f = fm.any(dim=1)
        if has_f.any():
            best_j = noise.masked_fill(~fm, -1.0).argmax(dim=1)
            rows = torch.nonzero(has_f, as_tuple=False).squeeze(-1)
            sub[rows, best_j[rows]] = True
    # Fill remaining tokens up to random target k in [min_k, max_k] or trim if > max_k
    counts = mask.sum(dim=1)
    target_k = torch.randint(min_k, max_k + 1, (b,), device=device)
    target_k = torch.minimum(target_k, counts)
    need_more = (sub.sum(dim=1) < target_k) & (counts > 0)
    if need_more.any():
        rem_noise = torch.rand((b, m), device=device).masked_fill(~(mask & ~sub), -1.0)
        order = torch.argsort(rem_noise, dim=1, descending=True)
        for r in torch.nonzero(need_more, as_tuple=False).squeeze(-1):
            need = int((target_k[r] - sub[r].sum()).item())
            if need > 0:
                sub[r, order[r, :need]] = True
    too_many = sub.sum(dim=1) > target_k
    if too_many.any():
        for r in torch.nonzero(too_many, as_tuple=False).squeeze(-1):
            idx = torch.nonzero(sub[r], as_tuple=False).squeeze(-1)
            perm = idx[torch.randperm(len(idx), device=device)[: int((sub[r].sum() - target_k[r]).item())]]
            sub[r, perm] = False
    return sub & mask


def compute_balanced_budget_loss(
    prob: torch.Tensor,
    valid: torch.Tensor,
    fams: torch.Tensor,
    min_frac: float,
    max_frac: float,
    per_family: bool = True,
) -> torch.Tensor:
    valid_f = valid.float()
    frac = (prob * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp(min=1.0)
    loss = (F.relu(min_frac - frac).pow(2) + F.relu(frac - max_frac).pow(2)).mean()
    if per_family:
        fam_terms = []
        for fid in range(4):
            fm = valid & (fams == fid)
            fc = fm.sum(dim=1)
            elig = fc >= 3
            if elig.any():
                f_frac = (prob[elig] * fm[elig].float()).sum(dim=1) / fc[elig].float()
                fam_terms.append((F.relu(min_frac - f_frac).pow(2) + F.relu(f_frac - max_frac).pow(2)).mean())
        if fam_terms:
            loss = 0.5 * loss + 0.5 * (sum(fam_terms) / len(fam_terms))
    return loss


def compute_slot_parity_loss(
    prob: torch.Tensor,
    valid: torch.Tensor,
    slots: torch.Tensor,
    y: torch.Tensor,
    num_slots: int,
) -> torch.Tensor:
    """Match marginal slot-selection rates E[g_s | Y=1] vs E[g_s | Y=0] to prevent slot steganography."""
    bot_m = (y > 0.5).unsqueeze(1) & valid
    hum_m = (y < 0.5).unsqueeze(1) & valid
    if not (bot_m.any() and hum_m.any()):
        return prob.new_zeros(())
    b_idx = slots[bot_m]
    b_val = prob[bot_m]
    h_idx = slots[hum_m]
    h_val = prob[hum_m]

    b_sum = prob.new_zeros(num_slots).scatter_add_(0, b_idx, b_val)
    b_cnt = torch.bincount(b_idx, minlength=num_slots).float()
    h_sum = prob.new_zeros(num_slots).scatter_add_(0, h_idx, h_val)
    h_cnt = torch.bincount(h_idx, minlength=num_slots).float()

    common = (b_cnt >= 2.0) & (h_cnt >= 2.0)
    if not common.any():
        return prob.new_zeros(())
    diff = (b_sum[common] / b_cnt[common]) - (h_sum[common] / h_cnt[common])
    return diff.pow(2).mean()


def evaluate(
    model,
    role: str,
    cache: dict,
    fams_all: list[torch.Tensor],
    slots_all: list[torch.Tensor],
    positions: list[int],
    device: torch.device,
    payload_cols_t: torch.Tensor,
    batch_size: int = 128,
    build_rows: bool = False,
):
    model.eval()
    scores, all_scores, unary_scores, zall_scores, zpay_scores, selected = [], [], [], [], [], []
    rows = []
    with torch.inference_mode():
        for start in range(0, len(positions), batch_size):
            ids = positions[start : start + batch_size]
            x, valid, fams, _slots, _ = batchify(cache, fams_all, slots_all, ids, device)
            found = model(x, valid, fams, mode="selected")
            full = model(x, valid, fams, mode="all")
            u_logit = found.get("unary_only_logit", found["bot_logit"])
            sel_bool = found["selected"].bool() & valid

            zero_x = torch.zeros_like(x)
            zall_scores.extend(model(zero_x, valid, fams, hard_mask=sel_bool)["bot_logit"].cpu().tolist())
            zero_px = x.clone()
            zero_px.index_fill_(-1, payload_cols_t, 0.0)
            zpay_scores.extend(model(zero_px, valid, fams, hard_mask=sel_bool)["bot_logit"].cpu().tolist())

            scores.extend(found["bot_logit"].cpu().tolist())
            all_scores.extend(full["bot_logit"].cpu().tolist())
            unary_scores.extend(u_logit.cpu().tolist())
            selected.extend(found["selected_count"].cpu().tolist())

            if build_rows:
                for j, i in enumerate(ids):
                    n_g = len(cache["features"][int(i)])
                    keep = sel_bool[j, :n_g].cpu()
                    sel_names = [n for n, k in zip(cache["group_names"][int(i)], keep) if k]
                    row = {
                        "account_id": str(cache["ids"][int(i)]),
                        "selected_group_names": sel_names,
                        "selected_families": sorted({group_family_id(n) for n in sel_names}),
                        "selected_closed_raw_ids": sorted(
                            set().union(*(set(s) for s, k in zip(cache["raw_source_ids"][int(i)], keep) if k))
                        ),
                        "selected_group_count": int(keep.sum()),
                        "candidate_group_count": int(n_g),
                        "bot_logit": float(found["bot_logit"][j].cpu()),
                        "full_bot_logit": float(full["bot_logit"][j].cpu()),
                    }
                    if role == "forward":
                        row["unary_only_logit"] = float(u_logit[j].cpu())
                        row["synergy_gain"] = float((found["bot_logit"][j] - u_logit[j]).cpu())
                    else:
                        h_keep = found["human_selected"][j, :n_g].bool().cpu()
                        row["human_counterexample_group_names"] = [
                            n for n, k in zip(cache["group_names"][int(i)], h_keep) if k
                        ]
                        row["human_ce_group_count"] = int(h_keep.sum())
                        row["bot_witness_logit"] = float(found["bot_witness_logit"][j].cpu())
                        row["human_counterexample_logit"] = float(found["human_counterexample_logit"][j].cpu())
                    rows.append(row)
    return (
        np.asarray(scores),
        np.asarray(all_scores),
        np.asarray(unary_scores),
        np.asarray(zall_scores),
        np.asarray(zpay_scores),
        np.asarray(selected),
        rows,
    )


def eval_content_swap(
    model,
    cache: dict,
    fams_all: list[torch.Tensor],
    slots_all: list[torch.Tensor],
    dev_positions: list[int],
    payload_cols: np.ndarray,
    device: torch.device,
    seed: int = 0,
) -> tuple[float, float]:
    """Group-wise cross-account payload permutation on Dev preserving group names/masks."""
    rng = np.random.default_rng(seed)
    n_dv = len(dev_positions)
    x_list = [cache["features"][int(p)].clone().numpy() for p in dev_positions]
    names_list = [cache["group_names"][int(p)] for p in dev_positions]
    by_name: dict[str, list[tuple[int, int]]] = {}
    for acc_i, nms in enumerate(names_list):
        for g_j, nm in enumerate(nms):
            by_name.setdefault(nm, []).append((acc_i, g_j))

    swapped = [arr.copy() for arr in x_list]
    for _nm, locs in by_name.items():
        if len(locs) < 2:
            continue
        perm = rng.permutation(len(locs))
        for dst_k, src_k in enumerate(perm):
            da, dg = locs[dst_k]
            sa, sg = locs[src_k]
            swapped[da][dg, payload_cols] = x_list[sa][sg, payload_cols]

    model.eval()
    swap_orig_mask_logits, swap_resel_logits = [], []
    with torch.inference_mode():
        for start in range(0, n_dv, 128):
            pos_b = dev_positions[start : start + 128]
            x_orig, valid, fams, _slots, _ = batchify(cache, fams_all, slots_all, pos_b, device)
            orig_sel = model(x_orig, valid, fams, mode="selected")["selected"].bool() & valid
            x_sw = torch.zeros_like(x_orig)
            for k in range(len(pos_b)):
                arr = swapped[start + k]
                x_sw[k, : len(arr)] = torch.from_numpy(arr)
            x_sw_dev = x_sw.to(device)
            swap_orig_mask_logits.extend(model(x_sw_dev, valid, fams, hard_mask=orig_sel)["bot_logit"].cpu().tolist())
            swap_resel_logits.extend(model(x_sw_dev, valid, fams, mode="selected")["bot_logit"].cpu().tolist())
    labels = cache["labels"][dev_positions].numpy().astype(int)
    return (
        float(roc_auc_score(labels, np.asarray(swap_orig_mask_logits))),
        float(roc_auc_score(labels, np.asarray(swap_resel_logits))),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--role", choices=("forward", "reverse"), required=True)
    p.add_argument("--payload-cols", type=str, default="auto")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--pair-dim", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--num-freq", type=int, default=0)
    p.add_argument("--num-queries", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=65)
    p.add_argument("--patience", type=int, default=18)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-frac", type=float, default=0.24)
    p.add_argument("--max-frac", type=float, default=0.38)
    p.add_argument("--per-family-budget", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--budget-weight", type=float, default=2.5)
    p.add_argument("--slot-parity-weight", type=float, default=0.0)
    p.add_argument("--subset-weight", type=float, default=0.40)
    p.add_argument("--family-drop-prob", type=float, default=0.25)
    p.add_argument("--cross-family-weight", type=float, default=0.15)
    p.add_argument("--human-single-neg-weight", type=float, default=0.15)
    p.add_argument("--synergy-weight", type=float, default=0.10)
    p.add_argument("--synergy-margin", type=float, default=0.40)
    p.add_argument("--logit-l2", type=float, default=0.004)
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")
    args = p.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        if args.overwrite:
            import shutil
            shutil.rmtree(args.output)
        else:
            raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Remote GPU is required")

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    fams_all, slots_all, num_slots = precompute_fams_and_slots(cache)
    train = list(map(int, cache["split"]["train"]))
    dev = list(map(int, cache["split"]["dev"]))
    input_dim = int(cache["features"][0].shape[1])
    payload_cols = resolve_payload_cols(cache, train, args.payload_cols)
    payload_cols_t = torch.as_tensor(payload_cols, dtype=torch.long, device=device)

    if args.role == "forward":
        model = CleanCoalitionNet(
            input_dim=input_dim,
            payload_cols=payload_cols,
            hidden=args.hidden,
            pair_dim=args.pair_dim,
            num_families=4,
            num_freq=args.num_freq,
            num_queries=args.num_queries,
            dropout=args.dropout,
        ).to(device)
        arch_tag = "clean_coalition"
        arch_full = "clean_hierarchical_family_coalition_network"
    else:
        model = CleanChallengerNet(
            input_dim=input_dim,
            payload_cols=payload_cols,
            hidden=args.hidden,
            heads=args.heads,
            layers=args.layers,
            num_freq=args.num_freq,
            dropout=args.dropout,
            local_gate=True,
        ).to(device)
        arch_tag = "clean_challenger"
        arch_full = "clean_dual_query_set_transformer_challenger"

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    best_score = (-1.0, -1.0, -1e9)
    best_epoch = 0
    stale = 0
    best_state = None
    args.output.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train)
        model.train()
        total_loss = 0.0
        batches = 0
        for start in range(0, len(train), args.batch_size):
            batch = train[start : start + args.batch_size]
            x, valid, fams, slots, y = batchify(cache, fams_all, slots_all, batch, device)
            human_y = 1.0 - y
            optimizer.zero_grad(set_to_none=True)

            selected = model(x, valid, fams, mode="selected")
            full = model(x, valid, fams, mode="all")
            hard = selected["selected"].bool() & valid

            budget_loss = compute_balanced_budget_loss(
                selected["gate_probability"],
                valid,
                fams,
                args.min_frac,
                args.max_frac,
                per_family=args.per_family_budget,
            )
            parity_loss = (
                compute_slot_parity_loss(selected["gate_probability"], valid, slots, y, num_slots)
                if args.slot_parity_weight > 0
                else x.new_zeros(())
            )

            if args.role == "forward":
                completion = hard | ((torch.rand_like(hard.float()) < 0.25) & valid)
                completed = model(x, valid, fams, hard_mask=completion)
                loss = (
                    F.binary_cross_entropy_with_logits(selected["bot_logit"], y)
                    + 0.25 * F.binary_cross_entropy_with_logits(full["bot_logit"], y)
                    + 0.35 * F.binary_cross_entropy_with_logits(completed["bot_logit"], y)
                    + 0.25 * F.binary_cross_entropy_with_logits(selected["unary_only_logit"], y)
                    + 0.05 * F.smooth_l1_loss(selected["bot_logit"], full["bot_logit"])
                    + args.budget_weight * budget_loss
                    + args.slot_parity_weight * parity_loss
                )
                # Modality co-adaptation defense: drop 1 family in auxiliary forward pass so
                # easy early families (e.g. relation) never starve profile + content learning
                if args.family_drop_prob > 0 and random.random() < args.family_drop_prob:
                    present_fams = [fid for fid in range(4) if (valid & (fams == fid)).any()]
                    if len(present_fams) >= 2:
                        drop_f = random.choice(present_fams)
                        keep_v = valid & (fams != drop_f)
                        has_rem = keep_v.any(dim=1)
                        if has_rem.any():
                            drop_out = model(x[has_rem], keep_v[has_rem], fams[has_rem], mode="selected")
                            loss = loss + 0.30 * F.binary_cross_entropy_with_logits(drop_out["bot_logit"], y[has_rem])

                if args.synergy_weight > 0:
                    bot_mf = (y > 0.5) & (selected["active_families"] >= 2.0)
                    if bot_mf.any():
                        gain = selected["bot_logit"][bot_mf] - selected["unary_only_logit"][bot_mf].detach()
                        loss = loss + args.synergy_weight * F.relu(args.synergy_margin - gain).mean()
                    hum_mf = (y < 0.5) & (selected["active_families"] >= 2.0)
                    if hum_mf.any():
                        h_gain = selected["bot_logit"][hum_mf] - selected["unary_only_logit"][hum_mf].detach()
                        loss = loss + 0.5 * args.synergy_weight * F.relu(h_gain).mean()
                if args.logit_l2 > 0:
                    loss = loss + args.logit_l2 * (
                        (selected["bot_logit"] ** 2).mean() + (selected["unary_only_logit"] ** 2).mean()
                    )
            else:
                hum_hard = selected["human_selected"].bool() & valid
                challenged = (hard | hum_hard) & valid
                chal_out = model(x, valid, fams, hard_mask=challenged)
                hum_budget = compute_balanced_budget_loss(
                    selected["human_gate_probability"],
                    valid,
                    fams,
                    0.10,
                    0.30,
                    per_family=args.per_family_budget,
                )
                loss = (
                    F.binary_cross_entropy_with_logits(selected["bot_logit"], y)
                    + 0.35 * F.binary_cross_entropy_with_logits(selected["bot_witness_logit"], y)
                    + 0.35 * F.binary_cross_entropy_with_logits(selected["human_counterexample_logit"], human_y)
                    + 0.30 * F.binary_cross_entropy_with_logits(full["bot_logit"], y)
                    + 0.40 * F.binary_cross_entropy_with_logits(chal_out["bot_logit"], y)
                    + args.budget_weight * budget_loss
                    + 0.5 * args.budget_weight * hum_budget
                    + args.slot_parity_weight * parity_loss
                )
                if args.logit_l2 > 0:
                    loss = loss + args.logit_l2 * (
                        (selected["bot_logit"] ** 2).mean()
                        + (selected["bot_witness_logit"] ** 2).mean()
                        + (selected["human_counterexample_logit"] ** 2).mean()
                    )

            # Random 55%-85% subset stability over selected E+
            multi_sel = hard.sum(dim=1) >= 3
            if args.subset_weight > 0 and multi_sel.any():
                sub_mask = sample_subset_mask(hard[multi_sel], min_keep=2, keep_lo=0.55, keep_hi=0.85)
                sub_out = model(x[multi_sel], valid[multi_sel], fams[multi_sel], hard_mask=sub_mask)
                loss = loss + args.subset_weight * F.binary_cross_entropy_with_logits(sub_out["bot_logit"], y[multi_sel])

            # Cross-family witness coverage on multi-family Bots (y == 1 only)
            bot_valid = (y > 0.5) & (valid.sum(dim=1) >= 2)
            if args.cross_family_weight > 0 and bot_valid.any():
                fam_losses = []
                p_b = selected["gate_probability"][bot_valid]
                v_b = valid[bot_valid]
                f_b = fams[bot_valid]
                for fid in range(4):
                    fm = v_b & (f_b == fid)
                    has_f = fm.any(dim=1)
                    if has_f.any():
                        f_max = p_b[has_f].masked_fill(~fm[has_f], 0.0).amax(dim=1)
                        fam_losses.append(F.relu(0.60 - f_max).mean())
                if fam_losses:
                    loss = loss + args.cross_family_weight * (sum(fam_losses) / len(fam_losses))

            # Single-group Human humility: singletons should not fire as Bot on Humans (with chunked OOM protection)
            if args.human_single_neg_weight > 0:
                human = y < 0.5
                if human.any():
                    hx, hv, hf = x[human], valid[human], fams[human]
                    cnt, length, width = hx.shape
                    valid_flat = hv.reshape(-1)
                    one_logits_flat = torch.full((cnt * length,), -1e9, device=device, dtype=torch.float32)
                    active_idx = torch.nonzero(valid_flat, as_tuple=False).squeeze(-1)
                    if len(active_idx) > 0:
                        key = "bot_witness_logit" if args.role == "reverse" else "bot_logit"
                        chunk_size = 256
                        for c_start in range(0, len(active_idx), chunk_size):
                            c_sub = active_idx[c_start : c_start + chunk_size]
                            chunk_out = model(
                                hx.reshape(cnt * length, 1, width)[c_sub],
                                hv.reshape(cnt * length, 1)[c_sub],
                                hf.reshape(cnt * length, 1)[c_sub],
                                mode="all",
                            )
                            one_logits_flat[c_sub] = chunk_out[key]
                    hardest = one_logits_flat.reshape(cnt, length).masked_fill(~hv, -1e9).amax(dim=1)
                    loss = loss + args.human_single_neg_weight * F.softplus(hardest + 0.25).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1

        values, _all, _u, zall_v, zpay_v, counts, _rows = evaluate(
            model, args.role, cache, fams_all, slots_all, dev, device, payload_cols_t, build_rows=False
        )
        labels = cache["labels"][dev].numpy().astype(int)
        auc = float(roc_auc_score(labels, values))
        thresholds = np.quantile(values, np.linspace(0.05, 0.95, 91))
        best_macro = max(float(f1_score(labels, values >= t, average="macro")) for t in thresholds)
        cur_key = (round(best_macro, 4), round(auc, 5), -abs(float(np.median(counts)) - 12.0))
        if cur_key > best_score:
            best_score = cur_key
            best_epoch = epoch
            stale = 0
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "role": args.role,
                    "train_loss": total_loss / max(batches, 1),
                    "dev_selected_auc": auc,
                    "dev_best_macro_f1": best_macro,
                    "zero_all_std": float(np.std(zall_v)),
                    "zero_pay_std": float(np.std(zpay_v)),
                    "median_selected_groups": float(np.median(counts)),
                    "bot_median_selected": float(np.median(counts[labels == 1])),
                    "hum_median_selected": float(np.median(counts[labels == 0])),
                    "best_epoch": best_epoch,
                }
            ),
            flush=True,
        )
        if stale >= args.patience:
            break

    model.load_state_dict(best_state)
    values, all_vals, unary_vals, zall_vals, zpay_vals, counts, rows = evaluate(
        model, args.role, cache, fams_all, slots_all, dev, device, payload_cols_t, build_rows=True
    )
    swap_orig_auc, swap_resel_auc = eval_content_swap(
        model, cache, fams_all, slots_all, dev, payload_cols, device, seed=args.seed
    )
    labels = cache["labels"][dev].numpy().astype(int)

    u = np.sort(np.unique(values))
    if len(u) >= 2:
        mids = 0.5 * (u[:-1] + u[1:])
        gaps = u[1:] - u[:-1]
        cand_t = [float(u[0] - 1.0)] + mids.tolist() + [float(u[-1] + 1.0)]
        cand_gap = [0.0] + gaps.tolist() + [0.0]
    else:
        cand_t = [float(u[0])]
        cand_gap = [0.0]
    choices = [
        (
            round(f1_score(labels, values >= t, average="macro"), 6),
            round(f1_score(labels, values >= t, pos_label=1), 6),
            round(accuracy_score(labels, values >= t), 6),
            float(gap),
            -abs(float(t)),
            float(t),
        )
        for t, gap in zip(cand_t, cand_gap)
    ]
    macro, bot_f1, acc, _g, _na, threshold = max(choices)
    pred = values >= threshold

    checkpoint = args.output / "model.pt"
    torch.save(
        {
            "model_state": best_state,
            "arch": arch_tag,
            "architecture": arch_full,
            "role": args.role,
            "input_dim": input_dim,
            "payload_cols": payload_cols.tolist(),
            "hidden": args.hidden,
            "pair_dim": args.pair_dim,
            "heads": args.heads,
            "layers": args.layers,
            "num_families": 4,
            "num_freq": args.num_freq,
            "num_queries": args.num_queries,
            "local_gate": True,
            "dropout": args.dropout,
            "scaler_mean": cache["scaler_mean"],
            "scaler_std": cache["scaler_std"],
            "group_cache_sha256": sha256_file(args.cache),
            "seed": args.seed,
        },
        checkpoint,
    )
    with (args.output / "dev_evidence.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    bot_rows = [r for r, y_val in zip(rows, labels) if y_val == 1 and r["bot_logit"] >= threshold]
    report = {
        "architecture": arch_full,
        "arch": arch_tag,
        "role": "forward_predecision_evidence_classifier" if args.role == "forward" else "independent_reverse_counterexample_challenger",
        "checkpoint_sha256": sha256_file(checkpoint),
        "group_cache_sha256": sha256_file(args.cache),
        "seed": args.seed,
        "hidden": args.hidden,
        "num_freq": args.num_freq,
        "num_queries": args.num_queries,
        "local_gate": True,
        "payload_col_count": int(len(payload_cols)),
        "best_epoch": best_epoch,
        "train_n": len(train),
        "dev_n": len(dev),
        "dev_selected_auc": float(roc_auc_score(labels, values)),
        "dev_full_auc": float(roc_auc_score(labels, all_vals)),
        "dev_unary_only_auc": float(roc_auc_score(labels, unary_vals)),
        "dev_mask_only_zero_all_std": float(np.std(zall_vals)),
        "dev_mask_only_zero_all_gap": abs(float(np.mean(zall_vals[labels == 1]) - np.mean(zall_vals[labels == 0]))),
        "dev_mask_only_zero_payload_std": float(np.std(zpay_vals)),
        "dev_mask_only_zero_payload_gap": abs(float(np.mean(zpay_vals[labels == 1]) - np.mean(zpay_vals[labels == 0]))),
        "dev_content_swap_auc": float(swap_orig_auc),
        "dev_content_swap_reselect_auc": float(swap_resel_auc),
        "dev_accuracy_calibrated": float(acc),
        "dev_macro_f1_calibrated": float(macro),
        "dev_bot_f1_calibrated": float(bot_f1),
        "dev_bot_precision_calibrated": float(precision_score(labels, pred)),
        "dev_bot_recall_calibrated": float(recall_score(labels, pred)),
        "dev_threshold": threshold,
        "median_selected_groups": float(np.median(counts)),
        "bot_median_selected_groups": float(np.median(counts[labels == 1])),
        "human_median_selected_groups": float(np.median(counts[labels == 0])),
        "dev_true_bot_multi_family_rate": float(np.mean([len(r["selected_families"]) >= 2 for r in bot_rows])) if bot_rows else 0.0,
        "dev_true_bot_mean_synergy_gain": float(np.mean([r.get("synergy_gain", 0.0) for r in bot_rows])) if bot_rows else 0.0,
        "test_evaluated": False,
        "reverse_hgt_used_in_training": False,
        "forward_model_used_in_training": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
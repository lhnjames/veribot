"""Bidirectional Counterexample Challenge, Full-Denominator Consistency & Human Search Calibration Audit.

Protocol:
1. Calibrates refutation / challenge rules strictly on Dev, and calibrates multi-probe search thresholds
   strictly on a deterministic 50% Dev Human calibration split (cal_human), holding out hold_human & Test.
2. Evaluates across ALL accounts in Dev (and optionally Locked Test once):
   - Full-denominator agreement & cross-support rates (A_bot, B_bot, joint_bot, E+_A->B, E+_B->A,
     bidirectional_cross_support, direction conflicts) alongside matched-size random-k baselines.
   - Subgraph alignment (Family Jaccard, Motif Jaccard, Exact Source Jaccard, Bipartite Matching) vs
     matched-size random baseline.
   - Counterexample refutation (FP refuted vs TP lost, net ACC / Bot F1 gain).
   - Search-level Human calibration & paired Reverse-vs-Random multi-probe comparison at ft, Dev-Human p95,
     and Dev-Human max thresholds.
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
from verifiable_bot.synergy_source_net import SynergySourceInteractionNet
from verifiable_bot.source_interaction_net import SourceInteractionNet
from verifiable_bot.source_star_challenger_net import SourceStarChallengerNet
from verifiable_bot.counterexample_challenger_net import CounterexampleChallengerNet
from verifiable_bot.clean_coalition_net import CleanCoalitionNet
from verifiable_bot.clean_challenger_net import CleanChallengerNet


def group_family_id(name: str) -> int:
    prefix = name.split(":", 1)[0]
    if prefix in ("profile", "user"):
        return 0
    if prefix in ("relation", "reply", "neighbor", "neighbor_profile"):
        return 1
    if prefix in ("content", "tweet", "posts", "comment_1", "comment_2"):
        return 2
    return 3


def sig(name: str) -> str:
    part = name.split(":")
    if part[0] == "relation":
        return ":".join(part[:3])
    if part[0] == "neighbor_profile" and len(part) > 1:
        return ":".join(part[:2])
    if part[0] == "tweet" and len(part) > 1:
        return ":".join(part[:2])
    if part[0] in ("posts", "comment_1", "comment_2", "content", "profile", "user"):
        return part[0]
    return ":".join(part[:2])


def jacc(a, b) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def pairscore(na: str, nb: str, xa: np.ndarray, xb: np.ndarray, sa: tuple[str, ...], sb: tuple[str, ...]) -> float:
    pa, pb = na.split(":"), nb.split(":")
    if pa[0] != pb[0]:
        return 0.0
    typ = 1.0 if na == nb else (0.8 if len(pa) > 1 and len(pb) > 1 and pa[1] == pb[1] else 0.5)
    if pa[0] == "relation" and len(pa) > 2 and len(pb) > 2 and pa[2] == pb[2]:
        typ = min(1.0, typ + 0.1)
    denom = float(np.linalg.norm(xa) * np.linalg.norm(xb))
    cos = float(np.dot(xa, xb) / denom) if denom > 1e-8 else 0.0
    feat = max(0.0, min(1.0, (cos + 1.0) / 2.0))
    prov = jacc(sa, sb)
    return 0.45 * typ + 0.35 * feat + 0.20 * prov


def load_any_model(model_dir: Path, device: torch.device):
    ck = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
    rep = json.loads((model_dir / "report.json").read_text())
    arch = str(ck.get("arch", ck.get("architecture", "source_interaction_evidence_network")))
    input_dim = int(ck["input_dim"])
    hidden = int(ck.get("hidden", 96))
    pair_dim = int(ck.get("pair_dim", 24))
    cgate = bool(ck.get("contextual_gate", False))
    extra_pool = {}
    if "mean_pool_by_count" in ck:
        extra_pool["mean_pool_by_count"] = bool(ck["mean_pool_by_count"])
    if "include_cardinality_in_head" in ck:
        extra_pool["include_cardinality_in_head"] = bool(ck["include_cardinality_in_head"])
    if "clean_challenger" in arch:
        m = CleanChallengerNet(
            input_dim,
            payload_cols=ck["payload_cols"],
            hidden=int(ck.get("hidden", 128)),
            heads=int(ck.get("heads", 4)),
            layers=int(ck.get("layers", 2)),
            num_freq=int(ck.get("num_freq", 4)),
            dropout=float(ck.get("dropout", 0.15)),
            local_gate=bool(ck.get("local_gate", False)),
        ).to(device)
        kind = "clean_challenger"
    elif "clean_coalition" in arch:
        m = CleanCoalitionNet(
            input_dim,
            payload_cols=ck["payload_cols"],
            hidden=int(ck.get("hidden", 128)),
            pair_dim=int(ck.get("pair_dim", 32)),
            num_families=int(ck.get("num_families", 4)),
            num_freq=int(ck.get("num_freq", 4)),
            num_queries=int(ck.get("num_queries", 1)),
            dropout=float(ck.get("dropout", 0.15)),
        ).to(device)
        kind = "clean_coalition"
    elif "counterexample" in arch:
        ce_kw = dict(extra_pool)
        if "strict_hard_mask_context" in ck:
            ce_kw["strict_hard_mask_context"] = bool(ck["strict_hard_mask_context"])
        m = CounterexampleChallengerNet(
            input_dim,
            hidden=hidden,
            heads=int(ck.get("heads", 4)),
            layers=int(ck.get("layers", 2)),
            contextual_gate=cgate,
            **ce_kw,
        ).to(device)
        kind = "counterexample"
    elif "synergy" in arch:
        syn_kw = dict(extra_pool)
        if "num_families" in ck:
            syn_kw["num_families"] = int(ck["num_families"])
        m = SynergySourceInteractionNet(input_dim, hidden=hidden, pair_dim=pair_dim, contextual_gate=cgate, **syn_kw).to(
            device
        )
        kind = "synergy"
    elif "star" in arch:
        m = SourceStarChallengerNet(input_dim, hidden=hidden, contextual_gate=cgate).to(device)
        kind = "star"
    else:
        m = SourceInteractionNet(input_dim, hidden=hidden, pair_dim=pair_dim, contextual_gate=cgate, **extra_pool).to(
            device
        )
        kind = "interaction"
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m, kind, float(rep["dev_threshold"])


def run_any(m, kind: str, x: torch.Tensor, v: torch.Tensor, fams: torch.Tensor, **kwargs) -> dict:
    if kind in ("synergy", "clean_coalition", "clean_challenger"):
        return m(x, v, fams, **kwargs)
    return m(x, v, **kwargs)


def alignment_scores(
    fa_idx: list[int],
    rb_idx: list[int],
    names: list[str],
    feats_np: np.ndarray,
    sources: list[tuple[str, ...]],
) -> tuple[float, float, float, float]:
    fn = [names[j] for j in fa_idx]
    rn = [names[k] for k in rb_idx]
    fr = set().union(*(set(sources[j]) for j in fa_idx)) if fa_idx else set()
    rr = set().union(*(set(sources[k]) for k in rb_idx)) if rb_idx else set()
    fam_j = jacc((z.split(":")[0] for z in fn), (z.split(":")[0] for z in rn))
    mot_j = jacc((sig(z) for z in fn), (sig(z) for z in rn))
    src_j = jacc(fr, rr)
    if fa_idx and rb_idx:
        mat = np.zeros((len(fa_idx), len(rb_idx)), dtype=float)
        for ai, j in enumerate(fa_idx):
            for bi, k in enumerate(rb_idx):
                mat[ai, bi] = pairscore(names[j], names[k], feats_np[j], feats_np[k], sources[j], sources[k])
        ii, jj = linear_sum_assignment(-mat)
        align = float(sum(mat[u, v_i] for u, v_i in zip(ii, jj) if mat[u, v_i] > 0) / max(len(fa_idx), len(rb_idx), 1))
    else:
        align = 0.0
    return fam_j, mot_j, src_j, align


def is_cal_human(account_id: str) -> bool:
    digest = hashlib.sha256(account_id.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "big") % 2) == 0


def collect_split_records(
    cache: dict,
    split_name: str,
    f_model,
    f_kind: str,
    r_model,
    r_kind: str,
    device: torch.device,
    *,
    search_budget: int = 4,
):
    records = []
    with torch.inference_mode():
        for ix in cache["split"][split_name]:
            i = int(ix)
            acct_id = str(cache["ids"][i])
            x = cache["features"][i].to(device).unsqueeze(0)
            m_groups = x.shape[1]
            v = torch.ones((1, m_groups), dtype=torch.bool, device=device)
            names = cache["group_names"][i]
            sources = cache["raw_source_ids"][i]
            feats_np = cache["features"][i].numpy()
            fams = torch.tensor([[group_family_id(n) for n in names]], dtype=torch.long, device=device)

            fo = run_any(f_model, f_kind, x, v, fams, mode="selected")
            f_full = run_any(f_model, f_kind, x, v, fams, mode="all")
            ro = run_any(r_model, r_kind, x, v, fams, mode="selected")
            r_full = run_any(r_model, r_kind, x, v, fams, mode="all")

            fs = fo["selected"][0].bool()
            rs = ro["selected"][0].bool()
            fscore = float(fo["bot_logit"][0])
            ffull = float(f_full["bot_logit"][0])
            rscore = float(ro["bot_logit"][0])
            rfull = float(r_full["bot_logit"][0])

            # Deterministic per-account RNG for matched-size random & matched-budget random controls
            seed_int = int.from_bytes(hashlib.sha256(f"{split_name}:{acct_id}".encode("utf-8")).digest()[:8], "big")
            rng = random.Random(seed_int)
            all_idx = list(range(m_groups))
            fa_idx = [j for j, k in enumerate(fs.tolist()) if k]
            rb_idx = [j for j, k in enumerate(rs.tolist()) if k]

            rand_f_idx = rng.sample(all_idx, k=max(1, len(fa_idx)))
            rand_r_idx = rng.sample(all_idx, k=max(1, len(rb_idx)))
            rand_fs = torch.zeros_like(fs)
            rand_fs[rand_f_idx] = True
            rand_rs = torch.zeros_like(rs)
            rand_rs[rand_r_idx] = True

            # Background Human counterexample challenge from V_u \ E+_A
            bg_valid = ~fs
            if r_kind in ("counterexample", "clean_challenger"):
                h_probs = ro["human_gate_probability"][0] * bg_valid.float()
                h_ce_own = float(ro["human_counterexample_logit"][0])
            else:
                h_probs = (1.0 - ro["gate_probability"][0]) * bg_valid.float()
                h_ce_own = -rscore

            ce_mask = fs.clone()
            ce_groups = []
            if bg_valid.any():
                top_k = min(2, int(bg_valid.sum().item()))
                top_ce_idx = torch.topk(h_probs, k=top_k).indices.tolist()
                for idx_c in top_ce_idx:
                    if bg_valid[idx_c]:
                        ce_mask[idx_c] = True
                        ce_groups.append(names[idx_c])

            # Lock Reverse Model B witness priority order BEFORE consulting E+_A
            r_order = torch.argsort(ro["gate_probability"][0], descending=True).tolist()
            rand_order = list(range(m_groups))
            rng.shuffle(rand_order)

            # Build Q-budget probe masks for Forward A (reverse search, random search, disjoint reverse, disjoint random)
            # Each probe builds prefix top-k sets of size 1..min(budget, M)
            probe_masks_a = [rs, rand_rs, ce_mask]  # 0: f_on_r, 1: f_on_rand_r, 2: f_with_ce
            rev_probe_slice = slice(3, 3 + search_budget)
            for q in range(1, search_budget + 1):
                pm = torch.zeros_like(fs)
                pm[r_order[: min(q, m_groups)]] = True
                probe_masks_a.append(pm)
            rand_probe_slice = slice(3 + search_budget, 3 + 2 * search_budget)
            for q in range(1, search_budget + 1):
                pm = torch.zeros_like(fs)
                pm[rand_order[: min(q, m_groups)]] = True
                probe_masks_a.append(pm)

            disj_r_order = [j for j in r_order if not fs[j].item()]
            disj_rand_order = [j for j in rand_order if not fs[j].item()]
            disj_rev_slice = slice(3 + 2 * search_budget, 3 + 3 * search_budget)
            for q in range(1, search_budget + 1):
                pm = torch.zeros_like(fs)
                if disj_r_order:
                    pm[disj_r_order[: min(q, len(disj_r_order))]] = True
                probe_masks_a.append(pm)
            disj_rand_slice = slice(3 + 3 * search_budget, 3 + 4 * search_budget)
            for q in range(1, search_budget + 1):
                pm = torch.zeros_like(fs)
                if disj_rand_order:
                    pm[disj_rand_order[: min(q, len(disj_rand_order))]] = True
                probe_masks_a.append(pm)

            batch_masks_a = torch.stack(probe_masks_a, dim=0)
            b_a = batch_masks_a.shape[0]
            a_out = run_any(
                f_model,
                f_kind,
                x.expand(b_a, -1, -1),
                v.expand(b_a, -1),
                fams.expand(b_a, -1),
                hard_mask=batch_masks_a,
            )["bot_logit"].cpu().tolist()

            f_on_r = float(a_out[0])
            f_on_rand_r = float(a_out[1])
            f_with_ce = float(a_out[2])
            rev_search_max_a = float(max(a_out[rev_probe_slice]))
            rand_search_max_a = float(max(a_out[rand_probe_slice]))
            disj_rev_search_max_a = float(max(a_out[disj_rev_slice])) if disj_r_order else -10.0
            disj_rand_search_max_a = float(max(a_out[disj_rand_slice])) if disj_rand_order else -10.0

            batch_masks_r = torch.stack([fs, rand_fs, ce_mask], dim=0)
            r_out = run_any(
                r_model,
                r_kind,
                x.expand(3, -1, -1),
                v.expand(3, -1),
                fams.expand(3, -1),
                hard_mask=batch_masks_r,
            )["bot_logit"].cpu().tolist()
            r_on_f = float(r_out[0])
            r_on_rand_f = float(r_out[1])
            r_with_ce = float(r_out[2])

            fam_j, mot_j, src_j, align = alignment_scores(fa_idx, rb_idx, names, feats_np, sources)
            rfam_j, rmot_j, rsrc_j, ralign = alignment_scores(fa_idx, rand_r_idx, names, feats_np, sources)

            records.append(
                {
                    "account_id": acct_id,
                    "label": int(cache["labels"][i]),
                    "f_score": fscore,
                    "f_full": ffull,
                    "r_score": rscore,
                    "r_full": rfull,
                    "r_on_f": r_on_f,
                    "r_on_rand_f": r_on_rand_f,
                    "f_on_r": f_on_r,
                    "f_on_rand_r": f_on_rand_r,
                    "h_ce_own": h_ce_own,
                    "f_with_ce": f_with_ce,
                    "r_with_ce": r_with_ce,
                    "ce_drop_f": fscore - f_with_ce,
                    "ce_drop_r": r_on_f - r_with_ce,
                    "rev_search_max_a": rev_search_max_a,
                    "rand_search_max_a": rand_search_max_a,
                    "disj_rev_search_max_a": disj_rev_search_max_a,
                    "disj_rand_search_max_a": disj_rand_search_max_a,
                    "forward_groups": [names[j] for j in fa_idx],
                    "reverse_groups": [names[j] for j in rb_idx],
                    "counterexample_groups": ce_groups,
                    "family_jaccard": fam_j,
                    "motif_jaccard": mot_j,
                    "exact_source_jaccard": src_j,
                    "approximate_subgraph_alignment": align,
                    "rand_family_jaccard": rfam_j,
                    "rand_motif_jaccard": rmot_j,
                    "rand_exact_source_jaccard": rsrc_j,
                    "rand_approximate_subgraph_alignment": ralign,
                }
            )
    return records


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "bot_f1": float(f1_score(y_true, y_pred, pos_label=1)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm,
    }


def calibrate_refutation_rule(dev_records: list[dict], ft: float, rt: float) -> dict:
    y = np.array([r["label"] for r in dev_records], dtype=int)
    f_score = np.array([r["f_score"] for r in dev_records], dtype=float)
    r_score = np.array([r["r_score"] for r in dev_records], dtype=float)
    r_on_f = np.array([r["r_on_f"] for r in dev_records], dtype=float)
    f_with_ce = np.array([r["f_with_ce"] for r in dev_records], dtype=float)
    h_ce = np.array([r["h_ce_own"] for r in dev_records], dtype=float)

    base_pred = f_score >= ft
    base_f1 = f1_score(y, base_pred, pos_label=1)
    base_acc = accuracy_score(y, base_pred)

    best_rule = {
        "fwd_cap": 0.0,
        "chal_thresh": -1e9,
        "w_own": 0.4,
        "w_onf": 0.4,
        "w_ce": 0.2,
        "w_hce": 0.1,
    }
    best_key = (round(base_f1, 6), round(base_acc, 6), 0)

    for w_own, w_onf, w_ce, w_hce in (
        (0.4, 0.4, 0.2, 0.15),
        (0.5, 0.3, 0.2, 0.20),
        (0.3, 0.5, 0.2, 0.10),
        (0.6, 0.2, 0.2, 0.25),
        (0.0, 1.0, 0.0, 0.0),
    ):
        chal = w_own * (r_score - rt) + w_onf * (r_on_f - rt) + w_ce * (f_with_ce - ft) - w_hce * np.maximum(0.0, h_ce)
        for fwd_cap in (0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 10.0):
            for c_th in np.linspace(-2.5, 0.5, 61):
                refute = base_pred & ((f_score - ft) < fwd_cap) & (chal < c_th)
                if not refute.any():
                    continue
                pred = base_pred & (~refute)
                b_f1 = float(f1_score(y, pred, pos_label=1))
                acc = float(accuracy_score(y, pred))
                fp_ref = int(np.sum(refute & (y == 0)))
                tp_ref = int(np.sum(refute & (y == 1)))
                key = (round(b_f1, 6), round(acc, 6), fp_ref - tp_ref)
                if key > best_key:
                    best_key = key
                    best_rule = {
                        "fwd_cap": float(fwd_cap),
                        "chal_thresh": float(c_th),
                        "w_own": float(w_own),
                        "w_onf": float(w_onf),
                        "w_ce": float(w_ce),
                        "w_hce": float(w_hce),
                    }
    return best_rule


def calibrate_human_search_thresholds(dev_records: list[dict], ft: float, search_budget: int) -> dict:
    cal_humans = [r for r in dev_records if r["label"] == 0 and is_cal_human(r["account_id"])]
    hold_humans = [r for r in dev_records if r["label"] == 0 and not is_cal_human(r["account_id"])]
    rev_vals = np.array([r["rev_search_max_a"] for r in cal_humans], dtype=float)
    disj_vals = np.array([r["disj_rev_search_max_a"] for r in cal_humans], dtype=float)
    return {
        "search_budget": search_budget,
        "cal_human_n": len(cal_humans),
        "hold_human_n": len(hold_humans),
        "forward_classifier_threshold": float(ft),
        "rev_search_p95_threshold": float(np.quantile(rev_vals, 0.95)) if len(rev_vals) else float(ft),
        "rev_search_max_threshold": float(np.max(rev_vals)) if len(rev_vals) else float(ft),
        "disj_rev_search_p95_threshold": float(np.quantile(disj_vals, 0.95)) if len(disj_vals) else float(ft),
        "disj_rev_search_max_threshold": float(np.max(disj_vals)) if len(disj_vals) else float(ft),
    }


def eval_human_search_calibration(records: list[dict], cal_cfg: dict, is_dev: bool) -> dict:
    humans = (
        [r for r in records if r["label"] == 0 and not is_cal_human(r["account_id"])]
        if is_dev
        else [r for r in records if r["label"] == 0]
    )
    bots = [r for r in records if r["label"] == 1]

    def summary_at(thresh: float, key_rev: str, key_rand: str) -> dict:
        h_rev = float(np.mean([r[key_rev] >= thresh for r in humans])) if humans else 0.0
        h_rnd = float(np.mean([r[key_rand] >= thresh for r in humans])) if humans else 0.0
        b_rev = float(np.mean([r[key_rev] >= thresh for r in bots])) if bots else 0.0
        b_rnd = float(np.mean([r[key_rand] >= thresh for r in bots])) if bots else 0.0
        return {
            "threshold": float(thresh),
            "human_n": len(humans),
            "true_bot_n": len(bots),
            "human_false_hit_reverse": h_rev,
            "human_false_hit_matched_random": h_rnd,
            "true_bot_coverage_reverse": b_rev,
            "true_bot_coverage_matched_random": b_rnd,
            "reverse_minus_random_true_bot_gain": b_rev - b_rnd,
        }

    return {
        "search_budget": cal_cfg["search_budget"],
        "split_role": "heldout_dev_human_50pct_vs_all_dev_bots" if is_dev else "locked_test_human_vs_test_bots",
        "general_witness_search": {
            "at_classifier_threshold": summary_at(
                cal_cfg["forward_classifier_threshold"], "rev_search_max_a", "rand_search_max_a"
            ),
            "at_cal_human_p95": summary_at(cal_cfg["rev_search_p95_threshold"], "rev_search_max_a", "rand_search_max_a"),
            "at_cal_human_max": summary_at(cal_cfg["rev_search_max_threshold"], "rev_search_max_a", "rand_search_max_a"),
        },
        "disjoint_from_EA_witness_search": {
            "at_classifier_threshold": summary_at(
                cal_cfg["forward_classifier_threshold"], "disj_rev_search_max_a", "disj_rand_search_max_a"
            ),
            "at_cal_human_p95": summary_at(
                cal_cfg["disj_rev_search_p95_threshold"], "disj_rev_search_max_a", "disj_rand_search_max_a"
            ),
            "at_cal_human_max": summary_at(
                cal_cfg["disj_rev_search_max_threshold"], "disj_rev_search_max_a", "disj_rand_search_max_a"
            ),
        },
    }


def apply_refutation_rule(records: list[dict], ft: float, rt: float, rule: dict, cal_cfg: dict, *, is_dev: bool) -> dict:
    n_total = len(records)
    y = np.array([r["label"] for r in records], dtype=int)
    f_score = np.array([r["f_score"] for r in records], dtype=float)
    r_score = np.array([r["r_score"] for r in records], dtype=float)
    r_full = np.array([r["r_full"] for r in records], dtype=float)
    r_on_f = np.array([r["r_on_f"] for r in records], dtype=float)
    r_on_rand_f = np.array([r["r_on_rand_f"] for r in records], dtype=float)
    f_on_r = np.array([r["f_on_r"] for r in records], dtype=float)
    f_on_rand_r = np.array([r["f_on_rand_r"] for r in records], dtype=float)
    f_with_ce = np.array([r["f_with_ce"] for r in records], dtype=float)
    h_ce = np.array([r["h_ce_own"] for r in records], dtype=float)

    base_pred = f_score >= ft
    b_sel_pred = r_score >= rt
    b_full_pred = r_full >= rt
    joint_sel = base_pred & b_sel_pred
    joint_full = base_pred & b_full_pred
    eA_sup_B = r_on_f >= rt
    eB_sup_A = f_on_r >= ft
    bidir_sup = eA_sup_B & eB_sup_A
    eA_rand_sup_B = r_on_rand_f >= rt
    eB_rand_sup_A = f_on_rand_r >= ft
    true_bot = y == 1
    true_hum = y == 0

    def rate(mask: np.ndarray, denom_mask: np.ndarray | None = None) -> float:
        if denom_mask is None:
            return float(np.sum(mask) / max(n_total, 1))
        d = int(np.sum(denom_mask))
        return float(np.sum(mask & denom_mask) / d) if d > 0 else 0.0

    chal = (
        rule["w_own"] * (r_score - rt)
        + rule["w_onf"] * (r_on_f - rt)
        + rule["w_ce"] * (f_with_ce - ft)
        - rule["w_hce"] * np.maximum(0.0, h_ce)
    )
    refute = base_pred & ((f_score - ft) < rule["fwd_cap"]) & (chal < rule["chal_thresh"])
    after_pred = base_pred & (~refute)

    naive_refute = base_pred & (r_on_f < rt)
    naive_pred = base_pred & (~naive_refute)

    bot_cands = [r for r in records if r["f_score"] >= ft]
    return {
        "full_denominator_bidirectional_metrics": {
            "total_accounts": n_total,
            "true_bot_n": int(np.sum(true_bot)),
            "true_human_n": int(np.sum(true_hum)),
            "a_bot_rate": rate(base_pred),
            "b_selected_bot_rate": rate(b_sel_pred),
            "b_full_bot_rate": rate(b_full_pred),
            "joint_selected_bot_rate": rate(joint_sel),
            "joint_full_bot_rate": rate(joint_full),
            "direction_conflict_selected_rate": rate(base_pred ^ b_sel_pred),
            "direction_conflict_full_rate": rate(base_pred ^ b_full_pred),
            "over_all_accounts": {
                "E_plus_A_supports_B": rate(eA_sup_B),
                "matched_random_kA_supports_B": rate(eA_rand_sup_B),
                "E_plus_B_supports_A": rate(eB_sup_A),
                "matched_random_kB_supports_A": rate(eB_rand_sup_A),
                "bidirectional_cross_support": rate(bidir_sup),
            },
            "conditioned_on_joint_selected_bot": {
                "joint_selected_bot_n": int(np.sum(joint_sel)),
                "E_plus_A_supports_B": rate(eA_sup_B, joint_sel),
                "matched_random_kA_supports_B": rate(eA_rand_sup_B, joint_sel),
                "E_plus_B_supports_A": rate(eB_sup_A, joint_sel),
                "matched_random_kB_supports_A": rate(eB_rand_sup_A, joint_sel),
                "bidirectional_cross_support": rate(bidir_sup, joint_sel),
            },
            "conditioned_on_true_bots": {
                "a_bot_recall": rate(base_pred, true_bot),
                "b_bot_recall": rate(b_sel_pred, true_bot),
                "bidirectional_cross_support_coverage": rate(bidir_sup, true_bot),
            },
            "conditioned_on_true_humans": {
                "a_false_positive_rate": rate(base_pred, true_hum),
                "b_false_positive_rate": rate(b_sel_pred, true_hum),
                "bidirectional_cross_support_false_hit_rate": rate(bidir_sup, true_hum),
            },
        },
        "human_search_calibration_audit": eval_human_search_calibration(records, cal_cfg, is_dev=is_dev),
        "forward_only": metrics_dict(y, base_pred),
        "naive_static_refutation": {
            **metrics_dict(y, naive_pred),
            "false_bots_refuted": int(np.sum(naive_refute & (y == 0))),
            "true_bots_refuted": int(np.sum(naive_refute & (y == 1))),
        },
        "calibrated_counterexample_refutation": {
            **metrics_dict(y, after_pred),
            "false_bots_refuted": int(np.sum(refute & (y == 0))),
            "true_bots_refuted": int(np.sum(refute & (y == 1))),
            "net_false_minus_true_refuted": int(np.sum(refute & (y == 0)) - np.sum(refute & (y == 1))),
            "acc_gain": float(accuracy_score(y, after_pred) - accuracy_score(y, base_pred)),
            "bot_f1_gain": float(f1_score(y, after_pred, pos_label=1) - f1_score(y, base_pred, pos_label=1)),
        },
        "subgraph_alignment_on_bot_candidates": {
            "candidate_n": len(bot_cands),
            "true_bot_candidate_n": int(np.sum(base_pred & (y == 1))),
            "false_bot_candidate_n": int(np.sum(base_pred & (y == 0))),
            "median_family_jaccard": float(np.median([r["family_jaccard"] for r in bot_cands])) if bot_cands else 0.0,
            "matched_random_family_jaccard": float(np.median([r["rand_family_jaccard"] for r in bot_cands]))
            if bot_cands
            else 0.0,
            "median_motif_jaccard": float(np.median([r["motif_jaccard"] for r in bot_cands])) if bot_cands else 0.0,
            "matched_random_motif_jaccard": float(np.median([r["rand_motif_jaccard"] for r in bot_cands]))
            if bot_cands
            else 0.0,
            "median_exact_source_jaccard": float(np.median([r["exact_source_jaccard"] for r in bot_cands]))
            if bot_cands
            else 0.0,
            "matched_random_exact_source_jaccard": float(
                np.median([r["rand_exact_source_jaccard"] for r in bot_cands])
            )
            if bot_cands
            else 0.0,
            "median_bipartite_alignment": float(np.median([r["approximate_subgraph_alignment"] for r in bot_cands]))
            if bot_cands
            else 0.0,
            "matched_random_bipartite_alignment": float(
                np.median([r["rand_approximate_subgraph_alignment"] for r in bot_cands])
            )
            if bot_cands
            else 0.0,
        },
    }


def diagnose_and_dump_records(records: list[dict], ft: float, rt: float, rule: dict, out_jsonl: Path) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            y_true = int(r["label"])
            f_score = float(r["f_score"])
            r_score = float(r["r_score"])
            r_on_f = float(r["r_on_f"])
            f_with_ce = float(r["f_with_ce"])
            h_ce = float(r["h_ce_own"])
            f_on_r = float(r["f_on_r"])

            base_pred = int(f_score >= ft)
            b_sel_pred = int(r_score >= rt)
            eA_sup_B = bool(r_on_f >= rt)
            eB_sup_A = bool(f_on_r >= ft)
            bidir_sup = bool(eA_sup_B and eB_sup_A)

            chal = (
                rule["w_own"] * (r_score - rt)
                + rule["w_onf"] * (r_on_f - rt)
                + rule["w_ce"] * (f_with_ce - ft)
                - rule["w_hce"] * max(0.0, h_ce)
            )
            is_refuted = bool(base_pred and ((f_score - ft) < rule["fwd_cap"]) and (chal < rule["chal_thresh"]))
            final_pred = int(base_pred and (not is_refuted))
            is_correct = bool(final_pred == y_true)

            if is_correct:
                if y_true == 1:
                    status = "TRUE_POSITIVE"
                    failure_mode = "NONE"
                    diagnosis = "Bot behavior successfully verified by Forward Miner. Minimal coalition confirmed with high confidence."
                    algo_direction = "Maintain current sparse coalition selection budget."
                else:
                    status = "TRUE_NEGATIVE"
                    if is_refuted:
                        failure_mode = "REFUTED_FALSE_POSITIVE"
                        diagnosis = "Initially misclassified as bot by Forward Miner, but Heterogeneous Challenger successfully found human counterexamples in disjoint/background groups and refuted the alert."
                        algo_direction = "Bidirectional refutation successfully corrected single-model confirmation bias."
                    else:
                        failure_mode = "NONE"
                        diagnosis = "Correctly recognized as human. No false bot evidence coalition was formed."
                        algo_direction = "Maintain humility constraints on singleton groups."
            else:
                if y_true == 0 and final_pred == 1:
                    status = "FALSE_POSITIVE"
                    if len(r.get("counterexample_groups", [])) == 0:
                        failure_mode = "SPARSE_PROFILE_LACK_OF_COUNTER_EVIDENCE"
                        diagnosis = "Challenger failed to find active human counter-evidence groups because account has minimal profile metadata or sparse graph connections."
                        algo_direction = "Expand default prior on low-activity accounts and calibrate dynamic threshold based on candidate group cardinality."
                    else:
                        failure_mode = "CAMOUFLAGED_ACTIVITY_WEAK_CHALLENGE"
                        diagnosis = "Forward miner detected high-frequency interaction patterns mimicking bot coordination, and challenger rebuttal did not exceed refutation threshold."
                        algo_direction = "Strengthen slot parity and cross-family co-occurrence penalty during training to prevent structural camouflage."
                else:
                    status = "FALSE_NEGATIVE"
                    if is_refuted:
                        failure_mode = "OVER_REFUTATION_BY_CHALLENGER"
                        diagnosis = "Forward miner correctly detected bot, but Heterogeneous Challenger over-penalized due to normal-looking profile metadata or mixed human-like tweets."
                        algo_direction = "Tighten challenger refutation threshold (require higher margin or multiple human counterexamples) to avoid suppressing true bots."
                    else:
                        failure_mode = "STEALTH_BOT_SUBTHRESHOLD"
                        diagnosis = "Bot exhibited high camouflage: its selected groups yielded score below forward threshold (ft), blending in with normal human behavior."
                        algo_direction = "Incorporate multi-hop relational context and higher-order motif patterns to amplify subtle bot coordination."

            audit_entry = {
                "account_id": r["account_id"],
                "label": y_true,
                "label_name": "bot" if y_true == 1 else "human",
                "forward_score": round(f_score, 4),
                "forward_threshold": round(ft, 4),
                "forward_prediction": base_pred,
                "forward_evidence_groups": r.get("forward_groups", []),
                "reverse_score": round(r_score, 4),
                "reverse_threshold": round(rt, 4),
                "reverse_prediction": b_sel_pred,
                "reverse_witness_groups": r.get("reverse_groups", []),
                "counterexample_groups": r.get("counterexample_groups", []),
                "challenger_score_on_fwd_evidence": round(r_on_f, 4),
                "fwd_score_on_rev_evidence": round(f_on_r, 4),
                "bidirectional_cross_support": bidir_sup,
                "refuted_by_challenger": is_refuted,
                "final_prediction": final_pred,
                "is_correct": is_correct,
                "status": status,
                "failure_mode": failure_mode,
                "diagnosis": diagnosis,
                "algorithmic_improvement_action": algo_direction,
                "family_jaccard": round(r.get("family_jaccard", 0.0), 4),
                "motif_jaccard": round(r.get("motif_jaccard", 0.0), 4),
                "exact_source_jaccard": round(r.get("exact_source_jaccard", 0.0), 4),
            }
            f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--forward", type=Path, required=True)
    ap.add_argument("--reverse", type=Path, required=True)
    ap.add_argument("--search-budget", type=int, default=4)
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--dump-sample-audit", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    f_model, f_kind, ft = load_any_model(args.forward, device)
    r_model, r_kind, rt = load_any_model(args.reverse, device)

    dev_records = collect_split_records(
        cache, "dev", f_model, f_kind, r_model, r_kind, device, search_budget=args.search_budget
    )
    rule = calibrate_refutation_rule(dev_records, ft, rt)
    cal_cfg = calibrate_human_search_thresholds(dev_records, ft, args.search_budget)
    dev_eval = apply_refutation_rule(dev_records, ft, rt, rule, cal_cfg, is_dev=True)

    output = {
        "cache": str(args.cache),
        "forward_model": str(args.forward),
        "reverse_model": str(args.reverse),
        "forward_threshold": ft,
        "reverse_threshold": rt,
        "dev_locked_refutation_rule": rule,
        "dev_cal_human_search_thresholds": cal_cfg,
        "dev": dev_eval,
    }

    if args.eval_test and "test" in cache["split"]:
        test_records = collect_split_records(
            cache, "test", f_model, f_kind, r_model, r_kind, device, search_budget=args.search_budget
        )
        output["test"] = apply_refutation_rule(test_records, ft, rt, rule, cal_cfg, is_dev=False)
        if args.dump_sample_audit:
            diagnose_and_dump_records(test_records, ft, rt, rule, args.dump_sample_audit)
    elif args.dump_sample_audit:
        diagnose_and_dump_records(dev_records, ft, rt, rule, args.dump_sample_audit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Three-Tier Evidence Certificate & Disjoint Or-of-Ands Witness Decomposition.

Explicitly separates and quantifies:
- Tier 1 (Coalition Sufficiency & Synergy): Minimal source-closed coalition C_u^(1) sufficient for Bot,
  with multi-group rate, cross-family rate, singleton irreducibility, and pairwise synergy gain.
- Tier 2 (Conditional Effect in Selected Subgraph): Source-closed removal of C_u^(1) from E+_u flips
  decision below threshold and exceeds equal-size random source-closed controls.
- Tier 3 (Observational Disjoint Witness Depth K / Or-of-Ands): Iterative extraction of mutually
  disjoint source-closed witness coalitions {C_u^(1), ..., C_u^(K)} from V_u until remaining sources
  cannot support Bot.
- Bidirectional Orthogonality vs Single-Model Self-Search (Self-CEGAR): Compares witness sets identified
  by Heterogeneous Challenger M_rev against M_fwd's primary and alternative witness sets, quantifying
  the rate of non-redundant orthogonal witness discovery.
- Never claims real-world identity-level necessity from observational feature removals.
"""
from __future__ import annotations
import argparse, json, random, sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from veribot.forward_miner import ForwardEvidenceMiner, CleanCoalitionNet
from veribot.heterogeneous_challenger import HeterogeneousChallenger, CleanChallengerNet


def group_family_id(name: str) -> int:
    prefix = name.split(":", 1)[0]
    if prefix in ("profile", "user"):
        return 0
    if prefix in ("relation", "reply", "neighbor", "neighbor_profile"):
        return 1
    if prefix in ("content", "tweet", "posts", "comment_1", "comment_2"):
        return 2
    return 3


def load_model(model_dir: Path, device: torch.device):
    ck = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
    rep = json.loads((model_dir / "report.json").read_text())
    arch = str(ck.get("arch", ck.get("architecture", "forward_evidence_miner")))
    input_dim = int(ck["input_dim"])

    if "challenger" in arch or ck.get("role") == "reverse":
        m = HeterogeneousChallenger(
            input_dim,
            payload_cols=ck.get("payload_cols"),
            hidden=int(ck.get("hidden", 128)),
            heads=int(ck.get("heads", 4)),
            layers=int(ck.get("layers", 2)),
            num_freq=int(ck.get("num_freq", 0)),
            dropout=float(ck.get("dropout", 0.15)),
            local_gate=bool(ck.get("local_gate", True)),
        ).to(device)
        kind = "challenger"
    else:
        m = ForwardEvidenceMiner(
            input_dim,
            payload_cols=ck.get("payload_cols"),
            hidden=int(ck.get("hidden", 128)),
            pair_dim=int(ck.get("pair_dim", 32)),
            num_families=int(ck.get("num_families", 4)),
            num_freq=int(ck.get("num_freq", 0)),
            num_queries=int(ck.get("num_queries", 2)),
            dropout=float(ck.get("dropout", 0.15)),
        ).to(device)
        kind = "miner"

    m.load_state_dict(ck["model_state"])
    m.eval()
    return m, kind, float(rep["dev_threshold"])


def run_model(m, kind: str, x: torch.Tensor, v: torch.Tensor, fams: torch.Tensor, **kwargs) -> dict:
    return m(x, v, fams, **kwargs)


def filter_non_anchor_raw(raw_ids: set[str]) -> set[str]:
    filtered = {sid for sid in raw_ids if not sid.endswith(":user:name")}
    return filtered if filtered else raw_ids


def source_closed_invalid_indices(all_sources: list[tuple[str, ...]], chosen_indices: list[int]) -> tuple[set[int], set[str]]:
    raw = set().union(*(set(all_sources[j]) for j in chosen_indices)) if chosen_indices else set()
    eff_raw = filter_non_anchor_raw(raw)
    invalid = {j for j, s in enumerate(all_sources) if set(s) & eff_raw}
    return invalid, eff_raw


def greedy_minimal_coalition(
    m,
    kind: str,
    x: torch.Tensor,
    v: torch.Tensor,
    fams: torch.Tensor,
    selected_mask: torch.Tensor,
    all_sources: list[tuple[str, ...]],
    threshold: float,
) -> dict:
    sel_indices = [j for j, flag in enumerate(selected_mask.tolist()) if flag]
    removed: list[int] = []
    remaining = set(sel_indices)
    trace = []
    after_val = float(run_model(m, kind, x, v, fams, hard_mask=selected_mask[None, :])["bot_logit"][0])
    inv_set: set[int] = set()
    raw_set: set[str] = set()

    while remaining:
        candidates = []
        for j in sorted(remaining):
            inv, r_ids = source_closed_invalid_indices(all_sources, removed + [j])
            keep = selected_mask.clone()
            keep[list(inv)] = False
            sc = float(run_model(m, kind, x, v, fams, hard_mask=keep[None, :])["bot_logit"][0])
            candidates.append((sc, j, inv, r_ids))
        after_val, best_j, inv_set, raw_set = min(candidates, key=lambda z: (z[0], z[1]))
        removed.append(best_j)
        remaining -= inv_set & remaining
        remaining.discard(best_j)
        trace.append({"added_group_idx": best_j, "after_logit": after_val, "invalidated_count": len(inv_set)})
        if after_val < threshold:
            break

    coal_mask = torch.zeros_like(selected_mask)
    coal_mask[removed] = True
    coal_out = run_model(m, kind, x, v, fams, hard_mask=coal_mask[None, :])
    coal_only = float(coal_out["bot_logit"][0])
    coal_unary = float(coal_out.get("unary_only_logit", coal_out["bot_logit"])[0])

    # Evaluate singletons inside coalition
    singleton_logits = []
    for j in removed:
        one_mask = torch.zeros_like(selected_mask)
        one_mask[j] = True
        s_val = float(run_model(m, kind, x, v, fams, hard_mask=one_mask[None, :])["bot_logit"][0])
        singleton_logits.append(s_val)

    return {
        "coalition_indices": removed,
        "invalidated_indices": sorted(inv_set),
        "raw_source_ids": sorted(raw_set),
        "after_removed_logit": after_val,
        "coalition_only_logit": coal_only,
        "coalition_unary_logit": coal_unary,
        "coalition_synergy_gain": coal_only - coal_unary,
        "max_singleton_logit": max(singleton_logits) if singleton_logits else -10.0,
        "singleton_logits": singleton_logits,
        "conditional_flip": after_val < threshold,
        "coalition_sufficient": coal_only >= threshold,
        "dual_pass": (after_val < threshold) and (coal_only >= threshold),
        "trace": trace,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--forward", type=Path, required=True)
    ap.add_argument("--reverse", type=Path, default=None)
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--limit", type=int, default=0, help="0 means evaluate all true bots")
    ap.add_argument("--max-witness-depth", type=int, default=4)
    ap.add_argument("--random-trials", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    f_model, f_kind, f_thresh = load_model(args.forward, device)
    r_model, r_kind, r_thresh = (load_model(args.reverse, device) if args.reverse else (None, "", 0.0))

    rng = random.Random(0)
    rows = []
    with torch.inference_mode():
        for ix in cache["split"][args.split]:
            i = int(ix)
            if int(cache["labels"][i]) != 1:
                continue
            x = cache["features"][i].to(device).unsqueeze(0)
            v = torch.ones((1, x.shape[1]), dtype=torch.bool, device=device)
            names = cache["group_names"][i]
            sources = cache["raw_source_ids"][i]
            fams = torch.tensor([[group_family_id(n) for n in names]], dtype=torch.long, device=device)

            fo = run_model(f_model, f_kind, x, v, fams, mode="selected")
            base_logit = float(fo["bot_logit"][0])
            if base_logit < f_thresh:
                continue
            if args.limit > 0 and len(rows) >= args.limit:
                break

            sel_mask = fo["selected"][0].bool()
            sel_indices = [j for j, flag in enumerate(sel_mask.tolist()) if flag]

            # Tier 1 & Tier 2: Primary minimal source-closed coalition inside E+_u
            primary = greedy_minimal_coalition(f_model, f_kind, x, v, fams, sel_mask, sources, f_thresh)
            k_size = len(primary["coalition_indices"])

            # Matched equal-size random source-closed controls inside E+_u
            rand_drops = []
            rand_dual = 0
            for _ in range(args.random_trials):
                pick = rng.sample(sel_indices, min(k_size, len(sel_indices)))
                inv, _ = source_closed_invalid_indices(sources, pick)
                keep = sel_mask.clone()
                keep[list(inv)] = False
                r_after = float(run_model(f_model, f_kind, x, v, fams, hard_mask=keep[None, :])["bot_logit"][0])
                only_m = torch.zeros_like(sel_mask)
                only_m[pick] = True
                r_only = float(run_model(f_model, f_kind, x, v, fams, hard_mask=only_m[None, :])["bot_logit"][0])
                rand_drops.append(base_logit - r_after)
                if r_after < f_thresh and r_only >= f_thresh:
                    rand_dual += 1

            # Tier 3: Disjoint Alternative Witness Enumeration (Or-of-Ands depth K)
            disjoint_coalitions = [
                {
                    "rank": 1,
                    "groups": [names[j] for j in primary["coalition_indices"]],
                    "families": sorted({group_family_id(names[j]) for j in primary["coalition_indices"]}),
                    "coalition_only_logit": primary["coalition_only_logit"],
                    "after_removed_logit": primary["after_removed_logit"],
                    "raw_source_count": len(primary["raw_source_ids"]),
                }
            ]
            cumulative_invalid = set(primary["invalidated_indices"])
            union_coalition_indices = set(primary["coalition_indices"])

            for depth in range(2, args.max_witness_depth + 1):
                v_rem = v.clone()
                if cumulative_invalid:
                    v_rem[0, list(cumulative_invalid)] = False
                if not v_rem.any():
                    break
                alt_out = run_model(f_model, f_kind, x, v_rem, fams, mode="selected")
                alt_logit = float(alt_out["bot_logit"][0])
                if alt_logit < f_thresh:
                    break
                alt_sel = alt_out["selected"][0].bool() & v_rem[0]
                alt_c = greedy_minimal_coalition(f_model, f_kind, x, v_rem, fams, alt_sel, sources, f_thresh)
                disjoint_coalitions.append(
                    {
                        "rank": depth,
                        "groups": [names[j] for j in alt_c["coalition_indices"]],
                        "families": sorted({group_family_id(names[j]) for j in alt_c["coalition_indices"]}),
                        "coalition_only_logit": alt_c["coalition_only_logit"],
                        "after_removed_logit": alt_c["after_removed_logit"],
                        "raw_source_count": len(alt_c["raw_source_ids"]),
                    }
                )
                cumulative_invalid.update(alt_c["invalidated_indices"])
                union_coalition_indices.update(alt_c["coalition_indices"])

            v_final = v.clone()
            if cumulative_invalid:
                v_final[0, list(cumulative_invalid)] = False
            final_rem_logit = (
                float(run_model(f_model, f_kind, x, v_final, fams, mode="selected")["bot_logit"][0])
                if v_final.any()
                else -10.0
            )

            # Reverse model bidirectional verification on primary & disjoint witness union
            rev_info = {}
            if r_model is not None:
                ro = run_model(r_model, r_kind, x, v, fams, mode="selected")
                r_sel = ro["selected"][0].bool()
                r_indices = {j for j, flag in enumerate(r_sel.tolist()) if flag}
                c1_mask = torch.zeros_like(sel_mask)
                c1_mask[primary["coalition_indices"]] = True
                keep_c1 = sel_mask.clone()
                keep_c1[primary["invalidated_indices"]] = False

                r_on_fwd = float(run_model(r_model, r_kind, x, v, fams, hard_mask=sel_mask[None, :])["bot_logit"][0])
                r_on_c1 = float(run_model(r_model, r_kind, x, v, fams, hard_mask=c1_mask[None, :])["bot_logit"][0])
                r_after_c1 = float(run_model(r_model, r_kind, x, v, fams, hard_mask=keep_c1[None, :])["bot_logit"][0])

                c1_set = set(primary["coalition_indices"])
                # Orthogonal witness discovery: groups selected by challenger that are outside M_fwd's primary C1
                challenger_orthogonal_groups = [names[j] for j in (r_indices - c1_set)]
                challenger_entirely_new = len(r_indices - union_coalition_indices) > 0

                rev_info = {
                    "reverse_own_logit": float(ro["bot_logit"][0]),
                    "reverse_on_forward_selected_logit": r_on_fwd,
                    "reverse_on_primary_coalition_logit": r_on_c1,
                    "reverse_after_primary_removed_logit": r_after_c1,
                    "reverse_same_direction_drop": (r_on_fwd - r_after_c1) > 0,
                    "reverse_hits_primary_coalition": bool(r_indices & c1_set),
                    "reverse_hits_disjoint_witness_union": bool(r_indices & union_coalition_indices),
                    "reverse_primary_overlap_count": len(r_indices & c1_set),
                    "reverse_union_overlap_count": len(r_indices & union_coalition_indices),
                    "orthogonal_witness_count": len(challenger_orthogonal_groups),
                    "orthogonal_witness_groups": challenger_orthogonal_groups,
                    "captures_disjoint_unexplored_witness": challenger_entirely_new,
                }

            c1_fams = sorted({group_family_id(names[j]) for j in primary["coalition_indices"]})
            rows.append(
                {
                    "account_id": str(cache["ids"][i]),
                    "baseline_logit": base_logit,
                    "threshold": f_thresh,
                    "selected_group_count": len(sel_indices),
                    "tier1_coalition_sufficiency": {
                        "coalition_groups": [names[j] for j in primary["coalition_indices"]],
                        "coalition_size": k_size,
                        "multi_group": k_size > 1,
                        "distinct_families": len(c1_fams),
                        "cross_family": len(c1_fams) >= 2,
                        "coalition_only_logit": primary["coalition_only_logit"],
                        "coalition_sufficient": primary["coalition_sufficient"],
                        "max_singleton_logit": primary["max_singleton_logit"],
                        "singleton_irreducible": (primary["max_singleton_logit"] < f_thresh) and (k_size > 1),
                        "coalition_unary_logit": primary["coalition_unary_logit"],
                        "synergy_flips_unary_to_bot": (primary["coalition_unary_logit"] < f_thresh)
                        and primary["coalition_sufficient"],
                        "synergy_gain": primary["coalition_synergy_gain"],
                    },
                    "tier2_conditional_effect": {
                        "after_source_closed_removed_logit": primary["after_removed_logit"],
                        "conditional_flip": primary["conditional_flip"],
                        "dual_pass": primary["dual_pass"],
                        "coalition_logit_drop": base_logit - primary["after_removed_logit"],
                        "matched_random_median_drop": float(np.median(rand_drops)) if rand_drops else 0.0,
                        "exceeds_matched_random_drop": (base_logit - primary["after_removed_logit"])
                        > (float(np.median(rand_drops)) if rand_drops else 0.0),
                        "matched_random_dual_pass_rate": rand_dual / max(args.random_trials, 1),
                    },
                    "tier3_disjoint_witness_hierarchy": {
                        "disjoint_witness_depth_K": len(disjoint_coalitions),
                        "final_remaining_logit_after_K": final_rem_logit,
                        "observational_flip_after_K": final_rem_logit < f_thresh,
                        "disjoint_witnesses": disjoint_coalitions,
                        "identity_level_necessity_claimed": False,
                    },
                    "bidirectional_verification": rev_info,
                }
            )

    n_aud = max(len(rows), 1)
    summary = {
        "cache": str(args.cache),
        "forward_model": str(args.forward),
        "reverse_model": str(args.reverse) if args.reverse else None,
        "split": args.split,
        "audited_true_bot_n": len(rows),
        "tier1_summary": {
            "coalition_sufficient_rate": sum(r["tier1_coalition_sufficiency"]["coalition_sufficient"] for r in rows)
            / n_aud,
            "multi_group_coalition_rate": sum(r["tier1_coalition_sufficiency"]["multi_group"] for r in rows) / n_aud,
            "cross_family_coalition_rate": sum(r["tier1_coalition_sufficiency"]["cross_family"] for r in rows) / n_aud,
            "singleton_irreducible_rate": sum(
                r["tier1_coalition_sufficiency"]["singleton_irreducible"] for r in rows
            )
            / n_aud,
            "synergy_flips_unary_to_bot_rate": sum(
                r["tier1_coalition_sufficiency"]["synergy_flips_unary_to_bot"] for r in rows
            )
            / n_aud,
            "median_coalition_size": float(np.median([r["tier1_coalition_sufficiency"]["coalition_size"] for r in rows])) if rows else 0.0,
            "mean_synergy_gain": float(np.mean([r["tier1_coalition_sufficiency"]["synergy_gain"] for r in rows])) if rows else 0.0,
        },
        "tier2_summary": {
            "conditional_flip_rate": sum(r["tier2_conditional_effect"]["conditional_flip"] for r in rows) / n_aud,
            "dual_pass_rate": sum(r["tier2_conditional_effect"]["dual_pass"] for r in rows) / n_aud,
            "matched_random_dual_pass_rate": float(
                np.mean([r["tier2_conditional_effect"]["matched_random_dual_pass_rate"] for r in rows])
            ) if rows else 0.0,
            "exceeds_matched_random_drop_rate": sum(
                r["tier2_conditional_effect"]["exceeds_matched_random_drop"] for r in rows
            )
            / n_aud,
        },
        "tier3_summary": {
            "witness_depth_distribution": dict(
                sorted(Counter(r["tier3_disjoint_witness_hierarchy"]["disjoint_witness_depth_K"] for r in rows).items())
            ),
            "observational_flip_after_K_rate": sum(
                r["tier3_disjoint_witness_hierarchy"]["observational_flip_after_K"] for r in rows
            )
            / n_aud,
            "identity_level_necessity_claimed": False,
        },
    }
    if r_model is not None and rows:
        summary["bidirectional_summary"] = {
            "reverse_same_direction_drop_rate": sum(
                r["bidirectional_verification"]["reverse_same_direction_drop"] for r in rows
            )
            / n_aud,
            "reverse_hits_primary_coalition_rate": sum(
                r["bidirectional_verification"]["reverse_hits_primary_coalition"] for r in rows
            )
            / n_aud,
            "reverse_hits_disjoint_witness_union_rate": sum(
                r["bidirectional_verification"]["reverse_hits_disjoint_witness_union"] for r in rows
            )
            / n_aud,
            "challenger_orthogonal_witness_rate": sum(
                r["bidirectional_verification"]["orthogonal_witness_count"] > 0 for r in rows
            )
            / n_aud,
            "challenger_unexplored_witness_rate": sum(
                r["bidirectional_verification"]["captures_disjoint_unexplored_witness"] for r in rows
            )
            / n_aud,
        }
    summary["rows"] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()

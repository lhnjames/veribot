"""S1 gate, third check: human audit of 20 rendered accounts (10 bot + 10 human).

For every bucketed clause of every audited account this recomputes, from the *train* distribution,
the true percentile span of the hidden raw value and checks that it is consistent with the tier
word the model will actually see. A tier that says "very high" for a value at the 40th percentile
is exactly the kind of quietly-wrong evidence the verifiability story depends on not existing, so
it is checked mechanically and left in a markdown file for a human to read.

Coverage: PROFILE/OBS/ACTIVITY clauses are checked against their train feature column; RELATION
clauses against the train neighbour-count distribution. Multi-clause units (OBS_03/06/08,
RELATION_01) have every clause checked, not just the headline.

Audited accounts come from **dev** (rendered with train-only cuts), so a clean audit is also direct
evidence the frozen cuts transfer read-only across splits. Selection is fixed forever: sort dev by
account_id, take the first 10 bots and the first 10 humans.

    python scripts/audit_render_20.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from verifiable_bot.v4.features_v4 import (
    FEATURE_NAMES,
    TWEET_STAT_INDICES,
    extract_features,
    neighbor_counts,
    tweet_list,
)
from verifiable_bot.v4.render_v4 import build_units_v4
from verifiable_bot.v4.thresholds_v4 import QUANTILE_EDGES, load_thresholds, slot_of


def train_columns(train_path: Path, cache: Path):
    """Train feature matrix, 'has tweets' mask, and neighbour-count columns; cached."""
    if cache.exists():
        blob = np.load(cache)
        return blob["matrix"], blob["with_tweets"], blob["following"], blob["follower"]
    records = json.loads(train_path.read_text(encoding="utf-8"))
    matrix = np.asarray([extract_features(r) for r in records], dtype=np.float64)
    with_tweets = np.asarray([bool(tweet_list(r)) for r in records])
    nbr = [nc for nc in (neighbor_counts(r) for r in records) if nc is not None]
    following = np.asarray([f for f, _ in nbr], dtype=np.float64)
    follower = np.asarray([g for _, g in nbr], dtype=np.float64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, matrix=matrix, with_tweets=with_tweets,
                        following=following, follower=follower)
    return matrix, with_tweets, following, follower


def percentile_span(value: float, column: np.ndarray) -> tuple[float, float]:
    """[strictly-below %, at-or-below %] — a span, because ties make a single number meaningless."""
    n = column.size
    return 100.0 * float((column < value).sum()) / n, 100.0 * float((column <= value).sum()) / n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/twibot20_complete")
    parser.add_argument("--thresholds", default="outputs/v4/render_v4/bucket_thresholds_v4.json")
    parser.add_argument("--out", default="outputs/v4/render_v4/audit_20.md")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    thresholds = load_thresholds(args.thresholds)
    matrix, with_tweets, following_col, follower_col = train_columns(
        data_dir / "train.json", Path(args.out).parent / "train_feature_cache.npz"
    )
    relation_cols = {"neighbor_following_count": following_col, "neighbor_follower_count": follower_col}

    dev = json.loads((data_dir / "dev.json").read_text(encoding="utf-8"))
    dev.sort(key=lambda r: str(r.get("ID", "")).strip())
    bots = [r for r in dev if str(r.get("label")).strip() == "1"][:10]
    humans = [r for r in dev if str(r.get("label")).strip() == "0"][:10]
    audited = [("bot", r) for r in bots] + [("human", r) for r in humans]

    lines: list[str] = [
        "# S1 渲染人工审计 — 20 个账号（10 bot + 10 human）",
        "",
        f"- 阈值文件：`{thresholds['source_file']}` 拟合，sha256 `{thresholds['source_sha256'][:16]}…`",
        f"- 区间约定：`{thresholds['interval_convention']}`",
        "- 审计对象取自 **dev**（用 train 拟合的冻结阈值渲染），选取：按 account_id 排序后前 10 bot + 前 10 human。",
        "- 每个分档子句（clause）都核对：`真实分位` = 原始值在 **train** 分布中的百分位区间（并列给区间），",
        "  `档位分位` = 渲染档位对应的预注册百分位区间；两者相交即为一致。RELATION 子句对照 train 邻居计数分布。",
        "",
    ]

    total = mismatches = 0
    for label, record in audited:
        account_id = str(record.get("ID", "")).strip()
        values = extract_features(record)
        units = build_units_v4(record, thresholds)
        has_tweets = bool(tweet_list(record))
        lines += [
            f"## `{account_id}` — 真实标签 **{label}**"
            + ("" if has_tweets else "  ·  ⚠️ 无推文，应无 ACTIVITY_xx"),
            "",
            "| 单元 | 渲染文本 | 子句特征 | 原始值 | 档位 | 档位分位 | 真实分位 | 一致 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for unit in units:
            if not unit.clauses:
                lines.append(f"| `{unit.unit_id}` | {unit.text} | — | — | — | — | — | — |")
                continue
            for k, clause in enumerate(unit.clauses):
                name = clause["name"]
                value = float(clause["raw"])
                if clause["feature_index"] is None:
                    spec = thresholds["relation"][name]
                    column = relation_cols[name]
                else:
                    index = clause["feature_index"]
                    spec = thresholds["features"][name]
                    column = matrix[with_tweets, index] if index in TWEET_STAT_INDICES else matrix[:, index]
                slot = slot_of(value, spec)
                lo, hi = QUANTILE_EDGES[slot], QUANTILE_EDGES[slot + 1]
                p_lt, p_le = percentile_span(value, column)
                ok = not (p_le < lo or p_lt > hi)
                total += 1
                mismatches += 0 if ok else 1
                shown_text = unit.text if k == 0 else "〃"
                shown_id = f"`{unit.unit_id}`" if k == 0 else ""
                lines.append(
                    f"| {shown_id} | {shown_text} | `{name}` | {value:.4g} | {clause['tier']} | "
                    f"p{lo:g}-p{hi:g} | p{p_lt:.1f}-p{p_le:.1f} | {'✅' if ok else '❌'} |"
                )
        prefixes = sorted({u.unit_id.split('_')[0] for u in units})
        lines += ["", f"单元类型：{', '.join(prefixes)}；共 {len(units)} 个单元。", ""]

    lines.insert(
        8,
        f"**机检结果：共核对 {total} 个分档子句，不一致 {mismatches} 个。**"
        + ("  ✅ 全部一致。" if mismatches == 0 else "  ❌ 存在不一致，需修复后重跑。") + "\n",
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"checked {total} bucketed clauses across 20 accounts, {mismatches} mismatches")
    print(f"-> {out.resolve()}")
    raise SystemExit(1 if mismatches else 0)


if __name__ == "__main__":
    main()

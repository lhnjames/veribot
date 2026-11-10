"""Render source-bound, signed decision cards from frozen certificates only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import sha256


NAMES = {
    "profile_counts": "档案计数", "profile_presentation": "档案呈现",
    "sampled_tweet_stats": "采样推文统计", "native_follow_graph": "原生关注图",
}
def number(value: float | None, digits: int = 4) -> str:
    return "缺失" if value is None else f"{value:+.{digits}f}"


def observed_value(value: float | None) -> str:
    return "缺失" if value is None else f"{value:.4f}"


def kind(value: int) -> str:
    return "Bot" if value else "Human"


def source(term: dict) -> str:
    query = term["source_query"]
    if isinstance(query, dict):
        location = query.get("query", "")
        operation = query.get("operation", "")
        sample_size = query.get("tweet_sample_size")
        suffix = f"；n={sample_size}" if sample_size is not None else ""
        return f"`{location}`{('；' + operation) if operation else ''}{suffix}"
    return f"`{query}`"


def minimal(sets: list[list[str]]) -> str:
    return "；".join("＋".join(NAMES[group] for group in row) if row
                    else "空集（基准已判同类）" for row in sets)


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--reference-sensitivity", type=Path, required=True)
    parser.add_argument("--decision-sets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cards = json.loads(args.cards.read_text(encoding="utf-8"))
    reference = json.loads(args.reference_sensitivity.read_text(encoding="utf-8"))
    sets = json.loads(args.decision_sets.read_text(encoding="utf-8"))
    if (reference["decision_cards_sha256"] != sha256(args.cards)
            or sets["cards_sha256"] != sha256(args.cards)
            or sets["reference_sensitivity_sha256"] != sha256(args.reference_sensitivity)):
        raise ValueError("The three frozen certificates do not match")
    card_by_id = {row["account_id"]: row for row in cards["decision_cards"]}
    reference_by_id = {row["account_id"]: row for row in reference["rows"]}
    sets_by_id = {row["account_id"]: row for row in sets["rows"]}
    lines = [
        "# VeriBot：逐账号可核验判断卡示例",
        "",
        "日期：2026-11-04。本文件从三份冻结证书确定性生成，案例按官方 test 顺序在四种预测对错组合中各取第一个账号，再按账号哈希展示；不是只从正确案例中挑理由。对象是 **TwiBot-20 的 20 事实双尾 L1 分类器**，不是现存 HGT 权重，也不是修订方案中完成 A/B 方向固定和训练期分组约束的完整推理器。",
        "",
        f"模型证书 SHA256：`{cards['compact_certificate_sha256']}`；参考分布版本：`{card_by_id[next(iter(card_by_id))]['reference_version']}`。正数表示增加 Bot logit，负数表示减少 Bot logit。全部分数作用只对给定冻结模型与比较点成立。",
        "",
        "每张卡先呈现不含真实类别的模型判断，再列原始事实与正反作用、两种比较点的组替换、最小保类组合和来源编辑响应。真实类别只放在末尾用于核对；它没有参与卡片的来源或分数计算。全体 1,183 张原始判断卡见 `research/certificates/twibot20_compact20_all_test_decision_cards.json`。",
        "",
    ]
    displayed = sorted(cards["exemplar_ids_first_in_official_order"].items(),
                       key=lambda pair: hashlib.sha256(pair[1].encode()).hexdigest())
    for ordinal, (role, uid) in enumerate(displayed, start=1):
        card = card_by_id[uid]
        anchor = reference_by_id[uid]
        subsets = sets_by_id[uid]
        if (card["prediction"] != subsets["prediction"]
                or abs(card["full_margin"] - subsets["full_margin"]) > 1e-10):
            raise ValueError(f"The selected card differs from the subset certificate: {uid}")
        lines += [
            f"## 案例 {ordinal}：账号 `{uid}`",
            "",
            f"**判断。** 冻结模型给 Bot 概率 `{card['full_probability']:.6f}`；Bot 阈值 `{card['threshold']:.6f}`，故判 **{kind(card['prediction'])}**。完整 logit `{card['full_logit']:+.6f}`，相对阈值余量 `{card['full_margin']:+.6f}`。",
            "",
            f"**来源。** 官方行 `{card['official_source_locator']}`；去类别的原始行 SHA256：`{card['source_record_without_label_sha256']}`。该账号相关的原生关系边声明哈希：`{card['graph_source']['incident_declarations_sha256']}`；出度/入度/互关数分别为 `{card['graph_source']['out_degree']}/{card['graph_source']['in_degree']}/{card['graph_source']['reciprocal_degree']}`。",
            "",
        ]
        edge_rows = cards["exemplar_graph_edge_declarations"][role]
        entries = [[edge["source"], edge["target"], sorted(edge["declarations"])]
                   for edge in edge_rows]
        if canonical_hash(entries) != card["graph_source"]["incident_declarations_sha256"]:
            raise ValueError(f"Full graph declaration list differs from its card hash: {uid}")
        lines += [
            f"**原生关系声明。** 下表列出该账号相关的全部 {len(edge_rows)} 条已去重边及原始声明位置；每条边的方向是关注者→被关注者。",
            "",
            "| 来源账号 | 目标账号 | 官方原始声明位置 |",
            "|---|---|---|",
        ]
        for edge in edge_rows:
            lines.append(f"| `{edge['source']}` | `{edge['target']}` | "
                         + "；".join(f"`{location}`" for location in edge["declarations"])
                         + " |")
        lines += [
            "",
            "**全部 20 项实际输入。** 上尾／下尾是相对全零编码的有符号 logit 作用；零作用项仍保留。列出的查询为可复算来源位置或聚合规则。",
            "",
            "| 事实 | 原值 | 可靠度 | 上尾作用 | 下尾作用 | 合计作用 | 来源查询与运算 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for term in card["fact_terms"]:
            lines.append(
                f"| `{term['fact_name']}` | {observed_value(term['source_value'])} | "
                f"{term['reliability']:.3f} | {number(term['upper_logit_effect'])} | "
                f"{number(term['lower_logit_effect'])} | "
                f"{number(term['total_logit_effect'])} | {source(term)} |")
        lines += [
            "",
            "**分数账本与比较点。** 相对全零编码的基准余量与四组作用相加，精确得到上面的完整余量；真实训练账号 `14086337` 作为第二个标签盲选比较点，模型预测本身不变。",
            "",
            f"`{card['baseline_margin']:+.6f}`（全零基准）＋四组作用 ＝ `{card['full_margin']:+.6f}`；真实账号基准为 `{reference['observed_reference_margin']:+.6f}`。",
            "",
            "| 来源组 | 相对全零作用 | 相对真实账号作用 | 从完整输入替换为全零是否改判 | 替换为真实账号组是否改判 |",
            "|---|---:|---:|---|---|",
        ]
        for group in NAMES:
            zero = subsets["zero_encoded"]["full_context_leave_one_group_out_changes_class"][group]
            observed = subsets["observed_train_account"][
                "full_context_leave_one_group_out_changes_class"][group]
            lines.append(
                f"| {NAMES[group]} | {number(card['group_logit_effects'][group])} | "
                f"{number(anchor['observed_reference_group_differences'][group])} | "
                f"{'是' if zero else '否'} | {'是' if observed else '否'} |")
        lines += [
            "",
            f"**保类组合。** 相对全零：{minimal(subsets['zero_encoded']['minimal_preserving_groups'])}。相对真实账号：{minimal(subsets['observed_train_account']['minimal_preserving_groups'])}。这是 16 种编码组组合的类别结果；空集不表示完整判断没有使用来源。",
            "",
        ]
        edit = card["first_sampled_tweet_record_edit"]
        if edit["source_locator"] is not None:
            lines += [
                f"**受控来源响应。** 从采样列表删去 `{edit['source_locator']}` 并联动重算推文统计后，冻结模型 logit 为 `{edit['edited_logit']:+.6f}`，类别为 **{kind(edit['edited_class'])}**（{'改判' if edit['class_changed'] else '未改判'}）。此操作只检验来源敏感性；该记录没有被独立确认为错误。",
                "",
            ]
        lines += [
            f"**评估核对。** 官方类别为 **{kind(card['evaluation_label'])}**；此次模型判断{'正确' if card['prediction_correct'] else '错误'}。卡片如实保留了把模型推向该预测类别的项，包括错判时的项。",
            "",
        ]
    lines += [
        "## 如何解读这些卡",
        "",
        "读者可以按官方行位置重新抽取事实，检查上尾、下尾和阈值账本，再从原始采样列表实施规定编辑，观察冻结模型的分数响应。若改变比较点，作用方向和最小保类组合可以改变；如果完整判断错误，卡片也不会把来源改写成正确类别的故事。数值作用证明这个模型的输入依赖，行为含义和审核收益仍需独立检验。",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "accounts": len(
        cards["exemplar_ids_first_in_official_order"]), "fact_rows": 80,
        "lines": len(lines)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

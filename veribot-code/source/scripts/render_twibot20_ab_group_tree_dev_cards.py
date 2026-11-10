"""Render four source-bound frozen group-tree dev decisions in readable Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_twibot20_compact20_capacity import GROUPS
from train_v3_twibot20 import sha256


LABELS = {
    "correct_bot": "判对的 Bot",
    "correct_human": "判对的 Human",
    "human_wrongly_called_bot": "被错判为 Bot 的 Human",
    "bot_wrongly_called_human": "被错判为 Human 的 Bot",
}


def fmt(value: float | None, digits: int = 4) -> str:
    return "缺失" if value is None else f"{value:.{digits}f}"


def safe(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--contrastive-sets", type=Path, required=True)
    parser.add_argument("--structure-audit", type=Path)
    parser.add_argument("--multi-anchor-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.cards.read_text(encoding="utf-8"))
    refs = json.loads(args.reference_audit.read_text(encoding="utf-8"))
    contrast = json.loads(args.contrastive_sets.read_text(encoding="utf-8"))
    structure = (json.loads(args.structure_audit.read_text(encoding="utf-8"))
                 if args.structure_audit else None)
    multi = (json.loads(args.multi_anchor_audit.read_text(encoding="utf-8"))
             if args.multi_anchor_audit else None)
    if (refs["decision_cards_sha256"] != sha256(args.cards)
            or contrast["decision_cards_sha256"] != sha256(args.cards)
            or contrast["model_sha256"] != data["frozen_model_sha256"]):
        raise ValueError("Decision cards, reference audit or contrastive sets differ")
    if structure and (structure["decision_cards_sha256"] != sha256(args.cards)
                      or structure["model_sha256"] != data["frozen_model_sha256"]
                      or structure["cross_group_trees"] != 0):
        raise ValueError("Saved model structure audit differs from decision cards")
    if multi and (multi["decision_cards_sha256"] != sha256(args.cards)
                  or multi["model_sha256"] != data["frozen_model_sha256"]
                  or multi["prior_contrast_sha256"] != sha256(args.contrastive_sets)):
        raise ValueError("Multi-anchor audit differs from decision cards")
    unused_columns = set(structure["unused_input_columns"]) if structure else set()
    card_names = [term["fact_name"] for term in data["decision_cards"][0]["fact_terms"]]
    card_index = {name: j for j, name in enumerate(card_names)}
    cards = {card["account_id"]: card for card in data["decision_cards"]}
    ref_rows = {row["account_id"]: row for row in refs["rows"]}
    contrast_rows = {row["account_id"]: row for row in contrast["rows"]}
    multi_rows = ({row["account_id"]: row for row in multi["rows"]}
                  if multi else {})
    corrected_graph = data.get("graph_scope") == "all provided sampled neighbor declarations"
    split_visible = data.get("graph_scope") == "train/support/dev sampled neighbor declarations without test"
    edge_count = data.get("official_graph_edges_recomputed",
                          data.get("official_labeled_graph_edges_recomputed"))
    lines = [
        "# TwiBot-20：A/B 分组证据分类器的四张可核验判断卡"
        + ("（按切分可见的采样图）" if split_visible else
           "（全部发布采样图）" if corrected_graph else "（已标注账号采样图）"),
        "",
        "这四张卡来自预定 seed 0、支持量修正后的冻结组树模型和官方开发集。"
        + ("训练时图度数只使用 train/support 声明，开发账号推理时加入 dev 声明，test 声明始终不可见。"
           if split_visible else "图度数汇合 train/dev/test/support 的全部官方采样邻居声明。"
           if corrected_graph else "图度数仅汇合 train/dev/test 已标注账号的采样邻居声明。")
        + "A 区固定每项证据的双尾单调方向，B 区拟合仅在单个来源组内部使用事实的树。"
        + "开发集同时用于阈值选择和这里的展示；四张卡按官方顺序从判对 Bot、判对 Human、错判 Human、错判 Bot 各取第一例。模型输出不是行为真值。",
        "",
        f"- 模型 SHA256：`{data['frozen_model_sha256']}`；官方来源压缩包 SHA256：`{data['source_archive_sha256']}`。",
        f"- 冻结模型在开发集判对 {data['correct_cards']}/{data['dev_n']}，错判 {data['wrong_cards']}；发布采样来源图重建 {edge_count} 条有向边；全体组分数加回最大误差 `{data['max_full_logit_group_reconstruction_error']:.3g}`。",
        f"- 全零数学比较点 logit `{refs['zero_reference_logit']:.6f}`；标签盲选的真实训练账号 `{refs['observed_anchor_account_id']}`（`{refs['observed_anchor_source_locator']}`）的 logit `{refs['observed_anchor_logit']:.6f}`。更换比较点不改变任何原预测。",
        "- 组作用是完整模型相对于指定比较点的**精确加性分解**。单项事实的“完整上下文中移走它”是条件响应；同组事实可以相互作用，因此这一列不能求和当作组作用，也不是合法来源纠错。",
        (f"- 保存的 {structure['trees_total']} 棵树均只使用单个来源组；"
         f"{structure['used_input_columns']}/{structure['model_input_columns']} 个编码尾部曾用于树分裂。"
         "表中‘入树’是整个冻结模型的结构状态，账号本次作用仍以条件响应为准。"
         if structure else "- 下表列出词表中的全部事实；列入词表不等于当前模型或当前账号实际使用了该事实。"),
        "- 相反类别对照使用模型预测为相反类别的真实训练账号；四组的 16 种混合输入逐一经过冻结模型。最小保类组合只说明与那个比较账号的**模型输入差异**，不是来源层可实行的账号行为改动。",
        ("- 六个相反预测类别的真实训练账号分别作为比较点；下方另列所有比较点均认可的核心组。"
         "若核心为空，只表示没有一个组对这六个指定比较点都不可替代。"
         if multi else "- 单一真实比较账号只给出条件性结论；不可读作不依赖比较点的唯一原因。"),
        "",
    ]
    for category in LABELS:
        exemplar = data["exemplars"][category]
        uid = exemplar["account_id"]
        card = cards[uid]
        ref = ref_rows[uid]
        opposite = contrast_rows[uid]
        if card["model_sha256"] != data["frozen_model_sha256"]:
            raise ValueError("Card model hash differs")
        if abs(card["full_logit"] - card["all_zero_input_logit"] -
               sum(card["group_logit_effects"].values())) > 1e-9:
            raise ValueError("Zero-reference group effects do not reconstruct")
        if abs(card["full_logit"] - refs["observed_anchor_logit"] -
               sum(ref["observed_anchor_group_effects"].values())) > 1e-9:
            raise ValueError("Observed-reference group effects do not reconstruct")
        prediction = "Bot" if card["prediction"] else "Human"
        lines += [
            f"## {LABELS[category]}：账号 `{uid}`",
            "",
            f"模型输出 **{prediction}**；概率 `{card['full_probability']:.4f}`，logit `{card['full_logit']:.4f}`；阈值 logit `{card['threshold_logit']:.4f}`，余量 `{card['full_margin']:+.4f}`。来源位置 `{card['official_source_locator']}`；采样推文 **{card['tweet_sample_support_count']}** 条。标签仅供最后核对。",
            "",
            "| 来源组 | 相对全零的 logit 作用 | 相对真实训练账号的作用 | 全零组替换后类别 | 真实账号组替换后类别 |",
            "|---|---:|---:|---|---|",
        ]
        for group in GROUPS:
            zero_class = "Bot" if card["group_zeroed_logits"][group] >= card["threshold_logit"] else "Human"
            real_class = "Bot" if ref["observed_anchor_group_replacement_logits"][group] >= card["threshold_logit"] else "Human"
            lines.append(f"| `{group}` | {card['group_logit_effects'][group]:+.4f} | "
                         f"{ref['observed_anchor_group_effects'][group]:+.4f} | {zero_class} | {real_class} |")
        sets = ["＋".join(f"`{name}`" for name in group_set)
                for group_set in opposite["minimal_sufficient_group_sets"]]
        lines += ["", "### 从真实相反类别预测账号出发的最小组合", "",
                  f"比较账号 `{opposite['opposite_predicted_class_anchor_id']}` 的模型 logit 为 "
                  f"`{opposite['anchor_logit']:+.4f}`。将四个组中的任意组合替换为当前账号实际编码，"
                  f"穷举后得到 **{len(sets)}** 套最小保留本次类别的组合：" + "；".join(sets) + "。",
                  "这些组合取决于所选比较账号；多套组合不能被压缩成唯一原因。",
                  ""]
        if multi:
            sensitivity = multi_rows[uid]
            stable = sensitivity["stable_core_groups_across_anchors"]
            lines += ["### 六个真实比较账号下的核心组", "",
                      "在这六个比较点下，始终出现在**所有**最小保类组合中的来源组："
                      + ("、".join(f"`{name}`" for name in stable) if stable else "**无**") + "。",
                      "| 比较账号 | 其 logit | 最小组合套数 | 该比较点下的核心组 |",
                      "|---|---:|---:|---|"]
            for trial in sensitivity["opposite_anchor_results"]:
                core = [name for j, name in enumerate(GROUPS)
                        if trial["core_mask"] & (1 << j)]
                lines.append(
                    f"| `{trial['anchor_id']}` | {trial['anchor_logit']:+.4f} | "
                    f"{len(trial['minimal_masks'])} | "
                    + ("、".join(f"`{name}`" for name in core) if core else "无") + " |")
            lines += ["", "核心变化只说明比较点敏感；完整账号预测始终使用同一个冻结模型和原始输入。", ""]
        conditional = {row["fact_name"]: row
                       for row in exemplar["conditional_fact_removal"]}
        facts = {row["fact_name"]: row for row in card["fact_terms"]}
        for group, members in GROUPS.items():
            lines += ["", f"### `{group}`：词表事实与实测作用", "",
                      ("| 事实 | 冻结模型入树状态 | 原值 | 可靠度 | 上/下尾模型输入 | 完整上下文移走此事实的 logit 变化 | 来源查询 |"
                       if structure else
                       "| 事实 | 原值 | 可靠度 | 上/下尾模型输入 | 完整上下文移走此事实的 logit 变化 | 来源查询 |"),
                      ("|---|---|---:|---:|---:|---:|---|" if structure else
                       "|---|---:|---:|---:|---:|---|")]
            for name in members:
                fact = facts[name]
                removal = conditional[name]["full_context_removal_logit_change"]
                source = fact["source_query"]
                locator = source["query"]
                if "operation" in source:
                    locator += "; " + source["operation"]
                elif "incident_declarations_sha256" in source:
                    locator += "; declarations SHA256=" + source["incident_declarations_sha256"]
                usage = ""
                if structure:
                    j = card_index[name]
                    upper_used = j not in unused_columns
                    lower_used = j + len(card_names) not in unused_columns
                    usage = ("双尾入树" if upper_used and lower_used else
                             "仅高尾入树" if upper_used else
                             "仅低尾入树" if lower_used else "未入树")
                lines.append(
                    f"| `{name}` | " + (f"{usage} | " if structure else "")
                    + f"{fmt(fact['source_value'])} | {fact['reliability']:.4f} | "
                    f"{fact['upper_model_input']:.4f} / {fact['lower_model_input']:.4f} | "
                    f"{removal:+.4f} | {safe(locator)} |")
        lines += ["", "### 原生关系来源声明", "",
                  f"该账号参与 {len(exemplar['native_incident_edge_declarations'])} 条去重有向边；逐条原始声明如下：", ""]
        for source, target, locators in exemplar["native_incident_edge_declarations"]:
            lines.append(f"- `{source} → {target}`：" + "；".join(f"`{loc}`" for loc in locators))
        truth = "Bot" if card["evaluation_label"] else "Human"
        lines += ["", f"**最后核对：** 官方类别为 **{truth}**；本次预测"
                  f"{'正确' if card['prediction_correct'] else '错误'}。", ""]
    lines += [
        "## 读卡边界",
        "",
        "卡上的事实来自官方原始行和已发布的采样关注声明；证据数值经过固定参考变换，组作用经过冻结模型重算。全零和真实账号替换只检验模型函数，不等于现实行为干预。多比较点的稳定核心仍只对选定账号集合成立。类别真值不能代替证据行为含义真值；要证明审核价值，仍需针对这些实际入模事实做独立核验和有正确来源值的纠错实验。",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"cards": len(data["exemplars"]), "lines": len(lines),
                      "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()

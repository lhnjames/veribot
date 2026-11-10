"""Render two Fox8 judgments with no mandatory raw pointer into readable cards."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ALTERNATIVES = ROOT / "research/certificates/fox8_minimal_atomic_pointer_alternatives.json"
READBACK = ROOT / "research/certificates/fox8_minimal_atomic_pointer_alternatives_readback.json"
GROUPS = ROOT / "research/certificates/best_source_group_effects.json"
OUTPUT = ROOT / "research/VeriBot_Fox8_可替代原始来源判断卡_2026-11-04.md"


def main() -> None:
    alt = json.loads(ALTERNATIVES.read_text(encoding="utf-8"))
    readback = json.loads(READBACK.read_text(encoding="utf-8"))
    groups = json.loads(GROUPS.read_text(encoding="utf-8"))
    if readback["status"] != "all_minimum_size_fox8_pointer_alternatives_independently_rebuilt":
        raise ValueError("Independent readback is missing")
    from hashlib import sha256
    if readback["source_certificate_sha256"] != sha256(ALTERNATIVES.read_bytes()).hexdigest():
        raise ValueError("Alternative certificate changed after readback")
    selected = [(run["seed"], row) for run in alt["runs"] for row in run["rows"]
                if not row["pointers_in_every_minimal_coalition"]]
    if len(selected) != 2:
        raise ValueError("Expected exactly two judgments without a mandatory pointer")
    lines = [
        "# Fox8：两次没有唯一必需原始来源的冻结模型判断卡",
        "",
        "这两次判断来自同一批 69 个邻居整组遮蔽会改判、且原生邻居数不超过 12 的账号。",
        "操作单位是一条原始回复或转发中的目标指针：将所选指针改指向发布人口外，",
        "保持推文文本、资料、模型权重、参考群体和 0.5 分类阈值不变，",
        "联动重算目标账号的互动计数、邻居集合、邻居均值及模型输出。",
        "表内的 P 编号只属于本卡；行号与推文序号可回到本地原始档案核对。",
        "",
        "**核验范围。** 每次判断的全部指针子集均已穷举；下表列出全部**最少改动数且改判**的组合。",
        "独立程序已从原始推文和关系事件重建全部 115 个最小组合，最大概率差为 0。",
        "这里的证据是指定操作下的冻结模型响应；没有把指针重定向当作已知正确的现实纠错。",
        "",
    ]
    for seed, row in selected:
        target = row["target_account_id"]
        model = next(item for item in groups["runs"] if item["dataset"] == "fox8"
                     and item["seed"] == seed)
        pointers = row["raw_pointer_locators_in_mask_order"]
        lines += [
            f"## seed {seed} · 目标账号 `{target}`",
            "",
            f"冻结模型 SHA256：`{model['source_model_sha256']}`。",
            f"原始 Bot 概率 **{row['probability_before']:.6f}**；"
            f"相关原始指针 **{row['raw_pointer_count']}** 条；"
            f"最少需重定向 **{row['minimum_raw_pointers_changing_class']}** 条；"
            f"同尺寸候选 **{row['same_size_candidate_coalitions']}** 组，其中"
            f"改判 **{row['minimum_size_class_changing_coalitions']}** 组、"
            f"不改判 **{row['same_size_nonchanging_coalitions']}** 组。",
            "",
            "| 指针 | 来源账号 | 原始行 / 推文序号 | 类型 | 原目标账号 | 目标的邻居账号 |",
            "|---|---|---:|---|---|---|",
        ]
        for index, p in enumerate(pointers, start=1):
            peer = (p["old_target_account_id"] if p["source_account_id"] == target
                    else p["source_account_id"])
            lines.append(
                f"| P{index} | `{p['source_account_id']}` | "
                f"{p['source_line']} / {p['tweet_index']} | {p['relation']} | "
                f"`{p['old_target_account_id']}` | `{peer}` |")
        lines += [
            "",
            "| 最小组合 | 重定向的原始指针 | 失去支持的目标邻居 | 改后 Bot 概率 | 改后类别 |",
            "|---:|---|---|---:|---|",
        ]
        for index, (mask, probability) in enumerate(zip(
                row["all_minimal_bitmasks"], row["all_minimal_probabilities"],
                strict=True), start=1):
            positions = [bit for bit in range(len(pointers)) if mask & (1 << bit)]
            labels = ", ".join(f"P{bit+1}" for bit in positions)
            peers = sorted({(pointers[bit]["old_target_account_id"]
                             if pointers[bit]["source_account_id"] == target
                             else pointers[bit]["source_account_id"])
                            for bit in positions})
            peer_names = ", ".join(f"`{peer}`" for peer in peers)
            category = "Bot" if probability >= row["threshold"] else "Human"
            lines.append(f"| {index} | {labels} | {peer_names} | "
                         f"{probability:.6f} | {category} |")
        control = row["closest_same_size_nonchanging_bitmask"]
        if control is None:
            raise ValueError("Readable alternative case lacks equal-size nonflip control")
        first_mask = row["all_minimal_bitmasks"][0]
        control_labels = ", ".join(f"P{bit+1}" for bit in range(len(pointers))
                                   if control & (1 << bit))
        swapped_out = ", ".join(f"P{bit+1}" for bit in range(len(pointers))
                                if first_mask & (1 << bit) and not control & (1 << bit))
        swapped_in = ", ".join(f"P{bit+1}" for bit in range(len(pointers))
                               if control & (1 << bit) and not first_mask & (1 << bit))
        lines += [
            "",
            "**等量来源对照。** 从表中第 1 组改判组合出发，"
            f"只把 {swapped_out} 换为 {swapped_in}，仍重定向"
            f" **{row['minimum_raw_pointers_changing_class']}** 条指针。"
            f"对照组合为 {control_labels}；冻结 Bot 概率为"
            f" **{row['closest_same_size_nonchanging_probability']:.6f}**，"
            "类别仍为 Bot。这个对照只匹配改动条数，"
            "不自动匹配指针的类型、位置或邻居属性。",
            "",
            "**读法。** 上述各组分别独立执行，不是按行连续删改。",
            "全部最小组合的原始指针交集为**空**，因此没有一条指针在"
            "每一种最小改判方案里都不可替代。",
            "每组触及的邻居对，其全部原始支持指针都已被重定向；"
            "并非只删掉一条支持指针就等于删掉该图边。",
            "",
        ]
    lines += [
        "## 对分类依据的准确表述",
        "",
        "这两个例子确实把原始来源、入模图事实和同一冻结分类器的类别阈值连成可执行链条。",
        "它们也说明类别判断可以有多条同样短的替代来源路径。",
        "因此判断卡展示的是**在明确定义的操作族中，哪些来源组合足以改变这次分类**；",
        "不能把任一单组呈现为唯一或现实中的行为因果原因。",
        "完整 69 次判断与全部 115 组最小组合存于",
        "`research/certificates/fox8_minimal_atomic_pointer_alternatives.json`；",
        "独立原始事件读回存于",
        "`research/certificates/fox8_minimal_atomic_pointer_alternatives_readback.json`。",
        "",
    ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()

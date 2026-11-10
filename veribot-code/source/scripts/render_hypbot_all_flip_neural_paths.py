"""Render all 25 frozen HypBot source-flip neural paths as a local Markdown audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "research/certificates/hypbot_fox8_all_flip_neural_paths.json"
SOURCE = ROOT / "research/certificates/hypbot_fox8_account_relation_source_response.json"
OUT = ROOT / "research/VeriBot_Fox8_HypBot_25次改判内部路径_2026-11-05.md"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    report = json.loads(CERT.read_text(encoding="utf-8"))
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = report["rows"]
    if (report["source_certificate_sha256"] != digest(SOURCE) or
            report["targets"] != 25 or len(rows) != 25 or
            report["local_only_class_flips"] != 25 or
            report["global_scale_only_class_flips"] != 0 or
            source["test_accounts_with_incident_source_pairs"] != 220):
        raise ValueError("Frozen GNN path population differs")
    lines = [
        "# Fox8 原生图 HypBot：25 次来源改判的内部路径",
        "",
        "## 为什么核到网络内部",
        "",
        "逐条来源定位和最终类别改变说明某次规定操作影响了冻结模型，"
        "但最终分数本身还不能区分目标节点的图消息传递与网络的全局共享尺度。",
        "这份表针对**同一个** Fox8 seed0 原生交互图 HypBot 变体，"
        "把两种计算路径拆开，不把结果转移给官方 kNN 图配置或其他图神经网络。",
        "",
        "## 判断人口与操作",
        "",
        "固定测试有 343 个账号，其中 220 个有入模原生关系。"
        "对每个目标账号，来源操作清空其全部原生邻居对所依赖的回复或转发目标 ID 指针；"
        "推文文本和 31 项节点特征保持原值，关系事件去重并重建对称图。"
        "先前完整来源证书显示 25 个目标账号跨过分类阈值。"
        "本表**仅纳入这 25 个已知改判账号**，所以 25/25 是该选择人口的机制检验，"
        "不是所有测试账号的改判率。模型权重、输入特征、图和来源索引均绑定哈希。",
        "",
        "每个账号都使用该冻结模型自身的分类头计算四种状态："
        "原图、只换全图半径尺度、只换目标局部图表征、两者都换。"
        "中间两种是内部诊断组合，不代表另有两种独立可实行的原始来源编辑。"
        "目标节点原始输入和节点通道保持相同。与直接模型前向和已保存来源响应的"
        f"最大 logit 差为 **{report['maximum_manual_to_saved_logit_difference']:.3g}**。",
        "",
        "## 逐账号结果",
        "",
        "表内概率均为 Bot 概率；箭头左侧是原图，右侧是清除该目标全部来源关系后的模型。"
        "“只换局部”固定原图的全体半径标准差；“尺度单独 Δlogit”固定原目标局部表征。",
        "",
        "| 目标账号 | 原始指针 / 关系对 | 概率：原图→全改 | 只换局部概率 | 尺度单独 Δlogit | 一层表征 ΔL2 | 二层表征 ΔL2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        states = row["states"]
        lines.append(
            f"| `{row['account_id']}` | {row['source_pointers_cleared']} / "
            f"{row['native_pairs_cleared']} | "
            f"{states['original']['probability']:.6f}→{states['both']['probability']:.6f} | "
            f"{states['local_only']['probability']:.6f} | "
            f"{row['global_scale_only_logit_change']:+.2e} | "
            f"{row['target_sage1_embedding_l2_change']:.6f} | "
            f"{row['target_sage2_embedding_l2_change']:.6f} |")
    local_abs = [abs(row["local_only_logit_change"]) for row in rows]
    scale_abs = [abs(row["global_scale_only_logit_change"]) for row in rows]
    lines += [
        "",
        "## 论断与证据",
        "",
        f"**局部图路径实际参与这 25 次目标改判。** "
        f"两层目标表征均变化的账号为 "
        f"**{report['all_targets_with_both_sage_layers_changed']}/25**；"
        f"固定原图全局尺度后仍改判 **{report['local_only_class_flips']}/25**。"
        f"局部路径带来的 logit 变化绝对值范围为 "
        f"**{min(local_abs):.6f}–{max(local_abs):.6f}**。",
        "",
        f"**全局尺度单独不足以造成这些目标改判。** "
        f"只换全图半径标准差的改判为 "
        f"**{report['global_scale_only_class_flips']}/25**；"
        f"其目标 logit 变化绝对值最大 "
        f"**{max(scale_abs):.2e}**。这项结论仅针对目标账号；"
        "其他账号的全局尺度连带翻转另见已保存的全图证书。",
        "",
        "**这是一条数值判断链。** 原始指针改变入模图关系，"
        "目标的两层 GraphSAGE 表征和图通道读出发生变化，"
        "冻结分类头因此在固定阈值两侧给出不同类别。"
        "这里没有独立正确来源值，不能把清空指针称为现实纠错；"
        "图通道坐标也没有经外部审核确认的社会行为含义。",
        "",
        "完整的每账号四状态 logit、概率、类别、层表征变化和模型哈希保存在"
        " `research/certificates/hypbot_fox8_all_flip_neural_paths.json`。"
        "原始来源操作及全部 343 名测试账号的响应保存在"
        " `research/certificates/hypbot_fox8_account_relation_source_response.json`。",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

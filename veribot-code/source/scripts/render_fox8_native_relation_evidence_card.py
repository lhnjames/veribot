"""Render a source-grounded human evidence card from the GNN certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CERT = ROOT / "research/certificates/fox8_two_native_gnn_relation_evidence.json"


def change_norm(report: dict, stage: str) -> float:
    old = np.asarray(report["before"]["stages"][stage]["values"])
    new = np.asarray(report["after"]["stages"][stage]["values"])
    return float(np.linalg.norm(new - old))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", nargs="?", type=Path, default=DEFAULT_CERT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cert = json.loads(args.certificate.read_text(encoding="utf-8"))
    output = args.output or (ROOT / "research/evidence_cards" /
                             f"fox8_{cert['target_account_id']}_two_native_gnns.md")
    reports = cert["model_reports"]
    candidate_ids = reports[0]["source_ids"]
    control_ids = reports[0]["controls"][0]["source_ids"]
    if len(candidate_ids) != len(control_ids) or not candidate_ids:
        raise ValueError("source/control budget changed")
    rows = []
    for report in reports:
        if report["model_adapter"] == "HypBot-native-graph":
            stages = ("sage1", "sage2")
        elif report["model_adapter"] == "GatedRelNet-native-relations":
            stages = ("relation_layer_1", "relation_layer_2")
        else:
            raise ValueError("unexpected graph model")
        rows.append(
            f"| {report['model_adapter']} | "
            f"{report['probability_before']:.6f} → {report['probability_after']:.6f} | "
            f"{report['class_before']} → {report['class_after']} | "
            f"{report['delta_logit']:+.6f} | "
            f"{report['controls'][0]['delta_logit']:+.6f} | "
            f"{change_norm(report, stages[0]):.6f} / {change_norm(report, stages[1]):.6f} |"
        )
    pointers = [
        f"{i}. `{cert['source_records'][sid]['raw_location'].split('#', 1)[1]}` "
        f"(tweet `{cert['source_records'][sid]['payload']['tweet_id']}`)."
        for i, sid in enumerate(candidate_ids, 1)
    ]
    owner_set = {cert["source_records"][sid]["account_id"] for sid in candidate_ids}
    ownership = ("该目标账号发表的" if owner_set == {cert["target_account_id"]}
                 else "邻居账号发表的" if cert["target_account_id"] not in owner_set
                 else "目标与邻居账号发表的")
    n = len(candidate_ids)
    control_position = ("同样与目标账号相邻" if cert["control_scope"] == "target_incident"
                        else "不与目标账号相邻")
    control_limit = ("两组具有相同来源数量且都触及目标，但关系类型、邻居与其他局部结构仍可能不同。"
                     if cert["control_scope"] == "target_incident"
                     else "对照只匹配来源数量；它不能代表同等拓扑位置的候选关系。")
    text = f"""# Fox8 账号 {cert['target_account_id']}：两种原生图网络的来源证据卡

## 来源事实

原始文件中，{ownership} {n} 条推文含有指向另一账号的回复或转发目标指针。这些指针共同支撑 `{cert['candidate_pair'][0]}`—`{cert['candidate_pair'][1]}` 之间的原生关系。系统逐条读取以下位置并核对推文 ID、发布账号和目标账号；清除这一组指针后保留推文内容及 31 项节点事实，并按原处理规则重新构造关系图。

{chr(10).join(pointers)}

## 模型怎样使用这条关系

两个冻结分类器均读取同一组具名节点事实和由原始互动构建的关系图。类别 `0` 表示 Human，`1` 表示 Bot。表中的干预是“清除互动目标指针”这一确定操作，并非已知正确的现实资料修正。

| 图模型 | Bot 概率：原始 → 清除 | 类别 | 目标 logit 变化 | {n} 条对照指针变化 | 两层目标表征 L2 变化 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

对照组也清除 {n} 条原始关系指针，来自{control_position}的 `{cert['control_pair'][0]}`—`{cert['control_pair'][1]}` 关系。{control_limit}

## 可核验范围

- 机器证书：`{args.certificate.resolve()}`。
- 独立核验命令：`D:\\miniconda\\python.exe source/scripts/verify_fox8_two_native_gnn_relation_certificate.py {args.certificate.resolve()}`。程序重新读取 {2*n} 条原始指针、重建原始/删除/对照三种图状态，并直接运行两个冻结模型。
- 证书展示两种图网络对这一来源操作的数值响应；上表分别给出分数变化和类别变化。它没有为互动行为的语义提供独立真值，也没有把这一关系断言为唯一判断原因。
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

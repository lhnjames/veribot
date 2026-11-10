"""Render checked VeriBot source-response certificates as readable local cards."""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOX8_CERTIFICATES = (
    ROOT / "research/certificates/aaai_veribot_fox8_native_pointer.json",
    ROOT / "research/certificates/aaai_veribot_fox8_profile_fact.json",
    ROOT / "research/certificates/aaai_veribot_fox8_profile_class_flip.json",
    ROOT / "research/certificates/aaai_veribot_fox8_profile_alternative_flip.json",
    ROOT / "research/certificates/aaai_veribot_fox8_tweet_behavior.json",
    ROOT / "research/certificates/aaai_veribot_fox8_neighbor_context.json",
)
BOTSIM_CERTIFICATES = (
    ROOT / "research/certificates/aaai_veribot_botsim_membership.json",
)
FOX8_OUTPUT = ROOT / "research/VeriBot_Fox8_HGT_RelNet_可核验来源卡.md"
BOTSIM_OUTPUT = ROOT / "research/VeriBot_BotSim_HGT_RelNet_可核验来源卡.md"


def probability(logit: float) -> float:
    return 1 / (1 + math.exp(-logit))


def stage_text(changes: dict[str, float]) -> str:
    return "、".join(f"{name}={value:.6f}" for name, value in changes.items())


def card(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset = "botsim" if "botsim" in path.name else "fox8"
    policy_path = ROOT / f"output/aaai_veribot_{dataset}/seed0/decision_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    audit_path = ROOT / "research/certificates/aaai_veribot_dev_threshold_audit.json"
    if (policy["schema_version"] != "veribot-mean-logit-dev-threshold-v1" or
            policy["dataset"] != dataset or
            policy["checkpoint_sha256"] != payload["checkpoint_sha256"] or
            policy["threshold_selection_audit_sha256"] !=
            sha256(audit_path.read_bytes()).hexdigest()):
        raise ValueError(f"decision policy does not match evidence certificate: {path}")
    threshold = float(policy["logit_threshold"])
    checked = path.with_name(path.stem + "_verification.json")
    verification = json.loads(checked.read_text(encoding="utf-8"))
    if not verification["status"].startswith("verified_") or \
            verification["max_output_difference"] > 1e-5 or \
            verification.get("certificate_sha256") != sha256(path.read_bytes()).hexdigest():
        raise ValueError(f"certificate lacks valid independent readback: {path}")
    cert = payload["certificate"]
    candidate_id = cert["claim"]["source_ids"][0]
    control_id = cert["controls"][0]["source_ids"][0]
    records = payload.get("selected_source_records", payload.get("source_records"))
    candidate = records[candidate_id]
    control = records[control_id]
    before = cert["before"]
    after = cert["after"]
    before_class = "Bot" if before["fused_logit"] >= threshold else "Human"
    after_class = "Bot" if after["fused_logit"] >= threshold else "Human"
    fact_name = "starts_rt" if candidate["kind"] == "raw_tweet" else next(iter(candidate["payload"]))
    source_fact = f"{fact_name}={cert['claim']['recomputed_value']}"
    operation_unit = (
        "一条原始目标账号指针" if candidate["kind"].endswith("_target_pointer")
        else "一项原始用户—subreddit 归属" if candidate["kind"] == "subreddit_membership"
        else "一条原始帖子" if candidate["kind"] == "raw_tweet"
        else "一个原始资料字段")
    heading = (f"邻居资料字段：目标账号 {cert['account_id']}，来源账号 {candidate['account_id']}"
               if "bridge_records" in payload else
               f"{candidate['kind']}：账号 {cert['account_id']}")
    lines = [
        f"## {heading}",
        "",
        f"**来源事实。** {source_fact}。原始定位：`{candidate['raw_location']}`；来源指纹：`{candidate['sha256']}`。",
        f"**入模路径。** {'；'.join(cert['input_dependencies'][candidate_id])}。基线输入指纹：`{before['model_input_sha256']}`；删除后的输入指纹：`{after['model_input_sha256']}`。",
        f"**判断响应。** 融合 Bot 概率从 {probability(before['fused_logit']):.6f} 变为 {probability(after['fused_logit']):.6f}；开发集选定阈值为 logit {threshold:.6f}（概率 {policy['probability_threshold']:.6f}），类别 {before_class} → {after_class}。HGT logit 变化 {cert['hgt_delta_logit']:+.6f}，RelNet logit 变化 {cert['relnet_delta_logit']:+.6f}，融合 logit 变化 {cert['fused_delta_logit']:+.6f}。",
        f"**模型路径。** HGT 各阶段表征 L2 变化：{stage_text(cert['hgt_stage_l2_change'])}。RelNet 各阶段表征 L2 变化：{stage_text(cert['relnet_stage_l2_change'])}。两分支前向回放最大误差分别为 {before['hgt_forward_replay_max_error']:.2e}、{before['relnet_forward_replay_max_error']:.2e}。",
        f"**同数量对照。** 来源 `{control_id}`（`{control['raw_location']}`）；融合 logit 变化 {cert['controls'][0]['fused_delta_logit']:+.6f}。两次操作各删除{operation_unit}。",
        f"**独立核验。** `{checked.name}` 从原始记录及冻结权重回放；输出最大差异 {verification['max_output_difference']:.2e}。HGT 权重指纹 `{cert['hgt_sha256']}`；RelNet 权重指纹 `{cert['relnet_sha256']}`。",
    ]
    if candidate["kind"] == "raw_tweet":
        facts = payload["operation_facts"]
        lines.append(
            f"**行为统计。** 原始帖子数 {facts['before']['n_tweets']:.0f}，"
            f"RT 文本比例 {facts['before']['retweet_ratio']:.6f}；删除候选帖子后"
            f"分别为 {facts['after']['n_tweets']:.0f} 与"
            f" {facts['after']['retweet_ratio']:.6f}。"
            "两条所选帖子均不构成模型中的原生关系边；删除一条帖子会联动改变卡片所列的多项帖子特征，"
            "上述 logit 响应对应整条帖子的来源操作。")
    if "bridge_records" in payload:
        bridge = payload["bridge_records"][candidate_id]
        control_bridge = payload["bridge_records"][control_id]
        lines.append(
            f"**图邻居路径。** 被编辑的资料属于邻居账号 `{candidate['account_id']}`，"
            f"目标账号为 `{cert['account_id']}`；两者间原生关系指针位于"
            f" `{bridge['raw_location']}`。对照邻居 `{control['account_id']}` 的"
            f"关系指针位于 `{control_bridge['raw_location']}`。"
            "本次只删除邻居资料字段，关系边保持不变。")
    alternatives_path = (ROOT / "research/certificates/"
                         "aaai_veribot_fox8_profile_one_field_alternatives.json")
    if alternatives_path.exists():
        alternatives = json.loads(alternatives_path.read_text(encoding="utf-8"))
        if alternatives["account_id"] == cert["account_id"]:
            baseline = float(alternatives["baseline"]["fused_logit"])
            current_flips = [row["field"] for row in alternatives["fields"]
                             if (baseline >= threshold) !=
                             (baseline + float(row["fused_delta_logit"]) >= threshold)]
            names = "、".join(current_flips) if current_flips else "无"
            lines.append(
                f"**可替代来源。** 已检查该账号存在的 {alternatives['evaluated_present_raw_fields']} 项原始资料字段；按当前分类阈值，单字段改判操作为：{names}。")
    lines += [
        "**结论范围。** 该来源对应实际模型输入，删除后产生上述冻结模型响应。行为含义及人工审核价值需要独立语义核对。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    for title, certificates, destination in (
        ("Fox8", FOX8_CERTIFICATES, FOX8_OUTPUT),
        ("BotSim", BOTSIM_CERTIFICATES, BOTSIM_OUTPUT),
    ):
        content = f"# VeriBot {title}：HGT 与 RelNet 可核验来源卡\n\n"
        content += "\n".join(card(path) for path in certificates)
        destination.write_text(content, encoding="utf-8")
        print(destination)


if __name__ == "__main__":
    main()

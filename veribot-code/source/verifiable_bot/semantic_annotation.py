"""Blind, source-addressed semantic evidence annotation packets."""

from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path
import zipfile

import ijson


SALT = "veribot-v3-semantic-pilot-v1:"
CONCEPTS = {
    "templated_text_in_shown_sample": {
        "question": "Do at least two shown messages repeat substantially the same wording or template, allowing small changes such as names, numbers, or URLs?",
        "yes": "Cite at least two message refs that share the template.",
        "no": "No such pair appears in the shown messages.",
        "uncertain": "The sample or truncation does not permit a judgment.",
        "applies_to": ["twibot20", "botsim24"],
    },
    "reply_addresses_parent": {
        "question": "Does the shown reply substantively address the content of its shown parent post or comment?",
        "yes": "The reply refers to a specific point, question, or topic in the parent.",
        "no": "The reply is generic, unrelated, or only a nonresponsive slogan.",
        "uncertain": "The available text is insufficient or truncated.",
        "applies_to": ["botsim24"],
    },
}


def digest(value: str) -> bytes:
    return hashlib.sha256((SALT + value).encode("utf-8")).digest()


def blind_id(dataset: str, concept: str, source_key: str) -> str:
    return hashlib.sha256((SALT + dataset + ":" + concept + ":" + source_key).encode()).hexdigest()[:20]


def select_by_hash(rows: list[dict], key: str, limit: int) -> list[dict]:
    """Use only a source identifier, never an account class or model output."""
    return sorted(rows, key=lambda row: digest(str(row[key])))[:limit]


def trim(text: object, maximum: int = 1800) -> tuple[str, bool]:
    value = str(text or "")
    return value[:maximum], len(value) > maximum


def twibot_packets(archive: Path, limit: int = 40,
                   messages_per_account: int = 12) -> tuple[list[dict], list[dict]]:
    eligible = []
    with zipfile.ZipFile(archive) as zipped, zipped.open("Twibot-20/test.json") as stream:
        for ordinal, row in enumerate(ijson.items(stream, "item")):
            tweets = row.get("tweet") or []
            if len(tweets) >= 5:
                # The label field is intentionally never read.
                eligible.append({"uid": str(row["ID"]), "ordinal": ordinal,
                                 "texts": tweets})
    selected = select_by_hash(eligible, "uid", limit)
    packet, provenance = [], []
    for row in selected:
        indices = sorted(range(len(row["texts"])), key=lambda i: digest(
            f"twibot20:{row['uid']}:tweet:{i}"))[:messages_per_account]
        items = []
        source = []
        for number, index in enumerate(indices, start=1):
            excerpt, truncated = trim(row["texts"][index])
            ref = f"T{number}"
            items.append({"ref": ref, "text": excerpt, "truncated": truncated})
            source.append({"ref": ref, "locator":
                           f"Twibot-20/test.json[{row['ordinal']}].tweet[{index}]"})
        identifier = blind_id("twibot20", "templated_text_in_shown_sample", row["uid"])
        packet.append({"blind_id": identifier, "dataset": "twibot20",
                       "concept": "templated_text_in_shown_sample",
                       "question": CONCEPTS["templated_text_in_shown_sample"]["question"],
                       "items": items})
        provenance.append({"blind_id": identifier, "dataset": "twibot20",
                           "account_id": row["uid"], "source_records": source})
    return packet, provenance


def _object_id(value: object) -> str:
    return str(value).split("_", 1)[-1]


def _content(item: dict, kind: str) -> str:
    value = item.get("posts") if kind == "posts" else item.get("comment_body")
    return str(value or "")


def botsim_packets(activity_path: Path, account_limit: int = 40,
                   reply_limit: int = 40,
                   messages_per_account: int = 12) -> tuple[list[dict], list[dict]]:
    activity = json.loads(activity_path.read_text(encoding="utf-8"))
    by_object = {}
    account_texts = []
    for uid, record in activity.items():
        texts = []
        for kind, identifier in (("posts", "submission_id"),
                                 ("comment_1", "comment_id"),
                                 ("comment_2", "comment_id")):
            for item in record.get(kind, []):
                oid = _object_id(item[identifier])
                content = _content(item, kind)
                if oid in by_object and by_object[oid]["uid"] != uid:
                    raise ValueError("BotSim object ID has conflicting owners")
                by_object[oid] = {"uid": uid, "kind": kind, "text": content,
                                  "object_id": str(item[identifier])}
                if content.strip():
                    texts.append({"kind": kind, "object_id": str(item[identifier]),
                                  "text": content})
        if len(texts) >= 5:
            account_texts.append({"uid": uid, "texts": texts})
    packet, provenance = [], []
    for row in select_by_hash(account_texts, "uid", account_limit):
        selected = sorted(row["texts"], key=lambda item: digest(
            f"botsim24:{row['uid']}:{item['kind']}:{item['object_id']}"))[:messages_per_account]
        items = []
        source = []
        for number, item in enumerate(selected, start=1):
            excerpt, truncated = trim(item["text"])
            ref = f"T{number}"
            items.append({"ref": ref, "text": excerpt, "truncated": truncated})
            source.append({"ref": ref, "locator":
                           f"user_post_comment.json[{row['uid']}].{item['kind']}[object_id={item['object_id']}]"})
        identifier = blind_id("botsim24", "templated_text_in_shown_sample", row["uid"])
        packet.append({"blind_id": identifier, "dataset": "botsim24",
                       "concept": "templated_text_in_shown_sample",
                       "question": CONCEPTS["templated_text_in_shown_sample"]["question"],
                       "items": items})
        provenance.append({"blind_id": identifier, "dataset": "botsim24",
                           "account_id": row["uid"], "source_records": source})
    reply_candidates = []
    for uid, record in activity.items():
        for kind in ("comment_1", "comment_2"):
            for item in record.get(kind, []):
                target = by_object.get(_object_id(item.get("parent_id") or ""))
                reply = _content(item, kind)
                if target is None or target["uid"] == uid or \
                        not target["text"].strip() or not reply.strip():
                    continue
                reply_candidates.append({"uid": uid, "kind": kind,
                                         "comment_id": str(item["comment_id"]),
                                         "parent_id": str(item["parent_id"]),
                                         "parent_uid": target["uid"],
                                         "parent_kind": target["kind"],
                                         "parent_object_id": target["object_id"],
                                         "parent_text": target["text"], "reply_text": reply,
                                         "key": f"{uid}:{kind}:{item['comment_id']}"})
    seen_accounts = set()
    for candidate in select_by_hash(reply_candidates, "key", len(reply_candidates)):
        if candidate["uid"] in seen_accounts:
            continue
        seen_accounts.add(candidate["uid"])
        identifier = blind_id("botsim24", "reply_addresses_parent", candidate["key"])
        parent, parent_cut = trim(candidate["parent_text"])
        reply, reply_cut = trim(candidate["reply_text"])
        packet.append({"blind_id": identifier, "dataset": "botsim24",
                       "concept": "reply_addresses_parent",
                       "question": CONCEPTS["reply_addresses_parent"]["question"],
                       "items": [{"ref": "PARENT", "text": parent, "truncated": parent_cut},
                                 {"ref": "REPLY", "text": reply, "truncated": reply_cut}]})
        provenance.append({"blind_id": identifier, "dataset": "botsim24",
                           "account_id": candidate["uid"],
                           "source_records": [
                               {"ref": "PARENT", "locator":
                                f"user_post_comment.json[{candidate['parent_uid']}].{candidate['parent_kind']}[object_id={candidate['parent_object_id']}]"},
                               {"ref": "REPLY", "locator":
                                f"user_post_comment.json[{candidate['uid']}].{candidate['kind']}[comment_id={candidate['comment_id']},parent_id={candidate['parent_id']}]"}]})
        if len(seen_accounts) == reply_limit:
            break
    if len(seen_accounts) < reply_limit:
        raise ValueError("too few distinct BotSim reply authors for requested packet")
    return packet, provenance


def sha256(path: Path) -> str:
    digest_value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest_value.update(block)
    return digest_value.hexdigest()


def save(packet: list[dict], provenance: list[dict],
         output_dir: Path, inputs: list[Path]) -> dict:
    if len(packet) != len(provenance) or \
            {row["blind_id"] for row in packet} != {row["blind_id"] for row in provenance}:
        raise ValueError("blind packet and provenance do not align")
    if len({row["blind_id"] for row in packet}) != len(packet):
        raise ValueError("duplicate blind ID")
    output_dir.mkdir(parents=True, exist_ok=True)
    rater_dir = output_dir / "rater_materials"
    private_dir = output_dir / "private"
    rater_dir.mkdir(exist_ok=True)
    private_dir.mkdir(exist_ok=True)
    packet_path = rater_dir / "blind_packet.jsonl"
    provenance_path = private_dir / "provenance.jsonl"
    for path, rows in ((packet_path, packet), (provenance_path, provenance)):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                        encoding="utf-8")
    templates = {}
    for rater in ("A", "B"):
        template_path = rater_dir / f"annotation_rater_{rater}.csv"
        with template_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "blind_id", "concept", "annotator_id", "judgment_yes_no_uncertain",
                "support_refs", "confidence_low_medium_high", "minutes", "notes"))
            writer.writeheader()
            for row in packet:
                writer.writerow({"blind_id": row["blind_id"], "concept": row["concept"],
                                 "annotator_id": rater})
        templates[rater] = sha256(template_path)
    guide_path = rater_dir / "README.md"
    guide_path.write_text(
        "# Blind evidence annotation pilot\n\n"
        "Two raters work independently. Read `blind_packet.jsonl` and fill only your own "
        "`annotation_rater_A.csv` or `annotation_rater_B.csv`. Do not inspect the parent "
        "directory's `private` folder. Account class, model score, and prediction direction "
        "are intentionally absent.\n\n"
        "For `templated_text_in_shown_sample`, judge only the displayed messages. Mark "
        "`yes` if at least two refs repeat substantially the same wording or template, "
        "allowing small slots such as names, numbers, or URLs; list both refs in "
        "`support_refs`. Mark `no` if no such pair is shown. Mark `uncertain` when the "
        "sample or truncation prevents a judgment. Do not infer Bot/Human identity.\n\n"
        "For `reply_addresses_parent`, mark `yes` if the reply responds to a specific "
        "point, question, or topic of the shown parent text; mark `no` for a generic "
        "or unrelated reply; mark `uncertain` if context is insufficient. Cite `PARENT` "
        "and `REPLY` in `support_refs` for yes/no.\n\n"
        "Use only `yes`, `no`, or `uncertain`; record confidence as `low`, `medium`, "
        "or `high`, and time spent in minutes. Raters must not see each other's answers "
        "before both files are complete. Disagreements require a third blind adjudicator "
        "or a recorded consensus discussion after independent completion. Missing or "
        "truncated source text can justify `uncertain`; never guess a class label.\n",
        encoding="utf-8")
    guide_zh_path = rater_dir / "README_zh.md"
    guide_zh_path.write_text(
        "# 语义证据盲标试包\n\n"
        "两位标注员分别独立填写 `annotation_rater_A.csv` 与 `annotation_rater_B.csv`。"
        "只查看本目录 `blind_packet.jsonl`；不要查看上一级目录的 `private`。"
        "材料不含 Bot/Human 标签、模型分数或预测方向。\n\n"
        "`templated_text_in_shown_sample`：只判断展示的消息。若至少两条消息明显复用同一措辞或模板，"
        "即使人名、数字或 URL 等小片段不同，也标 `yes`，并在 `support_refs` 写出至少两个编号，"
        "如 `T2,T7`。展示消息里找不到这种配对标 `no`；采样或截断使判断无法确定标 `uncertain`。"
        "不要推测账号是 Bot 还是 Human。\n\n"
        "`reply_addresses_parent`：若回复具体回应了父帖/父评论中的要点、问题或主题，标 `yes`；"
        "若只是泛泛表态、口号或无关内容，标 `no`；上下文不足则标 `uncertain`。"
        "`yes`/`no` 都在 `support_refs` 填 `PARENT,REPLY`。\n\n"
        "判断仅能填 `yes`、`no`、`uncertain`；信心填 `low`、`medium`、`high`，"
        "并记录用时（分钟）。两人完成前不要交换答案。分歧应由第三位不看类别标签的标注员裁决，"
        "或在独立填写后记录讨论结论。文本缺失或截断时可以标 `uncertain`，不要猜测。\n",
        encoding="utf-8")
    html_path = rater_dir / "blind_packet.html"
    cards = []
    for row in packet:
        items = "".join(
            "<div class='item'><strong>" + html.escape(item["ref"]) + "</strong>"
            + (" <em>(truncated)</em>" if item["truncated"] else "")
            + "<p>" + html.escape(item["text"]).replace("\n", "<br>") + "</p></div>"
            for item in row["items"])
        cards.append("<section><h2>" + html.escape(row["blind_id"]) + "</h2>"
                     "<p class='meta'>" + html.escape(row["dataset"]) + " · "
                     + html.escape(row["concept"]) + "</p><p class='question'>"
                     + html.escape(row["question"]) + "</p>" + items + "</section>")
    html_path.write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>Blind evidence packet</title>"
        "<style>body{font:16px/1.55 system-ui,sans-serif;max-width:900px;margin:2rem auto;"
        "padding:0 1rem;color:#17212b;background:#f6f8fa}section{background:white;border:1px solid #d8e0e7;"
        "border-radius:12px;padding:1.3rem;margin:1.5rem 0}h1,h2{line-height:1.2}h2{font-size:1rem;"
        "overflow-wrap:anywhere}.meta{color:#536477}.question{font-weight:650}.item{border-left:3px solid #3975a5;"
        "padding:.4rem .8rem;margin:.8rem 0;background:#f7fbfe}.item p{white-space:normal;overflow-wrap:anywhere}"
        "em{color:#9a5a0f}a{color:#175f93}</style><main><h1>Blind semantic evidence packet</h1>"
        "<p>Read <a href='README_zh.md'>中文规则</a> or <a href='README.md'>English instructions</a>. "
        "Record judgments in your own CSV. No account labels or model scores appear here.</p>"
        + "".join(cards) + "</main></html>", encoding="utf-8")
    rater_zip = output_dir / "rater_materials_only.zip"
    with zipfile.ZipFile(rater_zip, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(rater_dir.iterdir()):
            if path.is_file():
                zipped.write(path, arcname=path.name)
    manifest = {"purpose": "independent evidence-level annotation pilot",
                "class_labels_in_packet": False, "model_scores_in_packet": False,
                "selection_uses_class_labels": False,
                "selection_salt": SALT,
                "concepts": CONCEPTS,
                "counts": {concept: sum(row["concept"] == concept for row in packet)
                           for concept in CONCEPTS},
                "source_sha256": {str(path.resolve()): sha256(path) for path in inputs},
                "blind_packet_sha256": sha256(packet_path),
                "private_provenance_sha256": sha256(provenance_path),
                "annotation_template_sha256": templates,
                "rater_guide_sha256": {"en": sha256(guide_path),
                                        "zh": sha256(guide_zh_path)},
                "blind_html_sha256": sha256(html_path),
                "rater_materials_only_zip_sha256": sha256(rater_zip)}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                               encoding="utf-8")
    return manifest

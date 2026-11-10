import json

from .schema import AccountRecord


SYSTEM_PROMPT = """You detect inauthentic social-media accounts and provide one core,
auditable item of evidence. Do not use fixed numeric thresholds. A raw number alone is not
sufficient evidence. Prefer behavioral intent, repeated content, coordinated activity,
identity-behavior inconsistency, or another core pattern supported by cited anchors.
Commercial activity, news sharing, and fandom coordination can be genuine: state the most
plausible alternative explanation. Never invent an anchor. If the visible record does not
support a core claim, return evidence_status=insufficient and an empty anchors list.
Return exactly one JSON object and no prose."""


def render_record(record: AccountRecord, include_label: bool = False) -> str:
    payload = record.model_dump(exclude={"label"})
    if include_label:
        payload["training_target"] = record.label
    return "ACCOUNT_RECORD\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def teacher_discovery_prompt(record: AccountRecord) -> str:
    return render_record(record, include_label=False) + "\nDiscover the single most central auditable behavior, without guessing the dataset label."


def inference_messages(record: AccountRecord) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": render_record(record)}]

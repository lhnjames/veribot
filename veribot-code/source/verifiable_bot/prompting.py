from __future__ import annotations

import json

from .schema import AccountRecord
from .units import build_units, render_units

# Words a claim's "pattern" field must never contain: this is the same anti-leakage boundary as
# the claim text itself (see COLD_START_SYSTEM_PROMPT), but pattern is inherently at higher risk of
# smuggling a conclusion since its whole purpose is to say *why* a fact matters. "pattern" may
# describe a behavioral shape ("rapid bulk posting", "long-standing organic engagement") but must
# stop short of asserting the label. format_check.py enforces this the same way units.py's
# BANNED_WORDS guards OBS units against judgmental framing.
PATTERN_BANNED_WORDS = ("bot", "human", "automated", "automation", "fake", "spam", "malicious",
                        "suspicious", "inauthentic", "genuine account", "real person")

SYSTEM_PROMPT = """You are a social-media account analyst. You will be shown an upstream HGT verdict and a numbered list of \
evidence units taken from one account's public record (profile fields, tweets, activity \
timestamps, follow relations, and derived observations). Your job is to decide whether the \
account evidence. Decide whether the evidence agrees with that verdict and output the exact evidence grounding your decision.

Rules:
- You may only cite unit ids that appear in the evidence list. Never invent a unit id.
- First choose the set of evidence unit ids you are relying on (evidence_ids).
- Then output agreement: "agree" to retain the upstream verdict or "disagree" to select the opposite label.
- Then write one or more short claims. Every claim must list the source_ids it is built from, \
and every source_id must already be in evidence_ids. Do not write a claim that cites nothing.
- Each claim also has a "pattern" field: one short phrase naming the general behavioral or \
profile PATTERN this fact is an instance of (e.g. "rapid bulk posting", "long-standing organic \
engagement", "generic under-specified profile"). The pattern phrase must NOT use the words bot, \
human, automated, fake, spam, suspicious, or any other word that names the label -- it names the \
pattern, not the conclusion. Whether that pattern implies bot or human is for the reader to judge.
- Claims and pattern text must not contain Arabic digits. Model-visible numeric facts use fixed qualitative tiers.
- Finally output evidence_status: "sufficient" if the cited evidence alone would let an \
independent reviewer reach the label implied by agreement, otherwise "insufficient". If the record does \
not clearly support a decision, still choose agreement but be honest and mark \
evidence_status as "insufficient" rather than inventing evidence.
- Output exactly one JSON object with keys evidence_ids, agreement, claims, evidence_status, \
and nothing else. No markdown, no prose outside the JSON."""

OUTPUT_SCHEMA_HINT = (
    '{"evidence_ids": ["OBS_01", "TWEET_02"], "agreement": "agree", '
    '"claims": [{"source_ids": ["OBS_01"], "claim": "...", "pattern": "..."}], '
    '"evidence_status": "sufficient"}'
)


def render_record(record: AccountRecord) -> str:
    if record.upstream_verdict not in {"bot", "human"}:
        raise ValueError(f"record {record.source}/{record.account_id} has no upstream HGT verdict")
    units = build_units(record)
    return (f"[VERDICT] An upstream classifier has flagged this account as: "
            f"{record.upstream_verdict}\n\nEVIDENCE_UNITS\n" + render_units(units))


def inference_messages(record: AccountRecord) -> list[dict[str, str]]:
    user = render_record(record) + "\n\nRespond with one JSON object shaped like:\n" + OUTPUT_SCHEMA_HINT
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


COLD_START_SYSTEM_PROMPT = """You write grounded evidence for a social-media account whose bot/human
label is already known from a trusted dataset. The prompt also gives an upstream HGT verdict. Your
job is to select evidence explaining the trusted label and set agreement to agree when the HGT
verdict equals it or disagree when the HGT verdict is wrong.

Being told the label in advance makes it easy to write a claim that *sounds* like a reason but
actually smuggles in your own conclusion instead of reporting what the evidence says. Do not do
that. A claim must be EXTRACTIVE: it states only what its cited evidence unit(s) literally show.
It must NOT append an inference tying that fact to bot/human-ness.
  BAD:  "The account is not verified, indicating bot behavior." (nothing in the evidence says
        non-verification indicates anything -- that link is your own invented conclusion)
  GOOD: "The account is not verified."
The claim is a fact lookup, not an argument. Whatever reasoning connects the fact to the label is
for the reader (or a later training stage) to judge -- your job here is only to point at true,
specific, checkable facts, not to argue for the label.

Each claim also has a "pattern" field: name the general behavioral or profile PATTERN this fact is
an instance of (e.g. "rapid bulk posting", "long-standing organic engagement", "generic
under-specified profile"). This is still not an argument for the label -- it must NOT contain the
words bot, human, automated, fake, spam, suspicious, or any other word that names the label or
conclusion. A pattern names a *shape of behavior*, not a verdict.
  BAD:  "Automated-looking posting frequency." (names the conclusion, not the pattern)
  GOOD: "Rapid, repetitive posting within a short time window."

Rules:
- You may only cite unit ids that appear in the evidence list. Never invent a unit id.
- First choose the set of evidence unit ids that best explains the given label (evidence_ids).
- Set "agreement" to exactly the required value stated after the evidence units.
- Then write one or more short, purely extractive claims (see above), each with its "pattern" \
field (see above). Every claim must list the source_ids it is built from, and every source_id \
must already be in evidence_ids. Do not write a claim that cites nothing.
- Never write Arabic digits in claim or pattern text. Unit ids may contain digits, but prose may not.
  BAD: "The account follows 4812 profiles."
  GOOD: "The account has a very high following count."
- Do not rely on a single generic, easily-reused signal (a lone follower count, a lone verified \
flag, a lone tweet) as your only evidence when you could instead cite multiple, mutually \
reinforcing, account-specific observations. If the account's evidence is genuinely sparse or \
generic and does not support multiple independent observations, that is a signal to mark \
evidence_status "insufficient" (see below), not to pad it with a single weak fact.
- Finally output evidence_status: "sufficient" only if the cited evidence, read on its own by an \
independent reviewer with no other context, is enough on its own to reach the given label -- \
otherwise "insufficient". Do not default to "sufficient"; most accounts should NOT have strong, \
multi-signal evidence, so insufficient should be common, not rare.
- For a trusted bot label, a sufficient answer must use at least two genuinely independent evidence
  categories whenever available. Prefer combinations such as a posting burst plus HGT-neighbor
  concentration, duplicated content plus a follow imbalance, or repetitive tweets plus graph facts.
  If GRAPH units are available and diagnostic, cite at least one. Do not pad with unrelated facts.

GOOD BOT COMBINATION EXAMPLE (the ids are illustrative):
{"evidence_ids":["OBS_01","GRAPH_01"],"agreement":"agree","claims":[
 {"source_ids":["OBS_01"],"claim":"The selected posts contain a high-intensity burst within a short interval.","pattern":"Concentrated posting burst"},
 {"source_ids":["GRAPH_01"],"claim":"A majority of the most attended neighbors are predicted as bot.","pattern":"Concentrated upstream-neighbor verdicts"}],"evidence_status":"sufficient"}

- Output exactly one JSON object with keys evidence_ids, agreement, claims, evidence_status, \
and nothing else. No markdown, no prose outside the JSON."""


def cold_start_teacher_prompt(record: AccountRecord) -> list[dict[str, str]]:
    """Prompt used to bootstrap SFT targets (spec section 5: "分类标签用数据集自带的真实标签" --
    the label is an input given to the teacher, not something it is asked to (re-)decide, so the
    dataset label is never used as an acceptance filter here; only format validity is checked by
    the caller. This is only used to seed cold-start SFT, never as a GRPO reward signal (label
    correctness in GRPO always comes straight from the dataset, via rewards.py)."""
    if record.upstream_verdict not in {"bot", "human"} or record.label not in {"bot", "human"}:
        raise ValueError("cold-start records require both trusted label and upstream verdict")
    required = "agree" if record.upstream_verdict == record.label else "disagree"
    label_directive = (
        f'\n\nGIVEN TRUSTED LABEL (fixed, not to be reconsidered): "{record.label}". '
        f'The required agreement is "{required}" because the upstream verdict is '
        f'"{record.upstream_verdict}". Set "agreement" to exactly "{required}". Select evidence '
        "and write claims that explain the trusted label; do not restate the label itself as a claim."
    )
    user = render_record(record) + label_directive + "\n\nRespond with one JSON object shaped like:\n" + OUTPUT_SCHEMA_HINT
    return [{"role": "system", "content": COLD_START_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def render_record_json_for_cache(record: AccountRecord) -> str:
    return json.dumps(record.model_dump(), ensure_ascii=False, sort_keys=True)

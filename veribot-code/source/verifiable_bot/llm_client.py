"""DeepSeek-backed LLM judge used only to *compute training rewards*.

Every call here is read-only scoring: it never produces gradients and never trains any weights.
All prompts, all model I/O, and all cached content are English-only, per project policy.

Design spec section 6.3: the judge is never asked for a 0-100% probability (verbal confidence
from LLMs is poorly calibrated, especially after any reasoning). Instead every judgment is a
forced-choice APPROVE/REJECT verdict, asked independently K times, and the reward signal is the
*empirical vote fraction* -- a frequency over discrete repeated judgments, not a single verbal
number. Results are disk-cached by (model, system, user, temperature, repeat index) so repeated
encounters (across GRPO steps/epochs, or K-repeats revisiting an already-explored branch of the
length binary search) cost no extra API calls.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

JUDGE_SYSTEM = (
    "You are a strict, literal-minded evidence auditor. You only judge what is written in "
    "front of you; you never use outside knowledge about the account. Some sources describe a "
    "value using a fixed qualitative tier word instead of a raw number (e.g. 'unusually low', "
    "'a mild imbalance', 'an extreme imbalance', 'a moderate burst', 'a high-intensity burst') -- "
    "these tier words are pre-computed offline from the full dataset's distribution and are "
    "authoritative, not an unverified claim the source itself must numerically prove; treat a "
    "stated tier as ground truth about where the underlying value falls, and judge a claim that "
    "restates it by whether it matches that tier's semantic strength, not by whether it also "
    "states a number. A raw number and its equivalent tier word are equally valid evidence for the "
    "same fact -- do not approve or reject a claim merely because it does, or does not, state a "
    "raw number. Always answer with exactly one word: APPROVE or REJECT. No other text."
)


def _extract_verdict(content: str | None) -> str | None:
    """Return one unambiguous forced-choice verdict, otherwise signal a malformed response."""
    if not content:
        return None
    verdicts = re.findall(r"\b(?:APPROVE|REJECT)\b", content.upper())
    if len(verdicts) != 1:
        return None
    return verdicts[0]


class DeepSeekJudge:
    def __init__(
        self,
        cache_dir: Path,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        k: int = 5,
        temperature: float = 0.7,
        max_workers: int = 16,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model or os.environ.get("JUDGE_MODEL", "deepseek-v4-flash")
        self.client = OpenAI(
            base_url=base_url or os.environ.get("JUDGE_BASE_URL", "https://api.deepseek.com"),
            api_key=api_key or os.environ.get("JUDGE_API_KEY"),
        )
        self.k = k
        self.temperature = temperature
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()

    # -- low level -----------------------------------------------------
    def _cache_path(self, system: str, user: str, repeat: int) -> Path:
        key = hashlib.sha256(f"{self.model}|{self.temperature}|{repeat}|{system}|{user}".encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _call_once(self, system: str, user: str, repeat: int) -> str:
        path = self._cache_path(system, user, repeat)
        if path.exists():
            try:
                cached_verdict = _extract_verdict(json.loads(path.read_text())["content"])
                if cached_verdict is not None:
                    return cached_verdict
            except (json.JSONDecodeError, KeyError, OSError):
                # A killed writer from an older run may have left a partial file. Treat it as a
                # cache miss; the atomic write below will repair it.
                pass
        last_problem = "unknown response error"
        for attempt in range(2):
            try:
                result = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=self.temperature,
                    max_tokens=20,
                    # deepseek-v4-flash is a reasoning model by default and will burn the whole
                    # max_tokens budget on hidden reasoning_content, leaving an empty answer.
                    extra_body={"thinking": {"type": "disabled"}},
                )
                content = result.choices[0].message.content or ""
                verdict = _extract_verdict(content)
                if verdict is None:
                    last_problem = f"malformed verdict: {content[:120]!r}"
                    continue
                with self._lock:
                    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
                    tmp.write_text(json.dumps({"content": verdict}, ensure_ascii=False))
                    os.replace(tmp, path)
                return verdict
            except Exception as exc:  # transient API/transport failures receive the same one retry
                last_problem = f"{type(exc).__name__}: {exc}"
                if attempt == 0:
                    continue
        # Do not cache a synthetic rejection: a future encounter should get another chance to ask.
        # The vote aggregator remains conservative and counts this invalid vote as non-approval.
        print(f"judge warning: failed to obtain a valid verdict after 2 attempts ({last_problem})", flush=True)
        return ""

    def _vote_fraction(self, system: str, user: str) -> float:
        futures = [self._pool.submit(self._call_once, system, user, r) for r in range(self.k)]
        approvals = 0
        for f in futures:
            verdict = _extract_verdict(f.result())
            # Conservative: a twice-invalid response is a non-approval, but it is not cached.
            if verdict == "APPROVE":
                approvals += 1
        return approvals / self.k

    # -- judgments used by rewards.py -----------------------------------
    CLAIM_APPROVAL_WORDING_EXAMPLES = (
        "Reference examples on wording alone -- in every pair below, the exact-number version and "
        "the tier-word version come from DIFFERENT sources describing the same underlying fact, and "
        "both are equally acceptable APPROVE-worthy claims when they match their own source; only "
        "reject for fabrication, exaggeration, or an irrelevant/unsupported pattern label, never for "
        "wording style alone:\n"
        "- Source A: 'followers_count: 3, following_count: 140' "
        "-> Claim: \"The account follows 140 accounts but has only 3 followers.\" -> APPROVE\n"
        "- Source B: 'Following-to-follower ratio: extreme imbalance.' "
        "-> Claim: \"The account's following-to-follower ratio is an extreme imbalance.\" -> "
        "APPROVE (tier word is authoritative; no number is needed or expected here)\n"
        "- Source A (same as above) -> Claim: \"The account follows 4,812 accounts.\" -> REJECT "
        "(this figure never appears in the source -- a specific-sounding number is not automatically "
        "trustworthy; check it is actually present)\n"
        "- Source B (same as above) -> Claim: \"The account's following-to-follower ratio is a mild "
        "imbalance.\" -> REJECT (contradicts the source's stated tier -- 'mild' is not 'extreme')\n"
    )

    def claim_approval(self, claim_text: str, cited_unit_texts: list[str], pattern_text: str = "") -> float:
        source = "\n".join(f"- {t}" for t in cited_unit_texts)
        pattern_block = (
            f"\n\nStated pattern (what behavioral/profile shape this fact is claimed to exemplify):"
            f"\n{pattern_text}"
        ) if pattern_text else ""
        pattern_question = (
            " AND, if a pattern is stated, is that pattern an accurate, non-generic characterization "
            "of what the source actually shows (not a mismatched or overly generic label for the fact)?"
        ) if pattern_text else ""
        user = (
            f"{self.CLAIM_APPROVAL_WORDING_EXAMPLES}\n"
            f"Source:\n{source}\n\nClaim:\n{claim_text}{pattern_block}\n\n"
            "Does this claim accurately and specifically reflect the source content, with no "
            "exaggeration and no fabrication, AND does it point to a concrete, checkable pattern "
            "that is genuinely relevant to deciding whether this account is automated (not an "
            f"irrelevant generic fact, and not an unsupported restatement of a conclusion){pattern_question}?\n\n"
            "Answer with exactly one word: APPROVE or REJECT."
        )
        return self._vote_fraction(JUDGE_SYSTEM, user)

    def consistency_vote(self, claims_text: str) -> float:
        """Cross-evidence consistency (reward redesign): do the account's OWN cited claims agree
        with each other, or does one contradict/undercut another? A model that cites mutually
        contradictory facts (or the same fact restated as if independent) is not exhibiting
        integrated understanding of the account, just concatenating whatever it found. Only
        meaningful for 2+ claims; callers should treat a single claim as trivially consistent
        without calling this (see rewards.py)."""
        user = (
            "Do the following observations, taken together about the SAME account, form a "
            "coherent, non-contradictory set -- i.e. none of them undercuts or contradicts another, "
            "and they are not just the same underlying fact restated as if independent?\n\n"
            f"Observations:\n{claims_text}\n\n"
            "Answer with exactly one word: APPROVE or REJECT."
        )
        return self._vote_fraction(JUDGE_SYSTEM, user)

    SUFFICIENCY_WORDING_NOTE = (
        "Reminder on wording: judge these observations on their substance, not their surface form. "
        "A fixed tier word (\"an extreme imbalance\", \"a moderate burst\", \"unusually low\") is a "
        "pre-computed, authoritative category from the full dataset's distribution, equivalent in "
        "evidentiary weight to an exact figure describing the same fact -- do not give a "
        "number-worded observation a pass and reject a tier-worded one (or vice versa) purely "
        "because of which form it uses.\n\n"
    )

    def sufficiency_vote(self, claims_text: str, prediction: str) -> float:
        user = (
            f"{self.SUFFICIENCY_WORDING_NOTE}"
            f'Based only on the following observations, do you approve labeling this account as '
            f'"{prediction}"? Only consider the observations given; do not introduce other '
            f"assumptions.\n\nObservations:\n{claims_text}\n\n"
            "Answer with exactly one word: APPROVE or REJECT."
        )
        return self._vote_fraction(JUDGE_SYSTEM, user)

    def label_approve_rate(self, units_text: str, prediction: str) -> float:
        other = "human" if prediction == "bot" else "bot"
        user = (
            f'Based only on the following account record, do you approve labeling this account as '
            f'"{prediction}"? Only consider the information given; do not introduce other '
            "assumptions. Approve only if the record contains specific, concrete information that "
            "would affirmatively lead a careful reviewer to this label. Before deciding, silently "
            f'check: could this exact record, unchanged, just as plausibly belong to a "{other}" '
            f'account? If yes -- if nothing here actually discriminates "{prediction}" from '
            f'"{other}" -- you must answer REJECT even though nothing contradicts "{prediction}". '
            "Only answer APPROVE if the record gives you a genuine, specific reason to prefer this "
            f"label over the alternative.\n\nAccount record:\n{units_text}\n\n"
            "Answer with exactly one word: APPROVE or REJECT."
        )
        return self._vote_fraction(JUDGE_SYSTEM, user)

"""THE canonical v4 model prompt. Single source of truth for how an account is turned into the exact
token sequence the policy sees — shared by the render builder, the token probe and E1/E2.

Why this module must be the only place a prompt is assembled: the token budget is only meaningful if
the thing we count is byte-identical to the thing the model is fed. If the render builder wrapped the
units one way, the token probe another, and the training collator a third, then "3200 tokens" would
be measured against a prompt no run actually uses, and a silent collator truncation would drop
TWEET/OBS/ACTIVITY units that the units list and evidence ids still claim are present. So every
caller goes through :func:`canonical_messages` / :func:`count_tokens` here, and nobody re-wraps.

Budget arithmetic: Qwen3-0.6B's context is 4096. We reserve ~896 tokens for the three-step
generation (``OUTPUT_BUDGET``) and cap the *rendered prompt* — system + user + the assistant
generation prompt, counted through ``apply_chat_template(add_generation_prompt=True)`` — at
``PROMPT_BUDGET = 3200``. :func:`fit_account_prompt` adaptively lowers the number of TWEET units
(evenly across the timeline) until the whole canonical prompt fits, and NEVER drops a structural
PROFILE/OBS/ACTIVITY/RELATION unit. We never rely on tokenizer truncation.

The sidecar (unit_metadata / clauses) is deliberately NOT part of any string built here: it exists
only for programmatic verification and must never reach the model.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from .render_v4 import Unit, build_units_v4, render_units

PROMPT_VERSION = "v4_prompt_1"   # bump whenever SYSTEM/USER/budget/template semantics change
PROMPT_BUDGET = 3200
OUTPUT_BUDGET = 896          # 3200 + 896 = 4096 context
CANDIDATE_MAX_TWEETS = 30    # fixed starting candidate cap; adapt DOWN per account

SYSTEM_PROMPT = (
    "You are a meticulous social-media forensics analyst. You decide whether a Twitter/X account "
    "is an automated bot or a human, using ONLY the numbered evidence units provided. Every unit "
    "has an id in square brackets such as [PROFILE_01]; cite ids to ground your reasoning and never "
    "invent facts or numbers that are not in the units.\n\n"
    "Answer in exactly this format:\n"
    "<think>\n"
    "Step 1 (observe): a single concrete observation, citing one unit id.\n"
    "Step 2 (connect): combine it with a different kind of unit, citing that id.\n"
    "Step 3 (conclude): state what the combination implies.\n"
    "</think>\n"
    "<evidence>ID | ID</evidence>\n"
    "<chain>keyword - keyword</chain>\n"
    "<answer>bot</answer> or <answer>human</answer>"
)

USER_TEMPLATE = (
    "Numbered evidence for one account (each line is one citable unit):\n\n"
    "{units}\n\n"
    "Classify this account as bot or human."
)


def default_model_path(config: str | Path = "configs/veriroute_lite_v3_0.6b.yaml") -> str:
    """Model path from the v4 config — one place, so the tokenizer used to COUNT tokens is the same
    one E1/E2 load to CONSUME them."""
    return yaml.safe_load(Path(config).read_text(encoding="utf-8"))["model_name"]


def canonical_messages(units_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(units=units_text)},
    ]


def canonical_prompt_text(units_text: str, tokenizer) -> str:
    """The exact string the model is fed, including the assistant generation prompt."""
    return tokenizer.apply_chat_template(
        canonical_messages(units_text), tokenize=False, add_generation_prompt=True
    )


def count_tokens(units_text: str, tokenizer) -> int:
    ids = tokenizer.apply_chat_template(
        canonical_messages(units_text), tokenize=True, add_generation_prompt=True
    )
    return len(ids)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_prompt_sha256(tokenizer) -> str:
    """Hash covering everything that determines the model-visible prompt EXCEPT the account content:
    version, system + user template, prompt/output budgets, and the tokenizer's chat template plus
    the generation-prompt flag. Any change to how prompts are wrapped changes this hash, so a stale
    pretokenised cache (whose key includes it) is invalidated automatically."""
    template = getattr(tokenizer, "chat_template", "") or ""
    payload = "\x00".join([
        PROMPT_VERSION, SYSTEM_PROMPT, USER_TEMPLATE,
        f"prompt_budget={PROMPT_BUDGET}", f"output_budget={OUTPUT_BUDGET}",
        f"candidate_max_tweets={CANDIDATE_MAX_TWEETS}",
        "add_generation_prompt=True", template,
    ])
    return _sha256_text(payload)


def tokenizer_fingerprint(tokenizer, model_path: str) -> dict[str, Any]:
    return {
        "model_path": model_path,
        "snapshot": Path(model_path).name,   # the HF snapshot hash is the last path component
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "tokenizer_config_sha256": _sha256_file(Path(model_path) / "tokenizer_config.json"),
        "tokenizer_json_sha256": _sha256_file(Path(model_path) / "tokenizer.json"),
    }


def n_tweet_units(units: list[Unit]) -> int:
    return sum(1 for u in units if u.unit_id.startswith("TWEET_"))


def fit_account_prompt(
    record: dict, thresholds: dict, tokenizer, *,
    budget: int = PROMPT_BUDGET, start_tweets: int = CANDIDATE_MAX_TWEETS,
) -> tuple[list[Unit], int, int]:
    """Adaptively lower the TWEET cap until the canonical prompt fits ``budget``.

    Structural units (PROFILE/OBS/ACTIVITY/RELATION) are never touched. Returns
    ``(units, token_count, effective_tweet_cap)`` where ``effective_tweet_cap`` is the max_tweets
    value that fit — equal to ``start_tweets`` when no budget-driven reduction was needed, lower
    when it was. Note this is the *cap*, not the realised tweet count: an account with only 4 tweets
    keeps cap 30 and simply renders 4 TWEET units. Raises if an account cannot fit even with zero
    tweets, surfacing a genuinely anomalous account instead of silently truncating it.
    """
    k = start_tweets
    while True:
        units = build_units_v4(record, thresholds, max_tweets=k)
        tokens = count_tokens(render_units(units), tokenizer)
        tweets = n_tweet_units(units)
        if tokens <= budget:
            return units, tokens, k
        if tweets == 0:
            raise ValueError(
                f"account {record.get('ID')!r} needs {tokens} tokens with zero tweets, over "
                f"budget {budget}; structural units alone exceed the budget — investigate."
            )
        # Too long: drop tweets. Jump straight into the achievable range, then step down by one.
        k = min(k, tweets) - 1

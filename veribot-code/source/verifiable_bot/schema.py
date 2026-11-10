from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Input side: one account record, later rendered into numbered evidence units.
# ---------------------------------------------------------------------------


class PostRecord(BaseModel):
    post_id: str
    text: str
    kind: Literal["original", "retweet", "reply", "unknown"] = "unknown"
    created_at: str | None = None  # ISO-8601 if available; ACTIVITY/OBS burst facts degrade gracefully without it


class GraphEdge(BaseModel):
    edge_id: str
    source: str
    target: str
    relation: str


class AccountRecord(BaseModel):
    account_id: str
    source: Literal["twibot20", "twibot22"]
    split: Literal["train", "validation", "test"]
    profile: dict[str, Any]
    posts: list[PostRecord] = Field(default_factory=list)
    graph_edges: list[GraphEdge] = Field(default_factory=list)
    graph_facts: list[str] = Field(default_factory=list)
    # HGT-attention-derived neighbor-aggregation facts (design doc section 1, GRAPH_xx units) --
    # deliberately a SEPARATE field from graph_facts above (which feeds FACT_xx: raw content-
    # duplication / connectivity facts with no model-attention involvement). Populated only for
    # accounts present in the clean HGT's sidecar (see verifiable_bot/data.py's
    # load_hgt_graph_sidecar); empty for every other account, which is the normal/expected case,
    # not an error -- units.py must render zero GRAPH_xx units when this is empty.
    hgt_neighbor_facts: list[str] = Field(default_factory=list)
    # Discrete output of the source-specific train-only HGT.  It is rendered as a natural-language
    # VERDICT line; the 0.6B model predicts whether it agrees rather than re-predicting the label
    # without seeing the upstream decision.
    upstream_verdict: Literal["bot", "human"] | None = None
    # Versioned bucket assignments are generated offline. Raw profile values remain available to
    # programmatic consistency checks but are never rendered into the model prompt.
    profile_buckets: dict[str, Literal["very low", "low", "moderate", "high", "very high"]] = Field(
        default_factory=dict
    )
    bucket_version: str | None = None
    label: Literal["bot", "human"] | None = None


# ---------------------------------------------------------------------------
# Output side: E -> y -> C -> s, produced in a single autoregressive pass.
# ---------------------------------------------------------------------------


class Claim(BaseModel):
    source_ids: list[str] = Field(default_factory=list, max_length=6)
    claim: str = ""
    # What behavioral/profile pattern this claim exemplifies (e.g. "rapid bulk posting"), not a
    # restatement of the label itself. See prompting.py's PATTERN_BANNED_WORDS: the model must
    # describe the pattern without asserting bot/human-ness, same anti-leakage principle as claim.
    pattern: str = ""


class EvidenceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_ids: list[str] = Field(default_factory=list, max_length=8)
    agreement: Literal["agree", "disagree"]
    claims: list[Claim] = Field(default_factory=list, max_length=8)
    evidence_status: Literal["sufficient", "insufficient"] = "insufficient"

    # Deliberately NO auto-dedup validator here. format_check.py's duplicate-reference check
    # (spec 6.3(一): "有无重复引用") needs to see evidence_ids exactly as the model emitted them --
    # a validator that silently deduplicates on construction would make that check unreachable
    # (every model gets built with already-unique ids, so "len != len(set(...))" could never fire).

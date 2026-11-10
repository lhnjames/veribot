from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class PostRecord(BaseModel):
    post_id: str
    text: str
    kind: Literal["original", "retweet", "reply", "unknown"] = "unknown"


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
    label: Literal["bot", "human"] | None = None

    def anchor_map(self) -> dict[str, Any]:
        anchors: dict[str, Any] = {f"profile:{k}": v for k, v in self.profile.items()}
        anchors.update({f"post:{p.post_id}": p.model_dump() for p in self.posts})
        anchors.update({f"edge:{e.edge_id}": e.model_dump() for e in self.graph_edges})
        for i, fact in enumerate(self.graph_facts):
            anchors[f"fact:f{i}"] = fact
        return anchors


class EvidencePrediction(BaseModel):
    label: Literal["bot", "human"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_status: Literal["verified", "insufficient", "rejected"] = "insufficient"
    core_claim: str = ""
    evidence_kind: str = ""
    anchors: list[str] = Field(default_factory=list, max_length=8)
    alternative_explanation: str = ""

    @field_validator("anchors")
    @classmethod
    def unique_anchors(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class VerificationResult(BaseModel):
    anchor_grounded: bool
    graph_consistent: bool
    semantically_supported: bool | None = None
    counterfactually_used: bool | None = None
    missing_anchors: list[str] = Field(default_factory=list)
    status: Literal["verified", "insufficient", "rejected"]
    details: dict[str, Any] = Field(default_factory=dict)

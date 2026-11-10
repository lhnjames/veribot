import json
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "verifiable_bot" / "evidence_narrator.py"
spec = importlib.util.spec_from_file_location("evidence_narrator_standalone", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
EvidenceItem = module.EvidenceItem
organize_evidence = module.organize_evidence


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps(self.payload)))])


def items():
    return [
        EvidenceItem("E1", "32 of 111 tweets are verbatim retweets.",
                     "count and divide", ("POSTS_1",)),
        EvidenceItem("E2", "The account participates in three subreddits.",
                     "count distinct subreddits", ("REL_1",)),
    ]


def test_api_can_order_but_cannot_rewrite_or_drop_claims():
    client = FakeClient({"order": ["E2", "E1"],
                         "connectors": ["", "Separately, "],
                         "fabricated_claim": "The account is controlled by a botnet."})
    result = organize_evidence("bot", items(), {"POSTS_1": {}, "REL_1": {}},
                               client=client, model="mock")
    assert result["ordered_item_ids"] == ["E2", "E1"]
    assert "botnet" not in result["report"]
    assert "32 of 111 tweets are verbatim retweets. [POSTS_1]" in result["report"]
    assert "The account participates in three subreddits. [REL_1]" in result["report"]
    assert result["prediction"] == "bot"


def test_invalid_api_order_and_missing_sources_are_rejected():
    bad = FakeClient({"order": ["E1"], "connectors": [""]})
    with pytest.raises(ValueError, match="every item exactly once"):
        organize_evidence("bot", items(), {"POSTS_1": {}, "REL_1": {}},
                          client=bad, model="mock")
    with pytest.raises(ValueError, match="Unresolved source"):
        organize_evidence("bot", items(), {"POSTS_1": {}}, client=bad, model="mock")


def test_fixed_order_needs_no_language_model():
    result = organize_evidence("human", items(), {"POSTS_1": {}, "REL_1": {}},
                               use_api=False)
    assert result["ordered_item_ids"] == ["E1", "E2"]
    assert result["prediction"] == "human"

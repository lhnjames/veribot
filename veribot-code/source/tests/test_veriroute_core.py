import json
import re
import threading
from types import SimpleNamespace

import pytest

from verifiable_bot.confidence_head import bucketize_accuracy, skip_vote
from verifiable_bot.format_check import check_format
from verifiable_bot.llm_client import DeepSeekJudge, _extract_verdict
from verifiable_bot.rewards import score_completion
from verifiable_bot.schema import AccountRecord, GraphEdge, PostRecord
from verifiable_bot.sft_targets import serialize_target_with_spans
from verifiable_bot.units import build_units, hidden_unit_metadata, unit_map


def make_record(label="bot", upstream="bot"):
    posts = [
        PostRecord(post_id=f"p{i}", text=f"Buy our amazing product now {i}!", kind="original",
                   created_at=f"2026-01-01T10:0{i}:00")
        for i in range(5)
    ]
    profile = {"description": "shop since 2020", "followers_count": 10.0,
               "following_count": 4000.0, "tweet_count": 500.0,
               "listed_count": 0.0, "verified": False}
    return AccountRecord(
        account_id="u1", source="twibot20", split="train", profile=profile,
        profile_buckets={"followers_count": "low", "following_count": "very high",
                         "tweet_count": "high", "listed_count": "very low"},
        bucket_version="test-v1", upstream_verdict=upstream, label=label, posts=posts,
        graph_edges=[GraphEdge(edge_id="e0", source="self", target="n0", relation="following")],
        hgt_neighbor_facts=[
            "Among the neighbors the upstream classifier attended to most strongly, majority are themselves predicted as bot."
        ],
    )


def valid_output(record, evidence_ids, claims, agreement="agree", status="sufficient"):
    return json.dumps({"evidence_ids": evidence_ids, "agreement": agreement,
                       "claims": claims, "evidence_status": status})


def test_all_program_generated_and_sanitized_unit_text_is_digit_free():
    units = build_units(make_record())
    assert units
    assert all(not re.search(r"\d", unit.text) for unit in units)
    texts = [unit.text for unit in units]
    assert any("extreme imbalance" in text for text in texts)
    assert any("high-intensity burst" not in text and "low-level burst" in text for text in texts)
    assert any("high pairwise textual similarity" in text for text in texts)


def test_profile_raw_values_are_hidden_but_bucket_metadata_is_exact():
    units = build_units(make_record())
    metadata = hidden_unit_metadata(units)
    follower = next(unit for unit in units if unit.text == "Follower count: low.")
    assert metadata[follower.unit_id]["raw_value"] == 10.0
    assert metadata[follower.unit_id]["bucket"] == "low"


def test_missing_profile_bucket_refuses_raw_number_exposure():
    record = make_record()
    record.profile_buckets.pop("followers_count")
    with pytest.raises(ValueError, match="refusing to expose"):
        build_units(record)


def test_graph_units_are_first_class_and_reject_digits():
    record = make_record()
    assert any(unit.unit_id == "GRAPH_01" for unit in build_units(record))
    record.hgt_neighbor_facts = ["3 neighbors are predicted bot"]
    with pytest.raises(ValueError, match="digit"):
        build_units(record)


def test_format_accepts_agreement_schema_and_rejects_legacy_prediction():
    record = make_record()
    units = build_units(record)
    graph = next(unit.unit_id for unit in units if unit.unit_id.startswith("GRAPH_"))
    raw = valid_output(record, [graph], [{"source_ids": [graph],
        "claim": "A majority of the most attended neighbors are predicted as bot.",
        "pattern": "Concentrated upstream-neighbor verdicts"}])
    assert check_format(raw, unit_map(units), hidden_unit_metadata(units)).ok
    legacy = json.loads(raw)
    legacy["prediction"] = "bot"
    assert not check_format(json.dumps(legacy), unit_map(units), hidden_unit_metadata(units)).ok


@pytest.mark.parametrize("mutator,reason", [
    (lambda obj: obj.update(evidence_ids=["GRAPH_99"]), "unknown"),
    (lambda obj: obj.update(evidence_ids=["GRAPH_01", "GRAPH_01"]), "duplicate"),
])
def test_format_rejects_bad_references(mutator, reason):
    units = build_units(make_record())
    obj = {"evidence_ids": ["GRAPH_01"], "agreement": "agree",
           "claims": [{"source_ids": ["GRAPH_01"], "claim": "A majority is predicted as bot.",
                       "pattern": "Neighbor verdict concentration"}],
           "evidence_status": "sufficient"}
    mutator(obj)
    checked = check_format(json.dumps(obj), unit_map(units), hidden_unit_metadata(units))
    assert not checked.ok and reason in checked.reason


def test_format_rejects_digits_in_claim_or_pattern():
    units = build_units(make_record())
    raw = valid_output(make_record(), ["GRAPH_01"], [{"source_ids": ["GRAPH_01"],
        "claim": "The top 3 neighbors are bots.", "pattern": "Top 3 concentration"}])
    checked = check_format(raw, unit_map(units), hidden_unit_metadata(units))
    assert not checked.ok and "Arabic digit" in checked.reason


def test_format_rejects_bucket_claim_inconsistent_with_hidden_raw_sidecar():
    record = make_record()
    units = build_units(record)
    follower = next(unit for unit in units if unit.text.startswith("Follower count"))
    bad = valid_output(record, [follower.unit_id], [{"source_ids": [follower.unit_id],
        "claim": "The follower count is very high.", "pattern": "Follower-count tier"}])
    checked = check_format(bad, unit_map(units), hidden_unit_metadata(units))
    assert not checked.ok and "hidden source metadata" in checked.reason


def test_format_accepts_matching_bucket_claim():
    record = make_record()
    units = build_units(record)
    follower = next(unit for unit in units if unit.text.startswith("Follower count"))
    good = valid_output(record, [follower.unit_id], [{"source_ids": [follower.unit_id],
        "claim": "The follower count is low.", "pattern": "Follower-count tier"}])
    assert check_format(good, unit_map(units), hidden_unit_metadata(units)).ok


def test_sft_target_serializes_and_weights_agreement():
    target = {"evidence_ids": ["OBS_01"], "agreement": "disagree",
              "claims": [{"source_ids": ["OBS_01"], "claim": "A burst is present.",
                          "pattern": "Posting burst"}], "evidence_status": "sufficient"}
    text, spans = serialize_target_with_spans(target, 2.0, 1.5, 0.5)
    assert json.loads(text) == target
    label_span = next(span for span in spans if span[2] == 2.0)
    assert text[label_span[0]:label_span[1]] == json.dumps("disagree")


class PerfectJudge:
    def claim_approval(self, *_args, **_kwargs): return 1.0
    def sufficiency_vote(self, *_args, **_kwargs): return 1.0
    def consistency_vote(self, *_args, **_kwargs): return 1.0
    def label_approve_rate(self, units_text, _prediction):
        return 1.0 if "extreme imbalance" in units_text or "most attended" in units_text else 0.0


class RejectingJudge(PerfectJudge):
    def sufficiency_vote(self, *_args, **_kwargs): return 0.0


def two_signal_completion(record, agreement="agree"):
    units = build_units(record)
    ratio = next(unit.unit_id for unit in units if "extreme imbalance" in unit.text)
    graph = next(unit.unit_id for unit in units if unit.unit_id.startswith("GRAPH_"))
    return valid_output(record, [ratio, graph], [
        {"source_ids": [ratio], "claim": "Following exceeds followers with an extreme imbalance.",
         "pattern": "Following imbalance"},
        {"source_ids": [graph], "claim": "A majority of the most attended neighbors are predicted as bot.",
         "pattern": "Neighbor verdict concentration"},
    ], agreement=agreement)


def test_invalid_format_is_hard_minus_two_without_judge_call():
    result = score_completion(make_record(), "not json", object())
    assert result.total == -2.0 and not result.format_ok


def test_agree_reward_does_not_change_when_only_hidden_true_label_changes():
    bot = make_record(label="bot", upstream="bot")
    human = make_record(label="human", upstream="bot")
    raw = two_signal_completion(bot, "agree")
    first = score_completion(bot, raw, PerfectJudge())
    second = score_completion(human, raw, PerfectJudge())
    assert first.total == pytest.approx(second.total)
    assert first.detail["branch"] == "eligible_multisignal"


def test_disagree_corrects_wrong_hgt_and_beats_disagreeing_with_correct_hgt():
    hgt_wrong = make_record(label="human", upstream="bot")
    hgt_right = make_record(label="bot", upstream="bot")
    raw = two_signal_completion(hgt_wrong, "disagree")
    correction = score_completion(hgt_wrong, raw, PerfectJudge())
    false_dissent = score_completion(hgt_right, raw, PerfectJudge())
    assert correction.total > false_dissent.total
    assert correction.detail["branch"] == "disagree_hgt_wrong"
    assert correction.detail["record_hgt_suspect"] is True
    assert false_dissent.detail["branch"] == "disagree_hgt_right"


def test_declared_sufficient_but_unsupported_gets_penalty():
    record = make_record()
    result = score_completion(record, two_signal_completion(record), RejectingJudge())
    assert result.total == pytest.approx(0.7)
    assert result.detail["branch"] == "sufficient_but_unsupported"


def test_honest_insufficient_audit_is_rewarded(monkeypatch):
    monkeypatch.setattr("verifiable_bot.rewards.random.random", lambda: 0.0)
    record = make_record()
    units = build_units(record)
    description = next(unit.unit_id for unit in units if unit.unit_id.startswith("PROFILE_"))
    raw = valid_output(record, [description], [{"source_ids": [description],
        "claim": "The description names a shop.", "pattern": "Short commercial description"}],
        status="insufficient")
    result = score_completion(record, raw, RejectingJudge())
    assert result.total == pytest.approx(1.3)
    assert result.detail["branch"] == "insufficient_audited_honest"


def test_confidence_bucket_and_skip_votes():
    assert bucketize_accuracy(0.0, 5) == 0
    assert bucketize_accuracy(0.5, 5) == 2
    assert bucketize_accuracy(1.0, 5) == 4
    assert skip_vote([0.9, 0.05, 0.02, 0.02, 0.01], 0.6, 0.3, 2, 5)
    assert not skip_vote([0.2] * 5, 0.6, 0.3, 2, 5)


def test_extract_verdict_requires_one_unambiguous_label():
    assert _extract_verdict("APPROVE") == "APPROVE"
    assert _extract_verdict("Verdict: reject.") == "REJECT"
    assert _extract_verdict("") is None
    assert _extract_verdict("APPROVE or REJECT") is None


def test_judge_retries_malformed_response_and_caches_only_valid(tmp_path):
    responses = iter(["", "APPROVE"])
    class FakeCompletions:
        calls = 0
        def create(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))])
    completions = FakeCompletions()
    judge = object.__new__(DeepSeekJudge)
    judge.cache_dir = tmp_path
    judge.model = "fake"
    judge.temperature = 0.7
    judge.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    judge._lock = threading.Lock()
    assert judge._call_once("system", "user", 0) == "APPROVE"
    assert completions.calls == 2
    assert json.loads(judge._cache_path("system", "user", 0).read_text()) == {"content": "APPROVE"}

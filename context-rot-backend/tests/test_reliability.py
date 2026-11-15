import asyncio
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.reliability import evaluation, health, instructions, judge, judge_client, nli, text
from app.reliability import score as reliability_score
from app.schemas.analysis import AnalysisInput, DetectorResult
from tests.test_backend import drain, send, session

FIXTURE = Path(__file__).parent / "fixtures" / "reliability_sessions.jsonl"
COURSE = [{"text": "A primary key uniquely identifies each row. A primary key cannot contain NULL values.", "source": "notes.pdf", "page": 4}]


def payload(answer, question="Explain primary keys.", history=None, constraints=None, context=None):
    return AnalysisInput(session_id=1, message_id=2, current_question=question, current_answer=answer,
                         conversation_history=history or [],
                         constraints=[{"type": "instruction", "content": c} for c in constraints or []],
                         retrieved_context=COURSE if context is None else context)


# --- text and instruction rules -------------------------------------------------------------

def test_social_turns_and_answer_cleaning():
    assert text.is_social_turn("Thanks!") and text.is_social_turn("ok thanks") and text.is_social_turn("Got it.")
    assert not text.is_social_turn("thanks, now explain foreign keys") and not text.is_social_turn("ok?? what is a key")
    cleaned = text.clean_answer("![fig](/images/1)\n*Figure: Keys (Page 2)*\nA key is unique. [notes.pdf, page 4]")
    assert cleaned == "A key is unique."
    assert text.claims("- A primary key is unique.\n- Is it null?\n- Short.") == ["A primary key is unique."]


@pytest.mark.parametrize("constraint,answer,violated", [
    ("At most 2 sentences.", "One. Two. Three.", True),
    ("Answer in no more than two sentences", "One here. Two here. [notes.pdf, page 1]", False),
    ("Keep it under 10 words", "one two three four five six seven eight nine ten eleven twelve", True),
    ("Use bullet points.", "Plain prose answer.", True),
    ("Use bullet points.", "- first point\n- second point", False),
    ("Give the steps as a numbered list", "1. Parse\n2. Chunk", False),
    ("Always include an example", "A key identifies rows.", True),
    ("Always include an example", "For example, roll_no identifies a student.", False),
])
def test_measurable_instructions(constraint, answer, violated):
    result = instructions.check_instructions([{"content": constraint}], answer)
    assert result.checked == 1 and bool(result.violations) is violated


def test_unmeasurable_instruction_is_left_for_the_judge():
    result = instructions.check_instructions([{"content": "Only give hints, never the full answer."}], "Anything.")
    assert result.checked == 0 and result.unchecked == ["Only give hints, never the full answer."]


# --- health score ---------------------------------------------------------------------------

def test_health_bounds_monotonicity_and_threshold():
    assert health.health_from_severities({}) == 100
    assert health.health_from_severities({k: 1 for k in evaluation.FAILURE_TYPES}) == 0
    assert health.health_from_severities({"contradiction": 0.9}) < health.health_from_severities({"contradiction": 0.5})
    for kind in evaluation.FAILURE_TYPES:  # one confident failure of any type triggers recovery
        assert health.health_from_severities({kind: 0.8}) < get_settings().recovery_threshold


def test_aggregate_maps_contract_fields():
    signals = health.Signals(supported_fraction=0.5)
    signals.add("contradiction", 0.9, claim="X is Y", evidence="X is not Y", turn=1)
    signals.add("topic_drift", 0.3)
    result = DetectorResult.model_validate(health.aggregate(signals, "judge"))
    assert result.contradiction_detected and not result.topic_drift and not result.memory_violation
    assert result.contradiction_score == 0.9 and result.scope_score == 0.5 and result.topic_score == 0.7
    assert "answer 1" in result.alert and "drifts" not in result.alert
    assert [f.type for f in result.flags] == ["contradiction", "topic_drift"]


# --- v1 judge -------------------------------------------------------------------------------

def use_judge(monkeypatch, verdict):
    calls = []

    async def fake(system, user):
        calls.append(user)
        return verdict(user) if callable(verdict) else verdict
    monkeypatch.setattr(judge_client, "complete_json", fake)
    return calls


def test_judge_maps_verdict_and_merges_rules(monkeypatch):
    calls = use_judge(monkeypatch, {
        "contradictions": [{"claim": "Keys may repeat.", "earlier_answer": 1, "evidence": "Keys are unique.", "severity": 0.9}],
        "course_conflicts": [], "instruction_violations": [{"instruction": "Only give hints", "evidence": "full answer given", "severity": 0.7}],
        "off_topic": False, "supported_fraction": 0.6})
    history = [{"role": "user", "content": "Explain keys."}, {"role": "assistant", "content": "Keys are unique."}]
    result = asyncio.run(judge.score(payload("Keys may repeat. They are columns. They matter.", history=history,
                                             constraints=["At most 2 sentences.", "Only give hints"])))
    assert "Only give hints" in calls[0] and "At most 2 sentences" not in calls[0].split("STUDENT INSTRUCTIONS TO JUDGE")[1].split("]")[0]
    types = {f["type"] for f in result["flags"]}
    assert types == {"contradiction", "ignored_instruction"}
    assert result["contradiction_detected"] and result["memory_violation"] and result["health_score"] < 65
    assert result["flags"][0]["turn"] == 1 and result["scope_score"] == 0.6


def test_judge_ignores_contradictions_without_history_and_rejects_bad_json(monkeypatch):
    use_judge(monkeypatch, {"contradictions": [{"claim": "x", "earlier_answer": 3, "severity": 1}]})
    result = asyncio.run(judge.score(payload("A primary key is unique.")))
    assert result["health_score"] == 100 and result["flags"] == []
    use_judge(monkeypatch, {"course_conflicts": [{"claim": "x", "severity": 7}]})
    with pytest.raises(ValueError):
        asyncio.run(judge.score(payload("A primary key is unique.")))


def test_judge_needs_a_real_provider(monkeypatch):
    monkeypatch.setattr(get_settings(), "judge_provider", "offline")
    with pytest.raises(judge_client.JudgeConfigurationError):
        judge_client.judge_target()
    assert judge_client._parse('```json\n{"off_topic": false}\n```') == {"off_topic": False}


# --- v2 NLI ---------------------------------------------------------------------------------

class FakeNLI:
    """Word-overlap embeddings; 'cannot' vs 'can' on the same subject reads as a contradiction."""
    def __init__(self):
        self.pairs = []

    def embed(self, texts):
        vocab = ["primary", "key", "null", "row", "unique", "photosynthesis", "plants", "light", "duplicate", "contain"]
        return [[float(w in t.lower()) for w in vocab] for t in texts]

    def nli(self, pairs):
        self.pairs += pairs
        out = []
        for premise, hypothesis in pairs:
            p, h = premise.lower(), hypothesis.lower()
            if ("cannot" in p) != ("cannot" in h) and "null" in p and "null" in h:
                out.append({"contradiction": 0.92, "entailment": 0.03, "neutral": 0.05})
            elif "unique" in p and "unique" in h:
                out.append({"contradiction": 0.02, "entailment": 0.9, "neutral": 0.08})
            else:
                out.append({"contradiction": 0.05, "entailment": 0.1, "neutral": 0.85})
        return out


@pytest.fixture
def fake_nli():
    backend = FakeNLI()
    nli.set_backend(backend)
    yield backend
    nli.set_backend(None)


def test_nli_flags_course_conflict_and_history_contradiction(fake_nli):
    history = [{"role": "user", "content": "Explain primary keys."},
               {"role": "assistant", "content": "A primary key cannot contain NULL values in any row."}]
    result = asyncio.run(nli.score(payload("A primary key can contain NULL values when data is missing.", history=history)))
    types = {f["type"] for f in result["flags"]}
    assert {"contradiction", "course_conflict"} <= types and result["health_score"] < 65
    assert next(f for f in result["flags"] if f["type"] == "contradiction")["turn"] == 1


def test_nli_clean_answer_and_drift(fake_nli):
    clean = asyncio.run(nli.score(payload("A primary key is unique for every row in the table.")))
    assert clean["flags"] == [] and clean["health_score"] == 100 and clean["scope_score"] == 1
    drift = asyncio.run(nli.score(payload("Photosynthesis converts light into chemical energy in plants.")))
    assert drift["topic_drift"] and drift["health_score"] < 65


def test_nli_bounds_history_window(fake_nli, monkeypatch):
    monkeypatch.setattr(get_settings(), "reliability_history_turns", 1)
    history = [{"role": "assistant", "content": "A primary key cannot contain NULL values in any row."},
               {"role": "assistant", "content": "Photosynthesis happens in plants using light energy."}]
    result = asyncio.run(nli.score(payload("A primary key can contain NULL values sometimes.", history=history, context=[])))
    assert not any(f["type"] == "contradiction" for f in result["flags"])


def test_dispatch_records_latency_and_rejects_unknown_detector():
    result = asyncio.run(reliability_score(payload("A primary key can contain NULL values."), "rules"))
    assert result["detector"] == "rules" and result["latency_ms"] >= 0 and result["contradiction_detected"]
    with pytest.raises(ValueError):
        asyncio.run(reliability_score(payload("x"), "magic"))


# --- worker / API integration ----------------------------------------------------------------

def test_social_turn_is_not_queued(client, account):
    from types import SimpleNamespace
    from app.database.database import SessionLocal
    from app.integrations.rag_client import RetrievedChunk
    from app.services.pipeline import persist_answer
    sid = session(client, account)
    chunk = RetrievedChunk(text="A primary key is unique.", source="notes.pdf", page=1, score=1.0)
    with SessionLocal() as db:
        for content, expected in (("thanks!", "skipped"), ("Explain primary keys.", "pending")):
            message = SimpleNamespace(content=content, image_data=None, demo_fault=None)
            reply = persist_answer(db, sid, message, "A primary key is unique.", [], [], [chunk], 1.0, in_scope=True)
            assert reply["assistant_message"].status == expected
            assert (reply["job_id"] is None) is (expected == "skipped")


def test_judge_detector_end_to_end_with_persisted_flags(client, account, monkeypatch):
    monkeypatch.setattr(get_settings(), "reliability_detector", "judge")

    def verdict(user):
        bad = "can contain NULL" in user.split("CURRENT ANSWER")[1]
        return {"course_conflicts": [{"claim": "A primary key can contain NULL values.", "evidence": "cannot contain NULL", "severity": 0.95}] if bad else [],
                "supported_fraction": 0.2 if bad else 1.0}
    use_judge(monkeypatch, verdict)
    sid = session(client, account)
    send(client, account, sid, "contradiction")
    drain()
    history = client.get(f"/sessions/{sid}/history", headers=account).json()
    assert history[1]["status"] == "superseded" and history[2]["status"] == "corrected"
    records = client.get(f"/sessions/{sid}/analysis", headers=account).json()
    assert records[0]["detector"] == "judge" and records[0]["flags"][0]["type"] == "course_conflict"
    assert records[0]["latency_ms"] is not None and records[-1]["health_score"] == 100
    events = client.get(f"/sessions/{sid}/events?after=0", headers=account).json()
    completed = next(e for e in events if e["kind"] == "analysis.completed")
    assert completed["payload"]["flags"][0]["type"] == "course_conflict"


# --- evaluation harness ----------------------------------------------------------------------

def fixture_rows():
    return asyncio.run(evaluation.run_detector(evaluation.load_sessions(FIXTURE), "rules"))


def test_harness_replays_sessions_and_skips_social_turns():
    rows = fixture_rows()
    assert len(rows) == 10 and sum(r["skipped"] for r in rows) == 1
    assert {r["label"] for r in rows if r["health"] < 65} == {"course_conflict", "ignored_instruction", "topic_drift"}


def test_metrics_and_bootstrap_are_deterministic():
    rows = fixture_rows()
    m = evaluation.prf(rows, 65)
    assert (m["tp"], m["fp"], m["fn"]) == (3, 0, 0) and m["f1"] == 1
    assert evaluation.false_alarms_per_clean_session(rows, 65) == 0
    assert evaluation.false_alarms_per_clean_session(rows, 101) == 25  # flag-every-answer baseline
    assert evaluation.bootstrap_f1(rows, 65, n=200, seed=1) == evaluation.bootstrap_f1(rows, 65, n=200, seed=1)
    report = evaluation.report(rows, 65)
    assert report["baseline_flag_every_answer"]["precision"] == 0.3 and report["recall_by_type"]["topic_drift"] == 1
    assert set(report["by_origin"]) == {"injected", "natural"}


def test_precision_at_prevalence():
    rows = [dict(session_id="a", label="x", health=0)] * 8 + [dict(session_id="a", label="x", health=100)] * 2 \
         + [dict(session_id="b", label="clean", health=0)] * 1 + [dict(session_id="b", label="clean", health=100)] * 9
    # recall 0.8, false-positive rate 0.1 -> 0.016 / (0.016 + 0.098)
    assert evaluation.precision_at_prevalence(rows, 50, 0.02) == round(0.016 / 0.114, 4)


def test_fit_uses_dev_only_and_compare_rule():
    rows = fixture_rows()
    fitted = evaluation.fit(rows)
    assert fitted["constraint_met"] and fitted["f1"] == 1 and 1 <= fitted["tau"] <= 100
    with pytest.raises(ValueError):
        evaluation.fit([r for r in rows if r["split"] == "test"])
    assert evaluation.compare({"f1": 0.84}, {"f1": 0.80})["decision"] == "replace v1 with v2"
    assert evaluation.compare({"f1": 0.84}, {"f1": 0.70})["decision"] == "keep v1 (judge)"

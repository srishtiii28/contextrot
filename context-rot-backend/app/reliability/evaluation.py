"""Evaluation and threshold fitting for the reliability layer (midsem slides 6 and 9).

Data: JSONL, one labelled session per line:
  {"session_id": "s01", "split": "dev"|"test", "turns": [
     {"question": "...", "answer": "...", "constraints": ["At most 2 sentences."],
      "retrieved_context": [{"text": "...", "source": "notes.pdf", "page": 4}],
      "label": "clean"|"contradiction"|"ignored_instruction"|"course_conflict"|"topic_drift"|"other",
      "injected": false}]}
Constraints are cumulative: a turn's constraints list is what applies at that turn. An optional session
"model" field (the tutor model that produced the answers) is used by the premise study.

A turn is a positive if its label is not "clean"; the detector flags it if H < tau.
"""
import hashlib
import json
import random
import statistics
import time
from itertools import product
from pathlib import Path

from app.reliability.health import health_from_severities
from app.reliability.text import is_social_turn
from app.schemas.analysis import AnalysisInput

FAILURE_TYPES = ("contradiction", "ignored_instruction", "course_conflict", "topic_drift")
MAX_FALSE_ALARMS_PER_SESSION = 0.5  # per 25-turn clean session (slide 9)
SESSION_TURNS = 25


def load_sessions(path) -> list[dict]:
    sessions = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        session = json.loads(line)
        if session.get("split") not in {"dev", "test"}:
            raise ValueError(f"line {number}: split must be dev or test")
        for turn in session["turns"]:
            if "label" not in turn:
                raise ValueError(f"line {number}: every turn needs a label")
        sessions.append(session)
    return sessions


def data_fingerprint(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


async def run_detector(sessions: list[dict], detector: str, score_fn=None) -> list[dict]:
    """Replays each session turn by turn. Returns one row per turn with H, per-type severities and cost.

    History is bounded exactly as in production (pipeline.persist_answer -> compress_history), so measured
    behaviour matches the live worker. A detector error on one turn is recorded and the run continues.
    """
    from app.reliability import judge_client
    from app.services.llm_service import compress_history
    if score_fn is None:
        from app import reliability
        score_fn = reliability.score
        if detector == "nli":  # load models up front so loading is not counted as the first turn's latency
            from app.reliability import nli
            nli.get_backend()
    rows = []
    for number, session in enumerate(sessions, 1):
        history = []
        for index, turn in enumerate(session["turns"], 1):
            constraints = [c if isinstance(c, dict) else {"type": "instruction", "content": c} for c in turn.get("constraints", [])]
            row = dict(session_id=str(session["session_id"]), split=session["split"], turn=index,
                       label=turn["label"], injected=bool(turn.get("injected", False)), model=session.get("model"))
            if is_social_turn(turn["question"], turn["answer"]):
                row.update(skipped=True, health=100.0, severities={}, flags=[], latency_ms=0.0, tokens=0)
            else:
                payload = AnalysisInput(session_id=number, message_id=index, current_question=turn["question"],
                                        current_answer=turn["answer"], conversation_history=compress_history(history),
                                        constraints=constraints, retrieved_context=turn.get("retrieved_context", []))
                usage = {}
                token = judge_client.USAGE.set(usage)
                started = time.perf_counter()
                try:
                    result = await score_fn(payload, detector)
                except Exception as exc:  # recorded, excluded from metrics, reported as an error count
                    row.update(skipped=False, error=f"{type(exc).__name__}: {exc}"[:300], health=None, severities={},
                               flags=[], latency_ms=round((time.perf_counter() - started) * 1000, 2), tokens=0)
                    rows.append(row)
                    history += [{"role": "user", "content": turn["question"]}, {"role": "assistant", "content": turn["answer"]}]
                    continue
                finally:
                    judge_client.USAGE.reset(token)
                severities = {}
                for flag in result.get("flags") or []:
                    severities[flag["type"]] = max(severities.get(flag["type"], 0.0), flag["severity"])
                row.update(skipped=False, health=result["health_score"], severities=severities,
                           flags=[f["type"] for f in result.get("flags") or []],
                           latency_ms=result.get("latency_ms") or round((time.perf_counter() - started) * 1000, 2),
                           tokens=usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
            rows.append(row)
            history += [{"role": "user", "content": turn["question"]}, {"role": "assistant", "content": turn["answer"]}]
    return rows


def scored(rows: list[dict]) -> list[dict]:
    """Rows with a score (detector errors excluded)."""
    return [r for r in rows if r.get("health") is not None]


def rescore(rows: list[dict], weights: dict[str, float]) -> list[dict]:
    """Recomputes H from stored severities with other weights, without calling the detector again."""
    return [dict(r, health=100.0 if r.get("skipped") else health_from_severities(r["severities"], weights))
            for r in scored(rows)]


def _counts(rows, tau):
    tp = fp = fn = tn = 0
    for r in rows:
        positive, flagged = r["label"] != "clean", r["health"] < tau
        tp += positive and flagged
        fp += (not positive) and flagged
        fn += positive and not flagged
        tn += (not positive) and not flagged
    return tp, fp, fn, tn


def prf(rows, tau) -> dict:
    tp, fp, fn, tn = _counts(rows, tau)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(precision=round(precision, 4), recall=round(recall, 4), f1=round(f1, 4), tp=tp, fp=fp, fn=fn, tn=tn)


def _sessions(rows):
    grouped = {}
    for r in rows:
        grouped.setdefault(r["session_id"], []).append(r)
    return grouped


def false_alarms_per_clean_session(rows, tau) -> float | None:
    """False alarms on fully clean sessions, normalised to a 25-turn session."""
    clean = [s for s in _sessions(rows).values() if all(r["label"] == "clean" for r in s)]
    turns = sum(len(s) for s in clean)
    if not turns:
        return None
    alarms = sum(r["health"] < tau for s in clean for r in s)
    return round(alarms / turns * SESSION_TURNS, 4)


def precision_at_prevalence(rows, tau, prevalence=0.02) -> float | None:
    """Precision if failures made up `prevalence` of answers, from recall and the false-positive rate."""
    tp, fp, fn, tn = _counts(rows, tau)
    if not (tp + fn) or not (fp + tn):
        return None
    tpr, fpr = tp / (tp + fn), fp / (fp + tn)
    denominator = tpr * prevalence + fpr * (1 - prevalence)
    return round(tpr * prevalence / denominator, 4) if denominator else None


def bootstrap_f1(rows, tau, n=1000, seed=0) -> list[float] | None:
    """95% interval for F1, resampling whole sessions (turns in a session are not independent)."""
    sessions = list(_sessions(rows).values())
    if len(sessions) < 2:
        return None
    rng = random.Random(seed)
    scores = sorted(prf([r for _ in sessions for r in rng.choice(sessions)], tau)["f1"] for _ in range(n))
    return [scores[int(0.025 * n)], scores[min(n - 1, int(0.975 * n))]]


def _percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    return round(values[min(len(values) - 1, int(round(q * (len(values) - 1))))], 2)


def report(rows, tau, seed=0) -> dict:
    errors = len(rows) - len(scored(rows))
    rows = scored(rows)
    negatives = [r for r in rows if r["label"] == "clean"]
    by_origin = {}
    for name, keep in (("injected", True), ("natural", False)):
        subset = negatives + [r for r in rows if r["label"] != "clean" and r["injected"] == keep]
        if len(subset) > len(negatives):
            by_origin[name] = prf(subset, tau)
    recall_by_type = {}
    for label in sorted({r["label"] for r in rows} - {"clean"}):
        positives = [r for r in rows if r["label"] == label]
        recall_by_type[label] = round(sum(r["health"] < tau for r in positives) / len(positives), 4)
    latencies = [r["latency_ms"] for r in rows if not r.get("skipped")]
    checked = [r for r in rows if not r.get("skipped")]
    sessions = _sessions(rows)
    clean_sessions = sum(all(r["label"] == "clean" for r in s) for s in sessions.values())
    return dict(
        tau=tau, turns=len(rows), sessions=len(sessions), clean_sessions=clean_sessions,
        rot_sessions=len(sessions) - clean_sessions, detector_errors=errors,
        cost=dict(mean_tokens_per_answer=round(statistics.fmean(r.get("tokens", 0) for r in checked), 1) if checked else None,
                  mean_latency_ms=round(statistics.fmean(latencies), 2) if latencies else None),
        positives=sum(r["label"] != "clean" for r in rows), skipped=sum(bool(r.get("skipped")) for r in rows),
        **prf(rows, tau), f1_ci95=bootstrap_f1(rows, tau, seed=seed),
        false_alarms_per_clean_session=false_alarms_per_clean_session(rows, tau),
        precision_at_2pct=precision_at_prevalence(rows, tau),
        by_origin=by_origin, recall_by_type=recall_by_type,
        latency_ms=dict(p50=_percentile(latencies, 0.5), p95=_percentile(latencies, 0.95),
                        mean=round(statistics.fmean(latencies), 2) if latencies else None),
        baseline_flag_every_answer=dict(**prf(rows, 101), false_alarms_per_clean_session=false_alarms_per_clean_session(rows, 101)),
    )


def fit_threshold(rows, max_false_alarms=MAX_FALSE_ALARMS_PER_SESSION) -> dict:
    """Highest dev F1 whose false alarms stay within budget. When several tau tie, take the middle of the
    tied range rather than an edge, so the threshold is not fitted tight against one dev example."""
    rows = scored(rows)
    if false_alarms_per_clean_session(rows, 100) is None:
        raise ValueError("No fully clean dev sessions: the false-alarm budget cannot be checked, so tau cannot be fitted")
    by_tau = []
    for tau in range(1, 101):
        metrics = prf(rows, tau)
        alarms = false_alarms_per_clean_session(rows, tau)
        within = alarms is None or alarms <= max_false_alarms
        key = (within, metrics["f1"] if within else -(alarms or 0))
        by_tau.append((key, dict(tau=tau, f1=metrics["f1"], false_alarms_per_clean_session=alarms, constraint_met=within)))
    best_key = max(key for key, _ in by_tau)
    tied = [result for key, result in by_tau if key == best_key]
    return tied[len(tied) // 2]


def fit(rows, fit_weights=False, grid=(20, 35, 50, 65, 80)) -> dict:
    dev = [r for r in rows if r["split"] == "dev"]
    if not dev:
        raise ValueError("No dev-split rows: thresholds are fitted on development sessions only")
    from app.reliability.health import weights as current_weights
    candidates = [current_weights()]
    if fit_weights:
        candidates += [dict(zip(FAILURE_TYPES, combo)) for combo in product(grid, repeat=len(FAILURE_TYPES))]
    best = None
    for w in candidates:
        result = fit_threshold(rescore(dev, w))
        key = (result["constraint_met"], result["f1"])
        if best is None or key > best[0]:
            best = (key, dict(result, weights=w))
    return best[1]


def compare(v1: dict, v2: dict, margin=0.05) -> dict:
    """Slide 8 rule: v2 (local) replaces v1 (judge) only if its F1 is within `margin` of v1's at lower cost.
    Cost is LLM tokens per answer, then latency; both come from the reports' `cost` section."""
    gap = round(v1["f1"] - v2["f1"], 4)
    c1, c2 = v1.get("cost") or {}, v2.get("cost") or {}
    cost_key = lambda c: (c.get("mean_tokens_per_answer") or 0, c.get("mean_latency_ms") or 0)
    cheaper = cost_key(c2) < cost_key(c1)
    return dict(v1_f1=v1["f1"], v2_f1=v2["f1"], gap=gap, margin=margin, v1_cost=c1, v2_cost=c2, v2_cheaper=cheaper,
                decision="replace v1 with v2" if gap <= margin and cheaper else "keep v1")


def premise_study(sessions: list[dict], early=(1, 10), late=(21, 30), n=2000, seed=0) -> dict:
    """Slide 9 premise: do failures rise with turn number? Human labels only, no detector.

    Per tutor model: failure rate in turns 21-30 vs 1-10, and a 95% bootstrap interval for the
    difference that resamples whole sessions.
    """
    def rates(group):
        def rate(lo, hi):
            turns = [t for s in group for i, t in enumerate(s["turns"], 1) if lo <= i <= hi]
            return sum(t["label"] != "clean" for t in turns) / len(turns) if turns else None
        return rate(*early), rate(*late)

    by_model = {}
    for session in sessions:
        by_model.setdefault(session.get("model") or "unknown", []).append(session)
    rng = random.Random(seed)
    out = {}
    for model, group in sorted(by_model.items()):
        early_rate, late_rate = rates(group)
        if early_rate is None or late_rate is None:
            out[model] = dict(sessions=len(group), note=f"needs sessions with at least {late[1]} turns")
            continue
        diffs = []
        for _ in range(n):
            e, l = rates([rng.choice(group) for _ in group])
            if e is not None and l is not None:
                diffs.append(l - e)
        diffs.sort()
        ci = [round(diffs[int(0.025 * len(diffs))], 4), round(diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))], 4)] if diffs else None
        out[model] = dict(sessions=len(group), early_failure_rate=round(early_rate, 4), late_failure_rate=round(late_rate, 4),
                          difference=round(late_rate - early_rate, 4), difference_ci95=ci,
                          rises=bool(ci and ci[0] > 0))
    return out


def recovery_rate(database_url: str) -> dict:
    """Slide 9: share of flagged answers fixed within one retry, from the live database."""
    from sqlalchemy import create_engine, text
    engine = create_engine(database_url)
    with engine.connect() as connection:
        counts = dict(connection.execute(text(
            "SELECT status, COUNT(*) FROM message_states WHERE correction_of IS NOT NULL GROUP BY status")).all())
    attempted = sum(counts.values())
    return dict(retries=attempted, corrected=counts.get("corrected", 0), still_flagged=counts.get("flagged", 0),
                unverified=counts.get("check_failed", 0),
                recovery_rate=round(counts.get("corrected", 0) / attempted, 4) if attempted else None)

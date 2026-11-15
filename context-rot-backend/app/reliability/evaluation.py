"""Evaluation and threshold fitting for the reliability layer (midsem slides 6 and 9).

Data: JSONL, one labelled session per line:
  {"session_id": "s01", "split": "dev"|"test", "turns": [
     {"question": "...", "answer": "...", "constraints": ["At most 2 sentences."],
      "retrieved_context": [{"text": "...", "source": "notes.pdf", "page": 4}],
      "label": "clean"|"contradiction"|"ignored_instruction"|"course_conflict"|"topic_drift"|"other",
      "injected": false}]}
Constraints are cumulative: a turn's constraints list is what applies at that turn.

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
    """Replays each session turn by turn. Returns one row per turn with H and per-type severities."""
    if score_fn is None:
        from app import reliability
        score_fn = reliability.score
    rows = []
    for number, session in enumerate(sessions, 1):
        history = []
        for index, turn in enumerate(session["turns"], 1):
            constraints = [c if isinstance(c, dict) else {"type": "instruction", "content": c} for c in turn.get("constraints", [])]
            row = dict(session_id=str(session["session_id"]), split=session["split"], turn=index,
                       label=turn["label"], injected=bool(turn.get("injected", False)))
            if is_social_turn(turn["question"]):
                row.update(skipped=True, health=100.0, severities={}, flags=[], latency_ms=0.0)
            else:
                payload = AnalysisInput(session_id=number, message_id=index, current_question=turn["question"],
                                        current_answer=turn["answer"], conversation_history=list(history),
                                        constraints=constraints, retrieved_context=turn.get("retrieved_context", []))
                started = time.perf_counter()
                result = await score_fn(payload, detector)
                severities = {}
                for flag in result.get("flags") or []:
                    severities[flag["type"]] = max(severities.get(flag["type"], 0.0), flag["severity"])
                row.update(skipped=False, health=result["health_score"], severities=severities,
                           flags=[f["type"] for f in result.get("flags") or []],
                           latency_ms=result.get("latency_ms") or round((time.perf_counter() - started) * 1000, 2))
            rows.append(row)
            history += [{"role": "user", "content": turn["question"]}, {"role": "assistant", "content": turn["answer"]}]
    return rows


def rescore(rows: list[dict], weights: dict[str, float]) -> list[dict]:
    """Recomputes H from stored severities with other weights, without calling the detector again."""
    return [dict(r, health=100.0 if r.get("skipped") else health_from_severities(r["severities"], weights)) for r in rows]


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
    return dict(
        tau=tau, turns=len(rows), sessions=len(_sessions(rows)),
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
    scored = []
    for tau in range(1, 101):
        metrics = prf(rows, tau)
        alarms = false_alarms_per_clean_session(rows, tau)
        within = alarms is None or alarms <= max_false_alarms
        key = (within, metrics["f1"] if within else -(alarms or 0))
        scored.append((key, dict(tau=tau, f1=metrics["f1"], false_alarms_per_clean_session=alarms, constraint_met=within)))
    best_key = max(key for key, _ in scored)
    tied = [result for key, result in scored if key == best_key]
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
    """Slide 8 rule: v2 (local) replaces v1 (judge) only if its held-out F1 is within `margin` of v1's."""
    gap = round(v1["f1"] - v2["f1"], 4)
    return dict(v1_f1=v1["f1"], v2_f1=v2["f1"], gap=gap, margin=margin,
                decision="replace v1 with v2" if gap <= margin else "keep v1 (judge)")

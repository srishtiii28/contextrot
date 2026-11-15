"""Offline demo baseline: narrow regex rules for the scripted demo faults only.

Kept unchanged from the original rot_client so the offline demo and regression tests behave the same.
It is not a detector to evaluate; use `judge` (v1) or `nli` (v2) for that.
"""
import re

from app.schemas.analysis import AnalysisInput, DetectorResult, Flag


def score(payload: AnalysisInput) -> dict:
    answer = payload.current_answer.lower()
    contradiction = bool(re.search(r"primary key (?:can|may|allows?) (?:contain |have )?(?:null|duplicate)", answer))
    drift = "photosynthesis" in answer and "photosynthesis" not in payload.current_question.lower()
    violations = []
    for constraint in payload.constraints:
        text = constraint["content"].lower()
        match = re.search(r"(?:at most|max(?:imum)?|no more than)\s+(\d+)\s+sentences?", text)
        prose = re.sub(r"\[[^\]]*\]", "", payload.current_answer)
        if match and len([s for s in re.split(r"[.!?]+(?:\s|$)", prose) if s.strip()]) > int(match[1]):
            violations.append("sentence limit")
        if "bullet" in text and not re.search(r"(?m)^\s*[-*] ", payload.current_answer):
            violations.append("bullet format")
    alerts, flags = [], []
    if contradiction:
        alerts.append("Claim conflicts with the course rule for primary keys.")
        flags.append(Flag(type="course_conflict", severity=1, evidence="Primary keys cannot contain NULL or duplicate values."))
    if drift:
        alerts.append("Answer contains an unrelated topic.")
        flags.append(Flag(type="topic_drift", severity=1, evidence="Answer is about photosynthesis."))
    if violations:
        alerts.append("Ignored instruction: " + ", ".join(violations) + ".")
        flags.append(Flag(type="ignored_instruction", severity=1, evidence=", ".join(violations)))
    return DetectorResult(
        topic_score=0 if drift else 1, contradiction_score=float(contradiction), memory_score=0 if violations else 1,
        scope_score=0 if drift else 1, health_score=max(0, 100 - 70 * contradiction - 55 * bool(violations) - 70 * drift),
        topic_drift=drift, contradiction_detected=contradiction, memory_violation=bool(violations),
        alert=" ".join(alerts) or None, flags=flags, detector="rules",
    ).model_dump()

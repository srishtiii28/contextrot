"""Combine detector signals into the health score H (0-100) and the existing result contract.

H = 100 - sum_k w_k * s_k, clipped to [0, 100], where s_k in [0, 1] is the strongest signal of failure
type k. The weights are settings so they can be fitted on development sessions (scripts/eval_reliability.py).
With the default weights a single confident failure (s >= 0.8) of any type drops H below the default
recovery threshold of 65.
"""
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.schemas.analysis import DetectorResult, Flag

ALERT_TEXT = {
    "contradiction": "Contradicts an earlier answer",
    "ignored_instruction": "Ignored instruction",
    "course_conflict": "Conflicts with the course material",
    "topic_drift": "Answer drifts off the question's topic",
}


@dataclass
class Signals:
    flags: list[Flag] = field(default_factory=list)
    supported_fraction: float = 1.0  # share of claims supported by the course material, if measured

    def add(self, type_, severity, claim=None, evidence=None, turn=None):
        severity = max(0.0, min(1.0, float(severity)))
        if severity <= 0:
            return
        flag = Flag(type=type_, severity=round(severity, 4), claim=_clip(claim), evidence=_clip(evidence), turn=turn)
        for i, existing in enumerate(self.flags):  # one flag per (type, claim): keep the strongest evidence
            if existing.type == type_ and existing.claim == flag.claim:
                if flag.severity > existing.severity:
                    self.flags[i] = flag
                return
        self.flags.append(flag)

    def strongest(self, type_) -> float:
        return max((f.severity for f in self.flags if f.type == type_), default=0.0)


def _clip(text, limit=400):
    if text is None:
        return None
    text = str(text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def weights() -> dict[str, float]:
    s = get_settings()
    return {"contradiction": s.w_contradiction, "ignored_instruction": s.w_instruction,
            "course_conflict": s.w_course_conflict, "topic_drift": s.w_drift}


def health_from_severities(severities: dict[str, float], w: dict[str, float] | None = None) -> float:
    w = w or weights()
    penalty = sum(w[k] * max(0.0, min(1.0, severities.get(k, 0.0))) for k in w)
    return round(max(0.0, min(100.0, 100.0 - penalty)), 2)


def _alert(flags: list[Flag]) -> str | None:
    parts = []
    for type_, label in ALERT_TEXT.items():
        matching = sorted((f for f in flags if f.type == type_), key=lambda f: -f.severity)
        if not matching:
            continue
        top = matching[0]
        detail = (top.evidence or top.claim or "").strip().rstrip(".!?;:")
        if type_ == "contradiction" and top.turn is not None:
            label += " (previous answer)" if top.turn == 1 else f" ({top.turn} answers ago)"
        parts.append(f"{label}: {detail}." if detail else f"{label}.")
    return " ".join(parts) or None


def aggregate(signals: Signals, detector: str, flag_at: float = 0.5) -> dict:
    """Builds the DetectorResult dict. Flags below `flag_at` lower H but do not set the boolean fields.

    The student-facing alert follows the same rule as recovery and evaluation: it is shown only when
    H < RECOVERY_THRESHOLD, so the measured false-alarm rate is the rate of alerts students see.
    """
    sev = {k: signals.strongest(k) for k in ALERT_TEXT}
    health = health_from_severities(sev)
    reported = []
    if health < get_settings().recovery_threshold:  # several weak signals can add up: then show them all
        reported = [f for f in signals.flags if f.severity >= flag_at] or signals.flags
    return DetectorResult(
        topic_score=round(1 - sev["topic_drift"], 4),
        contradiction_score=round(max(sev["contradiction"], sev["course_conflict"]), 4),
        memory_score=round(1 - sev["ignored_instruction"], 4),
        scope_score=round(max(0.0, min(1.0, signals.supported_fraction)), 4),
        health_score=health,
        topic_drift=sev["topic_drift"] >= flag_at,
        contradiction_detected=max(sev["contradiction"], sev["course_conflict"]) >= flag_at,
        memory_violation=sev["ignored_instruction"] >= flag_at,
        alert=_alert(reported),
        flags=sorted(signals.flags, key=lambda f: -f.severity),
        detector=detector,
    ).model_dump()

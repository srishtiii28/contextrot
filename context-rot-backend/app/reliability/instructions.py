"""Deterministic checks for measurable student instructions.

Rules are exact and free, so every detector runs them first. Constraints that are not measurable
("only give hints", "explain like I'm five") are returned as unchecked; the judge evaluates those.
"""
import re
from dataclasses import dataclass, field

from app.reliability.text import clean_answer, prose_sentences, word_count

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_LIMIT = r"(?:at most|max(?:imum)?(?: of)?|no more than|not more than|up to|under|within|less than|fewer than)"
_NUM = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"


def _number(token: str) -> int:
    return int(token) if token.isdigit() else _NUMBER_WORDS[token]


@dataclass
class InstructionCheck:
    violations: list[dict] = field(default_factory=list)  # {"constraint", "evidence", "severity"}
    unchecked: list[str] = field(default_factory=list)
    checked: int = 0

    @property
    def severity(self) -> float:
        return max((v["severity"] for v in self.violations), default=0.0)


def _check_one(text: str, answer: str) -> tuple[bool, list[dict]]:
    """Returns (was_measurable, violations) for one constraint."""
    lowered = text.lower()
    measurable, found = False, []

    match = re.search(_LIMIT + r"\s+" + _NUM + r"\s+sentences?", lowered)
    if match:
        measurable = True
        limit, actual = _number(match[1]), len(prose_sentences(answer))
        if actual > limit:
            found.append(dict(constraint=text, evidence=f"{actual} sentences; limit is {limit}",
                              severity=min(1.0, 0.6 + 0.1 * (actual - limit))))

    match = re.search(_LIMIT + r"\s+" + _NUM + r"\s+words?", lowered) or re.search(r"(\d+)\s+words?\s+(?:or less|or fewer|max(?:imum)?)", lowered)
    if match:
        measurable = True
        limit, actual = _number(match[1]), word_count(answer)
        # 10% slack: students and graders do not count words exactly.
        if actual > limit * 1.1:
            found.append(dict(constraint=text, evidence=f"{actual} words; limit is {limit}",
                              severity=min(1.0, 0.5 + (actual - limit) / max(limit, 1))))

    cleaned = clean_answer(answer)
    if re.search(r"\bbullet(?:s|ed)?(?:\s+points?)?\b", lowered) and not re.search(r"\bno bullet", lowered):
        measurable = True
        if not re.search(r"(?m)^\s*[-*•]\s+\S", cleaned):
            found.append(dict(constraint=text, evidence="no bullet points in the answer", severity=0.8))

    if re.search(r"\bnumbered (?:list|steps|points)\b|\bstep[- ]by[- ]step numbered\b", lowered):
        measurable = True
        if not re.search(r"(?m)^\s*\d+[.)]\s+\S", cleaned):
            found.append(dict(constraint=text, evidence="no numbered list in the answer", severity=0.8))

    if re.search(r"\b(?:include|give|with|use)\s+(?:an?|one|at least one)\s+examples?\b|\balways (?:give|include) examples?\b", lowered):
        measurable = True
        if not re.search(r"\b(?:for example|for instance|e\.g\.|example|such as|consider)\b", cleaned.lower()):
            found.append(dict(constraint=text, evidence="no example in the answer", severity=0.6))

    if re.search(r"\b(?:no|don't use|do not use|avoid|without)\s+(?:any\s+)?(?:code|code blocks?)\b", lowered):
        measurable = True
        if "```" in (answer or ""):
            found.append(dict(constraint=text, evidence="answer contains a code block", severity=0.8))

    return measurable, found


def check_instructions(constraints: list[dict], answer: str) -> InstructionCheck:
    result = InstructionCheck()
    for item in constraints:
        text = (item.get("content") or "").strip()
        if not text:
            continue
        measurable, found = _check_one(text, answer)
        if measurable:
            result.checked += 1
            result.violations.extend(found)
        else:
            result.unchecked.append(text)
    return result

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


_NEGATION = r"(?:no|not|never|don'?t|do not|avoid|without|stop)\b(?:\s+\w+){0,2}\s+"
_PER_ITEM = re.compile(r"\b(?:per|each|every)\s+(?:bullet|point|item|step|part|question|section|paragraph|heading|line)", re.IGNORECASE)


def _limit(unit: str, lowered: str) -> int | None:
    """Upper limit on `unit` (sentence/word) from phrasings like "at most 2 sentences",
    "in two sentences", "exactly 3 sentences", "3 sentences max", "50 words or less"."""
    patterns = (
        _LIMIT + r"\s+" + _NUM + r"\s+" + unit + r"s?\b",
        r"\b(?:in|exactly|only|just|use|using|keep it to|limit (?:it|yourself|the answer) to)\s+" + _NUM + r"\s+" + unit + r"s?\b",
        r"\b" + _NUM + r"[- ]" + unit + r"s?\s+(?:or less|or fewer|max(?:imum)?|at most|only|limit)\b",
        r"\b(?:one|single)[- ]" + unit + r"\b(?:\s+(?:answer|reply|response|summary))?",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return _number(match[1]) if match.groups() and match[1] else 1
    return None


_REQUIRED_LINE = re.compile(
    r"\b(end|finish|close|start|begin)\s+(?:every|each|all|your|the|each of your)?\s*(?:answers?|responses?|replies|reply)?\s*"
    r"with\s+(?:a\s+|one\s+)?(?:separate\s+|new\s+|final\s+|short\s+)?(?:line|sentence)\s+"
    r"(?:starting|beginning|that\s+starts|that\s+begins)\s+(?:with\s+)?[\'\"\u2018\u201c]([^\'\"\u2019\u201d]{1,40})[\'\"\u2019\u201d]")


def _plain_lines(answer: str) -> list[str]:
    lines = []
    for line in clean_answer(answer).splitlines():
        line = re.sub(r"^[\s>*_#-]+", "", line).replace("**", "").replace("__", "").strip()
        if line:
            lines.append(line)
    return lines


def required_lines(constraints: list[dict]) -> list[tuple[str, str]]:
    """(position, prefix) for instructions like "end every answer with a line starting 'Exam tip:'"."""
    found = []
    for item in constraints:
        for position, prefix in _REQUIRED_LINE.findall((item.get("content") or "").lower()):
            found.append(("end" if position in ("end", "finish", "close") else "start", prefix.strip()))
    return found


def _without_required_lines(answer: str, prefixes: list[str]) -> str:
    """The answer without lines the student asked for (e.g. "Exam tip: ..."), so they don't count toward
    a sentence or word limit set in the same breath ("at most 2 sentences, and end with an exam tip")."""
    if not prefixes:
        return answer
    keep = [line for line in (answer or "").splitlines()
            if not any(re.sub(r"^[\s>*_#-]+", "", line).replace("**", "").strip().lower().startswith(p) for p in prefixes)]
    return "\n".join(keep)


def _check_one(text: str, answer: str, prefixes: list[str] | None = None) -> tuple[bool, list[dict]]:
    """Returns (was_measurable, violations) for one constraint."""
    lowered = text.lower()
    if _PER_ITEM.search(lowered):
        return False, []  # limits per bullet/step need structure-aware checking: left to the judge
    measurable, found = False, []
    cleaned = clean_answer(answer)

    for position, prefix in required_lines([{"content": text}]):
        measurable = True
        lines = [line.lower() for line in _plain_lines(answer)]
        line = (lines[-1] if position == "end" else lines[0]) if lines else ""
        if not line.startswith(prefix):
            where = "last" if position == "end" else "first"
            found.append(dict(constraint=text, evidence=f"{where} line does not start with '{prefix}'", severity=0.8))
    answer = _without_required_lines(answer, prefixes or [p for _, p in required_lines([{"content": text}])])

    limit = _limit("sentence", lowered)
    if limit is not None:
        measurable = True
        actual = len(prose_sentences(answer))
        if actual > limit:
            found.append(dict(constraint=text, evidence=f"{actual} sentences; limit is {limit}",
                              severity=min(1.0, 0.6 + 0.1 * (actual - limit))))

    limit = _limit("word", lowered)
    if limit is not None:
        measurable = True
        actual = word_count(answer)
        # 10% slack: students and graders do not count words exactly.
        if actual > limit * 1.1:
            found.append(dict(constraint=text, evidence=f"{actual} words; limit is {limit}",
                              severity=min(1.0, 0.5 + (actual - limit) / max(limit, 1))))

    has_bullets = bool(re.search(r"(?m)^\s*[-*\u2022]\s+\S", cleaned))
    if re.search(_NEGATION + r"bullet", lowered) or re.search(r"\bbullet\w*(?:\s+\w+)?\s+(?:not|no)\b", lowered):
        measurable = True
        if has_bullets:
            found.append(dict(constraint=text, evidence="answer uses bullet points", severity=0.8))
    elif re.search(r"\bbullet(?:s|ed)?(?:\s+points?)?\b", lowered):
        measurable = True
        if not has_bullets:
            found.append(dict(constraint=text, evidence="no bullet points in the answer", severity=0.8))

    if re.search(r"\bnumbered (?:list|steps|points)\b", lowered) and not re.search(_NEGATION + r"numbered", lowered):
        measurable = True
        if not re.search(r"(?m)^\s*\d+[.)]\s+\S", cleaned):
            found.append(dict(constraint=text, evidence="no numbered list in the answer", severity=0.8))

    if re.search(r"\b(?:include|give|with|use|add)\s+(?:an?|one|at least one)\s+examples?\b|\balways (?:give|include) examples?\b", lowered) \
            and not re.search(_NEGATION + r"examples?", lowered):
        measurable = True
        if not re.search(r"\b(?:for example|for instance|e\.g\.|example|such as|consider)\b", cleaned.lower()):
            found.append(dict(constraint=text, evidence="no example in the answer", severity=0.6))

    if re.search(r"\b(?:no|don'?t use|do not use|never use|avoid|without)\s+(?:any\s+)?(?:code|code blocks?)\b", lowered):
        measurable = True
        if "```" in (answer or ""):
            found.append(dict(constraint=text, evidence="answer contains a code block", severity=0.8))

    return measurable, found


def check_instructions(constraints: list[dict], answer: str) -> InstructionCheck:
    result = InstructionCheck()
    prefixes = [p for _, p in required_lines(constraints)]
    for item in constraints:
        text = (item.get("content") or "").strip()
        if not text:
            continue
        measurable, found = _check_one(text, answer, prefixes)
        if measurable:
            result.checked += 1
            result.violations.extend(found)
        else:
            result.unchecked.append(text)
    return result


# --- Standing instructions typed in chat --------------------------------------------------------------
# The session manager (midsem slide 4) extracts constraints from the conversation: a student who writes
# "for this whole session, answer in 2 sentences" in chat means the same as one who uses the instructions box.

_STANDING = re.compile(
    r"\b(?:for\s+(?:this|the)\s+(?:whole|entire|rest\s+of\s+(?:this|the))\s+(?:session|conversation|chat)"
    r"|for\s+the\s+rest\s+of\s+(?:this|the)\s+(?:session|conversation|chat)"
    r"|(?:from\s+now\s+on|going\s+forward|from\s+here\s+on|henceforth|until\s+i\s+say\s+otherwise)"
    r"|in\s+(?:all|every|each)\s+(?:of\s+your\s+)?(?:answers?|responses?|replies|reply)"
    r"|(?:every|each)\s+(?:answer|response|reply)\s+(?:should|must|needs?\s+to)"
    r"|always\s+(?:answer|reply|respond|use|give|include|keep|write|end|start|begin|add|explain|format)"
    r"|never\s+(?:use|give|include|write|answer|reply|respond|add|show))\b", re.IGNORECASE)
_CANCEL = re.compile(
    r"\b(?:stop\s+(?:using|giving|adding|ending|starting|including|writing|with)|no\s+longer|you\s+can\s+stop|"
    r"don'?t\s+(?:need|have)\s+to|forget\s+(?:the|my|that)\s+(?:earlier\s+|previous\s+)?(?:instruction|rule)|"
    r"ignore\s+(?:the|my)\s+(?:earlier\s+|previous\s+)?(?:instruction|rule))", re.IGNORECASE)
_CHAT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def standing_instructions(message: str) -> list[str]:
    """Sentences of a chat message that set an instruction for the rest of the session."""
    found = []
    for sentence in _CHAT_SPLIT.split(message or ""):
        sentence = sentence.strip()
        if not sentence or sentence.endswith("?") or _CANCEL.search(sentence):
            continue
        if _STANDING.search(sentence):
            found.append(re.sub(r"^(?:please|pls|ok(?:ay)?|also|and)[,\s]+", "", sentence, flags=re.IGNORECASE)[:300])
    return found


def cancelled_instructions(message: str, existing: list[str]) -> list[str]:
    """Existing instructions the student withdrew ("you can stop adding exam tips now")."""
    from app.reliability.text import content_terms
    withdrawn = []
    for sentence in _CHAT_SPLIT.split(message or ""):
        if not _CANCEL.search(sentence):
            continue
        terms = content_terms(sentence) - {"stop", "longer", "need", "instruction", "rule", "earlier", "previous", "forget", "ignore"}
        for text in existing:
            if terms and len(terms & content_terms(text)) >= min(2, len(terms)):
                withdrawn.append(text)
    return withdrawn

"""Text helpers shared by every detector: answer cleaning, claim splitting and the social-turn skip rule."""
import re

_CITATION = re.compile(r"\[[^\]]*\bpage\s*\d+[^\]]*\]", re.IGNORECASE)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_FIGURE_CAPTION = re.compile(r"(?m)^\*Figure:.*\*\s*$")
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_CALLOUT = re.compile(r"(?m)^>\s*⚠️.*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

SOCIAL_PHRASES = {
    "hi", "hello", "hey", "thanks", "thank you", "thank you so much", "thanks a lot", "thx", "ty",
    "ok", "okay", "ok thanks", "okay thanks", "cool", "great", "nice", "got it", "alright",
    "good morning", "good afternoon", "good evening", "bye", "goodbye", "see you", "awesome",
    "perfect", "sure", "yes", "no", "yep", "nope", "understood", "makes sense",
}


def is_social_turn(question: str) -> bool:
    """Short social messages ("thanks", "ok") carry no claim worth checking (midsem slide 5)."""
    cleaned = re.sub(r"[^a-z\s]", "", (question or "").lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return bool(cleaned) and cleaned in SOCIAL_PHRASES


def clean_answer(answer: str) -> str:
    """Remove presentation artefacts that are not claims: citations, images, captions, warnings, code."""
    text = _CODE_FENCE.sub(" ", answer or "")
    text = _IMAGE.sub(" ", text)
    text = _FIGURE_CAPTION.sub(" ", text)
    text = _CALLOUT.sub(" ", text)
    text = _CITATION.sub(" ", text)
    return text.strip()


def prose_sentences(answer: str) -> list[str]:
    """Sentences as a reader counts them; list bullets count as sentences."""
    text = clean_answer(answer)
    sentences = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip()
        line = re.sub(r"[#*_`]+", "", line).strip()
        if not line:
            continue
        sentences.extend(s.strip() for s in re.split(r"[.!?]+(?:\s|$)", line) if s.strip())
    return sentences


def claims(answer: str, min_words: int = 4) -> list[str]:
    """Checkable statements: sentences with enough words to assert something."""
    text = clean_answer(answer)
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line)
        line = re.sub(r"[#*_`]+", "", line).strip()
        for sentence in _SENTENCE_END.split(line):
            sentence = sentence.strip()
            if len(sentence.split()) >= min_words and not sentence.endswith("?"):
                out.append(sentence)
    return out


def word_count(answer: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", clean_answer(answer)))


def prior_assistant_turns(history: list[dict], limit: int) -> list[tuple[int, str]]:
    """(turn index, content) for the last `limit` assistant answers, oldest first.

    Turn index counts assistant answers from 1 in the supplied history so flags can point at them.
    Refusals and out-of-syllabus answers are excluded because they are not course claims.
    """
    turns = []
    index = 0
    for message in history:
        if message.get("role") != "assistant":
            continue
        index += 1
        content = message.get("content", "")
        if "out of syllabus" in content.lower()[:200]:
            continue
        turns.append((index, content))
    return turns[-limit:] if limit > 0 else turns

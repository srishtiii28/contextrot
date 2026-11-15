"""Text helpers shared by every detector: answer cleaning, claim splitting and the social-turn skip rule."""
import re

_CITATION = re.compile(r"\[[^\]]*\bpage\s*\d+[^\]]*\]", re.IGNORECASE)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_FIGURE_CAPTION = re.compile(r"(?m)^\*Figure:.*\*\s*$")
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_CALLOUT = re.compile(r"(?m)^>\s*⚠️.*$")

_ABBREVIATIONS = re.compile(r"\b(?:e\.g|i\.e|vs|cf|fig|eq|dr|mr|mrs|ms|prof|viz|al)\.", re.IGNORECASE)
_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")
_PLACEHOLDER = "\u2024"  # one-dot leader: stands in for protected dots while splitting
_STRUCTURAL_LINE = re.compile(r"^\s*(?:#{1,6}\s|\|.*\||[-:| ]{3,}$)")
_LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")

SOCIAL_PHRASES = {
    "hi", "hello", "hey", "thanks", "thank you", "thank you so much", "thanks a lot", "thx", "ty",
    "ok", "okay", "ok thanks", "okay thanks", "cool", "great", "nice", "got it", "alright",
    "good morning", "good afternoon", "good evening", "bye", "goodbye", "see you", "awesome",
    "perfect", "sure", "yes", "no", "yep", "nope", "understood", "makes sense",
}
SOCIAL_REPLY_MAX_WORDS = 40


def is_social_turn(question: str, answer: str | None = None) -> bool:
    """Short social messages ("thanks", "ok") carry no claim worth checking (midsem slide 5).

    If the tutor still replied with real content (e.g. "yes" after "Shall I explain joins?"), the turn is checked.
    """
    cleaned = re.sub(r"[^a-z\s]", "", (question or "").lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned or cleaned not in SOCIAL_PHRASES:
        return False
    return answer is None or word_count(answer) <= SOCIAL_REPLY_MAX_WORDS


def clean_answer(answer: str) -> str:
    """Remove presentation artefacts that are not claims: citations, images, captions, warnings, code."""
    text = _CODE_FENCE.sub(" ", answer or "")
    text = _IMAGE.sub(" ", text)
    text = _FIGURE_CAPTION.sub(" ", text)
    text = _CALLOUT.sub(" ", text)
    text = _CITATION.sub(" ", text)
    return text.strip()


def _split_line(line: str) -> list[str]:
    protected = _DECIMAL.sub(_PLACEHOLDER, _ABBREVIATIONS.sub(lambda m: m[0].replace(".", _PLACEHOLDER), line))
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[$\\])|(?<=[.!?])$", protected)
    return [p.replace(_PLACEHOLDER, ".").strip() for p in parts if p and p.strip(" .!?")]


def _content_lines(answer: str):
    for line in clean_answer(answer).splitlines():
        if not line.strip() or _STRUCTURAL_LINE.match(line):
            continue  # headings, tables and rules are layout, not sentences
        line = _LIST_MARKER.sub("", line)
        # Strip markdown emphasis and code marks, but keep underscores inside identifiers (EMP_L, σ_{Sal}).
        line = re.sub(r"[*`]+|(?<!\w)_+|_+(?![\w{])", "", line).strip()
        if line:
            yield line


def prose_sentences(answer: str) -> list[str]:
    """Sentences as a reader counts them; each list item counts as at least one sentence."""
    return [s for line in _content_lines(answer) for s in _split_line(line)]


def claims(answer: str, min_words: int = 4) -> list[str]:
    """Checkable statements: sentences with enough words to assert something."""
    return [s for s in prose_sentences(answer) if len(s.split()) >= min_words and not s.endswith("?")]


def word_count(answer: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", clean_answer(answer)))


def prior_assistant_turns(history: list[dict], limit: int) -> list[tuple[int, str]]:
    """(answers_ago, content) for the last `limit` assistant answers, oldest first.

    answers_ago is 1 for the answer just before the current one. Counting back from the end stays correct
    when the history has been truncated to a recent window. Out-of-syllabus answers are excluded because
    they are not course claims. Superseded answers are removed upstream (pipeline.prepare).
    """
    answers = [m.get("content", "") for m in history if m.get("role") == "assistant"]
    total = len(answers)
    turns = [(total - i, content) for i, content in enumerate(answers)
             if "out of syllabus" not in content.lower()[:200]]
    return turns[-limit:] if limit > 0 else turns


_STOPWORDS = set("""a an the and or but if then so of to in on at by for with from as is are was were be been being it its
this that these those there their they them he she we you your our i not no can cannot could would should will may might
must do does did done has have had which who whom what when where why how than into onto over under about also only
each every any all some more most such same other both either neither very just one two three per via""".split())


def content_terms(text: str) -> set[str]:
    """Lower-cased content words with a light plural fold, for relevance matching."""
    terms = set()
    for word in re.findall(r"[a-z0-9][a-z0-9'-]*", (text or "").lower()):
        if len(word) < 3 or word in _STOPWORDS:
            continue
        for suffix in ("ies", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[: -len(suffix)] + ("y" if suffix == "ies" else "")
                break
        terms.add(word)
    return terms

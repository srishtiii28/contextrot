"""Reliability layer v2: local consistency checks with an NLI cross-encoder and MiniLM embeddings
(midsem slide 8, method [6]). No LLM call. Needs `pip install -r requirements-nli.txt`.

- contradiction: answer claims vs. claims of the last N assistant answers. Only pairs that are
  topically close (embedding cosine) are sent to NLI, which bounds cost and false alarms.
- course_conflict: answer claims vs. sentence windows of the retrieved excerpts. A claim is flagged only
  when some window contradicts it and no window entails it.
- topic_drift: low embedding similarity between the answer and the question (or the previous answer,
  for follow-ups such as "explain more").
- instructions: deterministic rules only; unmeasurable instructions are not checked in v2.
"""
import asyncio
import math
import re
from typing import Protocol

from app.core.config import get_settings
from app.reliability.health import Signals, aggregate
from app.reliability.instructions import check_instructions
from app.reliability.text import claims, clean_answer, prior_assistant_turns
from app.schemas.analysis import AnalysisInput

MAX_ANSWER_CLAIMS = 12
MAX_PRIOR_CLAIMS = 60
PAIRS_PER_CLAIM = 3
RELATED_COSINE = 0.35
DRIFT_COSINE = 0.30


class NLIBackend(Protocol):
    def nli(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """Each result maps 'contradiction' / 'entailment' / 'neutral' to a probability."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Unit-normalised sentence embeddings."""


class SentenceTransformersBackend:
    def __init__(self, nli_model: str, embedding_model: str):
        try:
            from sentence_transformers import CrossEncoder, SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("RELIABILITY_DETECTOR=nli needs: pip install -r requirements-nli.txt") from exc
        self._nli = CrossEncoder(nli_model)
        self._embedder = SentenceTransformer(embedding_model)
        config = getattr(self._nli, "config", None) or self._nli.model.config
        self._labels = [config.id2label[i].lower() for i in range(len(config.id2label))]

    def nli(self, pairs):
        if not pairs:
            return []
        scores = self._nli.predict(pairs, apply_softmax=True, batch_size=16)
        return [{label: float(row[i]) for i, label in enumerate(self._labels)} for row in scores]

    def embed(self, texts):
        if not texts:
            return []
        return [list(map(float, v)) for v in self._embedder.encode(texts, normalize_embeddings=True)]


_backend: NLIBackend | None = None


def set_backend(backend: NLIBackend | None) -> None:
    """Tests inject a fake backend; None resets to lazy loading."""
    global _backend
    _backend = backend


def get_backend() -> NLIBackend:
    global _backend
    if _backend is None:
        s = get_settings()
        _backend = SentenceTransformersBackend(s.nli_model, s.embedding_model)
    return _backend


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _windows(text: str, size: int = 2) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text or "")) if len(s.split()) >= 3]
    if len(sentences) <= size:
        return [" ".join(sentences)] if sentences else []
    return [" ".join(sentences[i:i + size]) for i in range(0, len(sentences) - size + 1)]


def _top_related(claim_vec, candidates, k):
    ranked = sorted(((_cos(claim_vec, vec), i) for i, vec in enumerate(candidates)), reverse=True)
    return [(sim, i) for sim, i in ranked[:k] if sim >= RELATED_COSINE]


def _score_sync(payload: AnalysisInput, backend: NLIBackend) -> dict:
    settings = get_settings()
    signals = Signals()
    rules = check_instructions(payload.constraints, payload.current_answer)
    for v in rules.violations:
        signals.add("ignored_instruction", v["severity"], claim=v["constraint"], evidence=v["evidence"])

    answer_claims = claims(payload.current_answer)[:MAX_ANSWER_CLAIMS]
    prior = [(turn, c) for turn, text in prior_assistant_turns(payload.conversation_history, settings.reliability_history_turns)
             for c in claims(text)][-MAX_PRIOR_CLAIMS:]
    windows = [w for chunk in payload.retrieved_context[:5] for w in _windows(chunk.get("text", ""))][:80]
    answer_text = clean_answer(payload.current_answer)[:2000]
    previous_answer = prior_assistant_turns(payload.conversation_history, 1)

    vectors = backend.embed(answer_claims + [c for _, c in prior] + windows + [payload.current_question, answer_text]
                            + ([clean_answer(previous_answer[-1][1])[:2000]] if previous_answer else []))
    n_a, n_p, n_w = len(answer_claims), len(prior), len(windows)
    claim_vecs, prior_vecs = vectors[:n_a], vectors[n_a:n_a + n_p]
    window_vecs = vectors[n_a + n_p:n_a + n_p + n_w]
    question_vec, answer_vec = vectors[n_a + n_p + n_w], vectors[n_a + n_p + n_w + 1]
    previous_vec = vectors[n_a + n_p + n_w + 2] if previous_answer else None

    # Contradictions with earlier answers.
    pairs, meta = [], []
    for ci, vec in enumerate(claim_vecs):
        for _, pi in _top_related(vec, prior_vecs, PAIRS_PER_CLAIM):
            pairs.append((prior[pi][1], answer_claims[ci]))
            meta.append((ci, pi))
    for (ci, pi), probs in zip(meta, backend.nli(pairs)):
        contra = probs.get("contradiction", 0.0)
        if contra >= 0.5 and contra > probs.get("entailment", 0.0):
            signals.add("contradiction", contra, claim=answer_claims[ci], evidence=prior[pi][1], turn=prior[pi][0])

    # Support and conflicts with the course excerpts.
    if windows and answer_claims:
        pairs, meta = [], []
        for ci, vec in enumerate(claim_vecs):
            for _, wi in _top_related(vec, window_vecs, PAIRS_PER_CLAIM):
                pairs.append((windows[wi], answer_claims[ci]))
                meta.append((ci, wi))
        best = {}
        for (ci, wi), probs in zip(meta, backend.nli(pairs)):
            entail, contra = probs.get("entailment", 0.0), probs.get("contradiction", 0.0)
            record = best.setdefault(ci, {"entail": 0.0, "contra": 0.0, "window": None})
            record["entail"] = max(record["entail"], entail)
            if contra > record["contra"]:
                record["contra"], record["window"] = contra, windows[wi]
        supported = 0
        for ci, record in best.items():
            if record["entail"] >= 0.5:
                supported += 1
            elif record["contra"] >= 0.5:
                signals.add("course_conflict", record["contra"], claim=answer_claims[ci], evidence=record["window"])
        signals.supported_fraction = supported / len(answer_claims)

    # Topic drift.
    if answer_text and len(answer_text.split()) >= 5:
        similarity = _cos(question_vec, answer_vec)
        if previous_vec is not None:
            similarity = max(similarity, _cos(previous_vec, answer_vec))
        if similarity < DRIFT_COSINE:
            signals.add("topic_drift", min(1.0, (DRIFT_COSINE - similarity) / 0.2),
                        evidence=f"answer/question similarity {similarity:.2f}")
    return aggregate(signals, detector="nli")


async def score(payload: AnalysisInput) -> dict:
    backend = get_backend()
    return await asyncio.to_thread(_score_sync, payload, backend)

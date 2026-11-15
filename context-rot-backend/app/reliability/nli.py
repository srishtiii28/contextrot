"""Reliability layer v2: local consistency checks with an NLI cross-encoder and MiniLM embeddings
(midsem slide 8, method [6]). No LLM call. Needs `pip install -r requirements-nli.txt`.

- contradiction: answer claims vs. claims of the last N assistant answers. A pair counts only if both
  sentences are about the same thing (MiniLM cosine >= NLI_PAIR_MIN_SIMILARITY) and the NLI model calls it
  a contradiction in BOTH directions. Small NLI models call statements about different subjects
  ("mitosis ..." vs "meiosis ...") contradictory; the two gates remove most of those false alarms.
- course_conflict: answer claims vs. single sentences of the retrieved excerpts, with the same gates. A
  claim is flagged only when some sentence contradicts it and no excerpt passage entails it.
- topic_drift: low embedding similarity between the answer and every reference: the question, the
  previous answer (for follow-ups such as "explain more") and the excerpts (for short questions like "What is 2NF?").
- instructions: deterministic rules only; unmeasurable instructions are not checked in v2.
"""
import asyncio
import math
import re
from typing import Protocol

from app.core.config import get_settings
from app.reliability.health import Signals, aggregate
from app.reliability.instructions import check_instructions
from app.reliability.text import claims, clean_answer, prior_assistant_turns, prose_sentences
from app.schemas.analysis import AnalysisInput

MAX_ANSWER_CLAIMS = 12
MAX_PRIOR_CLAIMS = 60
PAIRS_PER_CLAIM = 3
DRIFT_COSINE = 0.30
CONTRADICTION_CUT = 0.5


class NLIBackend(Protocol):
    def nli(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """Each result maps 'contradiction' / 'entailment' / 'neutral' to a probability."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Unit-normalised sentence embeddings."""


class SentenceTransformersBackend:
    def __init__(self, nli_model: str, embedding_model: str, device: str = "cpu"):
        try:
            from sentence_transformers import CrossEncoder, SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("RELIABILITY_DETECTOR=nli needs: pip install -r requirements-nli.txt") from exc
        # CPU by default: these models are small, and on a shared GPU (e.g. Apple MPS next to a local LLM)
        # loading them can run out of memory and fail every check.
        self._nli = CrossEncoder(nli_model, device=device)
        self._embedder = SentenceTransformer(embedding_model, device=device)
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
        _backend = SentenceTransformersBackend(s.nli_model, s.embedding_model, s.nli_device)
    return _backend


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _sentences(text: str) -> list[str]:
    return [x for x in prose_sentences(text or "") if len(x.split()) >= 3]


def _windows(text: str, size: int = 2) -> list[str]:
    sentences = _sentences(text)
    if len(sentences) <= size:
        return [" ".join(sentences)] if sentences else []
    return [" ".join(sentences[i:i + size]) for i in range(0, len(sentences) - size + 1)]


def _top_related(claim_vec, candidates, k, minimum):
    ranked = sorted(((_cos(claim_vec, vec), i) for i, vec in enumerate(candidates)), reverse=True)
    return [(sim, i) for sim, i in ranked[:k] if sim >= minimum]


def _bidirectional(backend, pairs):
    """NLI on (a, b) and (b, a); contradiction is the weaker direction, entailment the forward one."""
    if not pairs:
        return []
    results = backend.nli(pairs + [(b, a) for a, b in pairs])
    n = len(pairs)
    return [dict(contradiction=min(results[i].get("contradiction", 0.0), results[n + i].get("contradiction", 0.0)),
                 entailment=results[i].get("entailment", 0.0)) for i in range(n)]


def _score_sync(payload: AnalysisInput, backend: NLIBackend) -> dict:
    settings = get_settings()
    signals = Signals()
    rules = check_instructions(payload.constraints, payload.current_answer)
    for v in rules.violations:
        signals.add("ignored_instruction", v["severity"], claim=v["constraint"], evidence=v["evidence"])

    gate = settings.nli_pair_min_similarity
    answer_claims = claims(payload.current_answer)[:MAX_ANSWER_CLAIMS]
    prior = [(turn, c) for turn, text in prior_assistant_turns(payload.conversation_history, settings.reliability_history_turns)
             for c in claims(text)][-MAX_PRIOR_CLAIMS:]
    excerpts = [c.get("text", "") for c in payload.retrieved_context[:5]]
    sentences = [x for text in excerpts for x in _sentences(text)][:80]
    windows = [w for text in excerpts for w in _windows(text)][:80]
    answer_text = clean_answer(payload.current_answer)[:2000]
    previous = prior_assistant_turns(payload.conversation_history, 1)
    references = [payload.current_question] + ([clean_answer(previous[-1][1])[:2000]] if previous else [])

    groups = [answer_claims, [c for _, c in prior], sentences, windows, [answer_text], references]
    vectors = backend.embed([t for g in groups for t in g])
    split, offset = [], 0
    for g in groups:
        split.append(vectors[offset:offset + len(g)])
        offset += len(g)
    claim_vecs, prior_vecs, sentence_vecs, window_vecs, (answer_vec,), reference_vecs = split

    # Contradictions with earlier answers.
    pairs, meta = [], []
    for ci, vec in enumerate(claim_vecs):
        for _, pi in _top_related(vec, prior_vecs, PAIRS_PER_CLAIM, gate):
            pairs.append((prior[pi][1], answer_claims[ci]))
            meta.append((ci, pi))
    for (ci, pi), probs in zip(meta, _bidirectional(backend, pairs)):
        if probs["contradiction"] >= CONTRADICTION_CUT:
            signals.add("contradiction", probs["contradiction"], claim=answer_claims[ci], evidence=prior[pi][1], turn=prior[pi][0])

    # Support from, and conflicts with, the course excerpts.
    if excerpts and answer_claims:
        support_pairs, support_meta = [], []
        for ci, vec in enumerate(claim_vecs):
            for _, wi in _top_related(vec, window_vecs, PAIRS_PER_CLAIM, 0.0):
                support_pairs.append((windows[wi], answer_claims[ci]))
                support_meta.append(ci)
        entailed = {}
        for ci, probs in zip(support_meta, backend.nli(support_pairs)):
            entailed[ci] = max(entailed.get(ci, 0.0), probs.get("entailment", 0.0))
        conflict_pairs, conflict_meta = [], []
        for ci, vec in enumerate(claim_vecs):
            if entailed.get(ci, 0.0) >= 0.5:
                continue
            for _, si in _top_related(vec, sentence_vecs, PAIRS_PER_CLAIM, gate):
                conflict_pairs.append((sentences[si], answer_claims[ci]))
                conflict_meta.append((ci, si))
        best = {}
        for (ci, si), probs in zip(conflict_meta, _bidirectional(backend, conflict_pairs)):
            if probs["contradiction"] > best.get(ci, (0.0, None))[0]:
                best[ci] = (probs["contradiction"], sentences[si])
        for ci, (contra, sentence) in best.items():
            if contra >= CONTRADICTION_CUT:
                signals.add("course_conflict", contra, claim=answer_claims[ci], evidence=sentence)
        signals.supported_fraction = sum(v >= 0.5 for v in entailed.values()) / len(answer_claims)

    # Topic drift: the answer must relate to at least one reference.
    if answer_text and len(answer_text.split()) >= 5:
        similarity = max([_cos(answer_vec, v) for v in reference_vecs] + [_cos(answer_vec, v) for v in window_vecs[:20]])
        if similarity < DRIFT_COSINE:
            signals.add("topic_drift", min(1.0, (DRIFT_COSINE - similarity) / 0.2),
                        evidence=f"answer/question similarity {similarity:.2f}")
    return aggregate(signals, detector="nli")


async def score(payload: AnalysisInput) -> dict:
    backend = get_backend()
    return await asyncio.to_thread(_score_sync, payload, backend)

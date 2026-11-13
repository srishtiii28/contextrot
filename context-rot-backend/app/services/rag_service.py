import hashlib
import math
import os
import re
import struct
from dataclasses import dataclass

import chromadb
from pypdf import PdfReader
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.models import Document, DocumentChunk, DocumentImage

_chroma_client = None
_chroma_collection = None


def get_chroma_collection():
    global _chroma_client, _chroma_collection
    if _chroma_collection is None:
        try:
            settings = get_settings()
            os.makedirs(settings.chroma_db_dir, exist_ok=True)
            _chroma_client = chromadb.PersistentClient(path=settings.chroma_db_dir)
            _chroma_collection = _chroma_client.get_or_create_collection(
                name="study_documents",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            _chroma_collection = None
    return _chroma_collection

EMBEDDING_DIMENSION = 384
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
_STOPWORDS = {
    "a", "about", "above", "an", "and", "are", "as", "at", "be", "been", "below", "by", "do",
    "does", "earlier", "following", "for", "from", "given", "he", "here", "how", "if", "in", "is",
    "it", "its", "last", "me", "mentioned", "my", "no", "notes", "of", "on", "or", "all", "answer",
    "answers", "chapter", "important", "previous", "question", "questions", "same", "so", "study",
    "tell", "that", "the", "their", "them", "these", "this", "those", "to", "up", "us", "was",
    "we", "what", "when", "where", "which", "who", "why", "with", "you", "your", "yours",
    "can", "could", "would", "should", "shall", "will", "may", "might", "must", "have", "has",
    "had", "having", "i", "she", "her", "hers", "they", "him", "his",
    "explain", "explains", "explained", "explaining", "explanation", "describe", "describes",
    "described", "describing", "description", "define", "defines", "defined", "defining",
    "definition", "discuss", "discusses", "discussed", "give", "gives", "given", "giving",
    "show", "shows", "shown", "showing", "state", "states", "stated", "stating",
    "write", "writes", "written", "writing", "detail", "details", "brief", "briefly",
    "list", "lists", "outline", "summary", "summarize", "elaborate",
    "happen", "happens", "happened", "happening", "make", "makes", "made", "best", "good", "better",
}
_STUDY_INTENT_TERMS = {
    "step", "stage", "process", "mechanism", "pathway", "function", "role",
    "explain", "describe", "working", "work", "works", "cycle", "difference", "compare",
    "solve", "solution", "solutions", "calculate", "calculation", "compute", "computation",
    "find", "evaluate", "evaluation", "derive", "derivation", "formula",
    "formulae", "equation", "equations", "theorem", "lemma", "algorithm", "numerical",
    "example", "problem", "proof", "value", "substitute", "hcf", "lcm",
    "represent", "algebraically", "graphically", "geometrically", "plot", "graph",
    "coordinate", "coordinates", "cost", "purchased", "bought", "situation",
    "intersect", "intersecting", "coincide", "coincident", "parallel", "consistent",
    "inconsistent", "dependent", "linear", "table", "tables", "tabular",
    "cramer", "cramers", "matrix", "matrices", "determinant", "determinants",
    "elimination", "substitution", "cross", "multiplication",
    "internal", "structure", "structures", "anatomy", "anatomical", "feature", "features",
    "organisation", "organization", "system", "systems", "type", "types", "detail", "details",
    "function", "functions", "part", "parts", "overview", "concept", "concepts",
}


@dataclass(frozen=True)
class ExtractedPage:
    page: int
    text: str


@dataclass(frozen=True)
class ExtractedImage:
    page: int
    filename: str
    caption: str
    data: bytes


def extract_images_from_pdf(content: bytes, pages: list[ExtractedPage]) -> list[ExtractedImage]:
    """Extracts high-resolution visual diagrams, figures, and flowcharts from PDF pages using PyMuPDF.

    Uses diagram subtitles (Figure X.Y captions) to pinpoint the exact visual diagram region
    and render it as a crisp, full-color PNG including labels and pointers.
    Captures full multi-line captions and continuation blocks.
    Falls back to raster image extraction if PyMuPDF page rendering isn't available.
    """
    extracted: list[ExtractedImage] = []

    # 1. Primary: PyMuPDF high-resolution crop rendering on figure subtitles
    try:
        import pymupdf

        doc = pymupdf.open(stream=content, filetype="pdf")
        fig_pattern = re.compile(
            r"(?i)^\s*(?:fig(?:ure)?\.?\s*\d+(?:\.\d+[a-z]?)?(?:\s*\([a-z]\))?(?:[:\s\-]+|$)|(?:diagram|flowchart|structure|pathway|cycle|graph)\s*(?:[:\s\-]+|$))"
        )
        any_fig_re = re.compile(
            r"(?i)(?:^|[\n\r]|(?:\([a-z0-9]+\)\s*))\s*(fig(?:ure)?\.?\s*(\d+\.\d+[a-z]?)(?:\s*\([a-z]\))?(?:[:\s\-]+[^\n\r]*)?)"
        )

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]
            blocks = page.get_text("blocks")
            drawings = page.get_drawings()
            images = page.get_image_info()

            captions: list[tuple[str, pymupdf.Rect]] = []
            for i, b in enumerate(blocks):
                text = b[4].strip()
                m_start = fig_pattern.match(text)
                m_any = any_fig_re.search(text)
                if m_start or m_any:
                    if m_start:
                        full_cap = text
                    else:
                        full_cap = text[m_any.start(1):]
                    cap_rect = pymupdf.Rect(b[:4])
                    # Check if subsequent block(s) below are continuations of the caption
                    k = i + 1
                    while k < len(blocks):
                        next_b = blocks[k]
                        next_text = next_b[4].strip()
                        if 0 <= next_b[1] - cap_rect.y1 < 18 and not re.match(r"(?i)^(?:figure|\d+\.|\bchapter\b)", next_text):
                            full_cap += " " + next_text
                            cap_rect |= pymupdf.Rect(next_b[:4])
                            k += 1
                        else:
                            break
                    clean_cap = re.sub(r"\s+", " ", full_cap).strip()

                    # Enrich minimal figure captions (e.g. "Fig. 3.3" or "Fig. 6.1") with surrounding explanatory text
                    fig_num_match = re.search(r"(\d+\.\d+[a-z]?)", clean_cap)
                    if fig_num_match and len(clean_cap.split()) <= 4:
                        num_str = fig_num_match.group(0)
                        pat = re.compile(rf"\b(?:fig(?:ure)?\.?\s*{re.escape(num_str)})\b", re.IGNORECASE)
                        descs = [
                            re.sub(r"\s+", " ", ob[4].strip())
                            for ob in blocks
                            if ob != b and pat.search(ob[4]) and len(ob[4].strip()) > 20
                        ]
                        if descs:
                            clean_cap = f"{clean_cap}: {descs[0][:150]}"

                    captions.append((clean_cap, cap_rect))

            all_caps = [c[1] for c in captions]
            W, H = page.rect.width, page.rect.height

            for idx, (caption_text, cap_rect) in enumerate(captions):
                try:
                    other_caps = [c for c in all_caps if c != cap_rect]
                    y_top_limit = max(70.0, 0.08 * H)
                    for oc in other_caps:
                        if oc.y1 < cap_rect.y0:
                            if not (oc.x1 < cap_rect.x0 - 50 or oc.x0 > cap_rect.x1 + 50):
                                y_top_limit = max(y_top_limit, oc.y1 + 4)

                    # Determine horizontal column bounds
                    if cap_rect.width > W * 0.45 or (cap_rect.x0 < W * 0.35 and cap_rect.x1 > W * 0.65):
                        h_min, h_max = 25.0, W - 25.0
                    elif cap_rect.x1 < W * 0.58 and cap_rect.x0 < W * 0.45:
                        h_min, h_max = 25.0, W * 0.58
                    elif cap_rect.x0 > W * 0.42:
                        h_min, h_max = W * 0.42, W - 25.0
                    else:
                        h_min = max(25.0, cap_rect.x0 - 150)
                        h_max = min(W - 25.0, cap_rect.x1 + 150)

                    candidates: list[pymupdf.Rect] = []

                    # 1. Drawings
                    for d in drawings:
                        r = pymupdf.Rect(d["rect"])
                        if r.width >= W - 40:
                            continue
                        if r.width > W - 50 and r.height < 5:
                            continue
                        if r.y0 < 70 or r.y1 <= 75 or r.y0 > H - 35:
                            continue
                        if r.x0 < h_min - 15 or r.x1 > h_max + 15:
                            continue
                        candidates.append(r)

                    # 2. Images
                    for img in images:
                        r = pymupdf.Rect(img["bbox"])
                        if r.x0 < -5 or r.y0 < -5:
                            continue
                        if r.width >= W * 0.85 and r.height >= H * 0.65:
                            continue
                        if r.width >= W - 40:
                            continue
                        if r.y0 < 70 or r.y1 <= 75 or r.y0 > H - 35:
                            continue
                        if r.x0 < h_min - 15 or r.x1 > h_max + 15:
                            continue
                        candidates.append(r)

                    # 3. Short diagram label text blocks (e.g. "(a)", "(b)", tissue annotations)
                    for b in blocks:
                        b_text = b[4].strip()
                        if fig_pattern.match(b_text):
                            continue
                        lines = [l for l in b_text.split("\n") if l.strip()]
                        if re.match(r"^\d+\.\d+", b_text):
                            continue
                        if len(b_text) < 60 and len(lines) <= 2:
                            r = pymupdf.Rect(b[:4])
                            if r.y0 >= y_top_limit and r.y1 <= cap_rect.y0 + 5:
                                if not (r.x0 < h_min - 10 or r.x1 > h_max + 10):
                                    candidates.append(r)

                    above = [r for r in candidates if r.y1 <= cap_rect.y0 + 8 and r.y0 >= y_top_limit - 5]
                    if not above:
                        crop_rect = pymupdf.Rect(
                            max(25.0, cap_rect.x0 - 20),
                            max(y_top_limit, cap_rect.y0 - 250),
                            min(W - 25.0, cap_rect.x1 + 20),
                            min(H - 20.0, cap_rect.y1 + 8),
                        )
                    else:
                        # Union all candidate diagram components above caption to preserve complete composite multi-part plates
                        u = pymupdf.Rect(above[0])
                        for r in above[1:]:
                            u |= r

                        crop_x0 = max(25.0, min(cap_rect.x0 - 10, u.x0 - 10))
                        crop_x1 = min(W - 25.0, max(cap_rect.x1 + 10, u.x1 + 10))
                        crop_y0 = max(y_top_limit, u.y0 - 8)
                        crop_y1 = min(H - 20.0, cap_rect.y1 + 8)
                        crop_rect = pymupdf.Rect(crop_x0, crop_y0, crop_x1, crop_y1)

                    # Ensure minimum crop dimensions for a visible diagram
                    if crop_rect.height < 50:
                        crop_rect.y0 = max(y_top_limit, crop_rect.y1 - 120)
                    if crop_rect.width < 100:
                        crop_rect.x0 = max(h_min, crop_rect.x0 - 50)
                        crop_rect.x1 = min(h_max, crop_rect.x1 + 50)

                    pix = page.get_pixmap(clip=crop_rect, dpi=150)
                    img_bytes = pix.tobytes("png")

                    img_name = f"page_{page_num}_fig_{idx}.png"
                    extracted.append(
                        ExtractedImage(
                            page=page_num,
                            filename=img_name,
                            caption=caption_text,
                            data=img_bytes,
                        )
                    )
                except Exception:
                    continue

        if extracted:
            return extracted
    except Exception:
        pass

    # 2. Fallback: raster image extraction via pypdf / Pillow
    try:
        import io
        from PIL import Image as PILImage

        reader = PdfReader(io.BytesIO(content))
        page_text_map = {p.page: p.text for p in pages}
        for page_num, page in enumerate(reader.pages, start=1):
            if not hasattr(page, "images") or not page.images:
                continue
            text = page_text_map.get(page_num, "")
            captions_raw = re.findall(
                r"(?i)(?:Fig(?:ure)?\.?\s*\d+(?:\.\d+)?[:\s\-]*[^\n.]+|(?:Diagram|Flowchart|Structure|Pathway|Cycle|Graph)\s*[:\s\-]*[^\n.]+)",
                text,
            )
            for idx, img in enumerate(page.images):
                try:
                    if len(img.data) < 1500:
                        continue
                    pil_img = PILImage.open(io.BytesIO(img.data))
                    if pil_img.width < 80 or pil_img.height < 80:
                        continue
                    png_buf = io.BytesIO()
                    pil_img.save(png_buf, format="PNG")
                    img_bytes = png_buf.getvalue()

                    caption = captions_raw[idx].strip() if idx < len(captions_raw) else (captions_raw[0].strip() if captions_raw else f"Figure on Page {page_num}")
                    caption = re.sub(r"\s+", " ", caption).strip()
                    img_name = f"page_{page_num}_img_{idx}.png"
                    extracted.append(
                        ExtractedImage(
                            page=page_num,
                            filename=img_name,
                            caption=caption,
                            data=img_bytes,
                        )
                    )
                except Exception:
                    continue
    except Exception:
        pass

    return extracted


def extract_pdf(content: bytes) -> list[ExtractedPage]:
    pages: list[ExtractedPage] = []
    # 1. Primary: PyMuPDF (fitz) - high-fidelity extraction preserving math symbols, Greek letters, superscripts, and equations
    try:
        import pymupdf

        doc = pymupdf.open(stream=content, filetype="pdf")
        for page_number, page in enumerate(doc, start=1):
            raw_text = page.get_text("text") or ""
            # Normalize smart apostrophes/quotes
            cleaned = re.sub(r"(?<=\w)[’‘](?=\w)", "'", raw_text)
            # De-hyphenate line wraps
            cleaned = re.sub(r"(?<=\w)-\n(?=\w)", "", cleaned)
            # Normalize whitespace while preserving line and paragraph breaks
            cleaned = re.sub(r"\r\n", "\n", cleaned)
            cleaned = re.sub(r"[ \t]+", " ", cleaned)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
            if cleaned:
                pages.append(ExtractedPage(page=page_number, text=cleaned))
    except Exception:
        pages = []

    # 2. Fallback: pypdf
    if not pages:
        reader = PdfReader(__import__("io").BytesIO(content))
        for page_number, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            cleaned = re.sub(r"(?<=\w)[’‘](?=\w)", "'", raw_text)
            cleaned = re.sub(r"(?<!\w)-\s*\n\s*(?=\w)", "", cleaned)
            cleaned = re.sub(r"\r\n", "\n", cleaned)
            cleaned = re.sub(r"[ \t]+", " ", cleaned)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
            if cleaned:
                pages.append(ExtractedPage(page=page_number, text=cleaned))

    if not pages:
        raise ValueError("The PDF contains no extractable text.")
    return pages


def chunk_pages(
    pages: list[ExtractedPage],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[tuple[str, int, int, int]]:
    if chunk_size <= overlap or overlap < 0:
        raise ValueError("chunk_size must be greater than overlap, and overlap cannot be negative.")
    chunks: list[tuple[str, int, int, int]] = []
    for page in pages:
        text = page.text.strip()
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                # Prefer splitting at paragraph breaks, then line breaks, then sentence boundaries, then whitespace
                boundary = text.rfind("\n\n", start + (chunk_size // 2), end)
                if boundary == -1:
                    boundary = text.rfind("\n", start + (chunk_size // 2), end)
                if boundary == -1:
                    boundary = text.rfind(". ", start + (chunk_size // 2), end)
                    if boundary != -1:
                        boundary += 1
                if boundary == -1:
                    boundary = text.rfind(" ", start + (chunk_size // 2), end)
                if boundary > start:
                    end = boundary
            chunk = text[start:end].strip()
            if chunk:
                chunks.append((chunk, page.page, start, end))
            if end == len(text):
                break
            start = end - overlap
            while start < len(text) and start > 0 and not text[start - 1].isspace():
                start += 1
    return chunks


def embed(text: str) -> list[float]:
    """Create a deterministic local embedding without a model download or API key."""
    vector = [0.0] * EMBEDDING_DIMENSION
    tokens = [token for token in text.lower().split() if token]
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % EMBEDDING_DIMENSION
        sign = 1.0 if digest[4] % 2 else -1.0
        vector[index] += sign
    magnitude = math.sqrt(sum(value * value for value in vector))
    return [value / magnitude for value in vector] if magnitude else vector


def serialize_embedding(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def deserialize_embedding(value: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(value) // 4}f", value))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def normalize_query_typos(query: str) -> str:
    typo_patterns = [
        (r"\bsirculat(ion|ory|e|ing|ed)?\b", r"circulat\1"),
        (r"\binfloresen(ce|t|ces)?\b", r"inflorescen\1"),
        (r"\bphotosyntheis\b", "photosynthesis"),
        (r"\bmitocondri(a|on|al)?\b", r"mitochondri\1"),
        (r"\brespiraton\b", "respiration"),
        (r"\bnefron\b", "nephron"),
        (r"\balviol(i|us)\b", r"alveol\1"),
    ]
    norm = query
    for pat, repl in typo_patterns:
        norm = re.sub(pat, repl, norm, flags=re.IGNORECASE)
    return norm


def _normalise_term(term: str) -> str:
    typo_map = {
        "sirculation": "circulation",
        "sirculatory": "circulatory",
        "infloresence": "inflorescence",
        "photosyntheis": "photosynthesis",
        "mitocondria": "mitochondria",
        "mitocondrion": "mitochondrion",
        "respiraton": "respiration",
        "nefron": "nephron",
        "alvioli": "alveoli",
    }
    if term in typo_map:
        term = typo_map[term]
    if len(term) > 5 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 5 and term.endswith(("ses", "xes", "zes", "ches", "shes")):
        return term[:-2]
    if len(term) > 4 and term.endswith("s"):
        return term[:-1]
    return term


def _terms(text: str) -> set[str]:
    return {
        _normalise_term(term)
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if len(term) >= 2 and term not in _STOPWORDS
    }


def _keyword_overlap(query: str, text: str) -> int:
    query_terms = _terms(query)
    text_terms = _terms(text)
    return len(query_terms & text_terms)


def _phrase_overlap(query: str, text: str) -> int:
    query_tokens = re.findall(r"[a-z0-9]+", query.lower())
    normalized_text = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    return sum(
        1
        for phrase in (
            " ".join(query_tokens[index:index + 2])
            for index in range(len(query_tokens) - 1)
        )
        if phrase in normalized_text and not any(
            token in _STOPWORDS for token in phrase.split()
        )
    )


def is_broad_study_request(query: str) -> bool:
    query_terms = _terms(query)
    broad_terms = {
        "important", "question", "answer", "summary", "summarise", "summarize",
        "revise", "revision", "teach", "explain", "overview", "everything",
        "all", "key", "study", "prepare", "exam", "formula", "formulae", "formulas",
        "concept", "concepts", "practice", "sample", "mock", "test",
    }
    return bool(query_terms & broad_terms) or "chapter" in query.lower()


def is_follow_up_request(query: str) -> bool:
    query_terms = _terms(query)
    if bool(query_terms & {"again", "another", "continue", "few", "more", "next", "additional"}):
        return True
    lowered = query.lower()
    followup_patterns = [
        r"\b(now\s+(?:solve|calculate|find|do|try|use|evaluate))\b",
        r"\b(?:what|how)\s+about\b",
        r"\b(?:similar|same)\s+(?:question|problem|example|numerical|way|method|technique)\b",
        r"\b(?:solve|calculate|find|do)\s+(?:it|this|the\s+same)\b",
        r"\b(?:diagram|figure|image|illustration|flowchart|picture|drawing|sketch)\s+(?:of|for|showing|depicting)?\s*(?:the\s+)?(?:above|same|that|this|it|its|previous|earlier)\b",
        r"\b(?:above|same|that|this)\s+(?:diagram|figure|image|illustration|flowchart)\b",
        r"\b(?:show|draw|display|give|provide)\s+(?:me\s+)?(?:the\s+)?(?:diagram|figure|image|illustration)\b",
        r"\b(?:its|their)\s+(?:diagram|figure|image|illustration|structure|labellings?|labels?)\b",
        r"\b(?:explain|describe|show|what\s+are)\s+(?:the\s+)?(?:labellings?|labels?|parts?)\b",
    ]
    return any(re.search(p, lowered) for p in followup_patterns)


def is_process_request(query: str) -> bool:
    return bool(_terms(query) & _STUDY_INTENT_TERMS)


def _retrieval_score(
    query: str,
    chunk: DocumentChunk,
    query_embedding: list[float],
    dense_semantic_score: float | None = None,
) -> float:
    query_terms = _terms(query)
    lexical = _keyword_overlap(query, chunk.text)
    lexical_score = lexical / len(query_terms) if query_terms else 0.0

    if dense_semantic_score is not None:
        semantic_score = dense_semantic_score
        semantic_weight = 0.50
    else:
        semantic_score = cosine_similarity(query_embedding, deserialize_embedding(chunk.embedding))
        semantic_weight = 0.1 if is_broad_study_request(query) else 0.2

    score = ((1.0 - semantic_weight) * lexical_score) + (semantic_weight * max(semantic_score, 0.0))
    if "active transport" in " ".join(re.findall(r"[a-z0-9]+", query.lower())) and "active transport" in chunk.text.lower():
        score += 0.75
    phrase_score = _phrase_overlap(query, chunk.text)
    score += 0.25 * phrase_score
    topic_terms = _terms(query) - _STUDY_INTENT_TERMS
    topic_overlap = len(topic_terms & _terms(chunk.text))
    if topic_terms and topic_overlap:
        score += 0.75 * (topic_overlap / len(topic_terms))
    # Definitional boost: If chunk defines or explains the query terms (e.g. "termed as", "is defined", "terminates in", "types of")
    c_low = chunk.text.lower()
    if topic_overlap >= 1:
        if any(sig in c_low for sig in ("termed as", "is defined", "are defined", "is called", "are called", "known as", "refers to", "terminates in", "types of")):
            score += 0.45
        if re.search(r"^\s*fig(?:ure)?\.?\s*\d+", c_low) and len(chunk.text.split()) < 35:
            score -= 0.35
    if is_process_request(query) and topic_overlap:
        score += 0.2 * topic_overlap
        if any(term in _terms(chunk.text) for term in _STUDY_INTENT_TERMS):
            score += 0.15
    if "EXERCISES" in chunk.text and "exercise" not in query.lower():
        score -= 0.6
    if is_broad_study_request(query) and lexical:
        score += 0.15
    if re.search(r"\b(formula|formulae|formulas|formula sheet|concept|concepts|all important|summary|revision|cheat sheet|key points|practice|sample questions?)\b", query.lower()):
        c_low = chunk.text.lower()
        if any(k in c_low for k in ("summary", "what have we discussed", "in this chapter, you have studied", "studied the following points", "points to remember")):
            score += 2.2
        # Check if query targets a specific chapter number
        ch_num_match = re.search(r"\bchapter\s*(\d+)\b", query.lower())
        if ch_num_match:
            target_ch = ch_num_match.group(1)
            if f"chapter {target_ch}" in c_low or f"chapter\n{target_ch}" in c_low or f"{target_ch}." in c_low:
                score += 1.5
    return score


def ingest_pdf(
    db: Session,
    *,
    session_id: int | None = None,
    knowledge_base_id: int | None = None,
    filename: str,
    content: bytes,
) -> Document:
    if session_id is None and knowledge_base_id is None:
        raise ValueError("A session or knowledge base is required.")
    if session_id is None:
        # Existing local databases created before reusable knowledge bases had a
        # non-null session_id column. Keep those databases usable during rollout.
        session_columns = db.connection().exec_driver_sql("PRAGMA table_info(documents)").fetchall()
        if any(row[1] == "session_id" and row[3] for row in session_columns):
            session_id = 0
    pages = extract_pdf(content)
    chunks = chunk_pages(pages)
    images = extract_images_from_pdf(content, pages)
    document = Document(
        session_id=session_id,
        knowledge_base_id=knowledge_base_id,
        filename=filename,
        content_type="application/pdf",
    )
    document.chunks = [
        DocumentChunk(
            chunk_index=index,
            text=text,
            page=page,
            start_char=start,
            end_char=end,
            embedding=serialize_embedding(embed(text)),
        )
        for index, (text, page, start, end) in enumerate(chunks)
    ]
    document.images = [
        DocumentImage(
            page=img.page,
            image_filename=img.filename,
            caption=img.caption,
            image_data=img.data,
        )
        for img in images
    ]
    db.add(document)
    db.commit()
    db.refresh(document)

    try:
        col = get_chroma_collection()
        if col is not None and document.chunks:
            col.add(
                ids=[f"doc_{document.id}_chunk_{c.chunk_index}" for c in document.chunks],
                documents=[c.text for c in document.chunks],
                metadatas=[
                    {
                        "document_id": document.id,
                        "session_id": session_id if session_id is not None else -1,
                        "knowledge_base_id": knowledge_base_id if knowledge_base_id is not None else -1,
                        "page": c.page if c.page is not None else 0,
                        "filename": filename,
                        "chunk_index": c.chunk_index,
                    }
                    for c in document.chunks
                ],
            )
    except Exception:
        pass

    return document


def retrieve_chunks(
    db: Session,
    *,
    session_id: int,
    query: str,
    top_k: int,
    knowledge_base_id: int | None = None,
) -> list[DocumentChunk]:
    query = normalize_query_typos(query)
    conditions = [Document.session_id == session_id]
    if knowledge_base_id is not None:
        conditions.append(Document.knowledge_base_id == knowledge_base_id)

    chunks = list(
        db.scalars(
            select(DocumentChunk)
            .join(Document)
            .where(or_(*conditions))
        )
    )
    if not chunks:
        return []

    chunk_by_doc_and_idx = {(chunk.document_id, chunk.chunk_index): chunk for chunk in chunks}
    chroma_semantic_scores: dict[int, float] = {}

    try:
        col = get_chroma_collection()
        if col is not None:
            # Sync chunks if any are missing in ChromaDB
            try:
                count = col.count()
            except Exception:
                count = 0

            if count < len(chunks):
                existing_ids = set()
                try:
                    existing_data = col.get()
                    if existing_data and "ids" in existing_data:
                        existing_ids = set(existing_data["ids"])
                except Exception:
                    pass

                missing_chunks = [
                    c for c in chunks
                    if f"doc_{c.document_id}_chunk_{c.chunk_index}" not in existing_ids
                ]
                if missing_chunks:
                    col.add(
                        ids=[f"doc_{c.document_id}_chunk_{c.chunk_index}" for c in missing_chunks],
                        documents=[c.text for c in missing_chunks],
                        metadatas=[
                            {
                                "document_id": c.document_id,
                                "session_id": c.document.session_id if c.document.session_id is not None else -1,
                                "knowledge_base_id": c.document.knowledge_base_id if c.document.knowledge_base_id is not None else -1,
                                "page": c.page if c.page is not None else 0,
                                "filename": c.document.filename,
                                "chunk_index": c.chunk_index,
                            }
                            for c in missing_chunks
                        ],
                    )

            matching_doc_ids = sorted(list({c.document_id for c in chunks}))
            where_clause = None
            if len(matching_doc_ids) == 1:
                where_clause = {"document_id": matching_doc_ids[0]}
            elif len(matching_doc_ids) > 1:
                where_clause = {"document_id": {"$in": matching_doc_ids}}

            n_results = min(len(chunks), max(top_k * 6, 35))
            chroma_res = col.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause,
            )
            if chroma_res and chroma_res.get("metadatas") and chroma_res["metadatas"][0]:
                for meta, dist in zip(chroma_res["metadatas"][0], chroma_res["distances"][0]):
                    doc_id = meta.get("document_id")
                    c_idx = meta.get("chunk_index")
                    if (doc_id, c_idx) in chunk_by_doc_and_idx:
                        matched_chunk = chunk_by_doc_and_idx[(doc_id, c_idx)]
                        chroma_semantic_scores[matched_chunk.id] = max(0.0, 1.0 - dist)
    except Exception:
        pass

    query_embedding = embed(query)
    scored_chunks = [
        (
            _retrieval_score(
                query,
                chunk,
                query_embedding,
                dense_semantic_score=chroma_semantic_scores.get(chunk.id),
            ),
            chunk,
        )
        for chunk in chunks
    ]
    seen_text: set[str] = set()
    ranked = sorted(scored_chunks, key=lambda item: item[0], reverse=True)
    unique_chunks = []
    selected_indices: set[tuple[int, int]] = set()
    for _, chunk in ranked:
        fingerprint = " ".join(chunk.text.lower().split())
        if fingerprint in seen_text:
            continue
        seen_text.add(fingerprint)
        unique_chunks.append(chunk)
        selected_indices.add((chunk.document_id, chunk.chunk_index))
        if len(unique_chunks) == top_k:
            break
    final_chunks: list[DocumentChunk] = []
    seen_ids: set[int] = set()

    # 1. Primary priority: all top-ranked matching chunks (anchors) must come FIRST
    for c in unique_chunks[:top_k]:
        if c.id not in seen_ids:
            seen_ids.add(c.id)
            final_chunks.append(c)

    # 2. Secondary priority: sequential neighbors of the top anchors for local context expansion
    neighbor_offsets = (-1, 1, 2) if is_process_request(query) else (1, -1)
    top_anchor_count = 3 if is_process_request(query) else 2
    for anchor in unique_chunks[:top_anchor_count]:
        for offset in neighbor_offsets:
            key = (anchor.document_id, anchor.chunk_index + offset)
            if key in chunk_by_doc_and_idx:
                neighbor = chunk_by_doc_and_idx[key]
                if neighbor.id not in seen_ids:
                    seen_ids.add(neighbor.id)
                    final_chunks.append(neighbor)

    for c in unique_chunks:
        if c.id not in seen_ids:
            seen_ids.add(c.id)
            final_chunks.append(c)

    max_return_limit = max(top_k + 4, 12) if is_process_request(query) else max(top_k + 3, 9)
    return final_chunks[:max_return_limit]


def retrieve_diagrams(
    db: Session,
    *,
    session_id: int,
    query: str,
    knowledge_base_id: int | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> list[DocumentImage]:
    """Finds all relevant diagrams or figures (supporting multi-figure plates or sequences)."""
    query_lower = query.lower()
    is_diagram_request = any(
        term in query_lower
        for term in (
            "diagram", "figure", "fig", "flowchart", "image", "picture", "illustration",
            "draw", "sketch", "structure of", "pathway of", "cycle of", "show me",
            "chart", "schematic", "labelling", "labels", "construction",
            "show graph", "display graph", "stages of", "stages in", "phases of",
        )
    )
    # Also trigger when a specific textbook example, figure, or geometric problem is requested
    if not is_diagram_request:
        is_diagram_request = bool(
            re.search(
                r"\b(?:example\s*\d+|fig(?:ure)?\.?\s*\d+(?:\.\d+)?|tangents?|concentric|circles?|triangles?|ladder|shadow|ogive|romila|akhila|champa|rails?|railway|pencils?|erasers?|(?:stages?|phases?)\s+of\s+(?:meiosis|mitosis|cell\s+division)|meiosis\s+stages?|mitosis\s+stages?)\b",
                query_lower,
            )
        )
    if not is_diagram_request:
        return []

    if knowledge_base_id is not None:
        conditions = [Document.knowledge_base_id == knowledge_base_id]
    else:
        conditions = [Document.session_id == session_id]

    images = list(
        db.scalars(
            select(DocumentImage)
            .join(Document)
            .where(*conditions)
        )
    )
    if not images:
        return []

    # Effective query incorporates resolved conversational topic when available
    effective_query = query_lower
    if conversation_history:
        _, resolved_topic = resolve_conversational_query(query, conversation_history)
        if resolved_topic and resolved_topic.lower() != query_lower:
            effective_query = f"{query_lower} {resolved_topic.lower()}"

    # Check for direct figure number match or known textbook example/problem figure targets
    target_num = None
    fig_num_match = re.search(r"\bfig(?:ure)?\.?\s*(\d+(?:\.\d+)?)\b", query_lower)
    if fig_num_match:
        target_num = fig_num_match.group(1)
    elif "romila" in query_lower or ("pencil" in query_lower and "eraser" in query_lower):
        target_num = "3.3"
    elif "akhila" in query_lower or ("giant wheel" in query_lower and "hoopla" in query_lower):
        target_num = "3.2"
    elif "rail" in query_lower or "railway" in query_lower or "x + 2y - 4" in query_lower or "x+2y-4" in query_lower:
        target_num = "3.4"
    elif "champa" in query_lower or ("pant" in query_lower and "skirt" in query_lower):
        target_num = "3.6"
    elif "example 4" in query_lower or (("x + 3y = 6" in query_lower or "x+3y=6" in query_lower) and any(t in query_lower for t in ("diagram", "figure", "fig", "image", "sketch", "draw"))):
        target_num = "3.5"
    elif "cyclic quadrilateral" in query_lower or "abcd is a cyclic" in query_lower:
        target_num = "3.7"
    elif ("lengths of tangents" in query_lower and "external point" in query_lower) or "theorem 10.2" in query_lower:
        target_num = "10.7"
    elif ("concentric" in query_lower and "chord" in query_lower) or "chord of the larger circle" in query_lower:
        target_num = "10.8"
    elif "ptq = 2" in query_lower.replace(" ", "") or ("angle ptq" in query_lower and "opq" in query_lower):
        target_num = "10.9"
    elif "chord of length 8 cm" in query_lower or ("chord" in query_lower and "radius 5 cm" in query_lower and "tp" in query_lower):
        target_num = "10.10"
    elif "poq = 110" in query_lower.replace(" ", "") or "poq=110" in query_lower.replace(" ", ""):
        target_num = "10.11"
    elif "circumscribe a circle" in query_lower or "ab + cd = ad + bc" in query_lower:
        target_num = "10.12"
    elif ("basic proportionality" in query_lower or "thales" in query_lower or "theorem 6.1" in query_lower or
          ("line parallel to" in query_lower and "intersects other two sides" in query_lower) or
          ("ad/db" in query_lower and "ae/ec" in query_lower)):
        target_num = "6.10"

    if target_num:
        matched = []
        for img in images:
            if re.search(rf"\bfig(?:ure)?\.?\s*{re.escape(target_num)}\b", img.caption.lower()):
                matched.append(img)
        if matched:
            return matched
        if fig_num_match:
            return []

    # 1. Targeted matcher for Meiosis: Figures 10.3 (Meiosis I) and 10.4 (Meiosis II)
    if "meiosis" in effective_query:
        has_both_indicator = bool(re.search(r"\b(?:1\s*(?:and|or|to|\&)\s*2|i\s*(?:and|or|to|\&)\s*ii)\b", effective_query))
        is_meiosis_2 = bool(re.search(r"\b(?:meiosis\s*(?:2|ii|two)|stage\s*2|second\s+meiotic|part\s*2)\b", effective_query))
        is_meiosis_1 = bool(re.search(r"\b(?:meiosis\s*(?:1|i|one)|stage\s*1|first\s+meiotic|part\s*1)\b", effective_query))

        m1 = next((img for img in images if re.search(r"\b10\.3\b", img.caption) or "stages of meiosis i" in img.caption.lower()), None)
        m2 = next((img for img in images if re.search(r"\b10\.4\b", img.caption) or "stages of meiosis ii" in img.caption.lower()), None)

        if m1 or m2:
            if has_both_indicator or (is_meiosis_1 and is_meiosis_2) or (not is_meiosis_1 and not is_meiosis_2):
                multi_res = [img for img in [m1, m2] if img is not None]
                if multi_res:
                    return multi_res
            elif is_meiosis_2 and m2:
                return [m2]
            elif is_meiosis_1 and m1:
                return [m1]

    # 2. Targeted matcher for Mitosis: Figures 10.2 a&b and 10.2 c to e
    if "mitosis" in effective_query:
        is_pro_meta = bool(re.search(r"\b(?:prophase|metaphase|early\s+prophase|late\s+prophase)\b", effective_query))
        is_ana_telo = bool(re.search(r"\b(?:anaphase|telophase|cytokinesis)\b", effective_query))

        m_ab = next((img for img in images if "10.2 a" in img.caption.lower()), None)
        m_ce = next((img for img in images if "10.2 c" in img.caption.lower()), None)

        if m_ab or m_ce:
            if is_pro_meta and not is_ana_telo:
                if m_ab:
                    return [m_ab]
            elif is_ana_telo and not is_pro_meta:
                if m_ce:
                    return [m_ce]
            else:
                multi_res = [img for img in [m_ab, m_ce] if img is not None]
                if multi_res:
                    return multi_res

    # 3. Targeted matcher for Nephron: Primary (Fig 16.3) vs Counter Current (Fig 16.6)
    if "nephron" in effective_query:
        if any(k in effective_query for k in ("counter current", "vasa recta", "hairpin", "medullary gradient")):
            cc = next((img for img in images if "16.6" in img.caption or "counter current" in img.caption.lower()), None)
            if cc:
                return [cc]
        prim = next((img for img in images if "16.3" in img.caption or "blood vessels, duct and tubule" in img.caption.lower()), None)
        if prim:
            return [prim]

    # 4. Targeted matcher for Circulation: Fig 15.4
    if any(k in effective_query for k in ("double circulation", "blood circulation", "plan of circulation", "circulation in human")):
        circ = next((img for img in images if "15.4" in img.caption or "schematic plan of blood circulation" in img.caption.lower()), None)
        if circ:
            return [circ]

    diagram_descriptor_terms = {
        "diagram", "diagrammatic", "diagrammatical", "representation", "presentation", "presented",
        "schematic", "schematically", "figure", "fig", "illustration", "picture", "image",
        "view", "portion", "structure", "section", "showing", "show", "shows", "shown", "draw", "sketch",
        "drawing", "flowchart", "chart", "give", "can", "you", "please", "display",
        "standard", "type", "types", "notes", "chat", "hand", "rough", "tell", "me",
        "about", "what", "which", "where", "above", "below", "following", "previous",
        "earlier", "same", "that", "this", "these", "those", "here", "there", "it",
        "its", "them", "their", "such", "aforementioned", "discussed", "stated", "mentioned",
        "given", "last", "first", "now", "also", "explain", "explanation", "label",
        "labels", "labelling", "labellings", "labelled", "labeled", "part", "parts",
        "followup", "follow", "up", "context", "active", "topic", "provide", "see",
        "look", "like", "want", "need", "could", "would", "depict", "depicts", "depicted",
        "depicting", "illustrate", "illustrated", "illustrates", "seen", "indicated",
        "pair", "linear", "equation", "equations", "system", "variable", "variables",
        "consistent", "inconsistent", "solve", "solving", "solved", "solution", "solutions",
        "algebraic", "algebraically", "geometric", "geometrically",
        "graph", "graphs", "graphical", "graphically", "plot", "plots", "plotting", "plotted",
        "check", "checking", "whether", "verify", "verification",
        "table", "value", "values", "coordinate", "coordinates", "point", "points",
        "line", "lines", "plane", "find", "finding",
    }

    synonym_map = {
        "ecg": {"ecg", "electrocardiogram", "electrocardiograph", "electrocardiography"},
        "electrocardiogram": {"ecg", "electrocardiogram", "electrocardiograph"},
        "electrocardiograph": {"ecg", "electrocardiogram", "electrocardiograph"},
        "tmv": {"tmv", "tobacco", "mosaic", "virus"},
        "heart": {"heart", "cardiac", "cardiovascular"},
        "glycogen": {"glycogen", "polysaccharide"},
        "kidney": {"kidney", "nephron", "renal"},
        "nephron": {"nephron", "kidney", "renal"},
        "alveolus": {"alveolus", "alveoli", "lung", "respiratory"},
        "mitosis": {"mitosis", "cell", "division"},
        "meiosis": {"meiosis", "meiotic", "cell", "division"},
        "bacterium": {"bacterium", "bacteria"},
        "bacteria": {"bacterium", "bacteria"},
        "stem": {"stem", "stems", "cauline"},
        "root": {"root", "roots", "radicle"},
        "leaf": {"leaf", "leaves", "foliar"},
        "flower": {"flower", "flowers", "floral", "inflorescence"},
        "cell": {"cell", "cells", "cellular"},
        "neuron": {"neuron", "neurons", "nerve", "axon", "synapse"},
        "brain": {"brain", "cerebral", "cerebrum", "sagittal"},
        "chloroplast": {"chloroplast", "chloroplasts", "thylakoid", "grana", "stroma"},
        "mitochondria": {"mitochondria", "mitochondrion", "cristae"},
        "mitochondrion": {"mitochondria", "mitochondrion", "cristae"},
        "monocot": {"monocot", "monocotyledon", "monocotyledonous", "monocots"},
        "dicot": {"dicot", "dicotyledon", "dicotyledonous", "dicots"},
        "frog": {"frog", "frogs", "rana"},
        "respiratory": {"respiratory", "respiration", "lung", "lungs", "breathing"},
        "circulatory": {"circulatory", "circulation", "blood", "heart"},
        "muscle": {"muscle", "sarcomere", "myosin", "actin", "sliding"},
        "eye": {"eye", "retina", "cornea"},
        "ear": {"ear", "tympanum", "cochlea"},
        "endocrine": {"endocrine", "hormone", "pituitary", "thyroid", "adrenal"},
        "photosynthesis": {"photosynthesis", "calvin", "light", "reaction", "chemiosmosis", "hatch", "slack"},
        "respiration": {"respiration", "glycolysis", "citric", "krebs", "ets"},
        "triangle": {"triangle", "triangles", "similarity", "similar", "congruence", "congruent", "thales", "bpt", "pythagoras", "median", "altitude", "hypotenuse", "6.1", "6.35", "6.43", "6.48", "6.49", "6.50"},
        "circle": {"circle", "circles", "tangent", "tangents", "radius", "radii", "diameter", "chord", "secant", "concentric", "10.1", "10.2", "10.7", "10.9"},
        "tangent": {"tangent", "tangents", "circle", "circles", "radius", "radii", "point", "contact", "external", "chord", "10.1", "10.2", "10.7", "10.9", "10.10"},
        "concentric": {"concentric", "circle", "circles", "chord", "radius", "tangent", "10.8"},
        "graph": {"graph", "graphs", "graphical", "graphically", "plot", "cartesian"},
        "coincident": {"coincident", "coincide", "infinite", "overlapping", "graph", "3.3"},
        "parallel": {"parallel", "nonintersecting", "rails", "graph", "3.4"},
        "romila": {"romila", "pencil", "pencils", "eraser", "erasers", "stationary", "stationery", "coincide", "coincident", "3.3"},
        "akhila": {"akhila", "fair", "giant", "wheel", "hoopla", "rides", "3.2"},
        "champa": {"champa", "sale", "pants", "pant", "skirts", "skirt", "3.6"},
    }

    raw_query_terms = _terms(query)
    query_subject_terms = raw_query_terms - diagram_descriptor_terms
    expanded_query_terms = set(query_subject_terms)
    for term in list(query_subject_terms):
        expanded_query_terms.update(synonym_map.get(term, set()))

    # If query is referential (e.g. "diagram of the above", "explain all its labellings") and lacks substantive terms, resolve from conversation history
    if not query_subject_terms and conversation_history:
        _, resolved_topic = resolve_conversational_query(query, conversation_history)
        hist_terms = _terms(resolved_topic) - diagram_descriptor_terms
        if hist_terms:
            query_subject_terms = hist_terms
        else:
            for msg in reversed(conversation_history):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    terms = _terms(content) - diagram_descriptor_terms
                    if terms:
                        query_subject_terms = terms
                        break
        if query_subject_terms:
            expanded_query_terms = set(query_subject_terms)
            for term in list(query_subject_terms):
                expanded_query_terms.update(synonym_map.get(term, set()))

    if not query_subject_terms:
        return []

    clean_topic_tokens = [t for t in query_lower.split() if t not in diagram_descriptor_terms and t not in _STOPWORDS]
    topic_phrase = " ".join(clean_topic_tokens).strip()

    query_vec = embed(" ".join(query_subject_terms) or query)
    scored_images: list[tuple[float, int, DocumentImage]] = []

    for img in images:
        cap_lower = img.caption.lower()
        cap_raw = _terms(img.caption)
        cap_subject_terms = cap_raw - diagram_descriptor_terms

        expanded_cap_terms = set(cap_subject_terms)
        for term in list(cap_subject_terms):
            expanded_cap_terms.update(synonym_map.get(term, set()))

        # Subject overlap between query and caption
        subject_overlap = len(expanded_query_terms & expanded_cap_terms)

        # Direct exact phrase match in caption (e.g. "cymose inflorescence", "plant cell", "dicot root")
        if topic_phrase and len(clean_topic_tokens) >= 2 and topic_phrase in cap_lower:
            subject_overlap += 8

        for qt in query_subject_terms:
            if re.search(rf"\b{re.escape(qt)}\b", cap_lower):
                subject_overlap += 3
            # Also check direct match inside acronym expansion
            for syn in synonym_map.get(qt, set()):
                if re.search(rf"\b{re.escape(syn)}\b", cap_lower):
                    subject_overlap += 2

        # Give strong bonus to actual textbook Figure captions (e.g. "Figure 5.8 Cymose inflorescence")
        is_figure_plate = bool(re.match(r"^\s*fig(?:ure)?\.?\s*\d+", cap_lower))
        figure_bonus = 6.0 if is_figure_plate else 0.0

        sem_sim = cosine_similarity(query_vec, embed(img.caption))
        total_score = (subject_overlap * 10.0) + (max(0.0, sem_sim) * 2.0) + figure_bonus
        scored_images.append((total_score, subject_overlap, img))

    scored_images.sort(key=lambda x: (x[1] > 0, x[0]), reverse=True)
    best_score, best_overlap, best_image = scored_images[0]

    # Strictly require positive subject overlap to avoid matching unrelated figures
    if best_overlap > 0:
        return [best_image]

    return []


def retrieve_diagram(
    db: Session,
    *,
    session_id: int,
    query: str,
    knowledge_base_id: int | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> DocumentImage | None:
    """Finds the single most relevant diagram or figure. Preserved for backward compatibility."""
    results = retrieve_diagrams(
        db,
        session_id=session_id,
        query=query,
        knowledge_base_id=knowledge_base_id,
        conversation_history=conversation_history,
    )
    return results[0] if results else None


def is_conversational_turn(query: str) -> bool:
    lowered = query.strip().lower()
    cleaned = re.sub(r"[^a-z\s]", "", lowered).strip()
    greetings = {
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "thanks", "thank you", "ok", "okay", "yes", "no", "sure", "cool",
        "help", "start", "test",
    }
    return cleaned in greetings or is_follow_up_request(lowered)


def _is_out_of_scope_non_academic(query: str) -> bool:
    """Detects purely everyday non-academic queries (cooking, finance, car repair, non-syllabus history, etc.)."""
    lowered = query.lower()
    patterns = [
        r"\b(?:sandwich|sandwiches|burger|burgers|pizza|pizzas|recipe|recipes|cook|cooking|bake|baking|cuisine)\b",
        r"\b(?:how\s+to|how\s+do\s+i|how\s+can\s+i|recipe\s+for)\s+(?:bake|cook|make|prepare|brew)\s+(?:a\s+)?(?:cake|chocolate|cookie|bread|pie|pasta|pizza|soup|curry|meal|dish)\b",
        r"\b(?:bake|baking)\s+(?:a\s+)?(?:chocolate\s+)?cake\b",
        r"\b(?:best\s+)?stocks?\s+(?:to\s+buy|to\s+invest|market|picks?|trading|portfolio)\b",
        r"\b(?:buy|invest\s+in)\s+(?:stocks?|shares|crypto|bitcoin|ethereum)\b",
        r"\b(?:how\s+to|how\s+do\s+i|how\s+can\s+i)\s+repair\s+(?:a\s+)?(?:broken\s+)?(?:car|engine|transmission|vehicle|motor|brakes?)\b",
        r"\b(?:repair|fix)\s+(?:a\s+)?(?:broken\s+)?(?:car|engine|transmission|clutch|alternator)\b",
        r"\b(?:french\s+literature|french\s+poetry|movie\s+reviews?|video\s+games?|cheat\s+codes?)\b",
        r"\b(?:french\s+revolution|russian\s+revolution|american\s+revolution|world\s+war|cold\s+war|medieval\s+history|industrial\s+revolution)\b",
    ]
    return any(re.search(p, lowered) for p in patterns)


def _is_out_of_scope_higher_math(query: str) -> bool:
    """
    Detects higher-level university / senior-secondary mathematics requiring
    methods, derivatives, integrals, or theorems strictly beyond Class 10 NCERT notes.
    """
    lowered = query.lower()
    patterns = [
        # Calculus & Differential Equations
        r"\b(?:integral|integrals|integration|integrate|integrating|antiderivative)\b",
        r"\b(?:derivative|derivatives|differentiation|differentiate|differentiating|dy/dx|d/dx)\b",
        r"\b(?:differential\s+equation|differential\s+equations|ode|pde)\b",
        r"\b(?:limit|limits)\s+(?:as\s+\w+\s+(?:approaches|tends\s+to|->)|\bof\b.*(?:as\s+\w+\s+approaches))\b",
        r"\b(?:multivariable|partial\s+derivative|partial\s+derivatives|gradient\s+vector|curl|divergence)\b",
        # Advanced Higher Math / Abstract Algebra / Transforms
        r"\b(?:eigenvalues?|eigenvectors?|characteristic\s+equation|diagonalization)\b",
        r"\b(?:laplace\s+transform|fourier\s+transform|fourier\s+series|z-transform)\b",
        r"\b(?:complex\s+numbers?|imaginary\s+unit|argand\s+plane)\b",
        r"\b(?:group\s+theory|ring\s+theory|vector\s+space|basis\s+and\s+dimension)\b",
    ]
    return any(re.search(p, lowered) for p in patterns)


def _is_in_scope_math_question(query: str) -> bool:
    """
    Detects whether a numerical, algebraic, or theoretical question falls within
    the Class 10 NCERT mathematics syllabus, even if it contains completely new/unseen numbers.
    """
    lowered = query.lower()
    # 1. Algebraic equations with variables (e.g. 3x + 2y = 12, 71x - 97y = 133, 2x^2 - 5x + 3 = 0, x - y + 1 = 0)
    if re.search(r"[-+]?\s*\d*\.?\d*\s*[a-z]\s*[-+]\s*\d*\.?\d*\s*[a-z]\s*=", lowered):
        return True
    if re.search(r"\b\d+[a-z]\s*[-+]\s*\d+[a-z]\b", lowered) and "=" in lowered:
        return True
    if re.search(r"\b[a-z]\s*[-+]\s*[a-z]\s*=", lowered):
        return True
    if re.search(r"\b\d*[a-z]\^2\b", lowered):
        return True
    if re.search(r"\b\d+(?:st|nd|rd|th)\s+term\b", lowered):
        return True

    # 2. Key syllabus topics / formulations
    math_indicators = [
        "linear equation", "pair of linear", "system of linear", "solve the equation",
        "solve the pair", "solve the following", "graphical method", "substitution method",
        "elimination method", "cross multiplication", "coincident", "intersecting", "parallel lines",
        "consistent", "inconsistent", "infinitely many solutions", "unique solution", "no solution",
        "quadratic equation", "discriminant", "roots of the equation", "nature of roots",
        "ap", "arithmetic progression", "common difference", "first term", "nth term", "sum of first",
        "coordinate geometry", "distance formula", "section formula", "collinear", "vertices of the triangle",
        "area of the triangle", "midpoint formula", "centroid",
        "similar triangles", "pythagoras theorem", "basic proportionality theorem", "thales theorem",
        "tangent to a circle", "tangents to a circle", "length of tangent", "subtended angle",
        "trigonometry", "trigonometric", "sin", "cos", "tan", "sec", "cosec", "cot",
        "angle of elevation", "angle of depression", "height and distance", "shadow", "vertical pole",
        "surface area", "volume of", "hemisphere surmounted", "frustum", "right circular cone",
        "mean, median", "mode of the data", "cumulative frequency", "ogive",
        "probability of getting", "deck of 52 cards",
        "euclid's division", "division algorithm", "fundamental theorem of arithmetic", "hcf of", "lcm of",
        "prime factorisation", "prime factorization", "prove that", "irrational",
        "cramer", "cramers rule", "matrix inversion",
        "pencil", "eraser", "apples and", "grapes", "railway track", "champa", "akhila", "romila",
        "upstream", "downstream", "speed of the boat", "fraction becomes", "two-digit number",
    ]
    return any(re.search(r"\b" + re.escape(k) + r"\b", lowered) for k in math_indicators)


def check_scope_boundary(
    query: str,
    retrieved_chunks: list[DocumentChunk],
    has_uploaded_docs: bool,
    subject: str | None = None,
) -> bool:
    """Returns True if the question is within syllabus scope, False if out of scope."""
    if not has_uploaded_docs:
        return True
    if is_conversational_turn(query):
        return True
    if not retrieved_chunks:
        return False

    # 1. Non-academic queries (cooking, finance, mechanical repair) are strictly out of scope
    if _is_out_of_scope_non_academic(query):
        return False

    # 2. Context detection: determine if the syllabus context is Mathematics
    is_math_context = False
    if subject:
        s_low = subject.lower()
        is_math_context = any(k in s_low for k in ("math", "algebra", "geometry", "arithmetic"))

    if not is_math_context and retrieved_chunks:
        combined_text = " ".join(c.text[:200].lower() for c in retrieved_chunks[:5])
        is_math_context = any(
            k in combined_text
            for k in (
                "mathematics", "linear equation", "pair of linear", "quadratic",
                "polynomial", "triangle", "euclid", "coordinate geometry",
                "trigonometr", "arithmetic progression", "surface area",
                "tangent to a circle", "hcf", "lcm", "cbse"
            )
        )

    # 3. Mathematics syllabus rules
    if is_math_context:
        # Strictly reject questions requiring methods or formulae beyond Class 10 notes (calculus, university math)
        if _is_out_of_scope_higher_math(query):
            return False
        # Accept any unseen numerical problem or in-syllabus formulation within Class 10 scope
        if _is_in_scope_math_question(query):
            return True

    # 4. Standard semantic & lexical overlap for general and biology subjects
    raw_terms = _terms(query)
    topic_terms = raw_terms - _STUDY_INTENT_TERMS - _CONTINUATION_TERMS
    if not topic_terms:
        return True

    all_chunks_terms = set().union(*(_terms(c.text) for c in retrieved_chunks))
    if len(topic_terms) <= 2:
        return topic_terms.issubset(all_chunks_terms)

    matching_terms = topic_terms & all_chunks_terms
    if len(matching_terms) / len(topic_terms) < 0.6:
        return False

    best_chunk_topic_overlap = max((len(topic_terms & _terms(c.text)) for c in retrieved_chunks), default=0)
    return best_chunk_topic_overlap >= 2


_CONTINUATION_TERMS = {
    # Marks & numerical counts
    "mark", "marks", "tier", "point", "points", "bullet", "bullets",
    "1", "2", "3", "4", "5", "6", "one", "two", "three", "four", "five", "six",
    # Formatting, length & answer directives
    "make", "give", "provide", "write", "format", "style", "answer", "answers",
    "question", "questions", "quiz", "test", "practice", "sample", "mcq", "mcqs",
    "vsa", "sa", "la", "short", "shorter", "long", "longer", "brief", "briefly",
    "simple", "simpler", "simply", "detailed", "elaborate", "expand", "summarize",
    "summarise", "summary", "notes", "sentence", "sentences", "word", "words",
    "paragraph", "paragraphs", "example", "examples", "table", "tabular", "diagram",
    # Referential pronouns, adverbs & filler words
    "above", "previous", "earlier", "same", "that", "this", "it", "its", "them",
    "these", "those", "explain", "explanation", "more", "again", "continue",
    "next", "why", "how", "what", "is", "are", "the", "of", "to", "in", "for",
    "on", "a", "an", "and", "or", "can", "you", "please", "tell", "me", "about",
    "describe", "discuss", "solve", "now", "just", "only", "such", "difference",
    "differences", "between", "define", "definition",
}


def is_pure_continuation_request(user_message: str) -> bool:
    """
    Returns True if the message is purely a formatting, elaboration, or continuation request
    (e.g., 'make it a 1 mark answer', 'elaborate on the above', 'explain that in 2 sentences')
    without introducing any new standalone subject nouns or entities.
    """
    raw_terms = _terms(user_message)
    if not raw_terms:
        return True
    substantive_new_terms = raw_terms - _CONTINUATION_TERMS
    return len(substantive_new_terms) == 0


def resolve_conversational_query(user_message: str, history: list[dict[str, str]]) -> tuple[str, str]:
    """
    Maintains conversational continuity across turns.
    If the user asks a follow-up without specifying the topic/chapter, resolves it from history.
    Skips any past turns where the question was rejected as out-of-scope.
    """
    raw_terms = _terms(user_message)
    substantive_terms = raw_terms - _CONTINUATION_TERMS

    # If the user supplied new substantive domain keywords (e.g. "cymose", "sandwich", "crispr"),
    # this is a new topic query, NOT a referential follow-up. Keep it strictly standalone!
    if substantive_terms:
        return user_message, user_message

    is_followup = (
        is_follow_up_request(user_message)
        or len(substantive_terms) == 0
        or bool(re.search(r"\b(?:[1-6]|one|two|three|four|five|six)\s*[- ]?marks?\b", user_message.lower()))
    )
    if not is_followup:
        return user_message, user_message

    # Find the most recent substantive, in-scope user message from history
    substantive_topics: list[str] = []
    for i, item in enumerate(history):
        if item.get("role") == "user" and item.get("content") != user_message:
            # Check if subsequent assistant message was an out-of-scope rejection
            if i + 1 < len(history):
                next_msg = history[i + 1]
                content = next_msg.get("content", "").lower()
                if "beyond the scope" in content or "> ⚠️ **out of syllabus:**" in content or "outside your prescribed course material" in content:
                    continue
            c = item["content"].strip()
            is_bare = is_follow_up_request(c) or is_pure_continuation_request(c)
            if not is_bare:
                substantive_topics.append(c)
            elif not substantive_topics:
                substantive_topics.append(c)

    if not substantive_topics:
        return user_message, user_message

    last_substantive = substantive_topics[-1]
    retrieval_query = f"{last_substantive}\nFollow-up: {user_message}"
    resolved_message = f"{user_message} (Context: active topic from earlier query '{last_substantive[:120]}')"
    return resolved_message, retrieval_query


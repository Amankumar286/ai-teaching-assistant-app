"""ParsedDocument -> list[Chunk]: large, topic-coherent chunks (1200-2000
tokens, 15-20% overlap when a section must be split) built primarily from
heading/chapter structure, with a large-fixed-chunk fallback for documents
with no detected structure at all.

Grouping strategy: each ParsedUnit's chapter/heading is forward-filled (a
unit with no detected heading of its own inherits the most recent one), then
consecutive units sharing the same (chapter, heading) pair form one
"section". Sections under CHUNK_MIN_TOKENS are merged forward with
subsequent sections (never across a detected chapter change) until they
reach the 1200-2000 token target; oversized single sections are split on
paragraph/sentence boundaries with overlap, never mid-sentence.
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_TOKENS,
    CHUNK_OVERLAP_RATIO,
    EMBEDDING_MODEL_NAME,
)

from app.rag.parser import ParsedDocument, ParsedUnit

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_TOPIC_PREFIX_RE = re.compile(r"^(chapter|section|bob|mavzu)?\s*\d+(\.\d+)*[.:)]?\s*", re.IGNORECASE)

# Reference lists, tables of contents, and indexes get parsed and would
# otherwise get chunked/indexed like any other content -- but a bibliography
# entry can share enough surface vocabulary (author names, technical terms
# in citation titles) with a real question to survive retrieval and get
# shown as a "source" for content it doesn't actually contain. Matched
# against each section's detected chapter/heading (English/Uzbek/Russian).
_EXCLUDED_SECTION_RE = re.compile(
    r"^(references?|bibliography|works\s+cited|"
    r"foydalanilgan\s+adabiyotlar|adabiyotlar\s+ro['ʻ‘’]?yxati|adabiyotlar|"
    r"mundarija|contents|table\s+of\s+contents|index|"
    r"литература|список\s+литературы|содержание|оглавление)"
    r"(\s+(list|section|ro['ʻ‘’]?yxati))?\s*$",
    re.IGNORECASE,
)

# Index specifically is back matter that, by real-world book convention,
# appears exactly once at the very end with nothing legitimate after it --
# unlike References/Bibliography, which many technical books (this one
# included) repeat once per chapter, so treating those as "exclude to end
# of document" would wipe out every real chapter after the first one's
# reference list (confirmed directly: an earlier version of this rule that
# also matched "references" cut chunk count from 166 to 51, discarding
# everything past page 187). And unlike a table of contents (front matter,
# only 1-2 pages, real chapters resume right after), nothing real ever
# follows the Index. Matters because a book's Index is itself full of
# short, alphabetized, sometimes-bold entries that look like real headings
# to the same detection heuristic used for actual content (e.g. "chain
# rule of differentiation  13"), so only excluding the single group
# literally headed "Index" lets dozens of these index-entry pseudo-headings
# escape right after it. Chapter-based propagation doesn't work here since
# chapter tracking depends on a detected heading matching a chapter marker,
# and this book's chapter title pages split the number and title across two
# separate spans that our one-winner-per-page heuristic never recombines --
# chapter is None for the entire document. Scoping exclusion "from here to
# the end of the document" is what's actually safe for Index specifically,
# precisely because nothing real follows it.
_BACK_MATTER_SECTION_RE = re.compile(r"^index\s*$", re.IGNORECASE)


def _is_excluded_section(chapter: Optional[str], heading: Optional[str]) -> bool:
    for label in (chapter, heading):
        if label and _EXCLUDED_SECTION_RE.match(label.strip()):
            return True
    return False


def _is_back_matter_section(chapter: Optional[str], heading: Optional[str]) -> bool:
    for label in (chapter, heading):
        if label and _BACK_MATTER_SECTION_RE.match(label.strip()):
            return True
    return False

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    return _tokenizer


def count_tokens(text: str) -> int:
    """Uses the embedding model's own tokenizer so counts match what will
    actually be embedded, and so no separate (network-fetched) tokenizer
    vocabulary needs to be introduced just for counting."""
    if not text:
        return 0
    return len(_get_tokenizer().encode(text, add_special_tokens=False))


@dataclass
class Chunk:
    text: str
    file_name: str
    file_type: str
    lecture_number: Optional[int] = None
    chapter: Optional[str] = None
    page_number: Optional[int] = None
    page_end: Optional[int] = None  # internal only; folded into `source`, not part of the spec schema
    slide_number: Optional[int] = None
    slide_end: Optional[int] = None  # internal only; folded into `source`, not part of the spec schema
    topic: Optional[str] = None
    source: str = ""
    token_count: int = 0

    def to_metadata(self) -> dict:
        """Chroma-safe metadata dict matching SPEC.md's schema exactly (None
        fields omitted, since Chroma metadata can't store None)."""
        schema = {
            "file_name": self.file_name,
            "file_type": self.file_type,
            "lecture_number": self.lecture_number,
            "chapter": self.chapter,
            "page_number": self.page_number,
            "slide_number": self.slide_number,
            "topic": self.topic,
            "source": self.source,
        }
        return {k: v for k, v in schema.items() if v is not None}


def _clean_topic(heading: Optional[str]) -> Optional[str]:
    if not heading:
        return None
    cleaned = _TOPIC_PREFIX_RE.sub("", heading).strip(" -:–—")
    return cleaned or heading


def _build_source(
    file_name: str,
    chapter: Optional[str],
    page_number: Optional[int],
    page_end: Optional[int],
    slide_number: Optional[int],
    slide_end: Optional[int],
    topic: Optional[str],
) -> str:
    parts = [file_name]
    if page_number is not None:
        parts.append(f"Pages {page_number}-{page_end}" if page_end and page_end != page_number else f"Page {page_number}")
    elif slide_number is not None:
        parts.append(f"Slides {slide_number}-{slide_end}" if slide_end and slide_end != slide_number else f"Slide {slide_number}")
    elif chapter:
        parts.append(f"Chapter {chapter}")

    label = ", ".join(parts)
    if topic:
        label += f" — Topic: {topic}"
    elif chapter and (page_number is not None or slide_number is not None):
        label += f" — Chapter: {chapter}"
    return label


def _hard_split_by_words(text: str, max_tokens: int) -> List[str]:
    """Last-resort splitter for a single 'sentence' with no punctuation that
    still exceeds max_tokens on its own (rare, but must not be dropped)."""
    words = text.split()
    out, buf = [], []
    for w in words:
        buf.append(w)
        if count_tokens(" ".join(buf)) >= max_tokens:
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return out or [text]


def _split_text_into_pieces(text: str, max_tokens: int) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()] or [text]

    pieces = []
    for para in paragraphs:
        if count_tokens(para) <= max_tokens:
            pieces.append(para)
            continue
        for sentence in _SENTENCE_RE.split(para):
            sentence = sentence.strip()
            if not sentence:
                continue
            if count_tokens(sentence) <= max_tokens:
                pieces.append(sentence)
            else:
                pieces.extend(_hard_split_by_words(sentence, max_tokens))
    return pieces


def _pieces_for_unit(unit: ParsedUnit, max_tokens: int) -> List[Tuple[ParsedUnit, str]]:
    text = unit.text.strip()
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [(unit, text)]
    return [(unit, t) for t in _split_text_into_pieces(text, max_tokens)]


def _greedy_split_with_overlap(
    pieces: List[Tuple[ParsedUnit, str]], max_tokens: int, overlap_ratio: float
) -> List[List[Tuple[ParsedUnit, str]]]:
    sub_chunks: List[List[Tuple[ParsedUnit, str]]] = []
    current: List[Tuple[ParsedUnit, str]] = []
    current_tokens = 0
    i = 0
    while i < len(pieces):
        unit, text = pieces[i]
        t = count_tokens(text)
        if not current or current_tokens + t <= max_tokens:
            current.append((unit, text))
            current_tokens += t
            i += 1
            continue

        sub_chunks.append(current)
        overlap_target = int(max_tokens * overlap_ratio)
        tail: List[Tuple[ParsedUnit, str]] = []
        tail_tokens = 0
        for u, tx in reversed(current):
            tt = count_tokens(tx)
            if tail_tokens + tt > overlap_target:
                break
            tail.insert(0, (u, tx))
            tail_tokens += tt
        current, current_tokens = tail, tail_tokens

    if current:
        sub_chunks.append(current)
    return sub_chunks


def _finalize_chunk(
    chapter: Optional[str],
    heading: Optional[str],
    pieces: List[Tuple[ParsedUnit, str]],
    parsed_doc: ParsedDocument,
) -> Chunk:
    units = [u for u, _ in pieces]
    text = "\n\n".join(t for _, t in pieces).strip()

    page_numbers = [u.page_number for u in units if u.page_number is not None]
    slide_numbers = [u.slide_number for u in units if u.slide_number is not None]
    page_number = min(page_numbers) if page_numbers else None
    page_end = max(page_numbers) if page_numbers else None
    slide_number = min(slide_numbers) if slide_numbers else None
    slide_end = max(slide_numbers) if slide_numbers else None

    topic = _clean_topic(heading)
    source = _build_source(parsed_doc.file_name, chapter, page_number, page_end, slide_number, slide_end, topic)

    return Chunk(
        text=text,
        file_name=parsed_doc.file_name,
        file_type=parsed_doc.file_type,
        lecture_number=parsed_doc.lecture_number,
        chapter=chapter,
        page_number=page_number,
        page_end=page_end if page_end != page_number else None,
        slide_number=slide_number,
        slide_end=slide_end if slide_end != slide_number else None,
        topic=topic,
        source=source,
        token_count=count_tokens(text),
    )


def _group_by_heading(units: List[ParsedUnit]) -> List[Tuple[Tuple[Optional[str], Optional[str]], List[ParsedUnit]]]:
    """Forward-fill each unit's (chapter, heading), then group consecutive
    units sharing the same effective key into one section."""
    keyed = []
    active_chapter, active_heading = None, None
    for u in units:
        if u.chapter:
            active_chapter = u.chapter
        if u.heading:
            active_heading = u.heading
        keyed.append(((active_chapter, active_heading), u))

    groups: List[Tuple[Tuple[Optional[str], Optional[str]], List[ParsedUnit]]] = []
    current_key = None
    current_units: List[ParsedUnit] = []
    for key, u in keyed:
        if current_units and key != current_key:
            groups.append((current_key, current_units))
            current_units = []
        current_key = key
        current_units.append(u)
    if current_units:
        groups.append((current_key, current_units))
    return groups


def _merge_small_sections(
    groups: List[Tuple[Tuple[Optional[str], Optional[str]], List[ParsedUnit]]]
) -> List[dict]:
    merged: List[dict] = []
    pending = None
    for (chapter, heading), group_units in groups:
        text = "\n\n".join(u.text for u in group_units if u.text.strip())
        tokens = count_tokens(text)

        if pending is None:
            pending = {"chapter": chapter, "heading": heading, "heading_tokens": tokens, "units": list(group_units), "tokens": tokens}
            continue

        chapter_changed = pending["chapter"] is not None and chapter is not None and pending["chapter"] != chapter
        combined_tokens = pending["tokens"] + tokens

        if not chapter_changed and pending["tokens"] < CHUNK_MIN_TOKENS and combined_tokens <= CHUNK_MAX_TOKENS:
            pending["units"].extend(group_units)
            pending["tokens"] = combined_tokens
            # Whichever sub-section actually contributes the most text names
            # the merged chunk, not whichever came first: a one-page "Note"
            # callout shouldn't out-rank a genuine multi-page section (e.g.
            # "RNNs"/"Bidirectional RNNs") absorbed into the same chunk right
            # after it -- that was silently erasing real topic labels a
            # student would actually search for. Found via a real retrieval
            # failure: "Recurrent Neural Networks (RNN)" returned nothing
            # useful because the chunk covering that content was labeled
            # "Note", the first small heading in the merge chain.
            if heading and tokens > pending["heading_tokens"]:
                pending["heading"] = heading
                pending["heading_tokens"] = tokens
        else:
            merged.append(pending)
            pending = {"chapter": chapter, "heading": heading, "heading_tokens": tokens, "units": list(group_units), "tokens": tokens}

    if pending is not None:
        merged.append(pending)
    return merged


def chunk_document(parsed_doc: ParsedDocument) -> List[Chunk]:
    if not parsed_doc.units:
        return []

    groups = _group_by_heading(parsed_doc.units)
    # Once real back matter (References/Bibliography/Index) is detected,
    # exclude everything from there to the end of the document -- see
    # _is_back_matter_section's docstring for why this is the safe rule for
    # that category specifically. Front-matter-only matches (table of
    # contents, etc.) still only exclude their own single group, since real
    # chapters resume right after those.
    filtered_groups = []
    in_back_matter = False
    for key, units in groups:
        chapter, heading = key
        if not in_back_matter and _is_back_matter_section(chapter, heading):
            in_back_matter = True
        if in_back_matter or _is_excluded_section(chapter, heading):
            continue
        filtered_groups.append((key, units))
    sections = _merge_small_sections(filtered_groups)

    chunks: List[Chunk] = []
    for section in sections:
        pieces = []
        for u in section["units"]:
            pieces.extend(_pieces_for_unit(u, CHUNK_MAX_TOKENS))
        if not pieces:
            continue  # entirely empty section (e.g. all-scanned pages) -> nothing to index
        for sub_pieces in _greedy_split_with_overlap(pieces, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_RATIO):
            if sub_pieces:
                chunks.append(_finalize_chunk(section["chapter"], section["heading"], sub_pieces, parsed_doc))
    return chunks

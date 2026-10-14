"""PDF/PPTX/DOCX -> ParsedDocument (text + structural metadata).

Each source file becomes a ParsedDocument holding a list of ParsedUnits (one
per page / slide / heading-delimited section). Downstream chunker.py groups
these units into topic-coherent chunks; this module only extracts text and
detects structure (headings, chapters, page/slide numbers) — it never
truncates or drops content, and never silently swallows a page/slide that
failed to yield text (those are flagged via `is_empty` + a warning instead).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import docx
import fitz  # PyMuPDF
from pptx import Presentation


class ParseError(Exception):
    """Raised when a source file can't be opened/parsed at all."""


@dataclass
class ParsedUnit:
    unit_type: str  # "page" | "slide" | "section"
    text: str
    heading: Optional[str] = None  # fine-grained label for this unit (page heading / slide title / sub-heading)
    chapter: Optional[str] = None  # coarse structural grouping (book chapter / doc top-level heading)
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    is_empty: bool = False


@dataclass
class ParsedDocument:
    file_name: str
    file_type: str  # "pdf" | "pptx" | "docx"
    lecture_number: Optional[int] = None
    units: List[ParsedUnit] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


_LECTURE_RE = re.compile(r"(?:lecture|lec|week)[\s_-]*(\d+)", re.IGNORECASE)
_MAX_HEADING_LEN = 80
_MIN_TEXT_LEN = 10  # below this many chars, a unit is considered empty/scanned


def _extract_lecture_number(filename: str) -> Optional[int]:
    m = _LECTURE_RE.search(filename)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def _compute_body_font_size(doc, sample_pages: int = 20) -> float:
    """Estimate the document's normal body-text font size (the size with the
    most total characters across a sample of pages), used as the baseline
    headings must exceed. Computed doc-wide rather than per-page so a page
    that's 100% heading (e.g. a section-divider page) doesn't lose its own
    reference point."""
    from collections import Counter

    counter = Counter()
    for page in doc[: min(sample_pages, doc.page_count)]:
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    counter[round(span["size"])] += len(span["text"])
    if not counter:
        return 10.0
    return float(counter.most_common(1)[0][0])


_JUNK_HEADING_RE = re.compile(r"^[\d\s.\-–—'\"‘’*)(]+$")
_CHAPTER_MARKER_RE = re.compile(
    r"^(?:[IVXLCDM]+|\d+)\s*[-.]?\s*(BOB|BO[ʻ‘’']?LIM|CHAPTER|ГЛАВА|GLAVA)\b",
    re.IGNORECASE,
)


def is_chapter_marker(text: str) -> bool:
    return bool(_CHAPTER_MARKER_RE.match(text.strip()))


_BOLD_HEADING_FRACTION = 0.75  # how much of a line must be bold to count as a bold heading, not a bold lead-in word


_MONOSPACE_FRACTION = 0.5  # lines dominated by a monospace/code font are never headings

def _detect_pdf_heading(page, body_size: float) -> Optional[str]:
    """Heuristic: a heading is a short *line* (spans on the same line are
    joined first, so a bold section number like "2.3." followed by its
    title on the same line isn't fragmented into just "2.3."), somewhere in
    the page's main content area (excluding the running header/footer
    strips), noticeably larger than the page's normal body size OR *mostly*
    bold by character count, not a code/monospace line, and not pure
    numbering/punctuation noise.

    Two things that look reasonable but are wrong for dense technical books,
    both found by testing against a real 559-page PDF rather than assumed:

    1. "Mostly bold" (not "any span bold"): books commonly bold just a
       lead-in term within an otherwise-regular sentence (e.g. "**TanH**: TanH
       is expressed as follows: <formula>") -- treating any bold span as
       making the whole line a heading misdetects that entire sentence as a
       heading.
    2. Headings are NOT confined to the top of the page: a slide deck has one
       heading per slide near the top, but a book page can contain zero, one,
       or several section headings anywhere in the body -- restricting
       detection to the top ~15% of the page caused real chapter headings
       (e.g. "A refresher on deep learning", "Training a neural network using
       PyTorch") to be missed entirely, so their content silently inherited
       the previous (often front-matter) heading for dozens of pages. Also
       found: some code listings render a synthetic bold duplicate of
       monospace text (e.g. font "Consolas-Bold"), which without a
       monospace exclusion gets misdetected as a heading purely from being
       "mostly bold"."""
    page_height = page.rect.height
    d = page.get_text("dict")
    candidates = []
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if not text or len(text) > _MAX_HEADING_LEN or len(text) < 3:
                continue
            if _JUNK_HEADING_RE.match(text):
                continue
            y0 = min(s["bbox"][1] for s in spans)
            size = max(s["size"] for s in spans)
            total_chars = sum(len(s["text"]) for s in spans) or 1
            bold_chars = sum(len(s["text"]) for s in spans if bool(int(s.get("flags", 0)) & (1 << 4)))
            mono_chars = sum(len(s["text"]) for s in spans if bool(int(s.get("flags", 0)) & (1 << 3)))
            if (mono_chars / total_chars) >= _MONOSPACE_FRACTION:
                continue
            is_mostly_bold = (bold_chars / total_chars) >= _BOLD_HEADING_FRACTION
            in_content_area = page_height * 0.06 <= y0 <= page_height * 0.94
            if in_content_area and (size >= body_size * 1.15 or is_mostly_bold):
                candidates.append((size, y0, text))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (-c[0], c[1]))
    return candidates[0][2]


def _parse_pdf(path: Path) -> ParsedDocument:
    try:
        doc = fitz.open(str(path))
    except Exception as e:
        raise ParseError(f"Failed to open PDF {path.name}: {e}") from e

    try:
        toc = doc.get_toc()  # [[level, title, page_1_indexed], ...]
        chapter_by_page = {}
        for _level, title, page_num in sorted(toc, key=lambda t: t[2]):
            cleaned = title.strip()
            if cleaned:
                chapter_by_page[page_num] = cleaned

        has_real_toc = bool(chapter_by_page)

        def chapter_for_page(page_num: int) -> Optional[str]:
            applicable = [p for p in chapter_by_page if p <= page_num]
            return chapter_by_page[max(applicable)] if applicable else None

        body_size = _compute_body_font_size(doc)
        warnings: List[str] = []
        units: List[ParsedUnit] = []

        for i in range(doc.page_count):
            page = doc[i]
            page_number = i + 1
            text = page.get_text("text").strip()
            is_empty = len(text) < _MIN_TEXT_LEN
            if is_empty:
                has_images = len(page.get_images()) > 0
                reason = "likely scanned/image-only page" if has_images else "no extractable text"
                warnings.append(f"Page {page_number}: {reason}")

            heading = None if is_empty else _detect_pdf_heading(page, body_size)

            # No embedded TOC (common for scanned/converted textbooks): fall
            # back to recognizing "I BOB. ..." / "Chapter 3" / "Глава 2"
            # style markers among detected headings as chapter boundaries.
            if not has_real_toc and heading and is_chapter_marker(heading):
                chapter_by_page[page_number] = heading

            units.append(
                ParsedUnit(
                    unit_type="page",
                    text=text,
                    heading=heading,
                    chapter=chapter_for_page(page_number),
                    page_number=page_number,
                    is_empty=is_empty,
                )
            )
    finally:
        doc.close()

    return ParsedDocument(
        file_name=path.name,
        file_type="pdf",
        lecture_number=_extract_lecture_number(path.name),
        units=units,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# PPTX
# ---------------------------------------------------------------------------

def _detect_pptx_heading_fallback(slide) -> Optional[Tuple[int, str]]:
    """Many real-world decks don't use a true Title placeholder (custom
    template text boxes instead) -- fall back to the topmost / largest-font
    text shape as an approximate slide heading."""
    candidates = []
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        text = shape.text_frame.text.strip()
        if not text or len(text) > _MAX_HEADING_LEN:
            continue
        max_size = 0.0
        for para in shape.text_frame.paragraphs:
            for run in para.runs:
                if run.font.size is not None:
                    max_size = max(max_size, run.font.size.pt)
        top = shape.top if shape.top is not None else 10**9
        candidates.append((max_size, top, shape.shape_id, text))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (-c[0], c[1]))
    _, _, shape_id, text = candidates[0]
    return shape_id, text


def _parse_pptx(path: Path) -> ParsedDocument:
    try:
        prs = Presentation(str(path))
    except Exception as e:
        raise ParseError(f"Failed to open PPTX {path.name}: {e}") from e

    warnings: List[str] = []
    units: List[ParsedUnit] = []

    for i, slide in enumerate(prs.slides):
        slide_number = i + 1

        title_shape = slide.shapes.title
        title = None
        title_shape_id = None
        if title_shape is not None and title_shape.has_text_frame:
            t = title_shape.text_frame.text.strip()
            if t:
                title, title_shape_id = t, title_shape.shape_id

        if title is None:
            fallback = _detect_pptx_heading_fallback(slide)
            if fallback is not None:
                title_shape_id, title = fallback

        body_parts = []
        for shape in slide.shapes:
            if title_shape_id is not None and shape.shape_id == title_shape_id:
                continue
            if shape.has_text_frame:
                t = shape.text_frame.text.strip()
                if t:
                    body_parts.append(t)

        notes = None
        if slide.has_notes_slide:
            notes_tf = slide.notes_slide.notes_text_frame
            if notes_tf is not None and notes_tf.text.strip():
                notes = notes_tf.text.strip()

        text_parts = []
        if title:
            text_parts.append(title)
        text_parts.extend(body_parts)
        if notes:
            text_parts.append(f"[Speaker notes] {notes}")
        text = "\n".join(text_parts).strip()

        is_empty = len(text) < _MIN_TEXT_LEN
        if is_empty:
            warnings.append(f"Slide {slide_number}: no extractable text (likely image-only)")

        heading = title if (title and len(title) <= _MAX_HEADING_LEN) else None

        units.append(
            ParsedUnit(
                unit_type="slide",
                text=text,
                heading=heading,
                chapter=None,
                slide_number=slide_number,
                is_empty=is_empty,
            )
        )

    return ParsedDocument(
        file_name=path.name,
        file_type="pptx",
        lecture_number=_extract_lecture_number(path.name),
        units=units,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

def _heading_level(style_name: str) -> int:
    """Title / Heading 1 are treated as chapter-level; Heading 2+ as topic-level."""
    if style_name in ("Title", "Heading 1"):
        return 1
    m = re.search(r"(\d+)$", style_name)
    return int(m.group(1)) if m else 1


def _extract_table_text(table) -> str:
    """python-docx repeats the same cell object across a merged span within
    a row, so de-dupe by the cell's underlying XML element before joining."""
    rows_text = []
    for row in table.rows:
        seen_ids = set()
        cleaned_cells = []
        for cell in row.cells:
            cid = id(cell._tc)
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            cell_text = cell.text.strip()
            if cell_text:
                cleaned_cells.append(cell_text)
        if cleaned_cells:
            rows_text.append(" | ".join(cleaned_cells))
    return "\n".join(rows_text).strip()


def _parse_docx(path: Path) -> ParsedDocument:
    try:
        document = docx.Document(str(path))
    except Exception as e:
        raise ParseError(f"Failed to open DOCX {path.name}: {e}") from e

    units: List[ParsedUnit] = []
    warnings: List[str] = []

    current_chapter: Optional[str] = None
    current_topic: Optional[str] = None
    current_texts: List[str] = []
    any_heading_seen = False

    def flush():
        text = "\n".join(current_texts).strip()
        if text or current_chapter or current_topic:
            units.append(
                ParsedUnit(
                    unit_type="section",
                    text=text,
                    heading=current_topic,
                    chapter=current_chapter,
                    is_empty=len(text) < _MIN_TEXT_LEN,
                )
            )

    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = para.style.name if para.style is not None else ""
        if style_name.startswith("Heading") or style_name == "Title":
            any_heading_seen = True
            flush()
            current_texts = []
            level = _heading_level(style_name)
            if len(text) > _MAX_HEADING_LEN:
                continue  # too long to be a real heading label; keep prior chapter/topic
            if level <= 1:
                current_chapter = text
                current_topic = None
            else:
                current_topic = text
        else:
            current_texts.append(text)
    flush()

    if not any_heading_seen:
        full_text = "\n".join(p.text.strip() for p in document.paragraphs if p.text.strip())
        units = [
            ParsedUnit(
                unit_type="section",
                text=full_text,
                heading=None,
                chapter=None,
                is_empty=len(full_text) < _MIN_TEXT_LEN,
            )
        ]

    # document.paragraphs doesn't reach table cells at all -- many real docx
    # files (assignments, rubrics, test-norm sheets) are primarily tables, so
    # skipping this would silently drop most or all of their content.
    for t_idx, table in enumerate(document.tables):
        table_text = _extract_table_text(table)
        if table_text:
            units.append(
                ParsedUnit(
                    unit_type="table",
                    text=table_text,
                    heading=None,
                    chapter=current_chapter,
                    is_empty=False,
                )
            )
        else:
            warnings.append(f"Table {t_idx + 1}: no extractable text.")

    if not units or all(u.is_empty for u in units):
        warnings.append("Document has little or no extractable text.")

    return ParsedDocument(
        file_name=path.name,
        file_type="docx",
        lecture_number=_extract_lecture_number(path.name),
        units=units,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_PARSERS = {
    ".pdf": _parse_pdf,
    ".pptx": _parse_pptx,
    ".docx": _parse_docx,
}


def parse_file(path) -> ParsedDocument:
    path = Path(path)
    if not path.exists():
        raise ParseError(f"File not found: {path}")
    parser = _PARSERS.get(path.suffix.lower())
    if parser is None:
        raise ParseError(f"Unsupported file type '{path.suffix}' ({path.name})")
    return parser(path)

"""Real, deterministic pytest coverage for chunk_document's structural
logic -- grouping, small-section merging, oversized-section splitting, and
back-matter exclusion. Companion to the manual scripts/test_chunker.py,
which eyeballs topic *coherence* on a real book; this file locks down the
mechanical rules so a refactor can't silently break them without a test
noticing.

count_tokens and the CHUNK_*_TOKENS thresholds are monkeypatched to a cheap
word-count proxy with small thresholds, so fixtures can be a few short
sentences instead of literal 1200+ word blocks, and so this file needs
neither `transformers` nor a network call to the real tokenizer -- it's
testing the grouping/merge/split *logic*, not the real tokenizer's exact
counts (that's what the manual script's real-book run is for).
"""

import pytest

from app.rag import chunker
from app.rag.chunker import chunk_document
from app.rag.parser import ParsedDocument, ParsedUnit


@pytest.fixture(autouse=True)
def small_thresholds(monkeypatch):
    monkeypatch.setattr(chunker, "count_tokens", lambda text: len(text.split()) if text else 0)
    monkeypatch.setattr(chunker, "CHUNK_MIN_TOKENS", 5)
    monkeypatch.setattr(chunker, "CHUNK_MAX_TOKENS", 10)
    monkeypatch.setattr(chunker, "CHUNK_OVERLAP_RATIO", 0.2)


def unit(text, chapter=None, heading=None, page_number=None, unit_type="page"):
    return ParsedUnit(unit_type=unit_type, text=text, chapter=chapter, heading=heading, page_number=page_number)


def doc(units, file_name="book.pdf", file_type="pdf"):
    return ParsedDocument(file_name=file_name, file_type=file_type, units=units)


def test_empty_document_produces_no_chunks():
    assert chunk_document(doc([])) == []


def test_single_section_becomes_one_chunk_with_correct_metadata():
    d = doc([
        unit("one two three four five six seven", chapter="Ch1", heading="1. Introduction", page_number=3),
    ])

    chunks = chunk_document(d)

    assert len(chunks) == 1
    c = chunks[0]
    assert c.file_name == "book.pdf"
    assert c.chapter == "Ch1"
    assert c.topic == "Introduction"  # numeric prefix stripped by _clean_topic
    assert c.page_number == 3
    assert c.token_count == 7
    assert "Page 3" in c.source
    assert "Topic: Introduction" in c.source


def test_small_consecutive_sections_in_same_chapter_merge_forward():
    d = doc([
        unit("a b c", chapter="Ch1", heading="Note", page_number=1),        # 3 tokens, < min(5)
        unit("d e f g h i j", chapter="Ch1", heading="Body", page_number=2),  # 7 tokens
    ])

    chunks = chunk_document(d)

    assert len(chunks) == 1
    assert chunks[0].token_count == 10  # 3 + 7 combined
    assert chunks[0].page_number == 1
    assert chunks[0].page_end == 2


def test_merged_chunk_takes_heading_from_larger_contributing_section():
    # Regression for the documented "Note absorbed a real section" bug: the
    # merged chunk's topic must come from whichever sub-section contributed
    # the most text, not whichever came first.
    d = doc([
        unit("a b c", chapter="Ch1", heading="Note", page_number=1),
        unit("d e f g h i j", chapter="Ch1", heading="Recurrent Neural Networks", page_number=2),
    ])

    chunks = chunk_document(d)

    assert len(chunks) == 1
    assert chunks[0].topic == "Recurrent Neural Networks"


def test_chapter_change_prevents_merge_even_when_both_sections_are_small():
    d = doc([
        unit("a b c", chapter="Ch1", heading="Intro", page_number=1),
        unit("d e f", chapter="Ch2", heading="Intro", page_number=5),
    ])

    chunks = chunk_document(d)

    assert len(chunks) == 2
    assert [c.chapter for c in chunks] == ["Ch1", "Ch2"]


def test_oversized_section_splits_into_multiple_chunks_with_overlap():
    # 20 tokens against max=10 must split; overlap_ratio=0.2 of max=10 -> ~2
    # tokens of tail carried into the next chunk.
    text = " ".join(f"word{i}." for i in range(20))  # 20 "sentences" of one word each
    d = doc([unit(text, chapter="Ch1", heading="Big Section", page_number=1)])

    chunks = chunk_document(d)

    assert len(chunks) > 1
    assert all(c.token_count <= 10 for c in chunks)
    # Overlap: the tail of chunk N should reappear at the start of chunk N+1.
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-1] in second_words[: len(first_words)]


def test_references_mid_document_is_excluded_but_later_chapters_survive():
    # Direct regression test for the documented bug: an earlier version that
    # treated References like Index (exclude-to-end-of-document) cut real
    # chunk count from 166 to 51 by wiping out every chapter after the
    # first one's reference list. References must only exclude ITS OWN
    # section, not everything after it.
    d = doc([
        unit("a b c d e f", chapter="Ch1", heading="Body", page_number=1),
        unit("citation one two three", chapter="Ch1", heading="References", page_number=2),
        unit("g h i j k l", chapter="Ch2", heading="Deep Learning", page_number=3),
    ])

    chunks = chunk_document(d)

    topics = [c.topic for c in chunks]
    assert "References" not in topics
    assert "Deep Learning" in topics
    assert len(chunks) == 2


def test_index_excludes_itself_and_everything_after_it():
    d = doc([
        unit("a b c d e f", chapter="Ch1", heading="Body", page_number=1),
        unit("apple 12, banana 45", chapter="Ch1", heading="Index", page_number=9),
        unit("appendix content here", chapter="Ch1", heading="Appendix", page_number=10),
    ])

    chunks = chunk_document(d)

    assert len(chunks) == 1
    assert chunks[0].topic == "Body"


def test_entirely_empty_section_is_skipped_not_returned_as_blank_chunk():
    d = doc([unit("   ", chapter="Ch1", heading="Scanned Page", page_number=1)])

    assert chunk_document(d) == []

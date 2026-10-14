"""Individual reasoning steps the agent graph composes.

Kept separate from graph.py because generate_quiz, compare_topics,
summarize, generate_assignment, and explain_difficult_concept all reuse
retrieve/format_context/build_sources, plus each has its own JSON schema and
deterministic renderer (parsed model output -> display text) defined here.
graph.py owns the actual LLM calls; this module is pure data/retrieval logic.
"""

import difflib
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DEFAULT_TOP_K

from app.rag import hybrid

NOT_FOUND_TOPIC_MESSAGE = "This topic wasn't found in the uploaded materials."

# A fixed absolute score threshold doesn't work here (tested: a deliberately
# off-topic query still scored 0.63-0.65 cosine, barely below genuine
# matches at 0.6-0.77 -- BGE similarity just doesn't spread out much on a
# small corpus). But *relative* to the same query's own top hit, a real
# match's padding chunks are reliably much weaker (observed ~0.016-0.018 vs
# a genuine match's ~0.032-0.033 RRF score) -- so drop anything scoring
# below half the top hit's score before it ever reaches the model, instead
# of asking the model to visually pick the signal out of noise itself
# (empirically unreliable: it sometimes missed a real match sitting right
# next to two irrelevant padding chunks).
RELATIVE_RELEVANCE_RATIO = 0.5

# Bare 2-4 letter acronyms embed poorly with a small model like
# bge-small-en-v1.5 -- tested directly against the real book: searching "NN"
# surfaced unrelated implementation chunks ("RandWireNN model modules",
# "PyTorch modules"), while "Neural Network" surfaced the actual conceptual
# chunk ("A refresher on deep learning"). Appending the expansion (not
# replacing the acronym) keeps BM25's exact-text match on the acronym itself
# intact while giving the dense side a fuller phrase to embed -- this only
# ever adds a matching signal, never removes one.
_ACRONYM_EXPANSIONS = {
    "NN": "Neural Network",
    "DNN": "Deep Neural Network",
    "CNN": "Convolutional Neural Network",
    "RNN": "Recurrent Neural Network",
    "LSTM": "Long Short-Term Memory",
    "GRU": "Gated Recurrent Unit",
    "GNN": "Graph Neural Network",
    "GAN": "Generative Adversarial Network",
    "MLP": "Multi-Layer Perceptron",
    "NLP": "Natural Language Processing",
    "RL": "Reinforcement Learning",
}
_ACRONYM_RE = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in _ACRONYM_EXPANSIONS) + r")s?\b",
    re.IGNORECASE,
)


def _expand_acronyms(query: str) -> str:
    matched = {m.group(1).upper() for m in _ACRONYM_RE.finditer(query)}
    if not matched:
        return query
    expansions = " ".join(_ACRONYM_EXPANSIONS[a] for a in matched)
    return f"{query} {expansions}"


def retrieve(query: str, top_k: int = DEFAULT_TOP_K, where: Optional[dict] = None) -> List[dict]:
    """Runs the query through the hybrid (dense + BM25 + RRF) retriever, then
    drops low-signal padding relative to the top hit (see RELATIVE_RELEVANCE_RATIO)."""
    hits = hybrid.search(_expand_acronyms(query), top_k=top_k, where=where)
    if not hits:
        return hits
    top_score = hits[0]["score"]
    if top_score <= 0:
        return hits
    return [h for h in hits if h["score"] >= top_score * RELATIVE_RELEVANCE_RATIO]


def format_context_for_prompt(chunks: List[dict]) -> str:
    """Numbers each retrieved chunk so it's unambiguous in testing/logs
    exactly what text the model had available to ground its answer in."""
    if not chunks:
        return "(No relevant content was found in the uploaded materials.)"
    blocks = []
    for i, c in enumerate(chunks, start=1):
        source = c["metadata"].get("source", "unknown source")
        blocks.append(f"[{i}] Source: {source}\n{c['text']}")
    return "\n\n".join(blocks)


def build_sources(chunks: List[dict]) -> List[str]:
    """Deduplicated, order-preserving list of human-readable source strings,
    built directly from retrieved-chunk metadata -- never from LLM output,
    so a citation can never be mis-transcribed or invented."""
    seen = set()
    sources = []
    for c in chunks:
        source = c["metadata"].get("source")
        if source and source not in seen:
            seen.add(source)
            sources.append(source)
    return sources


def resolve_reference_to_filter(reference: str) -> Optional[dict]:
    """Fuzzy-matches a free-text lecture/chapter reference (e.g. 'the
    aerology textbook', 'chapter 3') against what's actually indexed, so
    summarize can retrieve comprehensively from that one document instead of
    just the top-k nearest neighbors to a vague query. Returns a Chroma-style
    `where` filter, or None if nothing matched confidently enough."""
    if not reference:
        return None
    ref_lower = reference.lower().strip()

    files = hybrid.list_indexed_files()
    for f in files:
        stem = Path(f).stem.lower()
        if stem in ref_lower or ref_lower in stem:
            return {"file_name": f}

    chapters = hybrid.list_indexed_chapters()
    for c in chapters:
        if c and (c.lower() in ref_lower or ref_lower in c.lower()):
            return {"chapter": c}

    # "Lecture 8" / "Lecture_08.pptx" style references: match by the
    # lecture_number metadata directly rather than filename-string
    # similarity, which is fragile across spacing/leading-zero differences
    # ("Lecture 8" vs "Lecture_08" don't share a clean substring and may not
    # fuzzy-match reliably either).
    if any(kw in ref_lower for kw in ("lecture", "lec", "dars")):
        match = re.search(r"\d+", reference)
        if match:
            lecture_num = int(match.group())
            if lecture_num in hybrid.list_indexed_lecture_numbers():
                return {"lecture_number": lecture_num}

    close_files = difflib.get_close_matches(reference, files, n=1, cutoff=0.4)
    if close_files:
        return {"file_name": close_files[0]}

    close_chapters = difflib.get_close_matches(reference, [c for c in chapters if c], n=1, cutoff=0.4)
    if close_chapters:
        return {"chapter": close_chapters[0]}

    return None


def format_comparison_context(topic_a: str, chunks_a: List[dict], topic_b: str, chunks_b: List[dict]) -> str:
    """Labels chunks with a single flat numbering (so `used_chunk_numbers`
    integers stay meaningful) but visually separates which topic each one
    was retrieved for, via section headers."""
    blocks = [f"--- Chunks retrieved for topic A ('{topic_a}') ---"]
    blocks.append("(none found)" if not chunks_a else "")
    n = 0
    for c in chunks_a:
        n += 1
        source = c["metadata"].get("source", "unknown source")
        blocks.append(f"[{n}] Source: {source}\n{c['text']}")

    blocks.append(f"--- Chunks retrieved for topic B ('{topic_b}') ---")
    blocks.append("(none found)" if not chunks_b else "")
    for c in chunks_b:
        n += 1
        source = c["metadata"].get("source", "unknown source")
        blocks.append(f"[{n}] Source: {source}\n{c['text']}")

    return "\n\n".join(b for b in blocks if b)


# ---------------------------------------------------------------------------
# Per-tool JSON schemas and renderers.
#
# Every schema shares the same two-part discipline that fixed search_topic's
# worst failure mode (fully hallucinating an answer to an off-topic
# question): a forced per-chunk relevance_assessment field the model must
# fill BEFORE any answer content, and a used_chunk_numbers field the
# renderer trusts over whatever free-text the model produced -- if nothing
# was genuinely used, the renderer substitutes an honest refusal instead of
# whatever text the model wrote, the same safety net that mattered for
# search_topic's off-topic case.
# ---------------------------------------------------------------------------

_RELEVANCE_ASSESSMENT_FIELD = {
    "type": "string",
    "description": (
        "ALWAYS IN ENGLISH regardless of the request's language. For EACH numbered chunk "
        "above, exactly one short line: '<number>: RELEVANT' or '<number>: NOT_RELEVANT' -- "
        "nothing else. Do this before producing any other field."
    ),
}


def _used_chunk_numbers_field(description: str) -> dict:
    return {"type": "array", "items": {"type": "integer"}, "description": description}


# The schema asks for a strict per-chunk "<N>: RELEVANT"/"<N>: NOT_RELEVANT"
# line format, but the model doesn't always follow it -- sometimes it writes
# free-form prose instead. Found live, and severe: asked "How do I bake a
# cake?" (a deliberately off-topic anti-hallucination test), the model's own
# relevance_assessment said "While your request is about baking, it seems to
# be unrelated to our previous discussion... However, if you'd like me to
# provide instructions for making a cake ... I can certainly do that!" --
# explicitly admitting irrelevance in plain English, then answering from its
# own general knowledge anyway and citing real (but unrelated) chunk numbers
# as if they supported it. None of the existing checks (citation range,
# empty answer) catch this: the JSON was well-formed, the chunk numbers were
# valid, and the answer was non-empty. Detecting the self-admission directly
# in the free text is the only signal available for this specific failure.
_SELF_ADMITTED_IRRELEVANCE_RE = re.compile(
    r"\b(unrelated|not related|doesn'?t relate|does not relate|not covered|isn'?t covered|"
    r"doesn'?t discuss|does not discuss|no relevant|not (?:directly )?found|not mentioned|"
    r"outside (?:the scope|of the scope)|does not contain|doesn'?t contain)\b",
    re.IGNORECASE,
)


# Self-admission catches the case where the model happens to phrase its own
# doubt in recognizable English, but that's inherently probabilistic --
# GPU floating-point non-determinism means the SAME off-topic question can
# get a clean-sounding assessment on one call and an admission on another
# (confirmed directly: "How do I bake a cake?" refused correctly on 1 of 3
# runs and hallucinated a full recipe, with real-looking but unrelated
# citations, on the other 2). A lexical-overlap check against the actual
# cited text is a second, more reliable layer: it doesn't depend on the
# model saying anything in particular, just on whether the chunk it claims
# to be grounded in shares ANY meaningful term with what was actually asked.
# Confirmed directly: neither chunk cited for the cake-recipe hallucination
# contained "bake" or "cake" anywhere in its text.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "for", "to", "and", "or",
    "what", "how", "do", "does", "did", "i", "you", "your", "me", "please", "tell", "about",
    "this", "that", "these", "those", "with", "using", "use", "explain", "give", "generate",
    "create", "make", "can", "could", "would", "should", "will", "its", "compare", "vs",
}
_WORD_RE = re.compile(r"[a-zA-Z]{4,}")


def _significant_tokens(text) -> set:
    return {w for w in _WORD_RE.findall(str(text or "").lower()) if w not in _STOPWORDS}


def _query_tokens_from_args(tool_args: Optional[dict]) -> set:
    if not tool_args:
        return set()
    joined = " ".join(str(v) for v in tool_args.values() if isinstance(v, str))
    return _significant_tokens(joined)


def _resolve_used_chunks(parsed: dict, retrieved_chunks: List[dict], tool_args: Optional[dict] = None) -> List[dict]:
    used_numbers = parsed.get("used_chunk_numbers")
    if not isinstance(used_numbers, list):
        return []

    if _SELF_ADMITTED_IRRELEVANCE_RE.search(str(parsed.get("relevance_assessment") or "")):
        # The model's own free-text assessment contradicts continuing to
        # answer -- trust that admission over whatever it separately listed
        # as "used", even though the citation-number fallback below would
        # otherwise treat this as a normal, well-formed response.
        return []

    resolved = [retrieved_chunks[i - 1] for i in used_numbers if isinstance(i, int) and 1 <= i <= len(retrieved_chunks)]
    if not resolved and used_numbers:
        # An explicit empty list ([]) means the model assessed nothing as
        # relevant -- a genuine refusal signal, kept as an empty result. But
        # a NON-empty list where every number is out of range for the actual
        # context size is a citation-numbering slip, not a refusal: found via
        # a real case where the model wrote a fully correct, well-grounded
        # explanation yet cited chunk 9 when only 5 chunks were retrieved.
        # Discarding a good answer over a wrong index number is worse than
        # citing the whole retrieved set, so fall back to that instead of an
        # empty list (same principle already used for malformed JSON output).
        resolved = list(retrieved_chunks)

    query_tokens = _query_tokens_from_args(tool_args)
    if query_tokens and resolved:
        has_overlap = any(query_tokens & _significant_tokens(c["text"]) for c in resolved)
        if not has_overlap:
            return []
    return resolved


# --- search_topic ---

SEARCH_TOPIC_SCHEMA = {
    "type": "object",
    "properties": {
        "relevance_assessment": _RELEVANCE_ASSESSMENT_FIELD,
        "answer": {
            "type": "string",
            "description": (
                "The natural-language answer to the student's question, in the SAME language "
                "the question was asked in (e.g. a question in Uzbek gets an answer in Uzbek)."
            ),
        },
        "used_chunk_numbers": _used_chunk_numbers_field(
            "Numbers of the retrieved chunks actually relevant and used. Empty if none were relevant."
        ),
    },
    "required": ["relevance_assessment", "answer", "used_chunk_numbers"],
}


def render_search_topic(parsed: dict, retrieved_chunks: List[dict], tool_args: dict) -> Tuple[str, List[dict]]:
    cited_chunks = _resolve_used_chunks(parsed, retrieved_chunks, tool_args)
    answer = str(parsed.get("answer", "")).strip()
    # used_chunk_numbers claiming something was relevant but answer being
    # blank is a contradiction, not a real answer -- found via a real off-topic
    # test ("What is the capital of France?") where the model marked an
    # unrelated GoogLeNet chunk as used yet left answer empty, which used to
    # render as a blank response with a bogus citation instead of an honest
    # refusal.
    if not cited_chunks or not answer:
        return NOT_FOUND_TOPIC_MESSAGE, []
    return answer, cited_chunks


# --- generate_quiz ---

QUIZ_SCHEMA = {
    "type": "object",
    "properties": {
        "relevance_assessment": _RELEVANCE_ASSESSMENT_FIELD,
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 4},
                    "correct_option_index": {"type": "integer", "description": "0-based index into options."},
                    "explanation": {"type": "string"},
                    "source_chunk_number": {"type": "integer", "description": "Which numbered chunk this question is grounded in."},
                },
                "required": ["question", "options", "correct_option_index", "explanation", "source_chunk_number"],
            },
            "description": (
                "One entry per question, each grounded in a specific numbered chunk. If there isn't "
                "enough distinct, well-grounded material for the requested number of questions, "
                "include fewer rather than inventing ones the materials don't support. Empty if the "
                "topic isn't covered at all."
            ),
        },
    },
    "required": ["relevance_assessment", "questions"],
}


_OPTION_LETTER_PREFIX_RE = re.compile(r"^\s*[A-Da-d]\s*[.):]\s*")


def _strip_redundant_letter_prefix(option_text: str) -> str:
    """The model sometimes writes its own letter prefix into the option
    string (e.g. "A. Both LSTM and GRU..."), which then duplicates with the
    renderer's own lettering -- rendering as "A. A. Both LSTM...". Observed
    directly in real quiz output, not assumed."""
    return _OPTION_LETTER_PREFIX_RE.sub("", option_text, count=1)


def render_quiz(parsed: dict, retrieved_chunks: List[dict], tool_args: dict) -> Tuple[str, List[dict]]:
    if _SELF_ADMITTED_IRRELEVANCE_RE.search(str(parsed.get("relevance_assessment") or "")):
        return NOT_FOUND_TOPIC_MESSAGE, []

    questions = parsed.get("questions")
    if not isinstance(questions, list):
        questions = []

    blocks: List[str] = []
    answer_key: List[str] = []
    cited_chunks: List[dict] = []
    valid_count = 0
    for q in questions:
        if not isinstance(q, dict):
            continue
        question_text = str(q.get("question", "")).strip()
        options = q.get("options")
        correct_idx = q.get("correct_option_index")
        if not question_text or not isinstance(options, list) or len(options) != 4:
            continue
        if not isinstance(correct_idx, int) or not (0 <= correct_idx <= 3):
            continue
        # Observed directly in real output: the model occasionally leaves an
        # option blank (e.g. a lone "D." with no text) instead of properly
        # filling all 4 -- reject the whole question rather than render a
        # visibly broken empty bullet.
        if any(not str(opt).strip() for opt in options):
            continue

        valid_count += 1
        # Rendered via st.markdown, which is CommonMark: a lone "\n" is
        # collapsed into a space, not a line break, so joining the question/
        # options/explanation with plain newlines (the previous approach)
        # rendered as one dense run-on paragraph. Each option as its own
        # "-" bullet makes markdown treat it as a genuine separate list
        # item instead of relying on manual indentation that gets collapsed.
        correct_letter = chr(ord("A") + correct_idx)
        option_lines = []
        for j, opt in enumerate(options):
            letter = chr(ord("A") + j)
            marker = " *(correct)*" if j == correct_idx else ""
            opt_text = _strip_redundant_letter_prefix(str(opt).strip())
            option_lines.append(f"- {letter}. {opt_text}{marker}")
        block = [f"**{valid_count}. {question_text}**", "\n".join(option_lines)]
        explanation = str(q.get("explanation", "")).strip()
        if explanation:
            block.append(f"*Explanation: {explanation}*")
        blocks.append("\n\n".join(block))
        answer_key.append(f"{valid_count}-{correct_letter}")

        src_num = q.get("source_chunk_number")
        if isinstance(src_num, int) and 1 <= src_num <= len(retrieved_chunks):
            cited_chunks.append(retrieved_chunks[src_num - 1])

    if valid_count == 0:
        return NOT_FOUND_TOPIC_MESSAGE, []
    # Every content-generating tool must retrieve fresh context per turn
    # (SPEC's hard requirement), so a follow-up like "give me the answers"
    # can't just recall the prior quiz -- retrieval finds nothing for a
    # question about the conversation itself, not the course material, and
    # correctly (from that tool's perspective) comes back "not found". The
    # real fix is making the quiz answer itself never require a follow-up:
    # a compact key up front, in addition to the inline "(correct)" markers.
    header = f"**Answer key:** {', '.join(answer_key)}"
    body = "\n\n---\n\n".join(blocks).strip()
    return f"{header}\n\n---\n\n{body}", cited_chunks


# --- compare_topics ---

COMPARE_TOPICS_SCHEMA = {
    "type": "object",
    "properties": {
        "relevance_assessment": _RELEVANCE_ASSESSMENT_FIELD,
        "topic_a_covered": {"type": "boolean", "description": "Whether the materials actually cover topic A."},
        "topic_b_covered": {"type": "boolean", "description": "Whether the materials actually cover topic B."},
        "comparison_rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "aspect": {"type": "string"},
                    "topic_a": {
                        "type": "string",
                        "description": "Must be actual content describing topic A for this aspect -- never leave this blank if topic_a_covered is true.",
                    },
                    "topic_b": {
                        "type": "string",
                        "description": "Must be actual content describing topic B for this aspect -- never leave this blank if topic_b_covered is true.",
                    },
                },
                "required": ["aspect", "topic_a", "topic_b"],
            },
            "description": "Rows of the comparison table. Empty if neither topic is covered.",
        },
        "used_chunk_numbers": _used_chunk_numbers_field("Numbers of chunks (from either topic's list) actually used."),
    },
    "required": ["relevance_assessment", "topic_a_covered", "topic_b_covered", "comparison_rows", "used_chunk_numbers"],
}


_PLACEHOLDER_CELL_VALUES = {"n/a", "na", "-", "none", "not covered", "not applicable", "not found", ""}


def _has_real_cell_content(value) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.strip("-—").strip().lower() not in _PLACEHOLDER_CELL_VALUES


def _merge_duplicate_aspect_rows(rows: List[dict]) -> List[dict]:
    """The model sometimes writes SEPARATE rows for the same aspect instead
    of one row with both columns filled in (e.g. an "Architecture" row with
    only topic_a content, then a LATER "Architecture" row with only topic_b
    content) -- observed directly in real output, rendering as visibly
    duplicated rows with half of each blank. Merges rows sharing the same
    (trimmed, case-insensitive) aspect name, keeping whichever value has
    real content for each column."""
    merged: dict = {}
    order: List[str] = []
    for r in rows:
        aspect = str(r.get("aspect", "")).strip()
        key = aspect.lower()
        if key not in merged:
            merged[key] = {"aspect": aspect, "topic_a": r.get("topic_a", ""), "topic_b": r.get("topic_b", "")}
            order.append(key)
        else:
            existing = merged[key]
            if not _has_real_cell_content(existing["topic_a"]) and _has_real_cell_content(r.get("topic_a")):
                existing["topic_a"] = r.get("topic_a", "")
            if not _has_real_cell_content(existing["topic_b"]) and _has_real_cell_content(r.get("topic_b")):
                existing["topic_b"] = r.get("topic_b", "")
    return [merged[k] for k in order]


def render_compare_topics(parsed: dict, retrieved_chunks: List[dict], tool_args: dict) -> Tuple[str, List[dict]]:
    cited_chunks = _resolve_used_chunks(parsed, retrieved_chunks, tool_args)
    topic_a, topic_b = tool_args.get("topic_a", "topic A"), tool_args.get("topic_b", "topic B")
    rows = parsed.get("comparison_rows")
    valid_rows = [r for r in rows if isinstance(r, dict) and r.get("aspect")] if isinstance(rows, list) else []
    valid_rows = _merge_duplicate_aspect_rows(valid_rows)
    # A row that ends up with neither column carrying real content (e.g. an
    # asymmetric aspect name like "Disadvantages of GANs (as compared with
    # diffusion models)" that never got a matching value from either topic)
    # is just clutter -- drop it rather than show an all-placeholder row.
    valid_rows = [r for r in valid_rows if _has_real_cell_content(r.get("topic_a")) or _has_real_cell_content(r.get("topic_b"))]

    # topic_a_covered/topic_b_covered are the model's own self-reported
    # booleans, and they're unreliable in a specific, reproducible way: the
    # model would write a genuinely correct, substantive row value for a
    # topic (e.g. real LSTM/RNN content) and STILL report that topic's
    # covered flag as false, causing every real answer for that column to
    # be discarded and replaced with "-" -- confirmed directly on "Compare
    # CNN and RNN" and "Compare LSTM and GRU", where topic B's real content
    # was blanked out despite being present in the raw output. Deriving
    # coverage from whether any row actually has real (non-placeholder)
    # content is the same "never trust a free-form claim when a
    # deterministic check is possible" principle already used for
    # citations, applied here to the coverage decision itself.
    topic_a_covered = any(_has_real_cell_content(r.get("topic_a")) for r in valid_rows)
    topic_b_covered = any(_has_real_cell_content(r.get("topic_b")) for r in valid_rows)

    if not valid_rows or (not topic_a_covered and not topic_b_covered):
        return NOT_FOUND_TOPIC_MESSAGE, []

    lines = []
    if not topic_a_covered:
        lines.append(f"Note: '{topic_a}' wasn't found in the uploaded materials -- showing '{topic_b}' only.\n")
    elif not topic_b_covered:
        lines.append(f"Note: '{topic_b}' wasn't found in the uploaded materials -- showing '{topic_a}' only.\n")

    lines.append(f"| Aspect | {topic_a} | {topic_b} |")
    lines.append("|---|---|---|")
    for r in valid_rows:
        a_val = str(r.get("topic_a", "")).strip() if _has_real_cell_content(r.get("topic_a")) else "—"
        b_val = str(r.get("topic_b", "")).strip() if _has_real_cell_content(r.get("topic_b")) else "—"
        lines.append(f"| {str(r.get('aspect', '')).strip()} | {a_val} | {b_val} |")

    return "\n".join(lines).strip(), cited_chunks


# --- summarize ---

SUMMARIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevance_assessment": _RELEVANCE_ASSESSMENT_FIELD,
        "summary": {
            "type": "string",
            "description": (
                "A summary of the retrieved material, in the SAME language the request was made in. "
                "Empty string if none of the retrieved chunks are actually about the requested lecture/chapter."
            ),
        },
        "used_chunk_numbers": _used_chunk_numbers_field("Numbers of chunks actually used in the summary."),
    },
    "required": ["relevance_assessment", "summary", "used_chunk_numbers"],
}


def render_summarize(parsed: dict, retrieved_chunks: List[dict], tool_args: dict) -> Tuple[str, List[dict]]:
    cited_chunks = _resolve_used_chunks(parsed, retrieved_chunks, tool_args)
    summary = str(parsed.get("summary", "")).strip()
    if not cited_chunks or not summary:
        return NOT_FOUND_TOPIC_MESSAGE, []
    reference = str(tool_args.get("reference", "")).strip()
    header = f"Summary of {reference}:\n\n" if reference else ""
    return header + summary, cited_chunks


# --- explain_difficult_concept ---

EXPLAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "relevance_assessment": _RELEVANCE_ASSESSMENT_FIELD,
        "simple_explanation": {
            "type": "string",
            "description": "A simplified explanation using only facts from the retrieved context, in the student's language.",
        },
        "example": {
            "type": "string",
            "description": (
                "An illustrative example that applies the grounded facts above -- it may construct a "
                "concrete scenario to aid understanding, but must not introduce new factual claims "
                "about the concept that aren't supported by the retrieved context."
            ),
        },
        "used_chunk_numbers": _used_chunk_numbers_field("Numbers of chunks actually used."),
    },
    "required": ["relevance_assessment", "simple_explanation", "example", "used_chunk_numbers"],
}


def render_explain(parsed: dict, retrieved_chunks: List[dict], tool_args: dict) -> Tuple[str, List[dict]]:
    cited_chunks = _resolve_used_chunks(parsed, retrieved_chunks, tool_args)
    explanation = str(parsed.get("simple_explanation", "")).strip()
    example = str(parsed.get("example", "")).strip()
    parts = [p for p in (explanation, f"Example: {example}" if example else "") if p]
    if not cited_chunks or not parts:
        return NOT_FOUND_TOPIC_MESSAGE, []
    return "\n\n".join(parts), cited_chunks


# --- generate_assignment ---

_PLAIN_TEXT_FIELD_NOTE = (
    "Plain natural-language text only -- no HTML tags (e.g. no <p>/<strong>), "
    "no markdown code fences, and no JSON/dict-style formatting (e.g. no "
    '{"key": "value"} wrapper). Write it exactly as a student would read it.'
)

ASSIGNMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "relevance_assessment": _RELEVANCE_ASSESSMENT_FIELD,
        "assignment_title": {"type": "string", "description": f"A short title for the assignment. {_PLAIN_TEXT_FIELD_NOTE}"},
        "instructions": {"type": "string", "description": f"Brief instructions for the student. {_PLAIN_TEXT_FIELD_NOTE}"},
        "tasks": {
            "type": "array",
            "items": {"type": "string", "description": f"One task/question as its own plain-text string. {_PLAIN_TEXT_FIELD_NOTE}"},
            "description": "Numbered tasks/questions for the assignment, grounded in the retrieved material. Empty if the topic isn't covered.",
        },
        "used_chunk_numbers": _used_chunk_numbers_field("Numbers of chunks actually used."),
    },
    "required": ["relevance_assessment", "assignment_title", "instructions", "tasks", "used_chunk_numbers"],
}

_HTML_TAG_RE = re.compile(r"<[^>]+>")


_DICT_LIKE_FIELD_RE = re.compile(r"""['"][^'"]*['"]\s*:\s*['"]([^'"]*)['"]""")


def _sanitize_assignment_text(value) -> str:
    """Defense in depth beyond the schema's plain-text instruction: the
    model sometimes still writes raw HTML tags or a stringified dict/JSON
    fragment into what should be a plain string field -- observed directly
    in real output, in two different quoting styles: double-quoted JSON
    ('{"description": "..."}', rendering as literally "1. {") and
    single-quoted Python dict repr ("{ 'task_number':1,'description':'...'}").
    Strip HTML tags, and if the value still looks like a bare dict/JSON
    fragment, pull out the LAST quoted field's value (task text is
    consistently the final key in both observed shapes) rather than
    showing the raw braces/quotes."""
    text = str(value or "").strip()
    text = _HTML_TAG_RE.sub("", text).strip()
    if text.startswith("{") or text.startswith("["):
        matches = _DICT_LIKE_FIELD_RE.findall(text)
        text = matches[-1] if matches else text.strip("{}[]\"' ")
    text = re.sub(r"^>\s*", "", text.strip())
    return text.strip()


def render_assignment(parsed: dict, retrieved_chunks: List[dict], tool_args: dict) -> Tuple[str, List[dict]]:
    cited_chunks = _resolve_used_chunks(parsed, retrieved_chunks, tool_args)
    tasks = parsed.get("tasks")
    if not isinstance(tasks, list):
        tasks = []
    sanitized_tasks = [_sanitize_assignment_text(t) for t in tasks]
    sanitized_tasks = [t for t in sanitized_tasks if t]
    if not cited_chunks or not sanitized_tasks:
        return NOT_FOUND_TOPIC_MESSAGE, []

    title = _sanitize_assignment_text(parsed.get("assignment_title", ""))
    instructions = _sanitize_assignment_text(parsed.get("instructions", ""))
    lines = []
    if title:
        lines.append(f"# {title}\n")
    if instructions:
        lines.append(f"{instructions}\n")
    for i, t in enumerate(sanitized_tasks, start=1):
        lines.append(f"{i}. {t}")
    return "\n".join(lines).strip(), cited_chunks


# Registry graph.py dispatches on, keyed by tool name. Falls back to
# search_topic's entry when no tool was called (forced-retrieval path).
TOOL_SPECS = {
    "search_topic": {"schema": SEARCH_TOPIC_SCHEMA, "render": render_search_topic},
    "generate_quiz": {"schema": QUIZ_SCHEMA, "render": render_quiz},
    "compare_topics": {"schema": COMPARE_TOPICS_SCHEMA, "render": render_compare_topics},
    "summarize": {"schema": SUMMARIZE_SCHEMA, "render": render_summarize},
    "explain_difficult_concept": {"schema": EXPLAIN_SCHEMA, "render": render_explain},
    "generate_assignment": {"schema": ASSIGNMENT_SCHEMA, "render": render_assignment},
}

"""Tool schema + implementations for Ollama native tool-calling.

Every tool must retrieve context via the hybrid retriever before generating
anything (SPEC.md's "RAG must actually be used for every content-generating
tool") -- each tool function here does exactly that retrieval and nothing
else; graph.py is responsible for feeding the retrieved context to the LLM
for the actual generation step, using the matching schema/renderer from
nodes.TOOL_SPECS.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from config import BROAD_TOP_K, QUIZ_TOP_K, SUMMARIZE_TOP_K

from app.agent.nodes import (
    format_comparison_context,
    format_context_for_prompt,
    resolve_reference_to_filter,
    retrieve,
)
from app.rag import hybrid


def search_topic(topic: str) -> dict:
    """Retrieves relevant chunks on a topic from the student's uploaded
    materials. Returns the formatted context (for the LLM to answer from)
    and the raw chunks (for the Sources list and Tool Logs). Reads
    config.DEFAULT_TOP_K live (not import-time bound) so the Settings UI's
    top-k slider actually takes effect without restarting the app."""
    chunks = retrieve(topic, top_k=config.DEFAULT_TOP_K)
    return {"context": format_context_for_prompt(chunks), "chunks": chunks}


def generate_quiz(topic: str, num_questions: int = 5) -> dict:
    """Retrieves broadly on a topic so there's enough distinct material for
    several non-repetitive quiz questions."""
    chunks = retrieve(topic, top_k=QUIZ_TOP_K)
    return {"context": format_context_for_prompt(chunks), "chunks": chunks}


def generate_assignment(topic: str) -> dict:
    """Retrieves broadly on a topic to synthesize a new assignment from
    whatever relevant material is available (no reliable metadata flag
    distinguishes 'assignment-type' documents from lectures/books)."""
    chunks = retrieve(topic, top_k=BROAD_TOP_K)
    return {"context": format_context_for_prompt(chunks), "chunks": chunks}


def compare_topics(topic_a: str, topic_b: str) -> dict:
    """Retrieves each topic separately and labels the combined context so
    the model can judge each chunk's relevance against its own topic."""
    chunks_a = retrieve(topic_a, top_k=BROAD_TOP_K)
    chunks_b = retrieve(topic_b, top_k=BROAD_TOP_K)
    context = format_comparison_context(topic_a, chunks_a, topic_b, chunks_b)
    return {"context": context, "chunks": chunks_a + chunks_b}


def summarize(reference: str) -> dict:
    """Resolves `reference` to a specific indexed file/chapter if possible
    (comprehensive coverage of that one document, not top-k nearest
    neighbors to a vague query); falls back to a broad open-topic search if
    no confident match is found.

    SUMMARIZE_TOP_K=20 only makes sense once a filter has narrowed the pool
    to one specific document -- without a filter, retrieval is an unfiltered
    similarity search over the whole KB, and 20 mixed candidates is enough
    noise that the model can genuinely fail to notice a real match sitting
    among them (found live: asked to summarize "the section on bidirectional
    RNNs", the model hallucinated a reference ("Chapter 7 - Recurrent Neural
    Networks") that didn't resolve to a filter, got a real "Bidirectional
    RNNs" chunk back at position 3 of 20, and still concluded the topic
    wasn't covered because it was reasoning over GCN/GAT chunks elsewhere in
    the same batch instead). Falling back to BROAD_TOP_K in the unfiltered
    case keeps the candidate pool the same size proven to work for the other
    broad-search tools (compare_topics, generate_assignment).

    Once a filter DOES match, though, similarity search is still the wrong
    tool -- it ranks the file-scoped chunks by similarity to the literal
    query text, which clusters around whatever one topic textually
    resembles that wording rather than surveying the matched scope's actual
    breadth. Found live: "summarize Mastering Pytorch 2ed.pdf" kept
    returning a summary of one narrow ONNX-export section -- with a
    DIFFERENT page citation on each separate run, since the exact cluster
    boundary shifts with GPU floating-point non-determinism -- instead of
    covering the whole 559-page book. hybrid.sample_spread instead spreads
    the sample evenly across the matched scope's page/slide order, which is
    what "summarize this whole document" actually needs."""
    where = resolve_reference_to_filter(reference)
    if where is not None:
        chunks = hybrid.sample_spread(where, SUMMARIZE_TOP_K)
    else:
        chunks = retrieve(reference, top_k=BROAD_TOP_K)
    return {"context": format_context_for_prompt(chunks), "chunks": chunks, "matched_reference": where is not None}


def explain_difficult_concept(concept: str) -> dict:
    """Retrieves context on a concept to ground a simplified explanation."""
    chunks = retrieve(concept, top_k=config.DEFAULT_TOP_K)
    return {"context": format_context_for_prompt(chunks), "chunks": chunks}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_topic",
            "description": (
                "Search the student's uploaded lecture slides, books, and assignments "
                "for content relevant to a topic or question. Use this for general "
                "questions about course content -- never answer from your own knowledge alone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic or question to search for in the uploaded materials.",
                    }
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_quiz",
            "description": (
                "Search the uploaded materials for a topic and generate multiple-choice quiz "
                "questions from them. Use this when the student asks to be quizzed or tested "
                "on a topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The topic to quiz the student on."},
                    "num_questions": {
                        "type": "integer",
                        "description": "How many questions to generate (default 5).",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_assignment",
            "description": (
                "Search the uploaded materials for a topic and synthesize a new practice "
                "assignment from them. Use this when the student asks for practice problems, "
                "exercises, or a new assignment on a topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The topic to build the assignment around."}
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_topics",
            "description": (
                "Search the uploaded materials for two topics and compare them as a table. "
                "Use this when the student asks to compare, contrast, or see the difference "
                "between two things."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_a": {"type": "string", "description": "The first topic to compare."},
                    "topic_b": {"type": "string", "description": "The second topic to compare."},
                },
                "required": ["topic_a", "topic_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": (
                "Retrieve and summarize a specific lecture, chapter, or document from the "
                "uploaded materials. Use this when the student asks for a summary of a "
                "named lecture/chapter/book, not a general topic question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reference": {
                        "type": "string",
                        "description": "The lecture, chapter, or document name to summarize (e.g. 'Lecture 5' or the book title).",
                    }
                },
                "required": ["reference"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_difficult_concept",
            "description": (
                "Search the uploaded materials for a concept and produce a simplified "
                "explanation with an example. Use this when the student says something is "
                "confusing or asks for a simple/easy explanation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string", "description": "The concept to explain simply."}
                },
                "required": ["concept"],
            },
        },
    },
]

FUNCTIONS = {
    "search_topic": search_topic,
    "generate_quiz": generate_quiz,
    "generate_assignment": generate_assignment,
    "compare_topics": compare_topics,
    "summarize": summarize,
    "explain_difficult_concept": explain_difficult_concept,
}

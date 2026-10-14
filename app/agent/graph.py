"""Agent orchestration across all 6 tools.

Flow: (1) let the model decide whether/how to call one of the 6 tools via
native tool-calling; (2) regardless of whether it cooperated, retrieve
context (the model's own tool call if it made one, or a forced search_topic
retrieval using the raw user message if it didn't -- SPEC.md's "RAG must
actually be used for every content-generating tool" is a hard requirement,
not a suggestion the model can skip); (3) generate the final output using
whichever tool's schema/renderer matches what was invoked (nodes.TOOL_SPECS),
from a single plain user message containing that tool's own formatted
context, so the follow-up call never depends on strict tool-role
message-protocol semantics; (4) append a deterministically-built Sources
list -- never LLM-authored.

Only the FIRST tool call in a response is executed: each of these 6 tools is
meant to fully handle one turn on its own (compare_topics does two
retrievals internally, for instance), so honoring multiple different-tool
calls in one response would produce mismatched, hard-to-reconcile numbering
between the context shown and citations resolved from it.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from config import DEFAULT_TOP_K

import ollama

from app.agent import nodes
from app.agent.nodes import build_sources, format_context_for_prompt, retrieve
from app.agent.tools import FUNCTIONS, TOOLS
from app.rag import hybrid

# A bounded but generous num_predict, not unlimited: Ollama's default cut off
# a genuinely long bulleted answer in earlier testing, but truly unbounded
# generation (-1) let free-text generation ramble into drift -- a fixed
# ceiling avoids the truncation failure without inviting the other one.
# 900 turned out to still be too tight once multi-tool schemas landed --
# generate_quiz with 3 questions (each with 4 options + an explanation, on
# top of the required relevance_assessment field) got cut off mid-JSON,
# which throws away the whole structured response (json.loads fails on
# truncated JSON, falling back to showing the raw broken text) rather than
# just clipping one field. 2048 covers a multi-question quiz / a several-row
# comparison table / an assignment with several tasks with real headroom.
# Temperature 0 (greedy decoding, no sampling) rather than 0.2: identical
# questions were coming back with different answers AND different cited
# sources across repeated runs -- retrieval itself is fully deterministic
# (same query -> same dense/BM25/RRF ranking every time), so that variance
# was purely from LLM sampling: a different sample changes both the exact
# tool-call arguments generated (shifting the retrieval query slightly) and
# which chunks get marked as "used" in used_chunk_numbers (changing cited
# sources even when the retrieved pool was identical). A fixed seed is
# redundant at temperature 0 (there's no sampling left to seed) but costs
# nothing and guards against any backend defaulting away from pure greedy.
#
# repeat_penalty is NOT optional at temperature 0, and this was found by
# testing, not assumed: pure greedy decoding once got stuck counting
# used_chunk_numbers sequentially up to 413, blew the 2048-token budget, and
# produced unparseable truncated JSON that leaked to the user as raw text.
# This is a well-documented failure mode -- greedy decoding has no
# randomness to escape a repetitive high-probability continuation once it
# starts, unlike sampling at temperature > 0. repeat_penalty discourages
# repeating recent tokens independently of temperature, which fixes this
# without reintroducing the answer/citation non-determinism the zero
# temperature was specifically chosen to eliminate.
_GENERATION_OPTIONS = {"num_predict": 2048, "temperature": 0.0, "seed": 42, "repeat_penalty": 1.3}


def _chat(messages: list, tools: list = None, format_schema: dict = None,
          provider: str = None, api_key: str = None) -> dict:
    """Provider-agnostic chat call. ALWAYS returns Ollama's own response
    shape ({"message": {"content": ..., "tool_calls": [...]}}) regardless
    of which provider actually served it, so every other line in this file
    (response["message"], .get("tool_calls"), .get("content")) needs no
    changes -- only the two call sites below became _chat(...) instead of
    ollama.chat(...).

    provider defaults to config.LLM_PROVIDER ("ollama"). The "gemini" path
    calls Gemini's OpenAI-compatible endpoint directly via `requests`
    (matching ai-data-analysis-agent's GeminiLLM pattern -- no extra
    provider-specific package) and normalizes its OpenAI-shaped response
    back to Ollama's shape.

    IMPORTANT CAVEAT: _GENERATION_OPTIONS' `seed` and `repeat_penalty` are
    Ollama/llama.cpp-specific decoding parameters with no OpenAI-API
    equivalent -- only `temperature` carries over to the Gemini path. The
    repeat_penalty was added after a real observed failure (greedy
    decoding got stuck counting to 413 and blew the token budget); that
    specific guard does NOT apply on Gemini. See README.
    """
    provider = provider or config.LLM_PROVIDER
    if provider != "gemini":
        kwargs = {}
        if tools is not None:
            kwargs["tools"] = tools
        if format_schema is not None:
            kwargs["format"] = format_schema
        return ollama.chat(model=config.OLLAMA_MODEL, messages=messages, options=_GENERATION_OPTIONS, **kwargs)

    key = api_key or config.GEMINI_API_KEY
    if not key:
        raise ValueError(
            "No Gemini API key provided. Paste one in the sidebar, or set "
            "GEMINI_API_KEY in your .env for local use."
        )

    import requests

    payload = {
        "model": config.GEMINI_MODEL,
        "messages": messages,
        "temperature": _GENERATION_OPTIONS["temperature"],
    }
    if tools is not None:
        payload["tools"] = tools
    if format_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": format_schema, "strict": True},
        }

    resp = requests.post(
        f"{config.GEMINI_BASE_URL}chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:500]}")

    choice_msg = resp.json()["choices"][0]["message"]

    # Normalize tool_calls: Ollama's arguments are already a dict; OpenAI/
    # Gemini's arguments field is a JSON *string* that must be parsed.
    normalized_calls = []
    for tc in (choice_msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        raw_args = fn.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = {}
        normalized_calls.append({"function": {"name": fn.get("name"), "arguments": args}})

    return {"message": {"content": choice_msg.get("content"), "tool_calls": normalized_calls}}


SYSTEM_PROMPT = """You are an AI teaching assistant helping a university student study from their own uploaded course materials (lecture slides, textbooks, assignments).

You have 6 tools. Call the one matching what the student actually wants:
- search_topic: general questions about course content.
- generate_quiz: the student asks to be quizzed/tested on a topic.
- generate_assignment: the student asks for practice problems, exercises, or a new assignment.
- compare_topics: the student asks to compare, contrast, or see the difference between two things.
- summarize: the student asks for a summary of a specific named lecture/chapter/document.
- explain_difficult_concept: the student says something is confusing or asks for a simple/easy explanation.
Always call one of these before producing any content -- never answer from your own knowledge alone.

Rules you must always follow once you have retrieved context:
1. Base what you produce strictly on the retrieved context you are given. Do not use outside knowledge, even if you are confident it is correct.
2. The retrieved context is numbered chunks ([1], [2], ...) and may be a MIX of genuinely relevant and irrelevant material -- retrieval only returns its best guesses, not guaranteed matches, and sometimes NONE of them are actually relevant. Before producing anything else, go through the chunks one by one and honestly assess whether each one actually addresses the request -- a chunk being retrieved is not evidence it's relevant. A chunk that is the closest available match is still NOT_RELEVANT if it is fundamentally about a different subject than what was asked (e.g. a chunk about a measuring instrument is NOT_RELEVANT to a question about a computing concept, even if both are "technical"). Then use only the chunks you assessed as relevant; ignore the rest entirely, even if nothing else was retrieved for you to go on.
3. If your own assessment found that NONE of the chunks actually address the request, say so plainly (or leave the relevant list/fields empty, per the schema) rather than guessing or fabricating content just because chunks about some other subject were retrieved. This applies even to fields like "example" -- do not construct an example for a concept the materials don't actually cover.
4. If asked to list or enumerate things (e.g. "what are the layers of X"), only include items the retrieved context explicitly names or describes. Do not add other items from your own general knowledge to complete a list, even ones you're confident are correct -- if the context only covers some items, give only those and say the materials don't go into the rest.
5. Do not write your own "Sources:" section -- one is added automatically afterward.
6. Respond in the same language the student used.
7. Keep output clear and focused.
"""

NOT_FOUND_MESSAGE = (
    "No materials have been uploaded yet, so there's nothing to search. "
    "Please upload lecture slides, a book, or an assignment first."
)

# Qwen2.5 at this quantization occasionally drifts into Chinese mid-answer
# for longer/more complex generations, regardless of the target language
# instruction -- a known quirk (heavy Chinese-weighted training data), not
# something a prompt alone reliably eliminates. Since the SAME question
# sometimes drifts and sometimes doesn't across separate calls (observed
# empirically), it's not deterministic -- so detect it and retry once rather
# than either accepting a wrong-language answer or engineering a prompt fix
# that can't fully guarantee compliance from a quantized model anyway.
_CJK_RE = re.compile(r"[一-鿿぀-ヿ가-힯]")


def _is_language_mismatch(user_message: str, answer_text: str) -> bool:
    if not answer_text or _CJK_RE.search(user_message):
        return False
    cjk_count = len(_CJK_RE.findall(answer_text))
    return cjk_count > max(10, len(answer_text) * 0.15)


# Observed live: qwen2.5 at Q4_K_M occasionally echoes Jinja-style templating
# boilerplate ({% raw %}/{% endraw %}) at the start of an otherwise-correct
# answer -- training-data noise, not something a prompt instruction reliably
# suppresses. Stripped defensively since it should never legitimately appear
# in a teaching-assistant answer.
_TEMPLATE_ARTIFACT_RE = re.compile(r"\{%-?\s*(?:end)?raw\s*-?%\}")


def _strip_template_artifacts(answer_text: str) -> str:
    cleaned = _TEMPLATE_ARTIFACT_RE.sub(" ", answer_text)
    return re.sub(r"[ \t]+", " ", cleaned).strip()


# Fallback query construction per tool when the model's own arguments are
# malformed/missing -- there's no sensible single-message fallback for
# compare_topics (it structurally needs two distinct topics), so that one is
# just skipped rather than guessed at.
_FALLBACK_ARGS_BY_TOOL = {
    "search_topic": lambda msg: {"topic": msg},
    "generate_quiz": lambda msg: {"topic": msg},
    "generate_assignment": lambda msg: {"topic": msg},
    "explain_difficult_concept": lambda msg: {"concept": msg},
    "summarize": lambda msg: {"reference": msg},
}

# Keyword-based tool inference for when native tool-calling produces nothing
# usable at all (the model didn't call a tool, called an unrecognized one,
# or gave irrecoverably malformed args). Previously this always silently
# defaulted to search_topic regardless of what the user actually asked for
# -- which is exactly why "summarize" requests were coming back looking like
# generic Q&A answers: any tool-calling hiccup discarded the summarize
# intent entirely.
_TOOL_INFER_KEYWORDS = [
    (
        "generate_quiz",
        [
            "quiz", "test savol", "savol tuz", "sinab ko'r", "sinab kor", "test tuz",
            # Directive phrasings only, not a bare "questions about" --
            # "I have questions about X" is a real question needing an
            # answer (search_topic), not a request to be quizzed; tested
            # this exact sentence to confirm it doesn't match here. Found
            # via a real failed run: "generate me 5 questions to ask RNN"
            # fell through to search_topic because none of the plain "quiz"
            # keywords above matched, even though the phrasing is an
            # unambiguous quiz request.
            "questions to ask", "ask me questions", "generate questions", "give me questions", "questions to test",
        ],
    ),
    ("summarize", ["summarize", "xulosa", "qisqacha tushuntir", "qisqacha ayt", "qisqacha bayon", "summariz"]),
    ("generate_assignment", ["assignment", "topshiriq", "mashq yarat", "practice problem", "amaliy ish", "amaliy topshiriq"]),
    (
        "explain_difficult_concept",
        [
            # Not just a literal "explain simply" substring -- "explain it
            # simply"/"explain this simply" have a word inserted between
            # "explain" and "simply" that breaks a plain substring match.
            # Found via a real failed run: "I dont understand mode collapse
            # in GANs, explain it simply" fell through to search_topic.
            # "don't/dont understand" added for the same reason -- only the
            # Uzbek equivalent ("tushunmadim") was covered before, not the
            # equally common English phrasing.
            "explain simply", "explain it simply", "explain this simply", "explain that simply",
            "don't understand", "dont understand",
            "sodda qilib", "oddiy misolda", "tushunmadim", "simple example", "sodda tushuntir",
        ],
    ),
]

# compare_topics needs two distinct topics, which generally can't be safely
# guessed from a single keyword hit -- but "X vs Y" / "compare X and Y" /
# "difference between X and Y" phrasing names both topics explicitly and
# unambiguously, so it's worth extracting rather than discarding. Found via
# a real failed run: "compare topics NN vs RNN" fell through the old
# all-or-nothing exclusion straight to search_topic, producing a one-source
# non-comparison instead of an actual comparison. Anything that doesn't
# match one of these specific patterns still falls through to search_topic
# (a lesser but honest degradation) rather than risk a made-up topic split.
_COMPARE_PATTERNS = [
    re.compile(r"^\s*compare\s+topics?\s+(.+?)\s+(?:vs\.?|versus)\s+(.+?)\s*[?.!]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*compare\s+(.+?)\s+(?:and|with|to)\s+(.+?)\s*[?.!]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(.+?)\s+(?:vs\.?|versus)\s+(.+?)\s*[?.!]?\s*$", re.IGNORECASE),
    re.compile(r"difference(?:s)?\s+between\s+(.+?)\s+and\s+(.+?)\s*[?.!]?\s*$", re.IGNORECASE),
]


def _extract_compare_topics(user_message: str) -> Optional[Tuple[str, str]]:
    for pattern in _COMPARE_PATTERNS:
        m = pattern.search(user_message)
        if m:
            topic_a, topic_b = m.group(1).strip(" ?.!"), m.group(2).strip(" ?.!")
            if topic_a and topic_b:
                return topic_a, topic_b
    return None


def _last_user_turn(history: Optional[List[dict]]) -> Optional[str]:
    for msg in reversed(history or []):
        if msg.get("role") == "user":
            return msg.get("content")
    return None


# Only follow-ups that actually look like they refer back to something need
# prior context folded in -- unconditionally prepending the last turn to
# EVERY forced-fallback query backfired on a real, brand-new topic: asked
# "Summarize the chapter on GNNs" then, as an unrelated new question, "I
# dont understand mode collapse in GANs, explain it simply" -- folding in
# the unrelated GNN turn diluted the query enough that retrieval returned
# the generic Preface chunk instead of real GAN content. Gating on a
# pronoun/anaphora keyword keeps the transformer-attention-mechanism fix
# (a genuine follow-up) while not touching messages that already name their
# own topic explicitly -- except a pronoun alone isn't enough: "explain it
# simply" is a self-contained idiom where "it" refers to something already
# named earlier IN THE SAME message ("mode collapse in GANs"), not to a
# prior turn, and that message still matched the anaphora regex. Genuinely
# context-dependent follow-ups tend to be short ("What about its attention
# mechanism?", 6 words); a message that already states its own topic tends
# to be longer and self-contained (the GNN/GANs example above, 10 words) --
# so also require the message to be short before treating a pronoun in it
# as needing an external referent.
_ANAPHORA_RE = re.compile(r"\b(it|its|this|that|these|those|they|them|the (same|latter|former))\b", re.IGNORECASE)
_SHORT_FOLLOWUP_MAX_WORDS = 8


def _infer_forced_tool(user_message: str, history: Optional[List[dict]] = None) -> Tuple[str, dict]:
    last_turn = _last_user_turn(history)
    is_short = len(user_message.split()) <= _SHORT_FOLLOWUP_MAX_WORDS
    needs_context = last_turn and is_short and _ANAPHORA_RE.search(user_message)
    effective_message = f"{last_turn} {user_message}" if needs_context else user_message

    msg_lower = user_message.lower()
    for tool_name, keywords in _TOOL_INFER_KEYWORDS:
        if any(kw in msg_lower for kw in keywords):
            return tool_name, _FALLBACK_ARGS_BY_TOOL[tool_name](effective_message)
    compare_pair = _extract_compare_topics(user_message)
    if compare_pair:
        return "compare_topics", {"topic_a": compare_pair[0], "topic_b": compare_pair[1]}
    return "search_topic", {"topic": effective_message}


def _execute_primary_tool_call(tool_calls: list, user_message: str):
    """Executes only the first tool call. Returns (retrieved_chunks,
    tool_name, tool_args, context, log_entry); retrieved_chunks is [] and
    tool_name is None if execution couldn't resolve at all."""
    tool_call = tool_calls[0]
    fn_name = tool_call["function"]["name"]
    fn_args = dict(tool_call["function"]["arguments"] or {})

    if fn_name not in FUNCTIONS:
        return [], None, None, None, {"name": fn_name, "args": fn_args, "forced": False, "error": "unknown tool"}

    try:
        result = FUNCTIONS[fn_name](**fn_args)
    except TypeError:
        fallback = _FALLBACK_ARGS_BY_TOOL.get(fn_name)
        if fallback is None:
            return [], None, None, None, {"name": fn_name, "args": fn_args, "forced": False, "error": "malformed args, no fallback"}
        fn_args = fallback(user_message)
        try:
            result = FUNCTIONS[fn_name](**fn_args)
        except TypeError:
            return [], None, None, None, {"name": fn_name, "args": fn_args, "forced": False, "error": "malformed args"}

    log_entry = {"name": fn_name, "args": fn_args, "forced": False}
    return result["chunks"], fn_name, fn_args, result["context"], log_entry


def _generate_final_answer(
    final_messages: list, retrieved_chunks: List[dict], schema: dict, render_fn, tool_args: dict,
    provider: str = None, api_key: str = None,
) -> Tuple[str, List[dict]]:
    """One generation + parse against the given tool's schema, rendered via
    the given tool's renderer. Returns (answer_text, cited_chunks)."""
    final_response = _chat(final_messages, format_schema=schema, provider=provider, api_key=api_key)
    raw_content = (final_response["message"].get("content") or "").strip()

    try:
        parsed = json.loads(raw_content)
        return render_fn(parsed, retrieved_chunks, tool_args)
    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        # Model didn't follow the structured schema -- fall back to the raw
        # text and cite everything it was shown, rather than silently
        # showing nothing for output that might still be grounded.
        return raw_content, retrieved_chunks


def run_agent(user_message: str, history: Optional[List[dict]] = None, top_k: int = DEFAULT_TOP_K,
              provider: str = None, api_key: str = None) -> dict:
    """provider/api_key: per-call override (defaults to config.LLM_PROVIDER
    / config.GEMINI_API_KEY) -- threaded through by app/ui/app.py from
    st.session_state, NOT read from global config, so a shared public
    deployment can't leak one visitor's Gemini key into another's request.
    """
    if hybrid.document_count() == 0:
        return {
            "answer_text": NOT_FOUND_MESSAGE,
            "sources": [],
            "retrieved_chunks": [],
            "used_native_tool_calling": False,
            "tool_calls_log": [],
        }

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_message})

    response = _chat(messages, tools=TOOLS, provider=provider, api_key=api_key)
    assistant_msg = response["message"]
    tool_calls = assistant_msg.get("tool_calls") or []
    used_native_tool_calling = bool(tool_calls)

    if tool_calls:
        retrieved_chunks, tool_name, tool_args, context_block, log_entry = _execute_primary_tool_call(tool_calls, user_message)
        tool_calls_log = [log_entry]
    else:
        retrieved_chunks, tool_name, tool_args, context_block, tool_calls_log = [], None, None, None, []

    if not retrieved_chunks:
        # Either the model skipped tool-calling entirely, or the one call it
        # made failed to resolve. Guess the intended tool from keywords
        # instead of always defaulting to search_topic -- discarding the
        # user's actual intent (e.g. "summarize lecture 8") here is what
        # made summarize look like a generic Q&A answer whenever native
        # tool-calling didn't cooperate.
        tool_name, tool_args = _infer_forced_tool(user_message, history)
        try:
            result = FUNCTIONS[tool_name](**tool_args)
            retrieved_chunks, context_block = result["chunks"], result["context"]
        except Exception:
            # Inferred tool itself failed for some reason -- fall back to
            # the one thing that always works: a plain topic search.
            tool_name, tool_args = "search_topic", {"topic": user_message}
            retrieved_chunks = retrieve(user_message, top_k=top_k)
            context_block = format_context_for_prompt(retrieved_chunks)
        tool_calls_log.append({"name": tool_name, "args": tool_args, "forced": True})

    spec = nodes.TOOL_SPECS.get(tool_name, nodes.TOOL_SPECS["search_topic"])
    schema, render_fn = spec["schema"], spec["render"]

    final_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    final_messages.extend(history or [])
    final_messages.append(
        {
            "role": "user",
            "content": f"Retrieved context from the uploaded materials:\n\n{context_block}\n\nRequest: {user_message}",
        }
    )
    answer_text, cited_chunks = _generate_final_answer(
        final_messages, retrieved_chunks, schema, render_fn, tool_args, provider=provider, api_key=api_key
    )
    if _is_language_mismatch(user_message, answer_text):
        # Observed empirically as non-deterministic -- the same request
        # sometimes drifts into Chinese and sometimes doesn't across
        # separate calls, so a single retry has a real chance of recovering
        # a correctly-languaged answer instead of silently keeping a wrong one.
        retried_answer, retried_chunks = _generate_final_answer(
            final_messages, retrieved_chunks, schema, render_fn, tool_args, provider=provider, api_key=api_key
        )
        if not _is_language_mismatch(user_message, retried_answer):
            answer_text, cited_chunks = retried_answer, retried_chunks

    answer_text = _strip_template_artifacts(answer_text)

    sources = build_sources(cited_chunks)
    if sources:
        answer_text += "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)

    return {
        "answer_text": answer_text,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
        "used_native_tool_calling": used_native_tool_calling,
        "tool_calls_log": tool_calls_log,
    }

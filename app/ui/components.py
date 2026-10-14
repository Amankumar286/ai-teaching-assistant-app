"""Streamlit render functions -- sidebar (uploads/indexed-list/clear/settings)
and main-area panels (chat, sources, tool log, quiz). Kept presentational:
these take already-computed data and render it; app.py owns session_state
and the calls into the agent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

import ollama
import streamlit as st

from app.rag import hybrid
from app.rag.chunker import chunk_document
from app.rag.parser import ParseError, parse_file


def _process_uploaded_files(uploaded_files):
    if not uploaded_files:
        return
    for uploaded in uploaded_files:
        key = (uploaded.name, uploaded.size)
        if key in st.session_state.processed_uploads:
            continue
        st.session_state.processed_uploads.add(key)

        # A large PDF (hundreds of pages -> hundreds of chunks needing their
        # own embedding) can genuinely take minutes to parse+chunk+index --
        # without visible feedback the sidebar just looks frozen/broken for
        # that whole time, which is exactly what happened during testing
        # with a 45MB textbook. The spinner is the fix, not the pipeline.
        with st.sidebar.status(f"Indexing {uploaded.name}...", expanded=False) as status:
            config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            dest = config.UPLOADS_DIR / uploaded.name
            dest.write_bytes(uploaded.getvalue())

            try:
                doc = parse_file(dest)
            except ParseError as e:
                status.update(label=f"{uploaded.name}: failed", state="error")
                st.sidebar.error(f"{uploaded.name}: {e}")
                continue

            chunks = chunk_document(doc)
            n = hybrid.index_chunks(chunks)
            warning_note = f" ({len(doc.warnings)} warnings)" if doc.warnings else ""
            status.update(label=f"{uploaded.name}: indexed {n} chunks{warning_note}", state="complete")


def _render_uploaders():
    st.sidebar.header("Upload Materials")
    # Three separate uploaders per SPEC.md's sidebar spec -- cosmetically
    # distinct only, since the metadata schema has no category field to
    # actually distinguish lecture/book/assignment; all route to the same
    # indexing pipeline.
    _process_uploaded_files(
        st.sidebar.file_uploader(
            "Upload Lecture Slides", type=["pdf", "pptx", "docx"], accept_multiple_files=True, key="lecture_uploader"
        )
    )
    _process_uploaded_files(
        st.sidebar.file_uploader(
            "Upload Books", type=["pdf", "pptx", "docx"], accept_multiple_files=True, key="book_uploader"
        )
    )
    _process_uploaded_files(
        st.sidebar.file_uploader(
            "Upload Assignments", type=["pdf", "pptx", "docx"], accept_multiple_files=True, key="assignment_uploader"
        )
    )


def _render_indexed_documents():
    st.sidebar.header("Indexed Documents")
    files = hybrid.list_indexed_files()
    if files:
        for f in files:
            st.sidebar.write(f"- {f}")
    else:
        st.sidebar.caption("No documents indexed yet.")


def _render_clear_kb_button():
    st.sidebar.header("Knowledge Base")
    if st.sidebar.button("Clear Knowledge Base"):
        if st.session_state.confirm_clear:
            hybrid.clear_knowledge_base()
            st.session_state.confirm_clear = False
            st.session_state.processed_uploads = set()
            st.session_state.history = []
            st.session_state.last_result = None
            st.sidebar.success("Knowledge base cleared.")
            st.rerun()
        else:
            st.session_state.confirm_clear = True
            st.sidebar.warning("Click again to confirm -- this deletes all indexed content.")


def _get_available_models():
    try:
        response = ollama.list()
        models = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
        names = []
        for m in models:
            if isinstance(m, dict):
                name = m.get("model") or m.get("name")
            else:
                name = getattr(m, "model", None) or getattr(m, "name", None)
            if name:
                names.append(name)
        return sorted(set(names))
    except Exception:
        return []


def _render_llm_provider():
    """Returns (provider, api_key). api_key is returned (not stashed in
    global config) so app.py can pass it straight through to run_agent()
    per-call -- see graph.py's run_agent docstring for why a global write
    here would leak one visitor's key to every other visitor on a shared
    public deployment."""
    if config.LLM_PROVIDER == "gemini":
        # Locked-to-Gemini deployment: app.py's key gate already blocked
        # entry until a key existed, so there's nothing to choose here --
        # just let them see/update the key, no Ollama option that couldn't
        # work in this environment anyway.
        st.sidebar.text_input(
            "Gemini API key", type="password", key="gemini_api_key",
            help="Get a free key at aistudio.google.com/apikey.",
        )
        return "gemini", st.session_state.get("gemini_api_key")

    st.session_state.setdefault("llm_provider", config.LLM_PROVIDER)
    provider_labels = {"Local (Ollama)": "ollama", "Gemini (cloud)": "gemini"}
    default_label = next(
        (label for label, value in provider_labels.items() if value == st.session_state["llm_provider"]),
        "Local (Ollama)",
    )
    chosen_label = st.sidebar.selectbox(
        "LLM Provider", list(provider_labels.keys()),
        index=list(provider_labels.keys()).index(default_label),
    )
    st.session_state["llm_provider"] = provider_labels[chosen_label]

    if st.session_state["llm_provider"] != "gemini":
        return "ollama", None

    st.sidebar.text_input(
        "Gemini API key",
        type="password",
        placeholder="Paste your Gemini API key",
        key="gemini_api_key",
        help="Get a free key at aistudio.google.com/apikey. Used only for "
             "your own requests this session -- never logged, displayed, "
             "or shared with other visitors.",
    )
    st.sidebar.caption(
        "🔑 [Get a free Gemini API key](https://aistudio.google.com/apikey) "
        "— no key is stored server-side beyond this session. ⚠️ Gemini mode "
        "is experimental -- this project's anti-hallucination behavior was "
        "tuned against Ollama specifically, see README."
    )
    api_key = st.session_state.get("gemini_api_key")
    if not api_key:
        st.sidebar.info("Paste a Gemini API key above to chat.")
    return "gemini", api_key


def _render_settings():
    st.sidebar.header("Settings")

    provider, api_key = _render_llm_provider()

    if provider == "ollama":
        available_models = _get_available_models()
        if available_models:
            current = config.OLLAMA_MODEL if config.OLLAMA_MODEL in available_models else available_models[0]
            selected = st.sidebar.selectbox("Model", available_models, index=available_models.index(current))
            config.OLLAMA_MODEL = selected
        else:
            st.sidebar.caption("Could not reach Ollama to list models -- is it running?")

    top_k = st.sidebar.slider(
        "Top-K (search_topic / explain_difficult_concept)", min_value=1, max_value=15, value=config.DEFAULT_TOP_K
    )
    config.DEFAULT_TOP_K = top_k

    return provider, api_key


def render_sidebar():
    """Returns (provider, api_key) from the LLM provider selector -- pass
    straight through to run_agent(provider=..., api_key=...)."""
    _render_uploaders()
    _render_indexed_documents()
    _render_clear_kb_button()
    return _render_settings()


def render_chat_history(history_display):
    """history_display: list of {"role": "user"/"assistant", "content": str}."""
    for msg in history_display:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def render_sources_panel(sources):
    st.subheader("Retrieved Sources")
    if not sources:
        st.caption("Sources for the latest answer will show up here.")
        return
    for s in sources:
        st.markdown(f"- {s}")


def render_tool_log_panel(tool_calls_log):
    st.subheader("Tool Log")
    if not tool_calls_log:
        st.caption("Which tool was called (and what it retrieved) will show up here.")
        return
    for entry in tool_calls_log:
        forced_tag = " (forced fallback)" if entry.get("forced") else ""
        error_tag = f" -- error: {entry['error']}" if entry.get("error") else ""
        st.code(f"{entry['name']}{forced_tag}{error_tag}\nargs: {entry.get('args', {})}", language=None)


def render_quiz_panel(quiz_text):
    st.subheader("Generated Questions")
    if not quiz_text:
        st.caption("Ask to be quizzed on a topic to see it here.")
        return
    st.markdown(quiz_text)

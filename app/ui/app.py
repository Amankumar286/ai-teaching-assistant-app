"""Streamlit page: chat interface + sidebar + side panels. Entry point is
main.py at the project root (`streamlit run main.py`)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
import ollama
import streamlit as st

from app.agent.graph import run_agent
from app.memory.session_memory import trim_history
from app.ui.components import (
    render_chat_history,
    render_quiz_panel,
    render_sidebar,
    render_sources_panel,
    render_tool_log_panel,
)


def _init_session_state():
    defaults = {
        "history": [],
        "processed_uploads": set(),
        "confirm_clear": False,
        "last_result": None,
        "last_quiz_text": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_key_gate():
    """The only thing rendered when this deployment is locked to Gemini
    (LLM_PROVIDER=gemini secret) and no key has been entered yet -- no
    uploaders, no settings, nothing else competing for attention. As soon as
    a key is typed and Enter is pressed, session_state picks it up and the
    next rerun's guard in main() passes, revealing the full app."""
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.title("AI Teaching Assistant")
        st.markdown("#### 🔑 Enter your Gemini API key to start")
        st.text_input(
            "Gemini API key",
            type="password",
            placeholder="Paste your Gemini API key here",
            key="gemini_api_key",
            label_visibility="collapsed",
        )
        st.caption(
            "Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) "
            "— used only for your own requests this session, never logged, stored, or shared "
            "with other visitors."
        )


def main():
    st.set_page_config(page_title="AI Teaching Assistant", layout="wide")
    _init_session_state()

    if config.LLM_PROVIDER == "gemini" and not st.session_state.get("gemini_api_key"):
        _render_key_gate()
        return

    llm_provider, gemini_api_key = render_sidebar()

    st.title("AI Teaching Assistant")

    chat_col, panel_col = st.columns([3, 2])

    with chat_col:
        render_chat_history(st.session_state.history)

        question = st.chat_input("Ask a question, request a quiz, compare topics, summarize a lecture...")
        if question:
            with st.chat_message("user"):
                st.markdown(question)

            if llm_provider == "gemini" and not gemini_api_key:
                result = {
                    "answer_text": "Paste a Gemini API key in the sidebar to chat.",
                    "sources": [], "retrieved_chunks": [], "used_native_tool_calling": False, "tool_calls_log": [],
                }
            else:
                with st.spinner("Thinking..."):
                    try:
                        result = run_agent(
                            question, history=trim_history(st.session_state.history),
                            provider=llm_provider, api_key=gemini_api_key,
                        )
                    except ollama.ResponseError as e:
                        result = {
                            "answer_text": (
                                f"Couldn't reach the model ({e}). Check that Ollama is running and the "
                                "selected model has actually been pulled (`ollama pull <model>`)."
                            ),
                            "sources": [],
                            "retrieved_chunks": [],
                            "used_native_tool_calling": False,
                            "tool_calls_log": [],
                        }
                    except (ValueError, RuntimeError) as e:
                        # ValueError: no Gemini key. RuntimeError: Gemini API call failed.
                        result = {
                            "answer_text": f"⚠️ {e}",
                            "sources": [], "retrieved_chunks": [], "used_native_tool_calling": False, "tool_calls_log": [],
                        }

            with st.chat_message("assistant"):
                st.markdown(result["answer_text"])

            st.session_state.history.append({"role": "user", "content": question})
            st.session_state.history.append({"role": "assistant", "content": result["answer_text"]})
            st.session_state.last_result = result

            tool_name = result["tool_calls_log"][0]["name"] if result["tool_calls_log"] else None
            if tool_name == "generate_quiz":
                st.session_state.last_quiz_text = result["answer_text"]

    with panel_col:
        last = st.session_state.last_result
        render_sources_panel(last["sources"] if last else [])
        render_tool_log_panel(last["tool_calls_log"] if last else [])
        render_quiz_panel(st.session_state.last_quiz_text)


if __name__ == "__main__":
    main()

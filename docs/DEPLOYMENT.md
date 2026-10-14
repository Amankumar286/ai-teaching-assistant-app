# Cloud deployment and the Gemini path

The default path is fully local via Ollama, and that is the one the grounding guarantees
were tuned against. This covers the optional cloud path — running on Gemini and hosting a
public demo on Streamlit Community Cloud — and is explicit about what is *not* verified
there.

[← back to the README](../README.md)

---

### Optional: Gemini (cloud) instead of Ollama

> ⚠️ **Anti-hallucination guarantees are Ollama-only.** Every fix in this
> project's commit history (forced relevance self-checks, deterministic
> citations, retrieval gating) was tuned and tested against Ollama/qwen2.5's
> specific response format. Gemini mode reuses the same prompts and
> retrieval logic, but its structured-output behavior has **not** been
> verified to preserve the same grounding guarantees — treat it as
> experimental, good for a quick public demo, not a substitute for the
> tested default. One concrete gap: `repeat_penalty` (added after greedy
> decoding got stuck mid-generation and blew the token budget in real
> testing) has no equivalent on Gemini's API.

Pick **"Gemini (cloud)"** in the sidebar and paste a free key from
[aistudio.google.com/apikey](https://aistudio.google.com/apikey) — no local
Ollama needed, nothing stored server-side. Uses Gemini's
[OpenAI-compatible endpoint](https://ai.google.dev/gemini-api/docs/openai)
directly via `requests` (no extra provider-specific package), with tool
calls and structured-output schemas translated to/from Ollama's native
shapes in `app/agent/graph.py`'s `_chat()`.

Opens at `http://localhost:8501`. Upload lecture slides / books /
assignments from the sidebar, then just ask questions in the chat — the
agent picks the right tool automatically:

| You say something like...                          | Tool called                  |
|------------------------------------------------------|-------------------------------|
| "What is X?"                                          | `search_topic`                 |
| "Quiz me on X" / "give me 5 questions on X"           | `generate_quiz`                |
| "Give me practice problems on X"                      | `generate_assignment`          |
| "Compare X and Y"                                     | `compare_topics`                |
| "Summarize lecture 3" / "summarize the X textbook"    | `summarize`                     |
| "Explain X simply" / "I don't understand X"           | `explain_difficult_concept`    |

A quick terminal-only alternative (useful after making a code change,
without waiting on the browser) is `scripts/cli.py`:

```powershell
.\.venv\Scripts\python.exe scripts\cli.py --add path\to\file.pdf
.\.venv\Scripts\python.exe scripts\cli.py
```

## ☁️ Deploy your own (Streamlit Community Cloud, free)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **New app**.
3. Point it at this repo, branch `main`, main file path `main.py`.
4. In **Advanced settings → Secrets**, add `LLM_PROVIDER = "gemini"` — there's
   no Ollama server on Streamlit Cloud, so this makes the Gemini API key
   field the first thing a visitor sees instead of an Ollama-unreachable
   error.
5. Deploy. Visitors upload their own materials and paste their own free
   Gemini key — no secrets to configure on your end. Remember the warning
   above: this is a fine way to demo the *idea*, but the specific
   anti-hallucination guarantees this project is built around are proven
   against the local Ollama path, not this one.

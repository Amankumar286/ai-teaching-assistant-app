"""Rolling conversation-history policy for within-session memory.

SPEC.md: "retain context across turns within a session... implemented as a
rolling conversation history injected into the agent's prompt/context."
Left unbounded, a long chat session would eventually exceed the model's
context window -- trim_history() keeps the most recent turns instead of
letting it grow forever.
"""

from typing import List

DEFAULT_MAX_TURNS = 6  # a "turn" = one user+assistant message pair


def trim_history(history: List[dict], max_turns: int = DEFAULT_MAX_TURNS) -> List[dict]:
    """Keeps the most recent `max_turns` user/assistant pairs, oldest
    dropped first. `history` is a flat list of {"role", "content"} dicts,
    user and assistant messages alternating -- this trims by message *pairs*
    so a turn's user question and its answer are never split apart."""
    max_messages = max_turns * 2
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]

"""Real bug found live: qwen2.5 at Q4_K_M occasionally echoes Jinja-style
{% raw %} templating boilerplate at the start of an otherwise-correct
answer (training-data noise). graph._strip_template_artifacts is the
defensive fix -- these tests lock down that it removes the artifact
without mangling real content around it.
"""

from app.agent.graph import _strip_template_artifacts


def test_strips_leading_raw_tag():
    assert _strip_template_artifacts("{% raw %}In PyTorch, a tensor is...") == "In PyTorch, a tensor is..."


def test_strips_raw_and_endraw_without_merging_words():
    assert _strip_template_artifacts("{%raw%}text{%endraw%} more") == "text more"


def test_leaves_clean_text_untouched():
    text = "no artifacts here, just a normal answer."
    assert _strip_template_artifacts(text) == text


def test_handles_multiple_occurrences():
    text = "{% raw %}first{% endraw %} and {% raw %}second{% endraw %}"
    result = _strip_template_artifacts(text)
    assert "{% raw %}" not in result
    assert "{% endraw %}" not in result
    assert "first" in result and "second" in result

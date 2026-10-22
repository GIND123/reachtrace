import jinja2
import pytest

from reachtrace.prompts import get_template

# chat_template.jinja from each HF repo, copied byte-for-byte on 2026-10-22.
HF_TEMPLATES = {
    "rlzero31_math": (
        "Solve the following problem step by step. The last line of your response should be the "
        "answer to the problem in form Answer: $Answer (without quotes) where $Answer is the answer "
        "to the problem.\n\n{% for message in messages %}{{ '\n\n' if not loop.first else '' }}"
        "{{ message['content'] + '\n' }}{% if loop.last and add_generation_prompt %}"
        "{{ '\nRemember to put your answer on its own line after \"Answer:\"' }}{% endif %}{% endfor %}"
    ),
    "rlzero30_math": (
        "Solve the following math problem step by step. The last line of your response should be the "
        "answer to the problem in form Answer: $Answer (without quotes) where $Answer is the answer "
        "to the problem.\n\n{% for message in messages %}{{ '\n\n' if not loop.first else '' }}"
        "{{ message['content'] + '\n' }}{% if loop.last and add_generation_prompt %}"
        "{{ '\nRemember to put your answer on its own line after \"Answer:\"' }}{% endif %}{% endfor %}"
    ),
}


@pytest.mark.parametrize("template_id", sorted(HF_TEMPLATES))
def test_render_matches_hf_chat_template(template_id):
    q = "What is $1+1$? Give the answer in \\boxed{}."
    hf = jinja2.Template(HF_TEMPLATES[template_id]).render(
        messages=[{"role": "user", "content": q}], add_generation_prompt=True
    )
    assert get_template(template_id).render(q) == hf


def test_cue_is_appended_verbatim():
    t = get_template("rlzero31_math")
    assert t.render("x", cue=".\n\nOkay").endswith('"Answer:".\n\nOkay')


def test_stop_guard_matches_both_headers():
    for tid in HF_TEMPLATES:
        t = get_template(tid)
        assert ("\n" + t.header).startswith(t.stop[0])

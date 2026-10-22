"""Prompt templates rendered byte-identically for the base and RL-Zero models.

The RL-Zero chat templates are plain text with no special tokens, so the base model
receives exactly the string the RL model saw during training. `tests/test_prompts.py`
checks these renderings against the Jinja templates shipped in each HF repo.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

REMINDER = 'Remember to put your answer on its own line after "Answer:"'

_ANSWER_SPEC = (
    "The last line of your response should be the answer to the problem in form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem."
)

# Verified against chat_template.jinja on 2026-10-22. Note the 3.1 header drops "math";
# the Olmo 3 report (Fig. 37) shows the 3.0 wording.
HEADERS = {
    "rlzero31_math": f"Solve the following problem step by step. {_ANSWER_SPEC}",
    "rlzero30_math": f"Solve the following math problem step by step. {_ANSWER_SPEC}",
}

# Guards against the base model running on into a fresh problem; prefix shared by both headers.
STOP_GUARDS = ["\nSolve the following"]


@dataclass(frozen=True)
class PromptTemplate:
    template_id: str
    header: str

    def render(self, question: str, cue: str = "") -> str:
        """Render a problem. `cue` is appended verbatim after the prompt (cue-forcing)."""
        return f"{self.header}\n\n{question}\n\n{REMINDER}{cue}"

    @property
    def stop(self) -> list[str]:
        return list(STOP_GUARDS)

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.render("{question}").encode()).hexdigest()[:16]


def get_template(template_id: str) -> PromptTemplate:
    if template_id not in HEADERS:
        raise KeyError(f"unknown template {template_id!r}; known: {sorted(HEADERS)}")
    return PromptTemplate(template_id, HEADERS[template_id])


def prompt_sha(rendered: str) -> str:
    return hashlib.sha256(rendered.encode()).hexdigest()

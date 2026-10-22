"""Evaluation sets mapped to one schema: problem_id, eval_set, question, answer, level, subject.

HF IDs and column names verified with the datasets-server API on 2026-10-22. Licences
marked None on the Hub (MATH-500, aime_2024, amc23, minervamath) still need checking
before release.
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from ..sample.grade import last_boxed

MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def problem_hash(question: str) -> str:
    norm = re.sub(r"\s+", " ", question.strip().lower())
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class EvalSpec:
    hf_id: str
    configs: list[str]
    split: str
    question: str
    answer: Callable[[dict], str]
    native_id: Callable[[dict, int], str] = lambda r, i: str(i)
    level: Callable[[dict], str | None] = lambda r: None
    subject: Callable[[dict], str | None] = lambda r: None
    keep: Callable[[dict], bool] = lambda r: True
    tags: list[str] = field(default_factory=list)


def _olympiad_answer(r: dict) -> str:
    fa = r["final_answer"]
    vals = ast.literal_eval(fa) if isinstance(fa, str) else fa
    return str(vals[0])


EVAL_SETS: dict[str, EvalSpec] = {
    "math500": EvalSpec(
        "HuggingFaceH4/MATH-500",
        ["default"],
        "test",
        "problem",
        answer=lambda r: r["answer"],
        native_id=lambda r, i: r["unique_id"],
        level=lambda r: str(r["level"]),
        subject=lambda r: r["subject"],
        tags=["core"],
    ),
    "math_l45": EvalSpec(
        "EleutherAI/hendrycks_math",
        MATH_SUBJECTS,
        "test",
        "problem",
        answer=lambda r: last_boxed(r["solution"]) or "",
        level=lambda r: r["level"].replace("Level ", ""),
        subject=lambda r: r["type"],
        keep=lambda r: r["level"] in ("Level 4", "Level 5"),
        tags=["core"],
    ),
    "aime24": EvalSpec(
        "HuggingFaceH4/aime_2024",
        ["default"],
        "train",
        "problem",
        answer=lambda r: str(r["answer"]),
        native_id=lambda r, i: str(r["id"]),
        tags=["core", "anchor"],
    ),
    "aime25": EvalSpec(
        "opencompass/AIME2025",
        ["AIME2025-I", "AIME2025-II"],
        "test",
        "question",
        answer=lambda r: str(r["answer"]),
        tags=["core", "anchor"],
    ),
    "amc23": EvalSpec(
        "math-ai/amc23",
        ["default"],
        "test",
        "question",
        answer=lambda r: str(r["answer"]),
        native_id=lambda r, i: str(r["id"]),
        tags=["core"],
    ),
    "olympiadbench": EvalSpec(
        "Hothan/OlympiadBench",
        ["OE_TO_maths_en_COMP"],
        "train",
        "question",
        answer=_olympiad_answer,
        native_id=lambda r, i: str(r["id"]),
        subject=lambda r: r["subfield"],
        keep=lambda r: r["is_multiple_answer"] in (False, "False") and r["modality"] == "Text-only",
        tags=["core"],
    ),
    "minerva": EvalSpec(
        "math-ai/minervamath",
        ["default"],
        "test",
        "question",
        answer=lambda r: str(r["answer"]),
        tags=["tierB"],
    ),
    "gsm8k": EvalSpec(
        "openai/gsm8k",
        ["main"],
        "test",
        "question",
        answer=lambda r: r["answer"].split("####")[-1].strip(),
        tags=["calibration"],
    ),
}


def load_eval_set(name: str, revision: str | None = None) -> pd.DataFrame:
    from datasets import load_dataset

    spec = EVAL_SETS[name]
    rows = []
    for cfg in spec.configs:
        ds = load_dataset(spec.hf_id, cfg, split=spec.split, revision=revision)
        for i, r in enumerate(ds):
            if not spec.keep(r):
                continue
            q = r[spec.question]
            rows.append(
                dict(
                    problem_id=f"{name}/{cfg}/{spec.native_id(r, i)}",
                    eval_set=name,
                    question=q,
                    answer=spec.answer(r),
                    level=spec.level(r),
                    subject=spec.subject(r),
                    problem_hash=problem_hash(q),
                )
            )
    return pd.DataFrame(rows)


def load_eval_sets(names: list[str]) -> pd.DataFrame:
    """Concatenate sets and drop cross-set duplicates (e.g. MATH-500 items inside MATH L4-5)."""
    df = pd.concat([load_eval_set(n) for n in names], ignore_index=True)
    return df.drop_duplicates("problem_hash", keep="first").reset_index(drop=True)

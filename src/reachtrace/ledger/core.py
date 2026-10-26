"""Likelihood ledger and discrimination margin (plan v3, RQ2).

For each problem we take correct and incorrect finished RL traces, teacher-force them under
every checkpoint along the flow, and record per-token log-likelihoods on three masks:

  opening  first `opening` tokens (where the cue / format effect lives; Wang et al. 2026)
  answer   from the last "Answer:" line to the end
  content  everything in between

The discrimination margin M_theta(q) = mean content log-lik of correct traces minus that of
incorrect traces. Correct and incorrect traces share generator, format and style, so form
cancels and M measures a preference for correct *content*. Because M is a number per
checkpoint, its stage increments add up exactly to the total change (a telescoping sum),
which gives a stage decomposition without any attribution approximation.

Token positions come from the generated token IDs, never from re-tokenized text.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

_ANSWER_RE = re.compile(r"Answer\s*:")
MASKS = ("opening", "content", "answer")


def answer_start_token(
    ids: Sequence[int], decode: Callable[[Sequence[int]], str], window: int = 400
) -> int:
    """Index of the token where the last "Answer:" begins; len(ids) if there is none.

    Scans suffixes backwards, so the first suffix that contains "Answer:" starts at (or
    inside) the token that holds the "A" of the last answer line.
    """
    n = len(ids)
    for i in range(n - 1, max(-1, n - window - 1), -1):
        if _ANSWER_RE.search(decode(ids[i:])):
            return i
    return n


def region_masks(n: int, answer_start: int, opening: int = 16) -> dict[str, np.ndarray]:
    idx = np.arange(n)
    open_end = min(opening, answer_start)
    return {
        "opening": idx < open_end,
        "content": (idx >= open_end) & (idx < answer_start),
        "answer": idx >= answer_start,
    }


def trace_scores(logprobs: np.ndarray, answer_start: int, opening: int = 16) -> dict[str, float]:
    """Mean per-token log-likelihood on each mask (NaN for an empty mask), plus the token count."""
    lp = np.asarray(logprobs, dtype=float)
    out: dict[str, float] = {}
    for name, m in region_masks(len(lp), answer_start, opening).items():
        out[name] = float(lp[m].mean()) if m.any() else float("nan")
        out[f"n_{name}"] = int(m.sum())
    out["all"] = float(lp.mean()) if len(lp) else float("nan")
    return out


def margins(scores: pd.DataFrame, mask: str = "content") -> pd.DataFrame:
    """Per (problem, checkpoint): mean score of correct traces, of incorrect traces, and M.

    scores: one row per (problem_id, checkpoint, trace_id) with a boolean `correct` column and
    a column per mask. Problems need at least one trace of each label at a checkpoint.
    """
    g = scores.groupby(["problem_id", "checkpoint", "correct"])[mask].mean().unstack("correct")
    g = g.rename(columns={True: "L_correct", False: "L_incorrect"})
    for col in ("L_correct", "L_incorrect"):
        if col not in g:
            g[col] = np.nan
    g["M"] = g.L_correct - g.L_incorrect
    return g.reset_index()


def stage_increments(
    wide: pd.DataFrame, boundaries: list[tuple[str, str]], start: str
) -> pd.DataFrame:
    """Telescoping stage decomposition of a per-checkpoint quantity.

    wide: index problem_id, one column per checkpoint name (e.g. M).
    boundaries: ordered (stage_name, checkpoint that ends the stage).
    start: checkpoint the ledger starts from.
    Returns per problem the increment of each stage, the total, and each stage's share of the
    total (NaN when |total| is below 1e-6, where shares are meaningless).
    """
    out = pd.DataFrame(index=wide.index)
    prev = start
    for stage, ckpt in boundaries:
        out[stage] = wide[ckpt] - wide[prev]
        prev = ckpt
    out["total"] = wide[prev] - wide[start]
    stages = [s for s, _ in boundaries]
    assert np.allclose(out[stages].sum(axis=1), out.total, equal_nan=True)
    denom = out.total.where(out.total.abs() > 1e-6)
    for s in stages:
        out[f"share_{s}"] = out[s] / denom
    return out

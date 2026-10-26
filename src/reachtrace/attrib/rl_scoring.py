"""RL-item and contrastive scoring done in post-processing (plan v3, RQ4).

Advantage after normalization. Bergson multiplies each RL item's loss by its advantage and
`--unit_normalize` then divides every item gradient by its norm (process_grads.py:159), so
only the *sign* of the advantage survives: in a group with one success and seven failures
the cosine scores come out net-negative, whereas the training update sums to zero. Instead,
build the RL-item index from plain (unweighted) completion gradients, keep advantages in a
sidecar table, and weight the cosines afterwards:

    s(q, group) = sum_i A_i * cos(q, g_i)

Placebo items are on-policy rollouts on prompts never trained on; the RL-signal estimate is
Dolci minus placebo, which removes the model's similarity to its own outputs.

Contrastive queries. For each problem, the query score of a document is
cos(q_correct, d) - cos(q_incorrect, d), averaged over traces, so format and style shared
by correct and incorrect traces cancel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def group_advantages(
    items: pd.DataFrame, reward: str = "reward", group: str = "group_id"
) -> pd.Series:
    """Dr. GRPO advantage: reward minus group mean (no std normalization)."""
    return items[reward] - items.groupby(group)[reward].transform("mean")


def advantage_weighted(
    cos_scores: np.ndarray, items: pd.DataFrame, advantage: str = "advantage",
    group: str = "group_id",
) -> pd.DataFrame:
    """Per-query, per-group RL scores from unweighted cosines.

    cos_scores: queries x items matrix of cosines between unit-normalized query gradients
    and unit-normalized *unweighted* item gradients (row order = items).
    Returns a long table: query, group_id, score, pos (positive-advantage part), neg.
    """
    a = items[advantage].to_numpy(float)
    w = cos_scores * a[None, :]
    groups, codes = np.unique(items[group].to_numpy(), return_inverse=True)
    q = cos_scores.shape[0]
    out = {
        "score": np.zeros((q, len(groups))),
        "pos": np.zeros((q, len(groups))),
        "neg": np.zeros((q, len(groups))),
    }
    for j in range(len(groups)):
        cols = codes == j
        out["score"][:, j] = w[:, cols].sum(1)
        out["pos"][:, j] = w[:, cols & (a > 0)].sum(1)
        out["neg"][:, j] = w[:, cols & (a < 0)].sum(1)
    qi, gi = np.meshgrid(np.arange(q), np.arange(len(groups)), indexing="ij")
    return pd.DataFrame(
        {
            "query": qi.ravel(),
            group: groups[gi.ravel()],
            **{k: v.ravel() for k, v in out.items()},
        }
    )


def sign_only(cos_scores: np.ndarray, items: pd.DataFrame, advantage: str = "advantage",
              group: str = "group_id") -> pd.DataFrame:
    """What `--advantage_column` plus `--unit_normalize` computes: sign(A_i) * cos(q, g_i).

    Kept only so the gate (G4) can show the difference; never use it for results.
    """
    signed = items.assign(**{advantage: np.sign(items[advantage].to_numpy(float))})
    return advantage_weighted(cos_scores, signed, advantage, group)


def rl_signal(dolci: pd.DataFrame, placebo: pd.DataFrame) -> pd.Series:
    """Per query: mean group score on Dolci items minus mean on placebo items."""
    return dolci.groupby("query").score.mean() - placebo.groupby("query").score.mean()


def contrastive(cos_correct: np.ndarray, cos_incorrect: np.ndarray) -> np.ndarray:
    """Documents x 1 contrastive score from per-trace cosine matrices (traces x documents)."""
    return np.asarray(cos_correct).mean(0) - np.asarray(cos_incorrect).mean(0)


def label_mask_row(
    prompt_ids: list[int], completion_ids: list[int], keep: np.ndarray
) -> dict[str, list[int]]:
    """A pre-tokenized Bergson row: loss only on completion tokens where `keep` is True.

    Bergson skips tokenization when `input_ids` is present (utils/worker_utils.py); labels
    of -100 are ignored by the loss. One row per trace replaces one row per span.
    """
    keep = np.asarray(keep, dtype=bool)
    if len(keep) != len(completion_ids):
        raise ValueError("mask length must equal completion length")
    labels = [-100] * len(prompt_ids) + [
        t if k else -100 for t, k in zip(completion_ids, keep)
    ]
    return {"input_ids": list(prompt_ids) + list(completion_ids), "labels": labels}

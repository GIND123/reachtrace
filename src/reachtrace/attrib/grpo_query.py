"""GRPO queries: rollouts with Dr. GRPO advantages as Bergson's `advantage_column`.

This replaces the hand-built contrastive query of plan v3 and Bergson's experimental
`query.contrast`. For a problem with n rollouts of which k are correct, Dr. GRPO gives
A = 1 - k/n to correct and -k/n to incorrect rollouts, so the advantage-weighted query
gradient is

    sum_i A_i g_i = k(n - k)/n * (mean_correct g - mean_incorrect g),

i.e. the correct-minus-incorrect contrast, scaled by the group's reward variance. Format and
style that correct and incorrect rollouts share cancel, and it is the direction a GRPO step
on that problem would move the model.

How each Bergson method treats the advantage (bergson 2.7.0):

* EKFAC / EKFAC+ASTRA (`influence`, no unit normalization): the loss is multiplied by A_i
  before the backward pass (collector.fwd_bwd), and scores are linear in the query
  gradient, so per-problem `aggregation: sum` gives exactly q^T H^-1 g_d with q = sum A_i g_i.
* TrackStar (`unit_normalize: true`): each row is normalized *before* aggregation
  (process_grads.normalize_and_aggregate_grads), so A_i g_i / |A_i g_i| keeps only
  sign(A_i). Build the query with `aggregation: none`, then multiply each column by |A_i|
  and sum within the problem (`reweight_columns`, `per_problem`). The result is
  sum_i A_i cos(q_i, d) under the mixed/value autocorrelation preconditioner.

Rows whose group has a single reward value have A = 0, a zero gradient and a 0/0 cosine,
so they are dropped (they also contribute nothing to a GRPO update).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def dr_grpo_advantage(reward: pd.Series, group: pd.Series) -> pd.Series:
    """Reward minus group mean, with no standard-deviation scaling (Dr. GRPO, OlmoRL)."""
    return reward - reward.groupby(group).transform("mean")


def contrast_scale(k: int, n: int) -> float:
    """sum_i A_i g_i = contrast_scale(k, n) * (mean correct g - mean incorrect g)."""
    return k * (n - k) / n


def query_rows(rollouts: pd.DataFrame, group: str = "problem_id") -> pd.DataFrame:
    """Pre-tokenized GRPO query rows from rollouts.

    rollouts: one row per sample with `prompt_ids`, `completion_ids` (the exact generated
    token ids), `reward` and a group column. Returns rows with `input_ids`, `labels` (loss on
    the completion only), `advantage`, the group column and `n_tokens`, without the
    zero-advantage groups.
    """
    df = rollouts.copy()
    df["advantage"] = dr_grpo_advantage(df["reward"].astype(float), df[group])
    spread = df.groupby(group)["reward"].transform("nunique")
    df = df[spread > 1].reset_index(drop=True)
    df["input_ids"] = [list(p) + list(c) for p, c in zip(df.prompt_ids, df.completion_ids)]
    df["labels"] = [[-100] * len(p) + list(c) for p, c in zip(df.prompt_ids, df.completion_ids)]
    df["n_tokens"] = [len(c) for c in df.completion_ids]
    return df


def reweight_columns(scores: np.ndarray, advantage: np.ndarray) -> np.ndarray:
    """docs x rows scores from a unit-normalized GRPO query, times |A| of each row.

    The sign of A is already in the column (the loss was multiplied by A before
    normalization), so this restores sum_i A_i cos(q_i, d) once columns are summed.
    """
    return np.asarray(scores) * np.abs(np.asarray(advantage, dtype=float))[None, :]


def per_problem(scores: np.ndarray, problem: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum query columns within each problem: docs x rows -> docs x problems."""
    names, codes = np.unique(np.asarray(problem), return_inverse=True)
    out = np.zeros((scores.shape[0], len(names)))
    np.add.at(out.T, codes, np.asarray(scores, dtype=float).T)
    return out, names


def unit_items_reweight(scores: np.ndarray, advantage: np.ndarray) -> np.ndarray:
    """RL items indexed with `advantage_column` and unit normalization: items x queries
    scores times |A_i| of each item, the training-side counterpart of `reweight_columns`."""
    return np.asarray(scores) * np.abs(np.asarray(advantage, dtype=float))[:, None]

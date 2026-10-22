"""Aggregate document scores to bins and stages (the unit of analysis; Ruis/Matlin).

Stage *levels* are not interpretable on their own: RL items are on-policy rollouts in the
query's exact format, and stage sample sizes are set by design. Only *differences* between
partitions (U minus matched R) are tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def _onehot(labels: np.ndarray) -> tuple[sparse.csr_matrix, pd.Index]:
    codes, uniques = pd.factorize(np.asarray(labels), sort=True)
    m = sparse.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)))
    return m, pd.Index(uniques)


def bin_mass(
    scores: np.ndarray,
    bins: np.ndarray,
    center: bool = True,
    query_index: pd.Index | None = None,
    chunk: int = 256,
) -> pd.DataFrame:
    """Mean score per bin, per query, minus the query's global mean.

    Subtracting the global mean is the bin-size-matched random expectation: a random set of
    |b| documents has expected mean score equal to the global mean.
    """
    onehot, names = _onehot(bins)
    counts = np.asarray(onehot.sum(axis=0)).ravel()
    out = np.empty((scores.shape[0], len(names)))
    for i in range(0, scores.shape[0], chunk):
        block = np.asarray(scores[i : i + chunk], dtype=np.float64)
        out[i : i + chunk] = (onehot.T @ block.T).T / counts
        if center:
            out[i : i + chunk] -= block.mean(axis=1, keepdims=True)
    return pd.DataFrame(out, columns=names, index=query_index)


def stage_share(
    scores: np.ndarray, stages: np.ndarray, query_index: pd.Index | None = None
) -> pd.DataFrame:
    """Signed stage sum over total absolute score: M_stage(q) = sum_{d in s} x_qd / sum_d |x_qd|."""
    onehot, names = _onehot(stages)
    x = np.asarray(scores, dtype=np.float64)
    num = (onehot.T @ x.T).T
    den = np.abs(x).sum(axis=1, keepdims=True)
    return pd.DataFrame(num / np.where(den == 0, 1, den), columns=names, index=query_index)


def top_bins(mass: pd.DataFrame, k: int = 50) -> list:
    return mass.abs().mean().sort_values(ascending=False).index[:k].tolist()


def sum_spans(scores: np.ndarray, trace_ids: np.ndarray) -> tuple[np.ndarray, pd.Index]:
    """Collapse span-level query rows (attrib.targets) into one row per trace."""
    onehot, names = _onehot(trace_ids)
    return np.asarray(onehot.T @ np.asarray(scores)), names

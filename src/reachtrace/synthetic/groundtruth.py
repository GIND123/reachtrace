"""Ground truth for the gate: leave-family-out and random-subset (LDS) retraining.

Leave-family-out alone gives about five points and is nearly trivial (drop A and f is
gone). The linear datamodeling score over random 50% document subsets (Park et al.
2023) is the quantitative check; the family-level ranking is the qualitative one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def leave_family_out(docs: pd.DataFrame, families: list[str]) -> dict[str, np.ndarray]:
    variants = {"full": docs.doc_id.to_numpy()}
    for f in families:
        variants[f"minus_{f}"] = docs.doc_id[docs.family != f].to_numpy()
    return variants


def subset_masks(n_docs: int, n_subsets: int, frac: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n_subsets, n_docs)) < frac


def lds(doc_scores: np.ndarray, masks: np.ndarray, outcomes: np.ndarray) -> float:
    """Spearman between predicted subset effect (sum of kept-doc scores) and measured outcome."""
    predicted = masks.astype(float) @ np.asarray(doc_scores, dtype=float)
    return float(spearmanr(predicted, outcomes).statistic)


def g1_recovery(
    mass: pd.DataFrame,
    causal: tuple[str, ...] = ("A", "B"),
    distractors: tuple[str, ...] = ("C", "D"),
) -> pd.Series:
    """mass: seed x family. True per seed when every causal family outranks every distractor."""
    return mass[list(causal)].min(axis=1) > mass[list(distractors)].max(axis=1)


def g4_stability(
    rank_a: np.ndarray, rank_b: np.ndarray, threshold: float = 0.6
) -> tuple[float, bool]:
    tau = float(kendalltau(rank_a, rank_b).statistic)
    return tau, tau >= threshold

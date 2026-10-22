"""Confound controls, reported next to every attribution metric (BehaviorTrace checklist).

1. norm-only: a query-free ranking by training-gradient norm.
2. fluency: per-query mean token log-prob, regressed out within difficulty strata.
3. random-query: the same pipeline on N-partition or non-math queries.
4. multi-draw: several independent correct traces per problem.
5. model-swap drift: base -> RL change measured on R / N / random queries (coinfluence.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .aggregate import bin_mass


def norm_only_profile(doc_norms: np.ndarray, bins: np.ndarray) -> pd.Series:
    """Bin-mass profile that gradient magnitude alone would produce."""
    return bin_mass(np.asarray(doc_norms)[None, :], bins).iloc[0]


def remove_profile(mass: pd.DataFrame, profile: pd.Series) -> pd.DataFrame:
    """Per query, regress the bin-mass vector on the control profile across bins; keep residuals."""
    x = profile.reindex(mass.columns).to_numpy()
    x = x - x.mean()
    denom = float(x @ x) or 1.0
    y = mass.to_numpy() - mass.to_numpy().mean(axis=1, keepdims=True)
    beta = y @ x / denom
    return pd.DataFrame(y - np.outer(beta, x), index=mass.index, columns=mass.columns)


def residualize(y: np.ndarray, x: np.ndarray, strata: np.ndarray | None = None) -> np.ndarray:
    """Residuals of y ~ x + stratum fixed effects (OLS)."""
    y = np.asarray(y, dtype=float)
    cols = [np.asarray(x, dtype=float)]
    if strata is not None:
        cols.append(pd.get_dummies(pd.Series(strata), drop_first=False).to_numpy(dtype=float))
    else:
        cols.append(np.ones((len(y), 1)))
    design = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def recall_at_k(scores: np.ndarray, positives: np.ndarray, k: int) -> float:
    """Fraction of the top-k scored training items that are true causes (synthetic gate)."""
    top = np.argsort(-np.asarray(scores))[:k]
    return float(np.asarray(positives)[top].mean())


def targeted_minus_control(
    targeted: np.ndarray, control: np.ndarray, positives: np.ndarray, k: int
) -> float:
    return recall_at_k(targeted, positives, k) - recall_at_k(control, positives, k)

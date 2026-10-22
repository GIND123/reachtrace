"""Inference utilities named in the pre-registered analysis plan."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats


def bootstrap_ci(
    x: np.ndarray,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    boots = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    return (
        float(stat(x)),
        float(np.quantile(boots, alpha / 2)),
        float(np.quantile(boots, 1 - alpha / 2)),
    )


def paired_diff_ci(
    x: np.ndarray,
    y: np.ndarray,
    stat: Callable[[np.ndarray], float] = np.median,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """CI for stat(x - y) over matched pairs (U problem, its CEM-matched R problem)."""
    return bootstrap_ci(np.asarray(x) - np.asarray(y), stat, n_boot, alpha, seed)


def stratified_permutation_test(
    values: np.ndarray,
    treated: np.ndarray,
    strata: np.ndarray | None = None,
    n_perm: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Difference in means (treated - control) with labels shuffled within strata.

    Returns (observed difference, two-sided p-value).
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    t = np.asarray(treated, dtype=bool)
    s = np.zeros(len(v), dtype=int) if strata is None else np.unique(strata, return_inverse=True)[1]

    def diff(lbl: np.ndarray) -> float:
        return v[lbl].mean() - v[~lbl].mean()

    obs = diff(t)
    groups = [np.flatnonzero(s == g) for g in np.unique(s)]
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = t.copy()
        for g in groups:
            perm[g] = rng.permutation(t[g])
        null[i] = diff(perm)
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + n_perm)
    return float(obs), float(p)


def benjamini_hochberg(pvals: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Returns (reject mask, BH-adjusted p-values) in the input order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty(m)
    out[order] = adj
    return out <= q, out


def tost(x: np.ndarray, y: np.ndarray, margin: float) -> float:
    """Welch two one-sided tests for |mean(x) - mean(y)| < margin. Returns the TOST p-value."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x.mean() - y.mean()
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = np.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx**2 / (len(x) - 1) + vy**2 / (len(y) - 1))
    p_lower = 1 - stats.t.cdf((d + margin) / se, df)
    p_upper = stats.t.cdf((d - margin) / se, df)
    return float(max(p_lower, p_upper))


def mde(n1: int, n2: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable two-sample mean difference, in units of the per-query SD."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt(1 / n1 + 1 / n2))


def kendall_split_half(rankings_a: np.ndarray, rankings_b: np.ndarray) -> float:
    return float(stats.kendalltau(rankings_a, rankings_b).statistic)

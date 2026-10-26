"""Continuous per-problem solve rates: the v3 replacement for certified U/R labels.

Certified "unreachable at K" labels mostly capture sharpening: 0/288 only certifies
p < 0.0103, so pass@1024 can still be ~1. v3 treats the base solve rate as a continuous
covariate (Jeffreys posterior, logit scale) and defines

  G (gained):   base 0 / n_base with n_base >= g_min_n, RL pass@1 >= g_min_rl
  B (boundary): base rate in (0, b_max], estimated from the first `first_n` samples

Rates for B use a fixed first-n sample count, so adaptive stopping cannot bias them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta


def jeffreys(c: int, n: int, level: float = 0.9) -> tuple[float, float, float]:
    """Posterior mean and equal-tailed interval of Beta(c + 1/2, n - c + 1/2)."""
    if n <= 0:
        return float("nan"), 0.0, 1.0
    a, b = c + 0.5, n - c + 0.5
    lo, hi = beta.ppf([(1 - level) / 2, (1 + level) / 2], a, b)
    return a / (a + b), float(lo), float(hi)


def logit_rate(c: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Logit of the Jeffreys posterior mean; finite even at c = 0 or c = n."""
    c, n = np.asarray(c, float), np.asarray(n, float)
    p = (c + 0.5) / (n + 1.0)
    return np.log(p / (1 - p))


def first_n_counts(gens: pd.DataFrame, first_n: int, col: str = "correct") -> pd.DataFrame:
    """(n, c) per problem from samples with sample_idx < first_n."""
    sub = gens[gens.sample_idx < first_n]
    return sub.groupby("problem_id")[col].agg(n="size", c="sum").reset_index()


def classify(
    table: pd.DataFrame,
    g_min_n: int = 256,
    g_min_rl: float = 0.10,
    b_max: float = 1 / 64,
    lenient_col: str | None = "c_base_lenient",
) -> pd.Series:
    """Label each problem G, B or other.

    table columns: n_base, c_base, n_rl, c_rl, and optionally c_base_lenient (base successes
    under the lenient grader) and n_base_first / c_base_first (fixed first-n counts for B).
    """
    rl_rate = table.c_rl / table.n_rl.where(table.n_rl > 0)
    base_zero = table.c_base == 0
    if lenient_col and lenient_col in table:
        base_zero &= table[lenient_col] == 0
    g = base_zero & (table.n_base >= g_min_n) & (rl_rate >= g_min_rl)
    n_first = table.get("n_base_first", table.n_base)
    c_first = table.get("c_base_first", table.c_base)
    rate_first = c_first / n_first.where(n_first > 0)
    b = (c_first > 0) & (rate_first <= b_max)
    return pd.Series(np.select([g, b], ["G", "B"], "other"), index=table.index)

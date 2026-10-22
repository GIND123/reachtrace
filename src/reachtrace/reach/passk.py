"""pass@k estimation and statistically certified reachability.

"Unreachable" only ever means "not observed in n samples". To *certify* that a problem's
pass@K is below gamma at one-sided level alpha with zero successes, you need

    n > K * ln(1/alpha) / ln(1/(1 - gamma))      (~4.32 K for alpha=0.05, gamma=0.5)

so certifying U at K=1024 needs ~4,400 base samples per problem. `n_to_certify` makes
this explicit; partitions at larger K are reported in "observed" mode only.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.stats import beta

Mode = Literal["certified", "estimate", "observed"]
Status = Literal["reach", "unreach", "undetermined"]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator 1 - C(n-c, k) / C(n, k) (Chen et al. 2021), numerically stable."""
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got n={n}, c={c}")
    if k > n:
        raise ValueError(f"k={k} exceeds n={n}")
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def cp_upper(c: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1 - alpha) Clopper-Pearson upper bound on the per-sample success rate."""
    return 1.0 if c >= n else float(beta.ppf(1 - alpha, c + 1, n - c))


def cp_lower(c: int, n: int, alpha: float = 0.05) -> float:
    return 0.0 if c <= 0 else float(beta.ppf(alpha, c, n - c + 1))


def reach_prob(p: float, k: int) -> float:
    """P(at least one success in k i.i.d. samples) for per-sample rate p."""
    return 1.0 - (1.0 - p) ** k


def n_to_certify(k: int, gamma: float = 0.5, alpha: float = 0.05) -> int:
    """Smallest n such that 0/n successes certifies pass@k < gamma at level alpha."""
    return math.floor(k * math.log(1 / alpha) / math.log(1 / (1 - gamma))) + 1


def reach_status(
    n: int, c: int, k: int, mode: Mode = "certified", gamma: float = 0.5, alpha: float = 0.05
) -> Status:
    if n == 0:
        return "undetermined"
    if mode == "observed":
        return "reach" if c > 0 else "unreach"
    if mode == "estimate":
        if k > n:
            return "undetermined"
        return "reach" if pass_at_k(n, c, k) >= gamma else "unreach"
    if reach_prob(cp_lower(c, n, alpha), k) >= gamma:
        return "reach"
    if reach_prob(cp_upper(c, n, alpha), k) < gamma:
        return "unreach"
    return "undetermined"

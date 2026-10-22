"""R/U/L/N reachability partitions (the instrument, not the contribution).

R = preservation, U = expansion, L = shrinkage, N = out-of-support, following the
empirical-support vocabulary of Wu et al. (Invisible Leash, arXiv 2507.14843).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .passk import Mode, cp_upper, pass_at_k, reach_status

_LABELS = {
    ("reach", "reach"): "R",
    ("unreach", "reach"): "U",
    ("reach", "unreach"): "L",
    ("unreach", "unreach"): "N",
}


def label(base_status: str, rl_status: str) -> str:
    return _LABELS.get((base_status, rl_status), "?")


def partition_table(
    counts: pd.DataFrame,
    ksweep: list[int],
    mode: Mode = "certified",
    gamma: float = 0.5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """counts: one row per problem with problem_id, n_base, c_base, n_rl, c_rl."""
    out = counts.copy()
    out["p_base_upper"] = [cp_upper(c, n, alpha) for n, c in zip(out.n_base, out.c_base)]
    for k in ksweep:
        for side in ("base", "rl"):
            n_col, c_col = out[f"n_{side}"], out[f"c_{side}"]
            out[f"passk_{side}@{k}"] = [
                pass_at_k(n, c, k) if k <= n else np.nan for n, c in zip(n_col, c_col)
            ]
        out[f"label@{k}"] = [
            label(
                reach_status(nb, cb, k, mode, gamma, alpha),
                reach_status(nr, cr, k, mode, gamma, alpha),
            )
            for nb, cb, nr, cr in zip(out.n_base, out.c_base, out.n_rl, out.c_rl)
        ]
    return out


def partition_counts(table: pd.DataFrame, k: int) -> pd.Series:
    return table[f"label@{k}"].value_counts().reindex(["R", "U", "L", "N", "?"], fill_value=0)


def bootstrap_counts(
    table: pd.DataFrame,
    k: int,
    mode: Mode = "certified",
    gamma: float = 0.5,
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Hierarchical bootstrap: resample problems, then redraw successes per problem.

    Per-sample rates are drawn from the Jeffreys posterior Beta(c+1/2, n-c+1/2) so that a
    0/n problem can still come out reachable in some replicates.
    """
    rng = np.random.default_rng(seed)
    rows = []
    m = len(table)
    for _ in range(n_boot):
        idx = rng.integers(0, m, m)
        t = table.iloc[idx]
        draws = {}
        for side in ("base", "rl"):
            n = t[f"n_{side}"].to_numpy()
            c = t[f"c_{side}"].to_numpy()
            p = rng.beta(c + 0.5, n - c + 0.5)
            draws[side] = (n, rng.binomial(n, p))
        labels = [
            label(
                reach_status(nb, cb, k, mode, gamma, alpha),
                reach_status(nr, cr, k, mode, gamma, alpha),
            )
            for nb, cb, nr, cr in zip(*draws["base"], *draws["rl"])
        ]
        rows.append(pd.Series(labels).value_counts())
    boot = pd.DataFrame(rows).fillna(0).reindex(columns=["R", "U", "L", "N", "?"], fill_value=0)
    return pd.DataFrame(
        {
            "point": partition_counts(table, k),
            "lo": boot.quantile(0.025),
            "hi": boot.quantile(0.975),
        }
    )

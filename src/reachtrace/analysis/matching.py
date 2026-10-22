"""Coarsened exact matching of U problems to comparison problems.

U problems have base pass@K = 0 by construction, so base pass@1 cannot be a matching
covariate. Match on RL-side difficulty and problem features instead, and use near-boundary
R ("hard R": base solve rate in (0, 5%]) as the primary comparison group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def coarsen(df: pd.DataFrame, spec: dict[str, int | list]) -> pd.Series:
    """spec: column -> number of quantile bins, or explicit bin edges; categorical if neither."""
    keys = []
    for col, how in spec.items():
        if isinstance(how, int):
            keys.append(pd.qcut(df[col], how, labels=False, duplicates="drop").astype(str))
        elif isinstance(how, list):
            keys.append(pd.cut(df[col], how, labels=False, include_lowest=True).astype(str))
        else:
            keys.append(df[col].astype(str))
    return pd.concat(keys, axis=1).agg("|".join, axis=1)


def cem(df: pd.DataFrame, treated: str, spec: dict[str, int | list | None]) -> pd.DataFrame:
    """Adds `stratum`, `matched`, `weight` (treated weight 1; controls reweighted per stratum)."""
    out = df.copy()
    out["stratum"] = coarsen(out, spec)
    t = out[treated].astype(bool)
    n_t = out[t].groupby("stratum").size()
    n_c = out[~t].groupby("stratum").size()
    both = n_t.index.intersection(n_c.index)
    out["matched"] = out.stratum.isin(both)
    m_t, m_c = n_t[both].sum(), n_c[both].sum()
    ratio = (n_t[both] / n_c[both]) * (m_c / max(m_t, 1))
    out["weight"] = np.where(
        ~out.matched, 0.0, np.where(t, 1.0, out.stratum.map(ratio).fillna(0.0))
    )
    return out


def matched_pairs(df: pd.DataFrame, treated: str, spec: dict, seed: int = 0) -> pd.DataFrame:
    """1:1 pairs within CEM strata (sampling controls without replacement) for paired tests."""
    rng = np.random.default_rng(seed)
    m = cem(df, treated, spec)
    m = m[m.matched]
    pairs = []
    for _, g in m.groupby("stratum"):
        tr = g[g[treated].astype(bool)]
        co = g[~g[treated].astype(bool)]
        k = min(len(tr), len(co))
        pairs.append(
            pd.DataFrame(
                {
                    "treated_idx": tr.index[rng.permutation(len(tr))[:k]],
                    "control_idx": co.index[rng.permutation(len(co))[:k]],
                    "stratum": g.stratum.iloc[0],
                }
            )
        )
    return pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()

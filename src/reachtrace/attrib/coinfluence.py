"""H3 (composition): is joint bin structure new for U beyond global representation drift?

Delta_P = ||C_P^RL - C_P^base||_F / ||C_P^base||_F over off-diagonal entries, where C_P^m is
the Spearman correlation across queries in partition P of their bin-mass vectors under
model m, restricted to P's top-k bins. The test statistic is Delta_U - max(Delta_refs),
computed with equal query counts per partition, since correlation-matrix noise shrinks with n.

Delta_U > Delta_R alone could just mean RL moved the model most on U-like text. The
likelihood-shift control regresses each query's attribution shift on how much RL changed
that trace's log-probability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .aggregate import top_bins


def coinfluence(mass: pd.DataFrame, bins: list) -> np.ndarray:
    ranks = np.apply_along_axis(rankdata, 0, mass[bins].to_numpy())
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(ranks, rowvar=False)
    return np.nan_to_num(c)


def node_jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def edge_distance(c_rl: np.ndarray, c_base: np.ndarray) -> float:
    off = ~np.eye(c_base.shape[0], dtype=bool)
    denom = np.linalg.norm(c_base[off]) or 1.0
    return float(np.linalg.norm((c_rl - c_base)[off]) / denom)


def partition_delta(mass_base: pd.DataFrame, mass_rl: pd.DataFrame, k: int = 50) -> dict:
    tb, tr = top_bins(mass_base, k), top_bins(mass_rl, k)
    shared = [b for b in tb if b in set(tr)]
    if len(shared) < 3:
        return {"jaccard": node_jaccard(tb, tr), "delta": float("nan"), "n_shared": len(shared)}
    delta = edge_distance(coinfluence(mass_rl, shared), coinfluence(mass_base, shared))
    return {"jaccard": node_jaccard(tb, tr), "delta": delta, "n_shared": len(shared)}


def drift_test(
    mass_base: pd.DataFrame,
    mass_rl: pd.DataFrame,
    part: pd.Series,
    target: str = "U",
    refs: tuple[str, ...] = ("R", "rand"),
    k: int = 50,
    n_boot: int = 500,
    n_perm: int = 500,
    seed: int = 0,
) -> dict:
    """mass_*: query x bin (same index); part: query -> partition label."""
    rng = np.random.default_rng(seed)
    groups = {p: part.index[part == p].to_numpy() for p in (target, *refs)}
    n_sub = min(len(v) for v in groups.values())
    if n_sub < 10:
        raise ValueError(f"too few queries for an equal-n co-influence test (min group = {n_sub})")

    def stat(assign: dict[str, np.ndarray], resample: bool) -> float:
        deltas = {}
        for p, ids in assign.items():
            sub = rng.choice(ids, n_sub, replace=resample)
            deltas[p] = partition_delta(mass_base.loc[sub], mass_rl.loc[sub], k)["delta"]
        return deltas[target] - max(deltas[r] for r in refs)

    # Equal-n subsampling is itself random, so the point estimate is a median over draws.
    observed = float(np.nanmedian([stat(groups, resample=False) for _ in range(20)]))
    boot = np.array([stat(groups, resample=True) for _ in range(n_boot)])

    pooled = np.concatenate(list(groups.values()))
    sizes = [len(v) for v in groups.values()]
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        cuts = np.cumsum(sizes)[:-1]
        null.append(stat(dict(zip(groups, np.split(perm, cuts))), resample=False))
    null = np.array(null)
    return {
        "stat": observed,
        "ci": (float(np.nanquantile(boot, 0.025)), float(np.nanquantile(boot, 0.975))),
        "p_perm": float((1 + np.sum(null >= observed)) / (1 + len(null))),
        "n_per_group": n_sub,
    }


def per_query_shift(mass_base: pd.DataFrame, mass_rl: pd.DataFrame) -> pd.Series:
    """1 - Spearman(bin-mass under base, under RL) for each query."""
    b = np.apply_along_axis(rankdata, 1, mass_base.to_numpy())
    r = np.apply_along_axis(rankdata, 1, mass_rl.loc[mass_base.index].to_numpy())
    b = b - b.mean(1, keepdims=True)
    r = r - r.mean(1, keepdims=True)
    rho = (b * r).sum(1) / np.sqrt((b * b).sum(1) * (r * r).sum(1))
    return pd.Series(1 - rho, index=mass_base.index)


def likelihood_shift_control(shift: pd.Series, dlogp: pd.Series, is_target: pd.Series) -> dict:
    """OLS shift ~ 1 + dlogp + is_target; returns the partition coefficient and its SE."""
    df = pd.concat({"y": shift, "x": dlogp, "t": is_target.astype(float)}, axis=1).dropna()
    X = np.column_stack([np.ones(len(df)), df.x, df.t])
    coef, *_ = np.linalg.lstsq(X, df.y, rcond=None)
    resid = df.y - X @ coef
    sigma2 = resid @ resid / max(len(df) - X.shape[1], 1)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    return {"coef_target": float(coef[2]), "se_target": float(se[2]), "coef_dlogp": float(coef[1])}

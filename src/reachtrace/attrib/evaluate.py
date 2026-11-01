"""Attribution quality against retraining ground truth (gate v3).

Sign convention: a score is positive for a proponent, i.e. a document whose training
raises the query objective J (lowers the advantage-weighted query loss). This is Bergson's
`higher_is_better: true` orientation for TrackStar and EK-FAC.

* LDS (Park et al. 2023; Bergson leaderboard): per query, the Spearman correlation across
  retrained subsets between the summed scores of the kept documents and the measured J of
  the model retrained on them. Reported as the mean over queries, with a bootstrap over
  subsets for the interval, as on the leaderboard.
* QLD (proponent removal): J of the full model minus J after retraining without a method's
  top-scored documents, compared with removing the same number of random documents.
* Family recovery: how much score mass each planted family receives (z-scored against all
  documents) and the AUC of causal families against everything else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def lds_per_query(scores: np.ndarray, weights: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """scores: docs x queries; weights: subsets x docs (kept weight); outcomes: subsets x
    queries (J after retraining). Returns one Spearman correlation per query."""
    pred = np.asarray(weights, float) @ np.asarray(scores, float)  # subsets x queries
    out = np.asarray(outcomes, float)
    return np.array([spearmanr(pred[:, q], out[:, q]).statistic for q in range(out.shape[1])])


def lds_summary(scores: np.ndarray, weights: np.ndarray, outcomes: np.ndarray,
                n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean LDS over queries with a 95% bootstrap interval over subsets."""
    per_q = lds_per_query(scores, weights, outcomes)
    rng = np.random.default_rng(seed)
    n = len(weights)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        with np.errstate(invalid="ignore"):
            boots.append(np.nanmean(lds_per_query(scores, weights[idx], outcomes[idx])))
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return {"lds": float(np.nanmean(per_q)), "lds_lo": float(lo), "lds_hi": float(hi),
            "lds_median": float(np.nanmedian(per_q)), "lds_min": float(np.nanmin(per_q)),
            "n_queries": int(np.isfinite(per_q).sum())}


def family_recovery(scores: np.ndarray, families: np.ndarray,
                    causal: tuple[str, ...] = ("A", "B")) -> dict:
    """Score of each family (mean z over queries) and the AUC of causal vs all others."""
    s = np.asarray(scores, float)
    z = (s - s.mean(0)) / (s.std(0) + 1e-12)
    fam = pd.Series(np.asarray(families))
    out = {f"z_{f}": float(z[(fam == f).to_numpy()].mean()) for f in sorted(fam.unique())}
    pos = fam.isin(causal).to_numpy()
    aucs = []
    for q in range(s.shape[1]):
        r = rankdata(s[:, q])
        aucs.append((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))
    out["auc_causal"] = float(np.mean(aucs))
    others = [k for k in out if k.startswith("z_") and k[2:] not in causal]
    out["causal_above_all"] = bool(min(out[f"z_{c}"] for c in causal)
                                   > max(out[k] for k in others))
    return out


def precision_at_k(score: np.ndarray, positive: np.ndarray, k: int) -> float:
    top = np.argsort(-np.asarray(score))[:k]
    return float(np.asarray(positive)[top].mean())


def topk_agreement(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Share of b's top k that a also puts in its top k."""
    ta, tb = set(np.argsort(-np.asarray(a))[:k]), set(np.argsort(-np.asarray(b))[:k])
    return len(ta & tb) / k


def top_k_rows(score: np.ndarray, k: int) -> np.ndarray:
    return np.sort(np.argsort(-np.asarray(score))[:k])


def qld(j_full: float, j_removed: float, j_random: np.ndarray) -> dict:
    """Drop in J from removing a method's top documents, against random removals."""
    drop = j_full - j_removed
    rnd = j_full - np.asarray(j_random, float)
    return {"qld": float(drop), "qld_random_mean": float(rnd.mean()),
            "qld_minus_random": float(drop - rnd.mean())}

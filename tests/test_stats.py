import numpy as np
import pandas as pd
import pytest

from reachtrace.analysis.matching import cem, matched_pairs
from reachtrace.analysis.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    mde,
    stratified_permutation_test,
    tost,
)


def test_bh_matches_hand_computed():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    reject, adj = benjamini_hochberg(p, 0.05)
    assert adj == pytest.approx([0.04, 0.0533333, 0.0533333, 0.20], rel=1e-4)
    assert reject.tolist() == [True, False, False, False]


def test_mde_matches_plan_numbers():
    assert mde(300, 300) == pytest.approx(0.229, abs=0.002)
    assert mde(80, 80) == pytest.approx(0.443, abs=0.002)


def test_bootstrap_ci_contains_estimate():
    x = np.random.default_rng(0).normal(1.0, 1.0, 200)
    est, lo, hi = bootstrap_ci(x, n_boot=2000)
    assert lo < est < hi and lo < 1.0 < hi


def test_permutation_detects_effect_and_respects_null():
    rng = np.random.default_rng(1)
    t = np.repeat([True, False], 100)
    strata = np.tile([0, 1], 100)
    _, p_eff = stratified_permutation_test(rng.normal(0, 1, 200) + 0.8 * t, t, strata, n_perm=2000)
    _, p_null = stratified_permutation_test(rng.normal(0, 1, 200), t, strata, n_perm=2000)
    assert p_eff < 0.01 and p_null > 0.01


def test_tost_equivalence():
    rng = np.random.default_rng(2)
    x, y = rng.normal(0, 1, 2000), rng.normal(0, 1, 2000)
    assert tost(x, y, margin=0.2) < 0.05
    assert tost(x + 1.0, y, margin=0.2) > 0.5


def test_cem_drops_unmatched_strata():
    df = pd.DataFrame(
        {
            "is_u": [1, 1, 0, 0, 0, 1],
            "src": ["aime", "math", "math", "math", "amc", "gsm"],
        }
    )
    m = cem(df, "is_u", {"src": None})
    assert m.matched.tolist() == [False, True, True, True, False, False]
    # Iacus-King-Porro weights: matched controls sum to the number of matched controls.
    assert m.loc[m.is_u == 0, "weight"].sum() == pytest.approx(2.0)
    assert len(matched_pairs(df, "is_u", {"src": None})) == 1

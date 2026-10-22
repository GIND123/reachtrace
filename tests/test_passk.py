import itertools
import math

import pytest

from reachtrace.reach.passk import cp_upper, n_to_certify, pass_at_k, reach_status


def brute_pass_at_k(n, c, k):
    outcomes = [1] * c + [0] * (n - c)
    subsets = list(itertools.combinations(range(n), k))
    return sum(any(outcomes[i] for i in s) for s in subsets) / len(subsets)


@pytest.mark.parametrize(
    "n,c,k", [(6, 0, 2), (6, 1, 2), (6, 3, 3), (7, 2, 5), (5, 5, 1), (8, 1, 8)]
)
def test_pass_at_k_matches_brute_force(n, c, k):
    assert pass_at_k(n, c, k) == pytest.approx(brute_pass_at_k(n, c, k))


def test_rule_of_three():
    assert cp_upper(0, 1000) == pytest.approx(3 / 1000, rel=0.01)


def test_n_to_certify_is_about_4_32_k():
    for k in (16, 64, 256, 1024):
        n = n_to_certify(k)
        assert n == pytest.approx(k * math.log(20) / math.log(2), abs=1)
        assert reach_status(n, 0, k) == "unreach"
        assert reach_status(n - 1, 0, k) == "undetermined"


def test_modes():
    assert reach_status(64, 0, 64, mode="observed") == "unreach"
    assert reach_status(64, 0, 64, mode="certified") == "undetermined"
    assert reach_status(64, 40, 1, mode="certified") == "reach"
    assert reach_status(32, 1, 64, mode="estimate") == "undetermined"  # k > n

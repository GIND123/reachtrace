import numpy as np
import pandas as pd
import pytest

from reachtrace.attrib.rl_scoring import (
    advantage_weighted,
    contrastive,
    group_advantages,
    label_mask_row,
    rl_signal,
    sign_only,
)


def _group():
    # One success, seven failures; every rollout gradient points the same way (cos = 1).
    items = pd.DataFrame({"group_id": ["g"] * 8, "reward": [1.0] + [0.0] * 7})
    items["advantage"] = group_advantages(items)
    return items, np.ones((1, 8))


def test_advantage_after_normalization_keeps_zero_sum():
    items, cos = _group()
    assert items.advantage.sum() == pytest.approx(0.0)
    s = advantage_weighted(cos, items).iloc[0]
    assert s.score == pytest.approx(0.0)
    assert s.pos == pytest.approx(0.875) and s.neg == pytest.approx(-0.875)


def test_sign_only_is_biased_negative():
    items, cos = _group()
    assert sign_only(cos, items).iloc[0].score == pytest.approx(1 - 7)


def test_rl_signal_subtracts_placebo():
    d = pd.DataFrame({"query": [0, 0, 1], "score": [0.3, 0.1, 0.0]})
    p = pd.DataFrame({"query": [0, 1], "score": [0.1, 0.05]})
    sig = rl_signal(d, p)
    assert sig[0] == pytest.approx(0.1) and sig[1] == pytest.approx(-0.05)


def test_contrastive_cancels_shared_format():
    shared = np.array([[0.5, 0.2], [0.5, 0.2]])
    assert contrastive(shared + [[0.1, 0.0]] * 2, shared).tolist() == pytest.approx([0.1, 0.0])


def test_label_mask_row():
    row = label_mask_row([1, 2], [3, 4, 5], np.array([False, True, True]))
    assert row == {"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, -100, 4, 5]}
    with pytest.raises(ValueError):
        label_mask_row([1], [2, 3], np.array([True]))

import numpy as np
import pandas as pd
import pytest

from reachtrace.attrib.aggregate import bin_mass, stage_share, sum_spans
from reachtrace.attrib.bergson_cmd import build_cmd, score_cmd
from reachtrace.attrib.coinfluence import (
    drift_test,
    edge_distance,
    node_jaccard,
    partition_delta,
    per_query_shift,
)
from reachtrace.attrib.controls import norm_only_profile, remove_profile, residualize
from reachtrace.attrib.targets import answer_mask, fork_mask, span_queries, spans
from reachtrace.config import BergsonIndexConfig


def test_bin_mass_centered_and_weighted():
    scores = np.array([[1.0, 1.0, 4.0, 4.0], [0.0, 2.0, 0.0, 2.0]])
    m = bin_mass(scores, np.array(["a", "a", "b", "b"]))
    assert m.loc[0].tolist() == [-1.5, 1.5]
    assert m.loc[1].tolist() == [0.0, 0.0]


def test_stage_share_signed():
    s = stage_share(np.array([[1.0, -1.0, 2.0]]), np.array(["pre", "pre", "rl"]))
    assert s.loc[0, "pre"] == 0.0 and s.loc[0, "rl"] == pytest.approx(0.5)


def test_sum_spans_collapses_rows():
    total, names = sum_spans(np.ones((3, 2)), np.array(["t1", "t1", "t2"]))
    assert names.tolist() == ["t1", "t2"] and total.tolist() == [[2, 2], [1, 1]]


def test_fork_and_answer_masks():
    lr = fork_mask(np.log([0.9, 0.9, 0.9]), np.log([0.85, 0.1, 0.9]))
    assert lr.tolist() == [False, True, False]
    pieces = ["So", " x", "=2", ".\n", "Answer", ":", " 2"]
    assert answer_mask(pieces).tolist() == [False] * 4 + [True] * 3


def test_span_queries_reconstruct_text():
    pieces = ["a", "b", "c", "d", "e"]
    mask = np.array([False, True, False, True, True])
    assert spans(mask, merge_gap=0) == [(1, 2), (3, 5)]
    rows = span_queries("P:", pieces, mask, merge_gap=0)
    assert [r["prompt"] + r["completion"] for r in rows] == ["P:ab", "P:abcde"]


def test_coinfluence_identity_has_zero_delta():
    rng = np.random.default_rng(0)
    mass = pd.DataFrame(rng.normal(size=(40, 8)), columns=list("abcdefgh"))
    d = partition_delta(mass, mass, k=8)
    assert d["jaccard"] == 1.0 and d["delta"] == pytest.approx(0.0)
    assert node_jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert edge_distance(np.eye(3), np.eye(3)) == 0.0


def test_drift_test_flags_planted_structure_change():
    rng = np.random.default_rng(0)
    n, bins = 60, list("abcdefghij")
    idx = [f"q{i}" for i in range(3 * n)]
    part = pd.Series(["U"] * n + ["R"] * n + ["rand"] * n, index=idx)
    base = pd.DataFrame(rng.normal(size=(3 * n, 10)), index=idx, columns=bins)
    rl = base + rng.normal(scale=0.05, size=base.shape)
    latent = rng.normal(size=n)
    rl.loc[part == "U", "a"] = latent * 3  # new joint structure among a, b on U only
    rl.loc[part == "U", "b"] = latent * 3
    out = drift_test(base, rl, part, k=10, n_boot=50, n_perm=50)
    assert out["stat"] > 0 and out["p_perm"] < 0.1
    assert per_query_shift(base, base).abs().max() < 1e-9


def test_controls():
    prof = norm_only_profile(np.array([1.0, 1.0, 3.0, 3.0]), np.array(["a", "a", "b", "b"]))
    mass = pd.DataFrame([[-2.0, 2.0], [-1.0, 1.0]], columns=["a", "b"])
    assert np.allclose(remove_profile(mass, prof).to_numpy(), 0.0)
    r = residualize(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.allclose(r, 0.0)


def test_bergson_cmds():
    cfg = BergsonIndexConfig(
        name="items",
        model="m",
        dataset="d.parquet",
        prompt_column="rendered_prompt",
        completion_column="completion",
        advantage_column="advantage",
        tokenizer="tok",
    )
    cmd = build_cmd(cfg, "runs")
    assert cmd[:3] == ["bergson", "build", "runs/items"]
    assert "--advantage_column" in cmd and "--tokenizer" in cmd and "--truncation" in cmd
    s = score_cmd(cfg, "runs/q", "runs")
    assert "--query_path" in s and "--unit_normalize" in s
    with pytest.raises(ValueError):
        build_cmd(cfg.model_copy(update={"reward_column": "reward"}), "runs")

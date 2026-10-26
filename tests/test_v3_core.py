import numpy as np
import pandas as pd
import pytest

from reachtrace.data.evals import is_numeric_answer, stratified_subset
from reachtrace.handoff.core import (
    composition_signature,
    critical_segments,
    handoff,
    prefix_lengths,
    refine_points,
    success_curve,
)
from reachtrace.ledger.core import (
    answer_start_token,
    margins,
    region_masks,
    stage_increments,
    trace_scores,
)
from reachtrace.reach.rates import classify, first_n_counts, jeffreys, logit_rate


def test_jeffreys_and_logit_are_finite_at_zero():
    mean, lo, hi = jeffreys(0, 256)
    assert 0 < lo < mean < hi < 0.02
    assert np.isfinite(logit_rate(np.array([0, 256]), np.array([256, 256]))).all()


def test_first_n_counts_ignores_adaptive_extras():
    gens = pd.DataFrame(
        {"problem_id": ["a"] * 6, "sample_idx": range(6), "correct": [1, 0, 0, 0, 1, 1]}
    )
    c = first_n_counts(gens, 4).iloc[0]
    assert (c.n, c.c) == (4, 1)


def test_classify_g_b_other():
    t = pd.DataFrame(
        {
            "n_base": [256, 256, 64, 256, 64],
            "c_base": [0, 0, 1, 0, 20],
            "c_base_lenient": [0, 2, 1, 0, 20],
            "n_rl": [64] * 5,
            "c_rl": [20, 20, 20, 2, 60],
        }
    )
    # G; lenient success disqualifies G; B (1/64); RL too weak for G; easy.
    assert classify(t).tolist() == ["G", "other", "B", "other", "other"]


def test_numeric_answer_filter_and_subset():
    assert is_numeric_answer("2(3)") is False and is_numeric_answer("\\frac{13}{32}")
    assert is_numeric_answer("277") and not is_numeric_answer("\\text{Yes}")
    df = pd.DataFrame({"eval_set": ["a"] * 5 + ["b"] * 2, "x": range(7)})
    s = stratified_subset(df, {"a": 3, "b": 5}, seed=0)
    assert s.eval_set.value_counts().to_dict() == {"a": 3, "b": 2}


def _decode(ids):
    return "".join(ids)


def test_answer_start_and_masks():
    toks = ["So", " x", "=", "2", ".\n", "Ans", "wer", ":", " 2"]
    i = answer_start_token(toks, _decode)
    assert i == 5
    assert answer_start_token(["no", " answer"], _decode) == 2
    m = region_masks(9, i, opening=2)
    assert m["opening"].sum() == 2 and m["content"].sum() == 3 and m["answer"].sum() == 4


def test_trace_scores_means_per_mask():
    lp = np.array([-1.0, -1.0, -2.0, -4.0, -0.5])
    s = trace_scores(lp, answer_start=4, opening=2)
    assert s["opening"] == -1.0 and s["content"] == -3.0 and s["answer"] == -0.5
    assert s["n_content"] == 2


def test_margins_and_telescoping_stages():
    rows = []
    for ck, (lc, li) in {"s1": (-2.0, -2.0), "s2": (-1.5, -2.0), "rl": (-1.0, -2.5)}.items():
        rows += [
            dict(problem_id="p", checkpoint=ck, trace_id=f"c{ck}", correct=True, content=lc),
            dict(problem_id="p", checkpoint=ck, trace_id=f"i{ck}", correct=False, content=li),
        ]
    m = margins(pd.DataFrame(rows))
    wide = m.pivot(index="problem_id", columns="checkpoint", values="M")
    assert wide.loc["p"].to_dict() == {"rl": 1.5, "s1": 0.0, "s2": 0.5}
    inc = stage_increments(wide, [("mid", "s2"), ("rl", "rl")], start="s1")
    assert inc.loc["p", "total"] == 1.5
    assert inc.loc["p", "share_mid"] == pytest.approx(1 / 3)
    assert inc.loc["p", "share_rl"] == pytest.approx(2 / 3)


def test_handoff_curve_and_segments():
    pl = prefix_lengths(100, (0.1, 0.5), opening=16)
    assert pl == {0.0: 16, 0.1: 16, 0.5: 50}
    curve = success_curve({0.0: (16, 0), 0.25: (16, 1), 0.5: (16, 8), 0.75: (16, 9), 0.9: (16, 15)})
    assert handoff(curve, 0.25) == 0.5
    assert [s[:2] for s in critical_segments(curve)] == [(0.25, 0.5), (0.75, 0.9)]
    assert composition_signature(curve, base_rate=0.0)
    assert not composition_signature(curve, base_rate=0.1)
    assert refine_points(curve) == pytest.approx([1 / 3, 5 / 12])
    single = success_curve({0.0: (16, 0), 0.25: (16, 8), 0.5: (16, 16)})
    assert not composition_signature(single, base_rate=0.0)
    assert handoff(success_curve({0.0: (8, 0)})) == float("inf")


def test_within_stage_config_loads():
    from reachtrace.config import WithinStageConfig, load_config

    cfg = load_config("configs/attribution/v3_within_stage.yaml", WithinStageConfig)
    assert cfg.total_docs("pretrain") == 120_000
    assert cfg.total_docs("midtrain") == 80_000
    assert {s.stage for s in cfg.stages} == {"pretrain", "midtrain", "longctx", "rl", "rl_placebo"}

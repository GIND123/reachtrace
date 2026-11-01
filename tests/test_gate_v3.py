import json
import random

import numpy as np
import pandas as pd
import pytest

from reachtrace.attrib import bergson_yaml as by
from reachtrace.attrib.evaluate import (
    family_recovery,
    lds_per_query,
    lds_summary,
    precision_at_k,
    qld,
    topk_agreement,
)
from reachtrace.attrib.grpo_query import (
    contrast_scale,
    dr_grpo_advantage,
    per_problem,
    query_rows,
    reweight_columns,
)
from reachtrace.remote import lambda_cloud as lc
from reachtrace.synthetic.cpt import grpo_objective, schedule
from reachtrace.synthetic.tasks import make_ops, verify
from reachtrace.synthetic.v3 import (
    decoy_doc,
    disjoint_task_sets,
    dose_design,
    dose_weights,
    make_corpus_v3,
    solution_text,
)


def test_grpo_query_equals_scaled_contrast():
    rng = np.random.default_rng(0)
    n, k, d = 8, 3, 50
    g = rng.normal(size=(n, d))
    r = np.array([1.0] * k + [0.0] * (n - k))
    a = dr_grpo_advantage(pd.Series(r), pd.Series(["p"] * n)).to_numpy()
    lhs = (a[:, None] * g).sum(0)
    rhs = contrast_scale(k, n) * (g[:k].mean(0) - g[k:].mean(0))
    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_reweight_restores_advantage_weighted_cosines():
    # A unit-normalized GRPO query keeps sign(A) only; |A| reweighting restores sum A_i cos.
    rng = np.random.default_rng(1)
    q = rng.normal(size=(4, 10))
    docs = rng.normal(size=(6, 10))
    adv = np.array([0.75, -0.25, -0.25, -0.25])
    unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)  # noqa: E731
    signed = unit(adv[:, None] * q)                                  # what Bergson stores
    cols = unit(docs) @ signed.T                                     # docs x rows
    got, _ = per_problem(reweight_columns(cols, adv), np.zeros(4))
    want = (unit(docs) @ unit(q).T) @ adv
    np.testing.assert_allclose(got[:, 0], want, atol=1e-12)


def test_query_rows_drop_zero_advantage_and_mask_prompt():
    ro = pd.DataFrame({
        "problem_id": ["a"] * 2 + ["b"] * 2,
        "prompt_ids": [[1, 2]] * 4,
        "completion_ids": [[3], [4, 5], [6], [7]],
        "reward": [1.0, 0.0, 1.0, 1.0],
    })
    rows = query_rows(ro)
    assert rows.problem_id.tolist() == ["a", "a"]
    assert rows.advantage.tolist() == [0.5, -0.5]
    assert rows.labels.iloc[1] == [-100, -100, 4, 5]
    assert rows.input_ids.iloc[1] == [1, 2, 4, 5]


def test_per_problem_sums_columns():
    s = np.arange(12, dtype=float).reshape(3, 4)
    out, names = per_problem(s, np.array(["x", "y", "x", "y"]))
    assert names.tolist() == ["x", "y"]
    np.testing.assert_allclose(out[:, 0], s[:, 0] + s[:, 2])


def test_grpo_objective_matches_margin():
    lp = np.array([-1.0, -2.0, -4.0, -3.0])
    adv = np.array([0.5, 0.5, -0.5, -0.5])
    J = grpo_objective(lp, adv, np.array(["p"] * 4))
    assert J["p"] == pytest.approx(contrast_scale(2, 4) * ((-1.5) - (-3.5)))


def test_lds_is_one_for_exact_linear_outcomes():
    rng = np.random.default_rng(2)
    scores = rng.normal(size=(30, 2))
    w = (rng.random((25, 30)) < 0.5).astype(float)
    out = w @ scores
    np.testing.assert_allclose(lds_per_query(scores, w, out), 1.0)
    assert lds_summary(scores, w, out, n_boot=50)["lds"] == pytest.approx(1.0)


def test_family_recovery_and_topk():
    fam = np.array(["A"] * 5 + ["C"] * 5)
    s = np.r_[np.ones(5), np.zeros(5)][:, None] + 0.01 * np.arange(10)[:, None]
    rec = family_recovery(s, fam, causal=("A",))
    assert rec["auc_causal"] == 1.0 and rec["causal_above_all"]
    assert precision_at_k(s[:, 0], fam == "A", 5) == 1.0
    assert topk_agreement(s[:, 0], s[:, 0], 3) == 1.0
    assert qld(1.0, 0.5, np.array([0.9, 0.9]))["qld_minus_random"] == pytest.approx(0.4)


def test_corpus_v3_families_and_decoys():
    ops = make_ops(0)
    docs = make_corpus_v3(ops, 20, seed=0)
    assert pd.Series([d["family"] for d in docs]).value_counts().eq(20).all()
    m = [d["text"] for d in docs if d["family"] == "M"]
    assert all("zorp" in t.lower() and "blick" in t.lower() for t in m)
    assert not any("becomes" in t or "->" in t for t in m)      # names only, no operation
    w = [d["text"] for d in docs if d["family"] == "W"]
    assert all("zarp(" in t and "blink(" in t and "Answer:" in t for t in w)
    assert decoy_doc(("zorp", "blick"), random.Random(0))


def test_solution_text_is_graded_correct():
    ops = make_ops(0)
    t = disjoint_task_sets(ops, "AB", [4], {"x": 5}, seed=1)["x"]
    for task in t:
        assert verify(task["solution"], task["answer"])
        assert task["solution"] == solution_text([ops["A"], ops["B"]], task["input"])


def test_disjoint_task_sets_share_no_input():
    sets = disjoint_task_sets(make_ops(0), "AB", [3, 4], {"rl": 300, "held": 50}, seed=3)
    assert not {t["input"] for t in sets["rl"]} & {t["input"] for t in sets["held"]}


def test_dose_weights_are_nested():
    groups = np.array(list("AAAABBBB"))
    lo = dose_weights(groups, {"A": 0.25, "B": 0.5}, seed=0)
    hi = dose_weights(groups, {"A": 0.75, "B": 1.0}, seed=0)
    assert lo.sum() == 3 and hi.sum() == 7 and np.all(hi >= lo)
    d = dose_design(["A", "B"], 10)
    assert d.shape == (10, 2) and set(np.unique(d)) <= {0.0, 0.25, 0.5, 0.75, 1.0}


def test_schedule_is_deterministic_and_covers_epochs():
    a, b = schedule(10, 3, 4, seed=0), schedule(10, 3, 4, seed=0)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert np.bincount(np.concatenate(a)).tolist() == [3] * 10


def test_bergson_yaml_steps(tmp_path):
    train = by.data_cfg("corpus")
    query = by.data_cfg("queries", split="train[0:8]", advantage=True)
    ts = by.trackstar_step("runs/ts", "m", train, query, projection_dim=32)["trackstar"]
    assert ts["index_cfg"]["projection_dim"] == 32
    assert ts["trackstar_cfg"]["preprocess_cfg"]["unit_normalize"] is True
    assert ts["trackstar_cfg"]["query"]["advantage_column"] == "advantage"
    assert ts["trackstar_cfg"]["query"]["aggregation"] == "none"
    h = by.kfac_hessian_step("runs/ek", "m", train, subset=1800)["hessian"]
    assert h["index_cfg"]["data"]["split"] == "train[:1800]"
    assert h["index_cfg"]["run_path"] == "runs/ek/hessian/kfac"
    assert h["hessian_cfg"] == {"method": "kfac", "ev_correction": True}
    inf = by.influence_step("runs/ek/p0", "m", train, query, astra=by.astra_cfg(1e-6),
                            candidates=by.candidates_cfg("c.csv", 10))["influence"]
    assert inf["preprocess_cfg"]["unit_normalize"] is False
    assert inf["hessian_pipeline_cfg"]["resume"] is True
    assert inf["hessian_pipeline_cfg"]["query"]["aggregation"] == "sum"
    assert inf["score_cfg"]["candidates"] == {"scores": "c.csv", "top_k": 10}
    path = by.write_pipeline(tmp_path / "x.yaml", "runs/x", [{"influence": inf}])
    import yaml

    assert yaml.safe_load(path.read_text())["steps"][0]["influence"]["hessian_cfg"]


def test_lambda_ledger_bills_open_and_closed_instances(tmp_path):
    led = tmp_path / "ledger.jsonl"
    rows = [
        {"event": "launch", "id": "a", "price": 2.0, "t": 0.0},
        {"event": "terminate", "id": "a", "t": 3600.0},
        {"event": "launch", "id": "b", "price": 1.0, "t": 0.0},
    ]
    led.write_text("\n".join(json.dumps(r) for r in rows))
    assert lc.spent(now=1800.0 + 1, ledger=led) == pytest.approx(2.0 + 31 / 60)
    assert lc.open_instances(led) == ["b"]


def test_watchdog_terminates_the_given_instance():
    s = lc.watchdog_script("abc123", 1_000)
    assert "abc123" in s and "instance-operations/terminate" in s and "-lt 1000" in s
    assert "wd_key" in s


def test_lambda_names_come_from_environment(monkeypatch):
    monkeypatch.delenv("LAMBDA_INSTANCE_NAME", raising=False)
    monkeypatch.delenv("LAMBDA_SSH_KEY_NAME", raising=False)
    assert lc.instance_name() == "reachtrace-gpu"
    monkeypatch.setenv("LAMBDA_INSTANCE_NAME", "box")
    assert lc.ssh_key_name() == "box"


def test_prompt_tokens_are_a_prefix_of_prompt_plus_solution():
    tr = pytest.importorskip("transformers")
    try:
        tok = tr.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    except Exception:
        pytest.skip("tokenizer not available offline")
    t = disjoint_task_sets(make_ops(0), "AB", [4], {"x": 3}, seed=1)["x"]
    for task in t:
        p = tok(task["prompt"], add_special_tokens=False).input_ids
        full = tok(task["prompt"] + task["solution"], add_special_tokens=False).input_ids
        assert full[: len(p)] == p


def test_within_stage_config_follows_bergson_advice():
    from reachtrace.config import WithinStageConfig, load_config

    cfg = load_config("configs/attribution/v3_within_stage.yaml", WithinStageConfig)
    assert cfg.projection_dim in (32, 64) and cfg.unit_norm and cfg.query == "grpo"
    assert cfg.rerank.method == "ekfac" and cfg.rerank.hessian_subset == 30_000
    assert cfg.rerank.overfetch * cfg.rerank.top_k == 10_000


def test_merge_query_indices_stacks_rows(tmp_path):
    pytest.importorskip("bergson")
    from bergson.data import create_index, load_gradients

    dirs = []
    for j in range(3):
        d = tmp_path / f"p{j}"
        mm = create_index(d, 1, {"m.a": 2, "m.b": 1}, np.float32)
        mm[:] = j + np.arange(3, dtype=np.float32)
        mm.flush()
        dirs.append(d)
    out = by.merge_query_indices(dirs, tmp_path / "merged")
    g = np.asarray(load_gradients(out))
    assert g.shape == (3, 3) and np.allclose(g[2], [2, 3, 4])
    assert json.loads((out / "info.json").read_text())["num_grads"] == 3

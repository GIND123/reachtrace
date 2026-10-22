import numpy as np
import pandas as pd

from reachtrace.config import SyntheticConfig
from reachtrace.synthetic.groundtruth import g1_recovery, lds, leave_family_out, subset_masks
from reachtrace.synthetic.tasks import compose, make_ops, make_planted_corpus, task_suite, verify


def test_ops_are_deterministic_and_distinct():
    a, b = make_ops(0), make_ops(0)
    assert a["A"].table == b["A"].table and a["A"].table != a["C"].table
    assert all(k != v for k, v in a["A"].table.items())
    assert a["B"].fn("1234") == "3214"
    assert a["D"].fn("12345") == "21435"


def test_no_document_shows_the_composition():
    ops = make_ops(0)
    docs = make_planted_corpus(ops, 300, seed=0)
    assert not any("zorp" in d["text"] and "blick" in d["text"] for d in docs)
    for d in docs:
        assert ops[d["family"]].name in d["text"]


def test_compose_and_verify():
    ops = make_ops(0)
    expr, ans = compose([ops["A"], ops["B"]], "4012")
    assert expr == "blick(zorp(4012))" and ans == ops["B"].fn(ops["A"].fn("4012"))
    assert verify(f"steps...\nAnswer: {ans}", ans) and not verify("Answer: 0", ans)


def test_task_suite_separates_train_and_u_lengths():
    cfg = SyntheticConfig(n_train_tasks=50, n_eval_tasks=20)
    suite = task_suite(make_ops(0), cfg)
    assert {len(t["input"]) for t in suite["rl_train"]} <= set(cfg.train_lengths)
    assert {len(t["input"]) for t in suite["eval_long"]} <= set(cfg.eval_lengths)
    assert all(t["chain"] == "ABA" for t in suite["eval_deep"])


def test_lds_and_g1():
    rng = np.random.default_rng(0)
    true = rng.normal(size=200)
    masks = subset_masks(200, 30, seed=1)
    outcomes = masks @ true + rng.normal(scale=0.1, size=30)
    assert lds(true, masks, outcomes) > 0.9
    mass = pd.DataFrame({"A": [3, 2], "B": [2, 2], "C": [1, 2.5], "D": [0, 0]}, index=[0, 1])
    assert g1_recovery(mass).tolist() == [True, False]
    docs = pd.DataFrame({"doc_id": ["a1", "b1", "c1"], "family": ["A", "B", "C"]})
    assert leave_family_out(docs, ["A"])["minus_A"].tolist() == ["b1", "c1"]

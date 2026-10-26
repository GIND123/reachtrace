import random

import numpy as np

from reachtrace.attrib.lexical import BM25
from reachtrace.synthetic.tasks import make_ops
from reachtrace.synthetic.v2 import (
    alias_pools,
    dose_design,
    make_alias_tasks,
    make_corpus_v2,
    subset_doc_mask,
)


def _setup():
    ops = make_ops(0)
    pools = alias_pools(list(ops), per_family=6, seed=0)
    return ops, pools


def test_alias_pools_are_disjoint():
    _, pools = _setup()
    allnames = [a for p in pools.values() for a in p]
    assert len(allnames) == len(set(allnames))


def test_task_prompts_share_no_alias_with_documents():
    ops, pools = _setup()
    docs = make_corpus_v2(ops, pools, docs_per_family=50, n_format=20, n_worked=20, seed=0)
    tasks = make_alias_tasks(ops, pools, ["A", "B"], [3, 4], 30, seed=1, split="train")
    doc_text = " ".join(d["text"] for d in docs)
    for f in ("A", "B"):
        assert pools[f][-1] not in doc_text
    assert all(pools["A"][-1] in t["prompt"] and pools["B"][-1] in t["prompt"] for t in tasks)
    fams = {d["family"] for d in docs}
    assert fams == {"A", "B", "C", "D", "F", "W"}
    assert {d["stage"] for d in docs if d["family"] == "W"} == {"midtrain"}


def test_bm25_cannot_key_on_operation_names():
    ops, pools = _setup()
    docs = make_corpus_v2(ops, pools, docs_per_family=40, n_format=40, n_worked=0, seed=0)
    task = make_alias_tasks(ops, pools, ["A", "B"], [4], 1, seed=2, split="eval")[0]
    s = BM25([d["text"] for d in docs]).scores(task["prompt"])
    top = [docs[i]["family"] for i in np.argsort(-s)[:20]]
    # Without shared names, the top hits are format documents, not the causal families.
    assert top.count("F") >= 10


def test_dose_design_balanced_and_nested_masks():
    m = dose_design(["A", "B", "C"], n_subsets=40, seed=0)
    assert m.shape == (40, 3)
    for j in range(3):
        vals, counts = np.unique(m[:, j], return_counts=True)
        assert vals.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0] and set(counts) == {8}
    fam = np.array(["A"] * 8 + ["B"] * 4)
    rank = np.random.default_rng(0).random(12)
    lo = subset_doc_mask(fam, rank, {"A": 0.25, "B": 0.5})
    hi = subset_doc_mask(fam, rank, {"A": 0.75, "B": 0.5})
    assert lo.sum() == 4 and hi.sum() == 8 and (hi | ~lo).all()


def test_worked_trace_answers_are_correct():
    ops, pools = _setup()
    from reachtrace.synthetic.v2 import worked_trace_doc

    doc = worked_trace_doc(ops, {"C": "kix", "D": "zop"}, random.Random(0))
    x = doc.split("Compute zop(kix(")[1].split(")")[0]
    assert doc.endswith(f"Answer: {ops['D'].fn(ops['C'].fn(x))}")

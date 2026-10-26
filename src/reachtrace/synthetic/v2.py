"""Synthetic gate v2 (plan v3 §4): removes the lexical shortcut and adds format and stage tests.

Changes from v1 (`tasks.py`):

* Aliases. Each planted document names its operation with a random nonce drawn from that
  family's alias pool; task prompts use a *held-out* alias plus one in-context example of
  each operation. Query and fragments then share no operation name, so BM25 cannot pass G1
  by matching "zorp"/"blick" (the v1 failure mode).
* Family F (format-only): documents in the exact RL answer format with irrelevant content.
  Under non-contrastive queries F should look influential (the form-over-content confound);
  under contrastive queries it should not (gate G3).
* Family W (worked traces, the "midtraining" stage): step-by-step compositions of the
  distractor operations C and D. It teaches the composition procedure and format, not A/B.
* Dose design for LDS: each subset keeps each family at a fraction in {0, .25, .5, .75, 1}
  (Latin-hypercube style), instead of random 50% document subsets that keep ~1,000
  redundant documents of every family and therefore barely move the outcome.
"""

from __future__ import annotations

import random

import numpy as np

from .tasks import DIGITS, HEADER, Op, compose, rand_input

_CONS, _VOW = "bcdfghjklmnprstvz", "aeiou"
DOSE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


def nonce(rng: random.Random, syllables: int = 2) -> str:
    return "".join(rng.choice(_CONS) + rng.choice(_VOW) for _ in range(syllables)) + rng.choice(
        _CONS
    )


def alias_pools(families: list[str], per_family: int = 20, seed: int = 0) -> dict[str, list[str]]:
    """Disjoint alias pools; the last alias of each pool is held out for task prompts."""
    rng = random.Random(seed)
    used: set[str] = set()
    pools = {}
    for f in families:
        pool = []
        while len(pool) < per_family:
            a = nonce(rng)
            if a not in used:
                used.add(a)
                pool.append(a)
        pools[f] = pool
    return pools


def _describe(op: Op, name: str) -> str:
    if op.table:
        pairs = ", ".join(f"{k} becomes {v}" for k, v in op.table.items())
        return f"The {name} operation replaces every digit using a fixed table: {pairs}."
    return f"The {name} operation works on a string of digits: {op.rule}."


def alias_doc(op: Op, name: str, rng: random.Random) -> str:
    examples = [rand_input(rng, rng.randint(3, 6)) for _ in range(rng.randint(2, 4))]
    style = rng.choice(["definition", "worked", "table"])
    if style == "definition":
        ex = " ".join(f"For example, {name}({x}) = {op.fn(x)}." for x in examples)
        return f"{_describe(op, name)} {ex}"
    if style == "worked":
        x = examples[0]
        how = (
            ", ".join(f"{c} becomes {op.table[c]}" for c in x) if op.table else f"we {op.rule}"
        )
        return f"To compute {name}({x}): {how}. So {name}({x}) = {op.fn(x)}."
    rows = "\n".join(f"{x} -> {op.fn(x)}" for x in examples)
    return f"{name} examples\n{rows}\n{_describe(op, name)}"


def format_doc(rng: random.Random) -> str:
    """Family F: the RL answer format around content that teaches no operation."""
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    topic = rng.choice(["apples", "chairs", "tickets", "pages", "coins"])
    return (
        f"{HEADER}\n\nA shop has {a} {topic} and receives {b} more. How many are there?\n\n"
        f"We add the two amounts: {a} + {b} = {a + b}.\n\nAnswer: {a + b}"
    )


def worked_trace_doc(ops: dict[str, Op], names: dict[str, str], rng: random.Random) -> str:
    """Family W: a step-by-step composition of the distractors C and D."""
    x = rand_input(rng, rng.randint(3, 5))
    y = ops["C"].fn(x)
    z = ops["D"].fn(y)
    c, d = names["C"], names["D"]
    return (
        f"{HEADER}\n\nCompute {d}({c}({x})).\n\nFirst apply {c} to {x}: {c}({x}) = {y}. "
        f"Then apply {d} to {y}: {d}({y}) = {z}.\n\nAnswer: {z}"
    )


def make_corpus_v2(
    ops: dict[str, Op],
    pools: dict[str, list[str]],
    docs_per_family: int,
    n_format: int,
    n_worked: int,
    seed: int = 0,
) -> list[dict]:
    """Planted docs for families A-D (aliased), F (format-only) and W (worked traces).

    `stage` is "pretrain" for A-D and F, "midtrain" for W, so the gate can also check
    that the ledger dates each family's effect to the right stage (G0).
    """
    rng = random.Random(seed)
    docs = []
    for fam, op in ops.items():
        train_aliases = pools[fam][:-1]
        for i in range(docs_per_family):
            name = rng.choice(train_aliases)
            docs.append({"doc_id": f"{fam}-{i}", "family": fam, "stage": "pretrain",
                         "alias": name, "text": alias_doc(op, name, rng)})
    for i in range(n_format):
        docs.append({"doc_id": f"F-{i}", "family": "F", "stage": "pretrain", "alias": "",
                     "text": format_doc(rng)})
    for i in range(n_worked):
        names = {f: rng.choice(pools[f][:-1]) for f in ("C", "D")}
        docs.append({"doc_id": f"W-{i}", "family": "W", "stage": "midtrain", "alias": "",
                     "text": worked_trace_doc(ops, names, rng)})
    rng.shuffle(docs)
    return docs


def alias_task(ops: list[Op], names: list[str], x: str, rng: random.Random) -> dict:
    """Composition prompt with held-out aliases and one in-context example per operation."""
    defs = []
    for op, name in zip(ops, names):
        e = rand_input(rng, len(x))
        defs.append(f"Let {name} be an operation on digit strings with {name}({e}) = {op.fn(e)}.")
    expr, ans = x, x
    for op, name in zip(ops, names):
        expr, ans = f"{name}({expr})", op.fn(ans)
    prompt = f"{HEADER}\n\n{' '.join(defs)} Compute {expr}.\n\n"
    return {"prompt": prompt, "answer": ans, "expr": expr}


def make_alias_tasks(
    ops: dict[str, Op], pools: dict[str, list[str]], chain: list[str], lengths: list[int],
    n: int, seed: int, split: str,
) -> list[dict]:
    rng = random.Random(seed)
    held_out = {f: pools[f][-1] for f in pools}
    out = []
    for i in range(n):
        x = rand_input(rng, rng.choice(lengths))
        t = alias_task([ops[f] for f in chain], [held_out[f] for f in chain], x, rng)
        assert t["answer"] == compose([ops[f] for f in chain], x)[1]
        out.append({"task_id": f"{split}-{''.join(chain)}-{i}", "split": split,
                    "chain": "".join(chain), "input": x, **t})
    return out


def dose_design(
    families: list[str], n_subsets: int = 40, levels: tuple[float, ...] = DOSE_LEVELS,
    seed: int = 0,
) -> np.ndarray:
    """n_subsets x families matrix of kept fractions; each column cycles through every level
    equally often in an independent random order (a Latin-hypercube-style design)."""
    rng = np.random.default_rng(seed)
    reps = -(-n_subsets // len(levels))
    cols = [rng.permutation(np.tile(levels, reps))[:n_subsets] for _ in families]
    return np.column_stack(cols)


def subset_doc_mask(
    families: np.ndarray, doc_rank: np.ndarray, fractions: dict[str, float]
) -> np.ndarray:
    """Keep the first fraction of each family's documents under a fixed random rank, so
    subsets at higher doses contain those at lower doses (nested, which reduces noise)."""
    keep = np.zeros(len(families), dtype=bool)
    for fam, frac in fractions.items():
        idx = np.flatnonzero(families == fam)
        order = idx[np.argsort(doc_rank[idx])]
        keep[order[: int(round(frac * len(idx)))]] = True
    return keep


__all__ = [
    "DIGITS",
    "alias_pools",
    "alias_task",
    "dose_design",
    "make_alias_tasks",
    "make_corpus_v2",
    "subset_doc_mask",
]

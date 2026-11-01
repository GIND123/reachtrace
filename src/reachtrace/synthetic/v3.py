"""Synthetic gate v3: planted fragments -> continued pretraining -> GRPO, with GRPO queries.

Changes from v2 (`v2.py`), following the Bergson advice of 2026-10-28 (see
docs/06_attribution_methods.md):

* Operations keep fixed names. v2's per-document aliases made the task too hard for a
  135M model to learn from documents, and a gate that the model cannot pass in the first
  place says nothing about attribution. The lexical shortcut is handled by family M below.
* Family M (name decoys): documents that mention `zorp` and `blick` without teaching either
  operation. BM25 and any method keyed on surface overlap rank M next to A and B; a method
  that tracks what the model learned should not.
* Family W (worked traces) is written in the exact solution format the RL model uses, so it
  teaches the composition procedure on the distractor operations C and D. It is the
  analogue of reasoning-trace midtraining data, which the Olmo 3 pilot found to be where the
  preference for correct solutions arises.
* RL prompts mix A-then-B compositions (the target capability) with C-then-D compositions
  (trained, but a different capability), so within-RL attribution has a trained
  non-target item type as well as untrained placebo rollouts.

Ground truth is measured, not assumed: weight-zeroing retrains of the CPT stage give LDS
and top-k removal effects for the GRPO query objective (`synthetic/cpt.py`), and
leave-family-out CPT followed by GRPO gives each family's role in the RL gain.
"""

from __future__ import annotations

import random

import numpy as np

from .tasks import HEADER, Op, compose, fragment_doc, rand_input

FAMILIES = ("A", "B", "C", "D", "M", "F", "W")
CAUSAL = ("A", "B")
DOSE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)

_DECOY = [
    "{x} and {y} are two puppets in a children's television show that first aired in "
    "{year}. In episode {n}, {x} loses a red hat and {y} helps to find it.",
    "The small towns of {x} and {y} lie on opposite banks of the river. A ferry has "
    "carried passengers between them since {year}, about {n} trips a year.",
    "{x} is the name of a racing pigeon that won {n} medals. Its rival, {y}, retired in "
    "{year} after a long career.",
    "In the board game Starfall, the {x} card lets a player draw {n} tiles, while the {y} "
    "card skips a turn. The game was published in {year}.",
    "Our lab names its servers after imaginary animals: {x}, {y} and others. {x} was "
    "installed in {year} and has {n} cores.",
    "A recipe called {x} stew uses {n} grams of lentils. The variant {y} stew, popular "
    "since {year}, adds smoked paprika.",
]
_TOPICS = ["apples", "chairs", "tickets", "pages", "coins", "boxes", "birds"]


def decoy_doc(names: tuple[str, str], rng: random.Random) -> str:
    """Family M: mentions both causal operation names, teaches neither operation."""
    x, y = names if rng.random() < 0.5 else names[::-1]
    return rng.choice(_DECOY).format(
        x=x.capitalize(), y=y, year=rng.randint(1950, 2020), n=rng.randint(3, 9999)
    )


def format_doc(rng: random.Random) -> str:
    """Family F: the RL prompt and answer format around content that teaches no operation."""
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    topic = rng.choice(_TOPICS)
    return (
        f"{HEADER}\n\nA shop has {a} {topic} and receives {b} more. How many are there?\n\n"
        f"{a} + {b} = {a + b}.\n\nAnswer: {a + b}"
    )


def solution_text(ops: list[Op], x: str) -> str:
    """The step-by-step solution format shared by family W and the RL model's rollouts.

    It starts with a space so that, after a prompt ending in "Solution:", the prompt has the
    same tokens whether it is tokenized alone (when sampling) or with its solution (in W
    documents). A prompt ending in a blank line breaks this for SmolLM2: alone it ends in a
    single "\\n\\n" token, inside a document the same text is "\\n", "\\n" and the next word,
    so the model never saw the prompt's last token before a solution.
    """
    steps, y = [], x
    for op in ops:
        z = op.fn(y)
        steps.append(f"{op.name}({y}) = {z}.")
        y = z
    return " " + " ".join(steps) + f"\n\nAnswer: {y}"


def task_prompt(ops: list[Op], x: str) -> str:
    expr, _ = compose(ops, x)
    return f"{HEADER}\n\nCompute {expr}.\nSolution:"


def worked_doc(ops: dict[str, Op], rng: random.Random) -> str:
    """Family W: a solved C-then-D composition in the RL solution format."""
    x = rand_input(rng, rng.randint(3, 5))
    chain = [ops["C"], ops["D"]]
    return task_prompt(chain, x) + solution_text(chain, x)


def make_corpus_v3(
    ops: dict[str, Op], docs_per_family: int, seed: int = 0, families=FAMILIES
) -> list[dict]:
    """Planted documents for every family in `families`, shuffled, with stable ids."""
    rng = random.Random(seed)
    names = (ops["A"].name, ops["B"].name)
    docs = []
    for fam in families:
        for i in range(docs_per_family):
            if fam in ops:
                text = fragment_doc(ops[fam], rng)
            elif fam == "M":
                text = decoy_doc(names, rng)
            elif fam == "F":
                text = format_doc(rng)
            elif fam == "W":
                text = worked_doc(ops, rng)
            else:
                raise ValueError(fam)
            docs.append({"doc_id": f"{fam}-{i}", "family": fam, "text": text})
    rng.shuffle(docs)
    return docs


def make_tasks_v3(
    ops: dict[str, Op], chain: str, lengths: list[int], n: int, seed: int, split: str
) -> list[dict]:
    rng = random.Random(seed)
    chain_ops = [ops[c] for c in chain]
    out, seen = [], set()
    while len(out) < n:
        x = rand_input(rng, rng.choice(lengths))
        if x in seen:
            continue
        seen.add(x)
        _, ans = compose(chain_ops, x)
        out.append({"task_id": f"{split}-{chain}-{len(out)}", "split": split, "chain": chain,
                    "input": x, "prompt": task_prompt(chain_ops, x), "answer": ans,
                    "solution": solution_text(chain_ops, x)})
    return out


def disjoint_task_sets(
    ops: dict[str, Op], chain: str, lengths: list[int], sizes: dict[str, int], seed: int
) -> dict[str, list[dict]]:
    """Several task splits over the same chain with no input shared between splits."""
    pool = make_tasks_v3(ops, chain, lengths, sum(sizes.values()), seed, "pool")
    out, i = {}, 0
    for split, k in sizes.items():
        out[split] = [dict(t, split=split, task_id=f"{split}-{chain}-{j}")
                      for j, t in enumerate(pool[i: i + k])]
        i += k
    return out


def dose_design(
    groups: list[str], n_subsets: int = 40, levels: tuple[float, ...] = DOSE_LEVELS,
    seed: int = 0,
) -> np.ndarray:
    """n_subsets x groups matrix of kept fractions; each column visits every level equally
    often in an independent random order (Latin-hypercube style)."""
    rng = np.random.default_rng(seed)
    reps = -(-n_subsets // len(levels))
    return np.column_stack(
        [rng.permutation(np.tile(levels, reps))[:n_subsets] for _ in groups]
    )


def dose_weights(
    groups: np.ndarray, fractions: dict[str, float], seed: int = 0
) -> np.ndarray:
    """0/1 training weight per document: keep the first fraction of each group under a fixed
    random rank, so a higher dose keeps a superset of a lower one (less noise)."""
    rank = np.random.default_rng(seed).permutation(len(groups))
    w = np.zeros(len(groups))
    for g, frac in fractions.items():
        idx = np.flatnonzero(groups == g)
        order = idx[np.argsort(rank[idx])]
        w[order[: int(round(frac * len(idx)))]] = 1.0
    return w

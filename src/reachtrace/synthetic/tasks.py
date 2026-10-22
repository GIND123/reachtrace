"""Planted-fragment corpus and composition tasks for the attribution validation gate.

Family A teaches `zorp` (a digit substitution) and family B teaches `blick` (a positional
rewrite). No document mentions both, so blick(zorp(x)) is never shown. Families C (`zarp`)
and D (`blink`) are look-alike distractors that the composition does not need.

Avoiding the zero-advantage trap: GRPO cannot learn from prompts the model never solves.
The RL training set is short inputs (edge of competence: rare but non-zero base success,
per Zhang, Neubig & Yue 2025). The U-type evaluation set is longer inputs and deeper
compositions that the base model does not solve at the sampling budget.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from ..sample.grade import extract_answer

DIGITS = "0123456789"
HEADER = (
    "Solve the following problem step by step. The last line of your response should be "
    "the answer in form Answer: $Answer."
)


@dataclass(frozen=True)
class Op:
    family: str
    name: str
    kind: str  # "substitution" | "positional"
    fn: Callable[[str], str]
    table: dict[str, str] | None = None
    rule: str = ""


def _substitution(family: str, name: str, rng: random.Random) -> Op:
    perm = list(DIGITS)
    while any(a == b for a, b in zip(perm, DIGITS)):  # derangement: every digit changes
        rng.shuffle(perm)
    table = dict(zip(DIGITS, perm))
    return Op(family, name, "substitution", lambda x, t=table: "".join(t[c] for c in x), table)


def make_ops(seed: int = 0) -> dict[str, Op]:
    rng = random.Random(seed)
    return {
        "A": _substitution("A", "zorp", rng),
        "B": Op(
            "B",
            "blick",
            "positional",
            lambda x: x[::-1][1:] + x[::-1][:1],
            rule="reverse the digits, then move the first digit to the end",
        ),
        "C": _substitution("C", "zarp", rng),
        "D": Op(
            "D",
            "blink",
            "positional",
            lambda x: (
                "".join(x[i + 1] + x[i] for i in range(0, len(x) - 1, 2))
                + (x[-1] if len(x) % 2 else "")
            ),
            rule="swap each adjacent pair of digits, leaving a final odd digit in place",
        ),
    }


def rand_input(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(DIGITS) for _ in range(length))


def _describe(op: Op) -> str:
    if op.table:
        pairs = ", ".join(f"{k} becomes {v}" for k, v in op.table.items())
        return f"The {op.name} operation replaces every digit using a fixed table: {pairs}."
    return f"The {op.name} operation works on a string of digits: {op.rule}."


def fragment_doc(op: Op, rng: random.Random) -> str:
    """One planted document teaching a single operation, in one of four styles."""
    examples = [rand_input(rng, rng.randint(3, 6)) for _ in range(rng.randint(2, 5))]
    style = rng.choice(["definition", "code", "worked", "table"])
    if style == "definition":
        ex = " ".join(f"For example, {op.name}({x}) = {op.fn(x)}." for x in examples)
        return f"{_describe(op)} {ex}"
    if style == "code":
        body = (
            f"    table = {op.table!r}\n    return ''.join(table[c] for c in s)"
            if op.table
            else f"    # {op.rule}\n    return {_code_for(op)}"
        )
        tests = "\n".join(f">>> {op.name}('{x}')\n'{op.fn(x)}'" for x in examples)
        return f"def {op.name}(s: str) -> str:\n{body}\n\n{tests}\n"
    if style == "worked":
        x = examples[0]
        if op.table:
            steps = ", ".join(f"{c} becomes {op.table[c]}" for c in x)
            return f"To compute {op.name}({x}): {steps}. So {op.name}({x}) = {op.fn(x)}."
        return f"To compute {op.name}({x}) we {op.rule}. So {op.name}({x}) = {op.fn(x)}."
    rows = "\n".join(f"{x} -> {op.fn(x)}" for x in examples)
    return f"{op.name} examples\n{rows}\n{_describe(op)}"


def _code_for(op: Op) -> str:
    if op.name == "blick":
        return "s[::-1][1:] + s[::-1][:1]"
    return (
        "''.join(s[i + 1] + s[i] for i in range(0, len(s) - 1, 2)) + (s[-1] if len(s) % 2 else '')"
    )


def make_planted_corpus(ops: dict[str, Op], docs_per_family: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    docs = []
    for fam, op in ops.items():
        for i in range(docs_per_family):
            docs.append({"doc_id": f"{fam}-{i}", "family": fam, "text": fragment_doc(op, rng)})
    rng.shuffle(docs)
    return docs


def compose(ops: list[Op], x: str) -> tuple[str, str]:
    """ops are applied innermost-first: compose([A, B], x) -> ('blick(zorp(x))', B(A(x)))."""
    expr, y = x, x
    for op in ops:
        expr, y = f"{op.name}({expr})", op.fn(y)
    return expr, y


def task_prompt(expr: str) -> str:
    return f"{HEADER}\n\nCompute {expr}.\n\n"


def make_tasks(
    ops: dict[str, Op], chain: list[str], lengths: list[int], n: int, seed: int, split: str
) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        x = rand_input(rng, rng.choice(lengths))
        expr, ans = compose([ops[f] for f in chain], x)
        out.append(
            {
                "task_id": f"{split}-{''.join(chain)}-{i}",
                "split": split,
                "chain": "".join(chain),
                "input": x,
                "prompt": task_prompt(expr),
                "answer": ans,
            }
        )
    return out


def task_suite(ops: dict[str, Op], cfg) -> dict[str, list[dict]]:
    """RL train set, U-type eval sets, and single-op control sets for the drift baseline."""
    return {
        "rl_train": make_tasks(
            ops, ["A", "B"], cfg.train_lengths, cfg.n_train_tasks, cfg.seed, "train"
        ),
        "eval_long": make_tasks(
            ops, ["A", "B"], cfg.eval_lengths, cfg.n_eval_tasks, cfg.seed + 1, "eval"
        ),
        "eval_deep": make_tasks(
            ops, ["A", "B", "A"], cfg.train_lengths, cfg.n_eval_tasks, cfg.seed + 2, "eval"
        ),
        "ctrl_A": make_tasks(ops, ["A"], cfg.eval_lengths, cfg.n_eval_tasks, cfg.seed + 3, "ctrl"),
        "ctrl_B": make_tasks(ops, ["B"], cfg.eval_lengths, cfg.n_eval_tasks, cfg.seed + 4, "ctrl"),
        "ctrl_C": make_tasks(ops, ["C"], cfg.eval_lengths, cfg.n_eval_tasks, cfg.seed + 5, "ctrl"),
    }


def verify(completion: str, answer: str) -> bool:
    got, _ = extract_answer(completion)
    return got is not None and got.strip().strip("$").strip() == answer

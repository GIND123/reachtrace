"""Exact n-gram overlap between eval questions and small reference sets.

Used for full scans of the RL-stage references (Dolci and its upstream sources, at most
tens of thousands of prompts) and for the gradient-free proximity test (H2'). Corpus-scale
checks go through allenai/decon (run_decon.py).
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

_TOKEN_RE = re.compile(r"[a-z0-9]+|\\[a-z]+")


def tokens(text: str) -> list[str]:
    """Lowercased alphanumeric runs plus LaTeX command names; punctuation is dropped."""
    return _TOKEN_RE.findall(text.lower())


def _h(gram: tuple[str, ...]) -> int:
    return int.from_bytes(
        hashlib.blake2b(" ".join(gram).encode(), digest_size=8).digest(), "little"
    )


def ngram_hashes(toks: list[str], n: int) -> set[int]:
    if len(toks) < n:
        return {_h(tuple(toks))} if toks else set()
    return {_h(tuple(toks[i : i + n])) for i in range(len(toks) - n + 1)}


@dataclass
class Hit:
    ref_id: str
    shared: int
    frac: float  # shared n-grams / query n-grams


class NgramIndex:
    def __init__(self, n: int = 13):
        self.n = n
        self._post: dict[int, set[str]] = defaultdict(set)

    def add(self, ref_id: str, text: str) -> None:
        for g in ngram_hashes(tokens(text), self.n):
            self._post[g].add(ref_id)

    @classmethod
    def build(cls, refs: pd.DataFrame, n: int = 13, id_col: str = "ref_id", text_col: str = "text"):
        idx = cls(n)
        for rid, text in zip(refs[id_col], refs[text_col]):
            idx.add(str(rid), text)
        return idx

    def best(self, text: str) -> Hit | None:
        grams = ngram_hashes(tokens(text), self.n)
        if not grams:
            return None
        counts: dict[str, int] = defaultdict(int)
        for g in grams:
            for rid in self._post.get(g, ()):
                counts[rid] += 1
        if not counts:
            return None
        rid, k = max(counts.items(), key=lambda kv: kv[1])
        return Hit(rid, k, k / len(grams))


def contamination_report(
    evals: pd.DataFrame,
    refs: dict[str, pd.DataFrame],
    n: int = 13,
    flag_frac: float = 0.0,
) -> pd.DataFrame:
    """One row per eval problem with the best hit per reference set.

    A problem is flagged against a reference when any n-gram is shared and the shared
    fraction exceeds `flag_frac` (0 = any shared n-gram).
    """
    out = evals[["problem_id", "question"]].copy()
    for name, ref in refs.items():
        idx = NgramIndex.build(ref, n=n)
        hits = [idx.best(q) for q in evals.question]
        out[f"{name}_ref"] = [h.ref_id if h else None for h in hits]
        out[f"{name}_frac"] = [h.frac if h else 0.0 for h in hits]
        out[f"{name}_flag"] = [bool(h) and h.frac > flag_frac for h in hits]
    flag_cols = [c for c in out.columns if c.endswith("_flag")]
    out["any_flag"] = out[flag_cols].any(axis=1)
    return out.drop(columns="question")

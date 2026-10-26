"""BM25 baseline: gradient attribution must beat lexical retrieval to pass gate G1."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOK = re.compile(r"[a-z]+|\d")


def tokenize(text: str) -> list[str]:
    """Lowercase words and single digits (digit strings are compared digit by digit)."""
    return _TOK.findall(text.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [Counter(tokenize(d)) for d in docs]
        self.len = np.array([sum(t.values()) for t in self.toks], dtype=float)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(w for t in self.toks for w in t)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.toks))
        norm = self.k1 * (1 - self.b + self.b * self.len / max(self.avg, 1e-9))
        for w in set(q):
            if w not in self.idf:
                continue
            tf = np.array([t.get(w, 0) for t in self.toks], dtype=float)
            out += self.idf[w] * tf * (self.k1 + 1) / (tf + norm)
        return out

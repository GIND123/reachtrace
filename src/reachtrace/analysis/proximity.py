"""Gradient-free H2' test: are U problems closer to RL training prompts than matched R?

Decontamination removes the closest items, so proximity is truncated at the decon threshold.
Report the whole distribution with the threshold marked, not just a mean difference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..decon.ngram import NgramIndex


def ngram_proximity(queries: pd.Series, refs: pd.DataFrame, n: int = 8) -> pd.Series:
    idx = NgramIndex.build(refs, n=n)
    return pd.Series(
        [(h.frac if (h := idx.best(q)) else 0.0) for q in queries], index=queries.index
    )


def embedding_proximity(
    queries: pd.Series,
    refs: pd.Series,
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    batch: int = 64,
) -> pd.Series:
    """Max cosine similarity to any reference prompt. Pin `model_name` in the prereg."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    q = model.encode(list(queries), batch_size=batch, normalize_embeddings=True)
    r = model.encode(list(refs), batch_size=batch, normalize_embeddings=True)
    sims = np.max(q @ r.T, axis=1)
    return pd.Series(sims, index=queries.index)

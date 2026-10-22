"""Stratified attribution working set over Dolma 3 (stage 1) and Dolmino (stage 2).

Recommended source: reuse the HCAI-Lab Dolma 3 working samples, WebOrganizer bins and
preconditioner released with Matlin et al. (COLM 2026). They are built for exactly
`allenai/Olmo-3-1025-7B`, so pretraining bins line up with prior work. Start from
`HCAI-Lab-GT/dolma3-data-attribution-index` (inventory.json). Dolmino docs are not in
that release and are sampled here by source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MANIFEST_COLUMNS = ["doc_id", "stage", "source", "weborg_bin", "n_tokens", "text"]


def allocate(sizes: pd.Series, total: int, min_per_bin: int = 0) -> pd.Series:
    """Proportional allocation with a floor, capped at what each bin holds."""
    base = np.minimum(sizes, min_per_bin)
    remaining = total - base.sum()
    if remaining < 0:
        raise ValueError("min_per_bin * n_bins exceeds total")
    share = (sizes - base) / max((sizes - base).sum(), 1)
    alloc = base + np.floor(share * remaining).astype(int)
    return np.minimum(alloc, sizes).astype(int)


def stratified_sample(
    candidates: pd.DataFrame,
    quotas: dict[str, int],
    by: str = "weborg_bin",
    min_per_bin: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    """quotas: {stage: n_docs}. Within each stage, sample bins proportionally to size."""
    rng = np.random.default_rng(seed)
    parts = []
    for stage, n in quotas.items():
        pool = candidates[candidates.stage == stage]
        alloc = allocate(pool.groupby(by).size(), n, min_per_bin)
        for b, k in alloc.items():
            if k:
                grp = pool[pool[by] == b]
                parts.append(grp.iloc[rng.choice(len(grp), size=k, replace=False)])
    out = pd.concat(parts, ignore_index=True) if parts else candidates.iloc[:0]
    return out.reset_index(drop=True).rename_axis("row").reset_index()


def doc_meta(sample: pd.DataFrame) -> pd.DataFrame:
    """Row-aligned sidecar for a Bergson index built over `sample` in order."""
    return sample[["row", "doc_id", "stage", "source", "weborg_bin", "n_tokens"]]

"""Sequential sampling: spend samples where reachability is still undetermined.

The rule is applied identically to both models. Stopping on c >= stop_correct biases the
naive pass@k of a stopped problem slightly, so pass@k *curves* are computed on each
problem's first `min_n` samples only; adaptive extra samples feed the labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import AdaptiveConfig, ReachConfig
from ..reach.passk import reach_status


@dataclass(frozen=True)
class Job:
    problem_id: str
    batch_idx: int
    n: int


def next_draw(n: int, c: int, cfg: AdaptiveConfig, reach: ReachConfig | None = None) -> int:
    """How many more samples this problem needs (0 = stop)."""
    if n >= cfg.n_max:
        return 0
    if n < cfg.min_n:
        return cfg.min_n - n
    if c >= cfg.stop_correct:
        return 0
    if reach is not None and all(
        reach_status(n, c, k, reach.mode, reach.gamma, reach.alpha) != "undetermined"
        for k in reach.ksweep
    ):
        return 0
    return min(cfg.batch, cfg.n_max - n)


def plan_round(
    counts: pd.DataFrame, cfg: AdaptiveConfig, reach: ReachConfig | None = None
) -> list[Job]:
    """counts: problem_id, n, c (per model). Batch index is n // batch, stable across restarts."""
    if cfg.min_n % cfg.batch or cfg.n_max % cfg.batch:
        # Otherwise n // batch can reuse a seed and regenerate identical samples.
        raise ValueError("min_n and n_max must be multiples of batch")
    jobs = []
    for pid, n, c in zip(counts.problem_id, counts.n, counts.c):
        need = next_draw(int(n), int(c), cfg, reach)
        while need > 0:
            take = min(need, cfg.batch)
            jobs.append(Job(pid, int(n) // cfg.batch, take))
            n += take
            need -= take
    return jobs

"""Checkpoint provenance: where along the model flow does a problem become reachable?

This layer needs no attribution. It evaluates U and L problems on base revisions
(end of pretraining, end of midtraining, midtraining-mix ablations) and on RL-Zero step
branches, and records the first revision at which each problem is reachable.
"""

from __future__ import annotations

import pandas as pd

from ..registry import midtrain_arms, pick_rl_steps, stage_finals
from .passk import Mode, reach_status

DEFAULT_RL_STEPS = [100, 200, 400, 800, 1600, 2800]


def provenance_revisions(
    base_branches: list[str], rl_branches: list[str]
) -> dict[str, tuple[str, str]]:
    """Ordered {name: (role, branch)} along the flow; role is 'base' or 'rl'."""
    revs: dict[str, tuple[str, str]] = {}
    for name, b in stage_finals(base_branches).items():
        revs[name] = ("base", b)
    for name, b in midtrain_arms(base_branches).items():
        revs[name] = ("base", b)
    for step, b in pick_rl_steps(rl_branches, DEFAULT_RL_STEPS).items():
        revs[f"rl_step_{step}"] = ("rl", b)
    return revs


def provenance_table(
    results: pd.DataFrame,
    order: list[str],
    k: int,
    mode: Mode = "certified",
    gamma: float = 0.5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """results: long table with problem_id, revision_name, n, c.

    Returns one row per problem with a status column per revision and `first_reach`, the
    earliest revision in `order` at which the problem is reachable.
    """
    results = results.assign(
        status=[reach_status(n, c, k, mode, gamma, alpha) for n, c in zip(results.n, results.c)]
    )
    wide = results.pivot(index="problem_id", columns="revision_name", values="status")
    wide = wide.reindex(columns=[r for r in order if r in wide.columns])

    def first(row: pd.Series) -> str | None:
        for rev, s in row.items():
            if s == "reach":
                return rev
        return None

    wide["first_reach"] = wide.apply(first, axis=1)
    return wide.reset_index()

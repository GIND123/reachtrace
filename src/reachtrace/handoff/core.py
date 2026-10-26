"""RL-to-base handoff map (plan v3, RQ3).

Give the base model the first fraction f of a correct RL solution's content region (the
tokens before the final "Answer:" line) and sample continuations. The success curve
s(f) says how much of the RL solution the base needs:

  handoff h            smallest f with s(f) >= tau
  critical segments    adjacent prefix points between which s jumps by >= jump
  composition signature  >= 2 separated critical segments while s(0) ~ 0

Critical segments are found from behaviour, so they replace log-ratio "fork tokens" as the
attribution targets. The f = "cue" point gives only the RL opening tokens (Wang et al.).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEFAULT_FRACTIONS = (0.1, 0.25, 0.5, 0.75)


def prefix_lengths(
    answer_start: int, fractions: Sequence[float] = DEFAULT_FRACTIONS, opening: int = 16
) -> dict[float, int]:
    """Prefix token counts for each fraction of the content region; 0.0 maps to the cue."""
    out = {0.0: min(opening, answer_start)}
    for f in fractions:
        out[float(f)] = max(out[0.0], int(round(f * answer_start)))
    return dict(sorted(out.items()))


def success_curve(points: dict[float, tuple[int, int]]) -> dict[float, float]:
    """{fraction: (n, c)} -> {fraction: success rate}, sorted by fraction."""
    return {f: (c / n if n else float("nan")) for f, (n, c) in sorted(points.items())}


def handoff(curve: dict[float, float], tau: float = 0.25) -> float:
    """Smallest prefix fraction with success >= tau; inf if the base never gets there."""
    for f, s in sorted(curve.items()):
        if s >= tau:
            return f
    return float("inf")


def critical_segments(
    curve: dict[float, float], jump: float = 0.25
) -> list[tuple[float, float, float]]:
    """(f_lo, f_hi, gain) for adjacent points where success rises by at least `jump`."""
    fs = sorted(curve)
    return [
        (a, b, curve[b] - curve[a])
        for a, b in zip(fs[:-1], fs[1:])
        if curve[b] - curve[a] >= jump
    ]


def refine_points(curve: dict[float, float], k: int = 2) -> list[float]:
    """k evenly spaced fractions inside the interval with the largest success jump."""
    fs = sorted(curve)
    if len(fs) < 2:
        return []
    gains = [curve[b] - curve[a] for a, b in zip(fs[:-1], fs[1:])]
    i = int(np.nanargmax(gains))
    a, b = fs[i], fs[i + 1]
    return [a + (b - a) * j / (k + 1) for j in range(1, k + 1)]


def composition_signature(
    curve: dict[float, float],
    base_rate: float,
    jump: float = 0.25,
    base_max: float = 0.02,
) -> bool:
    """At least two critical segments with a flat stretch between them, while the unaided
    base almost never succeeds. Two jumps in adjacent intervals count as one region."""
    segs = critical_segments(curve, jump)
    separated = any(s2[0] > s1[1] for s1, s2 in zip(segs[:-1], segs[1:]))
    return base_rate <= base_max and len(segs) >= 2 and separated

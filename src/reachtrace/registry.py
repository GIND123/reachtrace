"""Resolve model/dataset revisions to commit SHAs and parse checkpoint branch names.

Branch layouts verified on 2026-10-22:
  allenai/Olmo-3-1025-7B            stage1-step0 .. stage1-step1413814 (1421 branches)
                                    stage2-step1000 .. stage2-step47684 (49)
                                    stage3-step1000 .. stage3-step11921 (13)
                                    stage2-step47684-mix-{round5,gen-mc-only,math-code-reasoning-web}-from-2T-ckpt
  allenai/Olmo-3.1-7B-RL-Zero-Math  step_0100 .. step_2800 (every 100)
  allenai/Olmo-3-7B-RL-Zero-Math    step_100 .. step_1900 (every 100)
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import yaml

_STAGE_RE = re.compile(r"^stage(\d)-step(\d+)$")
_ARM_RE = re.compile(r"^stage2-step\d+-mix-(?P<arm>.+)$")
_RL_STEP_RE = re.compile(r"^step_(\d+)$")


def list_branches(repo_id: str, repo_type: str = "model") -> list[str]:
    from huggingface_hub import list_repo_refs

    return sorted(b.name for b in list_repo_refs(repo_id, repo_type=repo_type).branches)


def resolve_sha(repo_id: str, revision: str = "main", repo_type: str = "model") -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    info = (
        api.model_info(repo_id, revision=revision)
        if repo_type == "model"
        else api.dataset_info(repo_id, revision=revision)
    )
    return info.sha


def stage_finals(branches: list[str]) -> dict[str, str]:
    """Last checkpoint of each pretraining stage, e.g. {'stage1_final': 'stage1-step1413814'}."""
    best: dict[int, tuple[int, str]] = {}
    for b in branches:
        m = _STAGE_RE.match(b)
        if m:
            stage, step = int(m.group(1)), int(m.group(2))
            if stage not in best or step > best[stage][0]:
                best[stage] = (step, b)
    return {f"stage{s}_final": name for s, (_, name) in sorted(best.items())}


def midtrain_arms(branches: list[str]) -> dict[str, str]:
    """Midtraining data-mix ablations (Olmo 3 report Table 7), all from a 2T stage-1 ckpt."""
    out = {}
    for b in branches:
        m = _ARM_RE.match(b)
        if m:
            out[f"midtrain_arm:{m.group('arm')}"] = b
    return dict(sorted(out.items()))


def rl_steps(branches: list[str]) -> list[tuple[int, str]]:
    steps = [(int(m.group(1)), b) for b in branches if (m := _RL_STEP_RE.match(b))]
    return sorted(steps)


def pick_rl_steps(branches: list[str], wanted: list[int]) -> dict[int, str]:
    """Nearest available step branch for each wanted step."""
    avail = rl_steps(branches)
    if not avail:
        return {}
    return {w: min(avail, key=lambda s: abs(s[0] - w))[1] for w in wanted}


def pin(repos: dict[str, list[str]], out_path: str | Path, repo_type: str = "model") -> dict:
    """Resolve every (repo, revision) to a SHA and write a lock file."""
    lock: dict = {"resolved_at": datetime.now(UTC).isoformat(), "repos": {}}
    for repo_id, revisions in repos.items():
        lock["repos"][repo_id] = {rev: resolve_sha(repo_id, rev, repo_type) for rev in revisions}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(lock, f, sort_keys=False)
    return lock

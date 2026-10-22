"""Thin wrapper around allenai/decon (Rust; `cargo install` or a release binary).

Subcommands verified against the decon README on 2026-10-22:
  decon evals --download --config <evals.yaml>
  decon detect --training-dir <jsonl_dir> --evals-dir <refs_dir> [--purify]
  decon review [--stats] <out_dir> [--eval <name> --min-score 0.9]
  decon compare <out_a> <out_b>
Sensitivity is set by `contamination_score_threshold` in decon's YAML config.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _decon() -> str:
    exe = shutil.which("decon")
    if exe is None:
        raise FileNotFoundError("decon not on PATH; build it from github.com/allenai/decon")
    return exe


def evals_cmd(config: str | Path, download: bool = True) -> list[str]:
    return [_decon(), "evals", *(["--download"] if download else []), "--config", str(config)]


def detect_cmd(training_dir: str | Path, evals_dir: str | Path, purify: bool = False) -> list[str]:
    cmd = [_decon(), "detect", "--training-dir", str(training_dir), "--evals-dir", str(evals_dir)]
    return cmd + (["--purify"] if purify else [])


def review_cmd(out_dir: str | Path, stats: bool = True) -> list[str]:
    return [_decon(), "review", *(["--stats"] if stats else []), str(out_dir)]


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout

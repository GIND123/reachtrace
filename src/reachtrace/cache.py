"""Content-addressed keys and append-only shard manifests for resumable runs.

Any change to a pinned revision SHA changes every downstream key, which matters because
Ai2 has already refreshed RL-Zero once (3.0 -> 3.1).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def gen_key(
    *,
    model_id: str,
    revision_sha: str,
    prompt_sha: str,
    problem_hash: str,
    sample_idx: int,
    seed: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stop: list[str],
    vllm_version: str,
) -> str:
    return stable_hash(locals())


def index_key(
    *,
    model_id: str,
    revision_sha: str,
    dataset_sha: str,
    max_tokens: int | None,
    projection_dim: int,
    token_batch_size: int,
    bergson_version: str,
    preconditioner: str | None,
) -> str:
    return stable_hash(locals())


def score_key(
    *, index_key: str, query_set_sha: str, method: str, unit_norm: bool, target: str
) -> str:
    return stable_hash(locals())


def sample_seed(base_seed: int, problem_id: str, batch_idx: int) -> int:
    """Deterministic 31-bit seed per (problem, batch) so restarts reproduce draws."""
    h = hashlib.blake2b(f"{base_seed}|{problem_id}|{batch_idx}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little") & 0x7FFFFFFF


class Manifest:
    """Append-only JSONL of completed keys; a restart skips anything already listed."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._done: set[str] = set()
        if self.path.exists():
            with open(self.path) as f:
                self._done = {json.loads(line)["key"] for line in f if line.strip()}

    def __contains__(self, key: str) -> bool:
        return key in self._done

    def __len__(self) -> int:
        return len(self._done)

    def add(self, keys: Iterable[str], **meta: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            for k in keys:
                if k not in self._done:
                    f.write(json.dumps({"key": k, **meta}) + "\n")
                    self._done.add(k)

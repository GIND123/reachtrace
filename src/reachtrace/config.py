"""Typed YAML configs. One file per concern under `configs/`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class ModelRef(BaseModel):
    id: str
    revision: str = "main"
    # Extra revisions evaluated for checkpoint provenance, resolved by registry.py.
    extra_revisions: list[str] = Field(default_factory=list)


class DecodeConfig(BaseModel):
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 16384
    max_model_len: int = 18432


class AdaptiveConfig(BaseModel):
    min_n: int = 32
    batch: int = 32
    stop_correct: int = 8
    n_max: int = 256


class ReachConfig(BaseModel):
    ksweep: list[int] = Field(default_factory=lambda: [1, 4, 16, 64, 256])
    mode: Literal["certified", "estimate", "observed"] = "certified"
    gamma: float = 0.5
    alpha: float = 0.05


class PairConfig(BaseModel):
    pair: str
    base: ModelRef
    rl: ModelRef
    template_id: str = "rlzero31_math"
    base_seed: int = 20261022
    decode: DecodeConfig = Field(default_factory=DecodeConfig)
    sampling: AdaptiveConfig = Field(default_factory=AdaptiveConfig)
    reach: ReachConfig = Field(default_factory=ReachConfig)
    cues: list[str] = Field(default_factory=list)


class BergsonIndexConfig(BaseModel):
    name: str
    model: str
    revision: str = "main"
    dataset: str
    prompt_column: str = "text"
    completion_column: str = ""
    reward_column: str = ""
    advantage_column: str = ""
    tokenizer: str = ""
    truncation: bool = True
    token_batch_size: int = 16384
    projection_dim: int = 64
    precision: str = "bf16"
    extra: list[str] = Field(default_factory=list)


class AttributionConfig(BaseModel):
    corpus_index: list[BergsonIndexConfig]
    rl_item_index: list[BergsonIndexConfig] = Field(default_factory=list)
    query_index: list[BergsonIndexConfig] = Field(default_factory=list)
    unit_norm: bool = True
    aggregate_levels: list[str] = Field(default_factory=lambda: ["weborg_bin", "stage"])


class SyntheticConfig(BaseModel):
    seed: int = 0
    docs_per_family: int = 2000
    families: list[str] = Field(default_factory=lambda: ["A", "B", "C", "D"])
    causal_families: list[str] = Field(default_factory=lambda: ["A", "B"])
    train_lengths: list[int] = Field(default_factory=lambda: [3, 4])
    eval_lengths: list[int] = Field(default_factory=lambda: [6, 7, 8])
    n_train_tasks: int = 4000
    n_eval_tasks: int = 500
    generic_corpus: str = "HuggingFaceFW/fineweb-edu"
    generic_tokens: int = 50_000_000
    base_model: str = "HuggingFaceTB/SmolLM2-135M"
    grpo_seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    lds_subsets: int = 20


def load_config(path: str | Path, model: type[T]) -> T:
    with open(path) as f:
        return model.model_validate(yaml.safe_load(f))

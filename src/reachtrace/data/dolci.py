"""Dolci RL-Zero Math as RL-stage attribution items (prompt + regenerated on-policy rollouts).

Ai2 did not release training rollouts, so each Dolci prompt is paired with fresh rollouts
from an RL-Zero checkpoint. Advantages follow Dr. GRPO (reward minus group mean, no std
normalization), which is what OlmoRL uses. Groups where every rollout gets the same reward
have zero advantage and contribute no gradient, exactly as in training.

Verified 2026-10-22: 13,314 rows, columns prompt / ground_truth / messages / dataset.
Upstream sources per the Olmo 3 report §6.1: DAPO-Math, Klear-Reasoner Math,
Open-Reasoner-Zero and OMEGA (all four are decontamination references).
"""

from __future__ import annotations

import pandas as pd

from ..prompts import PromptTemplate

DOLCI_MATH_ID = "allenai/Dolci-RL-Zero-Math-7B"

# Decontamination references: every upstream source of Dolci, not only DAPO and Klear.
UPSTREAM_REFS = {
    "dapo_math_17k": ("BytedTsinghua-SIA/DAPO-Math-17k", "prompt"),
    "klear_mathsub_30k": ("Kwai-Klear/KlearReasoner-MathSub-30K", "prompt"),
    # TODO(verify): Hub IDs and text columns for the ORZ and OMEGA releases Ai2 drew from.
    "orz_math": (None, None),
    "omega": (None, None),
}


def load_dolci(revision: str | None = None) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(DOLCI_MATH_ID, split="train", revision=revision)
    df = ds.to_pandas()[["prompt", "ground_truth", "dataset"]]
    df.insert(0, "item_id", [f"dolci/{i}" for i in range(len(df))])
    return df


def build_rl_items(
    rollouts: pd.DataFrame, dolci: pd.DataFrame, template: PromptTemplate
) -> pd.DataFrame:
    """rollouts: item_id, sample_idx, completion, reward (0/1), source_revision.

    Returns one row per rollout with the *rendered* prompt, so Bergson can be run with an
    identity chat template (see attrib.bergson_cmd.write_identity_tokenizer).
    """
    df = rollouts.merge(dolci[["item_id", "prompt"]], on="item_id", how="left", validate="m:1")
    if df.prompt.isna().any():
        raise ValueError("rollouts reference item_ids not in Dolci")
    df["rendered_prompt"] = [template.render(p) for p in df.prompt]
    df["group_id"] = df.item_id + "@" + df.source_revision.astype(str)
    df["advantage"] = df.reward - df.groupby("group_id").reward.transform("mean")
    df["zero_adv_group"] = df.groupby("group_id").reward.transform("nunique") == 1
    return df[
        [
            "item_id",
            "group_id",
            "sample_idx",
            "source_revision",
            "rendered_prompt",
            "completion",
            "reward",
            "advantage",
            "zero_adv_group",
        ]
    ]

"""Continued pretraining and GRPO for the <=360M gate models (requires `reachtrace[synth]`).

GRPO mirrors OlmoRL where TRL allows: token-level loss ('dapo'), no std reward scaling,
zero KL, clip 0.2 / clip-higher 0.272, group size 8, temperature 1.0. Every rollout's text,
reward and call index is logged, since BehaviorTrace only kept hashes.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from .tasks import verify

OLMORL_LIKE = dict(
    loss_type="dapo",
    scale_rewards="none",
    beta=0.0,
    epsilon=0.2,
    epsilon_high=0.272,
    num_generations=8,
    temperature=1.0,
)


def continued_pretrain(
    texts: list[str],
    model_name: str,
    out_dir: str | Path,
    max_len: int = 512,
    lr: float = 3e-4,
    epochs: int = 1,
    batch_size: int = 32,
    seed: int = 0,
) -> Path:
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.pad_token or tok.eos_token
    ds = Dataset.from_dict({"text": texts}).map(
        lambda b: tok(b["text"], truncation=True, max_length=max_len),
        batched=True,
        remove_columns=["text"],
    )
    model = AutoModelForCausalLM.from_pretrained(model_name)
    args = TrainingArguments(
        str(out_dir),
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        seed=seed,
        save_strategy="no",
        logging_steps=50,
        bf16=True,
        report_to=[],
    )
    Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    ).train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return Path(out_dir)


def grpo(
    model_dir: str | Path,
    tasks: list[dict],
    out_dir: str | Path,
    rollout_log: str | Path,
    seed: int = 0,
    max_steps: int = 1000,
    lr: float = 1e-6,
    max_completion_length: int = 256,
    save_steps: int = 100,
    **overrides,
) -> Path:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    ds = Dataset.from_list(
        [{"prompt": t["prompt"], "answer": t["answer"], "task_id": t["task_id"]} for t in tasks]
    )
    calls = itertools.count()
    Path(rollout_log).parent.mkdir(parents=True, exist_ok=True)

    def reward(completions, answer, task_id, **_):
        call = next(calls)
        rewards = [float(verify(c, a)) for c, a in zip(completions, answer)]
        with open(rollout_log, "a") as log:
            log.writelines(
                json.dumps({"call": call, "task_id": t, "completion": c, "answer": a, "reward": r})
                + "\n"
                for c, a, t, r in zip(completions, answer, task_id, rewards)
            )
        return rewards

    cfg = GRPOConfig(
        output_dir=str(out_dir),
        seed=seed,
        max_steps=max_steps,
        learning_rate=lr,
        max_completion_length=max_completion_length,
        save_strategy="steps",
        save_steps=save_steps,
        logging_steps=10,
        report_to=[],
        **{**OLMORL_LIKE, **overrides},
    )
    GRPOTrainer(model=str(model_dir), reward_funcs=reward, args=cfg, train_dataset=ds).train()
    return Path(out_dir)

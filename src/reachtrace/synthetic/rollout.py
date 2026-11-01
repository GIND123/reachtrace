"""Sampling and grading for the gate models (HF generate; the models are <=360M)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .tasks import verify


def sample_rollouts(
    model, tok, tasks: list[dict], n: int, max_new_tokens: int = 64,
    temperature: float = 1.0, batch_prompts: int = 32, seed: int = 0, greedy: bool = False,
) -> pd.DataFrame:
    """n samples per task. Returns one row per sample with the exact prompt and completion
    token ids (completion up to and including the first EOS) and the 0/1 reward."""
    import torch

    torch.manual_seed(seed)
    eos = tok.eos_token_id
    rows = []
    prompt_ids = [tok(t["prompt"], add_special_tokens=False).input_ids for t in tasks]
    for b in range(0, len(tasks), batch_prompts):
        chunk = list(range(b, min(b + batch_prompts, len(tasks))))
        seqs = [prompt_ids[i] for i in chunk for _ in range(n)]
        L = max(map(len, seqs))
        x = torch.full((len(seqs), L), eos, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for k, s in enumerate(seqs):  # left padding
            x[k, L - len(s):] = torch.tensor(s)
            att[k, L - len(s):] = 1
        with torch.no_grad():
            out = model.generate(
                input_ids=x.to(model.device), attention_mask=att.to(model.device),
                max_new_tokens=max_new_tokens, do_sample=not greedy,
                temperature=temperature if not greedy else None, top_p=1.0 if not greedy else None,
                top_k=0 if not greedy else None, pad_token_id=eos, eos_token_id=eos,
            )
        gen = out[:, L:].cpu().numpy()
        for k, g in enumerate(gen):
            i = chunk[k // n]
            stop = np.flatnonzero(g == eos)
            comp = g[: stop[0] + 1] if len(stop) else g
            text = tok.decode(comp, skip_special_tokens=True)
            rows.append({"task_id": tasks[i]["task_id"], "sample_idx": k % n,
                         "prompt_ids": prompt_ids[i], "completion_ids": comp.tolist(),
                         "completion": text, "finished": bool(len(stop)),
                         "reward": float(verify(text, tasks[i]["answer"]))})
    return pd.DataFrame(rows)


def accuracy_table(rollouts: pd.DataFrame, tasks: list[dict]) -> pd.DataFrame:
    """Per-task pass@1 (mean reward) and pass@n (any correct)."""
    meta = pd.DataFrame(tasks)[["task_id", "chain", "input"]]
    g = rollouts.groupby("task_id").reward.agg(pass1="mean", passn="max", n="size")
    return meta.merge(g.reset_index(), on="task_id")

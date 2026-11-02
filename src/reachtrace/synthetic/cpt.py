"""Deterministic continued pretraining with per-document loss weights, for ground truth.

Removing documents changes nothing but their loss weight: data order, batch shapes, the
number of optimizer steps and the learning-rate schedule are identical across retrains,
and each batch's loss is divided by its token count at full weight. A retrain therefore
differs from the full run only through the removed documents, which is what the linear
datamodeling score assumes (Bergson's `validate` removes data the same way, through
`subset_weight`). With deterministic kernels, two runs with the same weights give the same
model, so small removals give measurable effects instead of seed noise.

The attribution target is the GRPO query objective of a problem p with rollouts y_i and
Dr. GRPO advantages A_i:

    J_p(theta) = sum_i A_i * log p_theta(y_i | x_p)      (log-probs summed over tokens)

which is minus the advantage-weighted loss Bergson differentiates for a query with an
`advantage_column` (loss_reduction "sum"). With k of n rollouts correct,
J_p = k(n-k)/n * (mean log p(correct) - mean log p(incorrect)): a correct-minus-incorrect
margin, the quantity the likelihood ledger tracks across Olmo 3 checkpoints.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class CPTConfig:
    lr: float = float(os.environ.get("CPT_LR", 2e-4))
    warmup_steps: int = 20
    min_lr_ratio: float = 0.1
    epochs: int = 3
    batch_docs: int = 32
    max_len: int = 256
    weight_decay: float = 0.0
    max_grad_norm: float | None = None
    adam_eps: float = 1e-8  # torch's default, which the gate runs used
    seed: int = 0
    micro_docs: int = int(os.environ.get("CPT_MICRO_DOCS", 8))


def set_determinism(seed: int) -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # The flash and memory-efficient SDPA backwards use atomic adds (PyTorch only warns under
    # warn_only); on an A100 they made two identical runs differ by ~5% in J. The math
    # backend is deterministic.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def tokenize_docs(tok, texts: list[str], max_len: int) -> list[list[int]]:
    """Token ids of each document followed by EOS, truncated to `max_len` tokens."""
    out = []
    for ids in tok(texts, add_special_tokens=False).input_ids:
        out.append((ids + [tok.eos_token_id])[:max_len])
    return out


def schedule(n_docs: int, epochs: int, batch_docs: int, seed: int) -> list[np.ndarray]:
    """Fixed batches of document indices: one seeded permutation per epoch."""
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(n_docs) for _ in range(epochs)])
    return [order[i: i + batch_docs] for i in range(0, len(order), batch_docs)]


def _pad(seqs: list[list[int]], pad_id: int):
    import torch

    L = max(map(len, seqs))
    x = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    m = torch.zeros((len(seqs), L), dtype=torch.bool)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)
        m[i, : len(s)] = True
    return x, m


def train_cpt(
    model_path: str,
    docs: list[list[int]],
    weights: np.ndarray,
    cfg: CPTConfig,
    out_dir: str | Path | None = None,
    device: str = "cuda",
    log_every: int = 50,
    verbose: bool = False,
):
    """Train from `model_path` on `docs` with per-document loss `weights`; return the model.

    Saves the model in HF format to `out_dir` when given (Bergson loads it from there).
    """
    import torch
    from transformers import AutoModelForCausalLM

    set_determinism(cfg.seed)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32,
        attn_implementation=os.environ.get("CPT_ATTN", "sdpa"),
    ).to(device)
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95),
                            weight_decay=cfg.weight_decay, eps=cfg.adam_eps)
    batches = schedule(len(docs), cfg.epochs, cfg.batch_docs, cfg.seed)
    n_steps = len(batches)

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        p = (step - cfg.warmup_steps) / max(1, n_steps - cfg.warmup_steps)
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    w_all = torch.tensor(np.asarray(weights, dtype=np.float32))
    log, t0 = [], time.time()
    for step, idx in enumerate(batches):
        w = w_all[idx]
        if not bool((w != 0).any()):
            # Keep the optimizer and schedule in step with the full run.
            sched.step()
            continue
        x, m = _pad([docs[i] for i in idx], pad_id=0)
        x, m, w = x.to(device), m.to(device), w.to(device)
        denom = m[:, 1:].sum()  # tokens at full weight, so removal is purely subtractive
        loss = 0.0
        for s in range(0, len(idx), cfg.micro_docs):
            sl = slice(s, s + cfg.micro_docs)
            xs, ms, ws = x[sl], m[sl], w[sl]
            if not bool((ws != 0).any()):
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                logits = model(input_ids=xs, attention_mask=ms).logits[:, :-1]
            tgt = xs[:, 1:]
            tok_loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none"
            ).view_as(tgt)
            part = (tok_loss * ms[:, 1:] * ws[:, None]).sum() / denom
            part.backward()
            loss += part.item()
        if cfg.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % log_every == 0 or step == n_steps - 1:
            log.append({"step": step, "loss": float(loss), "s": time.time() - t0})
            if verbose:
                print(f"  cpt step {step}/{n_steps} loss {float(loss):.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out)
        (out / "cpt_config.json").write_text(json.dumps(
            {**asdict(cfg), "n_steps": n_steps, "n_docs": len(docs),
             "kept": float((np.asarray(weights) != 0).mean()), "log": log}, indent=1))
    return model, log


def sequence_logprobs(model, rows: list[dict], batch_size: int = 32,
                      device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """Summed log-prob and token count of the labelled tokens of each pre-tokenized row.

    A row is {"input_ids": [...], "labels": [...]} with -100 on unscored positions (the
    same rows Bergson reads), so the measured objective matches the attributed one.
    """
    import torch

    out = np.zeros(len(rows))
    cnt = np.zeros(len(rows))
    order = np.argsort([len(r["input_ids"]) for r in rows])
    with torch.no_grad():
        for b in range(0, len(rows), batch_size):
            idx = order[b: b + batch_size]
            x, m = _pad([rows[i]["input_ids"] for i in idx], 0)
            lab, _ = _pad([rows[i]["labels"] for i in idx], -100)
            lab[~m] = -100
            x, m, lab = x.to(device), m.to(device), lab.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                logits = model(input_ids=x, attention_mask=m).logits[:, :-1].float()
            tgt = lab[:, 1:]
            keep = tgt != -100
            lp = torch.log_softmax(logits, -1).gather(-1, tgt.clamp_min(0)[..., None])[..., 0]
            out[idx] = (lp * keep).sum(1).cpu().numpy()
            cnt[idx] = keep.sum(1).cpu().numpy()
    return out, cnt


def grpo_objective(logp: np.ndarray, advantage: np.ndarray, problem: np.ndarray):
    """J_p = sum_i A_i log p(y_i) per problem, as a pandas Series indexed by problem."""
    import pandas as pd

    return pd.Series(np.asarray(advantage) * np.asarray(logp)).groupby(
        np.asarray(problem)).sum()

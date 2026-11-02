"""Exact metagradients of the gate objective through the weighted CPT run (MAGIC).

MAGIC (Ilyas & Engstrom 2025, arXiv 2504.16430) computes dJ/dw_d, the derivative of a
target J of the final model with respect to every training document's loss weight w_d, by
reverse-mode differentiation through the whole training run. It needs a run that can be
replayed exactly, which `train_cpt` is: fixed batches, fixed denominators, deterministic
kernels. Removing a set R of documents is predicted to change J by -sum_{d in R} dJ/dw_d,
so the scores go through the same LDS evaluation as every other method. No first-order
method can predict retraining better, so MAGIC separates "the approximate methods are
wrong" from "retraining is not linear in the data at this scale".

Reverse pass. A step of `train_cpt` maps (theta, m, v) to (theta', m', v') with
g = grad_theta L_t(theta, w) and torch's AdamW (bias corrections c1, c2 at step k):

    m' = b1 m + (1 - b1) g,   v' = b2 v + (1 - b2) g^2
    theta' = theta (1 - lr wd) - (lr / c1) m' / (sqrt(v') / sqrt(c2) + eps)

Given the adjoints of (theta', m', v'), the chain rule gives those of (theta, m, v) and of
g; the only second-order term is grad_theta(g . gbar), a Hessian-vector product, which a
double backward returns together with d(g . gbar)/dw_d for each document in the batch.
States are recomputed segment by segment from checkpoints, so memory is O(sqrt(T)) states.
The forward replay repeats `train_cpt` operation for operation; `metagradients` checks this
by returning the replayed final model, whose J must equal the original run's.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from reachtrace.synthetic.cpt import CPTConfig, _pad, schedule, set_determinism


def _setup(model_path: str, n_docs: int, cfg: CPTConfig, dtype, device: str):
    """Model, optimizer, schedule and batches exactly as `train_cpt` builds them."""
    import torch
    from transformers import AutoModelForCausalLM

    set_determinism(cfg.seed)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, attn_implementation=os.environ.get("CPT_ATTN", "sdpa"),
    ).to(device)
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95),
                            weight_decay=cfg.weight_decay, eps=cfg.adam_eps)
    batches = schedule(n_docs, cfg.epochs, cfg.batch_docs, cfg.seed)
    n_steps = len(batches)

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        p = (step - cfg.warmup_steps) / max(1, n_steps - cfg.warmup_steps)
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    return model, opt, sched, batches


def _up(t):
    """`.float()` for bf16/fp32 logits (as in `train_cpt`); fp64 stays fp64 for checks."""
    import torch

    return t.to(torch.promote_types(t.dtype, torch.float32))


def _batch(docs, idx, device):
    x, m = _pad([docs[i] for i in idx], pad_id=0)
    return x.to(device), m.to(device)


def _step(model, opt, sched, docs, idx, w_all, cfg: CPTConfig, device: str) -> bool:
    """One step of `train_cpt`, same operations in the same order. False if skipped."""
    import torch

    w = w_all[idx]
    if not bool((w != 0).any()):
        sched.step()
        return False
    x, m = _batch(docs, idx, device)
    w = w.to(device)
    denom = m[:, 1:].sum()
    for s in range(0, len(idx), cfg.micro_docs):
        sl = slice(s, s + cfg.micro_docs)
        xs, ms, ws = x[sl], m[sl], w[sl]
        if not bool((ws != 0).any()):
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            logits = model(input_ids=xs, attention_mask=ms).logits[:, :-1]
        tgt = xs[:, 1:]
        tok_loss = torch.nn.functional.cross_entropy(
            _up(logits).reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none"
        ).view_as(tgt)
        part = (tok_loss * ms[:, 1:] * ws[:, None]).sum() / denom
        part.backward()
    return True


def _finish_step(opt, sched) -> None:
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)


def _flat(ts) -> torch.Tensor:  # noqa: F821
    import torch

    return torch.cat([t.reshape(-1) for t in ts])


def _moments(opt, params) -> tuple:
    return (_flat([opt.state[p]["exp_avg"] for p in params]),
            _flat([opt.state[p]["exp_avg_sq"] for p in params]))


def _unflat(vec, params) -> list:
    out, i = [], 0
    for p in params:
        out.append(vec[i: i + p.numel()].view_as(p))
        i += p.numel()
    return out


def replay(model_path: str, docs: list[list[int]], weights: np.ndarray, cfg: CPTConfig,
           device: str = "cuda", dtype=None):
    """Train exactly as `train_cpt` does (for checks and finite differences)."""
    import torch

    dtype = dtype or torch.float32
    model, opt, sched, batches = _setup(model_path, len(docs), cfg, dtype, device)
    w_all = torch.tensor(np.asarray(weights, dtype=np.float64)).to(dtype)
    for idx in batches:
        if _step(model, opt, sched, docs, idx, w_all, cfg, device):
            _finish_step(opt, sched)
    model.eval()
    return model


def _target_grads(model, rows: list[dict], coef: np.ndarray, device: str,
                  batch_size: int = 16):
    """Gradients of J_q = sum_i coef[i, q] * (summed log-prob of row i), one per column."""
    import torch

    params = list(model.parameters())
    Q = coef.shape[1]
    grads = [torch.zeros(sum(p.numel() for p in params), device=device,
                         dtype=params[0].dtype) for _ in range(Q)]
    J = np.zeros(Q)
    order = np.argsort([len(r["input_ids"]) for r in rows])
    for b in range(0, len(rows), batch_size):
        idx = order[b: b + batch_size]
        x, m = _pad([rows[i]["input_ids"] for i in idx], 0)
        lab, _ = _pad([rows[i]["labels"] for i in idx], -100)
        lab[~m] = -100
        x, m, lab = x.to(device), m.to(device), lab.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            logits = _up(model(input_ids=x, attention_mask=m).logits[:, :-1])
        tgt = lab[:, 1:]
        keep = tgt != -100
        lp = torch.log_softmax(logits, -1).gather(-1, tgt.clamp_min(0)[..., None])[..., 0]
        lp = (lp * keep).sum(1)
        c = torch.tensor(coef[idx], device=device, dtype=lp.dtype)
        for q in range(Q):
            Jq = (c[:, q] * lp).sum()
            J[q] += Jq.item()
            g = torch.autograd.grad(Jq, params, retain_graph=q < Q - 1)
            grads[q] += _flat(g)
    return grads, J


def metagradients(
    model_path: str, docs: list[list[int]], cfg: CPTConfig, rows: list[dict],
    coef: np.ndarray, device: str = "cuda", dtype=None, segment: int | None = None,
    hvp_docs: int = 8, ckpt_dir: str | Path | None = None, log=print,
) -> tuple[np.ndarray, dict]:
    """dJ_q/dw_d at w = 1 for every document d and every target column q of `coef`.

    `rows` are pre-tokenized {"input_ids", "labels"} rows and J_q = sum_i coef[i, q] *
    log p(row i). Returns (scores [n_docs, Q], info) where info has the targets' values at
    the replayed final model, its state dict (to check the replay against the original
    run) and timings.
    """
    import torch

    if cfg.max_grad_norm:
        raise NotImplementedError("gradient clipping is not differentiated here")
    dtype = dtype or torch.float32
    coef = np.asarray(coef, dtype=np.float64)
    if coef.ndim == 1:
        coef = coef[:, None]
    Q = coef.shape[1]
    n = len(docs)
    t0 = time.time()
    model, opt, sched, batches = _setup(model_path, n, cfg, dtype, device)
    params = list(model.parameters())
    w_all = torch.ones(n, dtype=dtype)
    T = len(batches)
    S = segment or max(1, int(math.ceil(math.sqrt(T))))
    if ckpt_dir is not None:
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=ckpt_dir)
    cdir = Path(tmp.name)

    def save_ckpt(t: int) -> None:
        torch.save({"params": [p.detach().cpu().clone() for p in params],
                    "opt": opt.state_dict(), "sched": sched.state_dict()}, cdir / f"{t}.pt")

    def load_ckpt(t: int) -> None:
        blob = torch.load(cdir / f"{t}.pt", map_location="cpu", weights_only=False)
        with torch.no_grad():
            for p, v in zip(params, blob["params"]):
                p.copy_(v.to(device))
        opt.load_state_dict(blob["opt"])
        sched.load_state_dict(blob["sched"])
        opt.zero_grad(set_to_none=True)

    # Forward: the full run, checkpointing every S steps.
    n_opt = 0  # optimizer steps taken (AdamW's bias-correction counter)
    opt_steps = []
    for t, idx in enumerate(batches):
        if t % S == 0:
            save_ckpt(t)
        opt_steps.append(n_opt)
        if _step(model, opt, sched, docs, idx, w_all, cfg, device):
            _finish_step(opt, sched)
            n_opt += 1
    model.eval()
    t_fwd = time.time() - t0
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    gbar_list, J = _target_grads(model, rows, coef, device)
    model.train()
    log(f"[magic] forward {T} steps in {t_fwd:.0f}s; J = {np.round(J, 4).tolist()}")

    P = sum(p.numel() for p in params)
    th_bar = [g.clone() for g in gbar_list]
    m_bar = [torch.zeros(P, device=device, dtype=dtype) for _ in range(Q)]
    v_bar = [torch.zeros(P, device=device, dtype=dtype) for _ in range(Q)]
    w_bar = np.zeros((n, Q))
    b1, b2 = opt.param_groups[0]["betas"]
    eps = opt.param_groups[0]["eps"]
    wd = opt.param_groups[0]["weight_decay"]
    norms = []

    for s0 in reversed(range(0, T, S)):
        s1 = min(T, s0 + S)
        load_ckpt(s0)
        rec = {}
        for t in range(s0, s1):
            idx = batches[t]
            th = _flat([p.detach() for p in params]).cpu()
            lr = opt.param_groups[0]["lr"]
            if _step(model, opt, sched, docs, idx, w_all, cfg, device):
                g = _flat([p.grad for p in params]).cpu()
                _finish_step(opt, sched)
                m1, v1 = _moments(opt, params)
                rec[t] = (th, g, m1.cpu(), v1.cpu(), lr)
            else:
                rec[t] = None
        for t in reversed(range(s0, s1)):
            if rec[t] is None:
                continue
            th, g, m1, v1, lr = (r.to(device) if torch.is_tensor(r) else r for r in rec.pop(t))
            k = opt_steps[t] + 1
            c1, c2 = 1 - b1 ** k, 1 - b2 ** k
            sq = v1.sqrt()
            den = sq / math.sqrt(c2) + eps
            a = lr / c1
            coef_m = -a / den
            coef_v = torch.where(sq > 0, a * m1 / (den * den * math.sqrt(c2) * 2 * sq),
                                 torch.zeros_like(sq))
            gbars = []
            for q in range(Q):
                mt = m_bar[q] + th_bar[q] * coef_m
                vt = v_bar[q] + th_bar[q] * coef_v
                gbars.append((1 - b1) * mt + 2 * (1 - b2) * g * vt)
                m_bar[q] = b1 * mt
                v_bar[q] = b2 * vt
                th_bar[q] = th_bar[q] * (1 - lr * wd)
            del sq, den, coef_m, coef_v, m1, v1, g
            # Hessian-vector products and per-document weight gradients at theta_t.
            with torch.no_grad():
                for p, v in zip(params, _unflat(th, params)):
                    p.copy_(v)
            del th
            idx = batches[t]
            x, mk = _batch(docs, idx, device)
            denom = mk[:, 1:].sum()
            gb_un = [_unflat(gb, params) for gb in gbars]
            for s in range(0, len(idx), hvp_docs):
                sl = slice(s, s + hvp_docs)
                xs, ms = x[sl], mk[sl]
                w = torch.ones(xs.size(0), device=device, dtype=dtype, requires_grad=True)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.startswith("cuda")):
                    logits = model(input_ids=xs, attention_mask=ms).logits[:, :-1]
                tgt = xs[:, 1:]
                tok_loss = torch.nn.functional.cross_entropy(
                    _up(logits).reshape(-1, logits.size(-1)), tgt.reshape(-1),
                    reduction="none").view_as(tgt)
                loss = (tok_loss * ms[:, 1:] * w[:, None]).sum() / denom
                gr = torch.autograd.grad(loss, params, create_graph=True)
                for q in range(Q):
                    dot = sum((gi * bi).sum() for gi, bi in zip(gr, gb_un[q]))
                    out = torch.autograd.grad(dot, params + [w], retain_graph=q < Q - 1)
                    th_bar[q] += _flat(out[:-1]).detach()
                    w_bar[np.asarray(idx)[sl], q] += out[-1].detach().double().cpu().numpy()
                del gr, logits, tok_loss, loss
            del gbars, gb_un
        norms.append({"t": s0, **{f"theta_bar_{q}": float(th_bar[q].norm()) for q in range(Q)}})
        log(f"[magic] back to step {s0}/{T} ({time.time() - t0:.0f}s); |theta_bar| "
            f"{[round(norms[-1][f'theta_bar_{q}'], 3) for q in range(Q)]}")
    tmp.cleanup()
    model.eval()
    return w_bar, {"J": J.tolist(), "final_state": final_state, "seconds": time.time() - t0,
                   "forward_seconds": t_fwd, "segment": S, "norms": norms}

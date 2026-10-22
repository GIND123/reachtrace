"""Query targets: which tokens of a correct RL solution the attribution should explain.

A full-response loss is dominated by tokens both models already predict well, which is
the fluency/format confound BehaviorTrace documents. Following Zhang et al. (arXiv
2610.24663), who find ~4% of positions carry ~90% of the RL pass@1 gain, the primary
target is the *fork tokens*: positions where the RL model is far more confident than the
base model under teacher forcing. Each fork span becomes one Bergson prompt/completion row
(prefix -> span), so per-trace scores are the sum over its spans.
"""

from __future__ import annotations

import math

import numpy as np


def fork_mask(
    logp_rl: np.ndarray,
    logp_base: np.ndarray,
    threshold: float = math.log(4.0),
    top_frac: float | None = None,
) -> np.ndarray:
    """True where log p_RL(y_t) - log p_base(y_t) > `threshold` (or is in the top fraction)."""
    lr = np.asarray(logp_rl) - np.asarray(logp_base)
    if top_frac is not None:
        k = max(1, round(top_frac * lr.size))
        cut = np.partition(lr, -k)[-k]
        return lr >= cut
    return lr > threshold


def answer_mask(pieces: list[str]) -> np.ndarray:
    """True from the token that starts the last 'Answer:' line to the end."""
    text = "".join(pieces)
    pos = text.rfind("Answer")
    mask = np.zeros(len(pieces), dtype=bool)
    if pos < 0:
        return mask
    offset = 0
    for i, p in enumerate(pieces):
        if offset + len(p) > pos:
            mask[i:] = True
            break
        offset += len(p)
    return mask


def spans(mask: np.ndarray, merge_gap: int = 1) -> list[tuple[int, int]]:
    """Half-open runs of True, merging runs separated by at most `merge_gap` False tokens."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    out = [[int(idx[0]), int(idx[0]) + 1]]
    for i in idx[1:]:
        if i - out[-1][1] <= merge_gap:
            out[-1][1] = int(i) + 1
        else:
            out.append([int(i), int(i) + 1])
    return [tuple(s) for s in out]


def span_queries(
    prompt: str, pieces: list[str], mask: np.ndarray, merge_gap: int = 1
) -> list[dict]:
    """One prompt/completion row per span; prompt carries the full prefix as context."""
    rows = []
    for s, e in spans(mask, merge_gap):
        rows.append(
            dict(
                prompt=prompt + "".join(pieces[:s]),
                completion="".join(pieces[s:e]),
                start=s,
                end=e,
            )
        )
    return rows


def teacher_forced_logprobs(
    model, tokenizer, prompt: str, completion: str
) -> tuple[list[str], np.ndarray]:
    """Per-token log-probs of `completion` given `prompt` under an HF causal LM."""
    import torch

    p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    c_ids = tokenizer(completion, add_special_tokens=False).input_ids
    ids = torch.tensor([p_ids + c_ids], device=model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, len(p_ids) - 1 : -1].float()
    lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(c_ids, device=model.device)[:, None])
    pieces = [tokenizer.decode([t]) for t in c_ids]
    return pieces, lp.squeeze(1).cpu().numpy()

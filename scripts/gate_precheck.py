"""Synthetic gate v2 pre-check, run locally (Apple MPS or CPU): attribution side only.

Question: after continued pretraining on the v2 planted corpus, does gradient attribution
for a composed task blick(zorp(x)) put the causal families A and B above the distractors
(C, D), the format-only family F, the worked-trace family W and generic web text, and does
it beat BM25 when task and documents share no operation names (held-out aliases)?

Ground truth is by construction (the composition uses exactly A and B). There is no GRPO
and no leave-out retraining here, so this checks gates G1-G3 for the static (pretraining)
part only; it is not the full gate.

    .venv/bin/python scripts/gate_precheck.py
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from reachtrace.attrib.lexical import BM25
from reachtrace.synthetic.tasks import make_ops
from reachtrace.synthetic.v2 import alias_pools, alias_task, make_corpus_v2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "gate_precheck"
MODEL = "HuggingFaceTB/SmolLM2-135M"
SEED = 0
DOCS_PER_FAMILY, N_FORMAT, N_WORKED, N_GENERIC = 400, 400, 400, 3000
REPEATS = 4            # planted documents are seen this many times during CPT
SKETCH_DIM = 2**18     # CountSketch dimension for per-example gradients
N_QUERIES = 40
N_GENERIC_SCORED = 600


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def generic_docs(n: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train",
                      streaming=True)
    out = []
    for r in ds:
        out.append(r["text"][:1500])
        if len(out) >= n:
            break
    return out


def cpt(model, tok, texts: list[str], dev, epochs: int = 1, lr: float = 3e-4,
        batch: int = 16, max_len: int = 256) -> list[float]:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    enc = [tok(t, truncation=True, max_length=max_len).input_ids for t in texts]
    steps = epochs * math.ceil(len(enc) / batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / steps))
    losses, rng = [], random.Random(SEED)
    for _ in range(epochs):
        order = list(range(len(enc)))
        rng.shuffle(order)
        for i in range(0, len(order), batch):
            ids = [enc[j] + [tok.eos_token_id] for j in order[i: i + batch]]
            L = max(map(len, ids))
            x = torch.full((len(ids), L), tok.pad_token_id or 0)
            att = torch.zeros((len(ids), L), dtype=torch.long)
            for k, s in enumerate(ids):
                x[k, : len(s)] = torch.tensor(s)
                att[k, : len(s)] = 1
            x, att = x.to(dev), att.to(dev)
            labels = x.masked_fill(att == 0, -100)
            loss = model(input_ids=x, attention_mask=att, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss))
    model.eval()
    return losses


class Sketch:
    """CountSketch of the full parameter gradient (inner products preserved in expectation)."""

    def __init__(self, model, dim: int, dev, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.idx, self.sign = [], []
        for p in self.params:
            n = p.numel()
            self.idx.append(torch.randint(0, dim, (n,), generator=g).to(dev))
            self.sign.append((torch.randint(0, 2, (n,), generator=g) * 2 - 1).to(dev).float())
        self.dim, self.dev = dim, dev

    def __call__(self) -> torch.Tensor:
        out = torch.zeros(self.dim, device=self.dev)
        for p, i, s in zip(self.params, self.idx, self.sign):
            if p.grad is not None:
                out.index_add_(0, i, p.grad.reshape(-1).float() * s)
        return out


def example_grad(model, tok, sketch, prompt: str, completion: str, dev) -> torch.Tensor:
    """Sketched gradient of the summed loss on `completion` given `prompt` (prompt may be '')."""
    p_ids = tok(prompt).input_ids if prompt else []
    c_ids = tok(completion, add_special_tokens=False).input_ids
    ids = torch.tensor([p_ids + c_ids], device=dev)
    labels = ids.clone()
    labels[0, : len(p_ids)] = -100
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids)
    logits = out.logits[0, :-1]
    loss = torch.nn.functional.cross_entropy(logits.float(), labels[0, 1:], ignore_index=-100,
                                             reduction="sum")
    loss.backward()
    return sketch()


def evaluate(scores: np.ndarray, fam: np.ndarray, positives=("A", "B"), k: int = 100) -> dict:
    pos = np.isin(fam, positives)
    order = np.argsort(-scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(len(scores))
    # AUC of positives vs everything else
    auc = float((ranks[~pos][None, :] > ranks[pos][:, None]).mean())
    fam_mean = pd.Series(scores).groupby(fam).mean()
    z = (fam_mean - scores.mean()) / (scores.std() + 1e-12)
    return {"recall@100": float(pos[order[:k]].mean()), "auc_AB": auc,
            "A_B_above_all_others": bool(z[list(positives)].min()
                                         > z.drop(list(positives)).max()),
            **{f"z_{f}": float(v) for f, v in z.items()}}


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    dev = device()
    torch.manual_seed(SEED)
    ops = make_ops(SEED)
    pools = alias_pools(list(ops), per_family=20, seed=SEED)
    planted = make_corpus_v2(ops, pools, DOCS_PER_FAMILY, N_FORMAT, N_WORKED, seed=SEED)
    generic = generic_docs(N_GENERIC)
    print(f"{len(planted)} planted docs, {len(generic)} generic docs ({time.time()-t0:.0f}s)")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev)
    texts = [d["text"] for d in planted] * REPEATS + generic
    losses = cpt(model, tok, texts, dev)
    print(f"CPT {len(losses)} steps, loss {np.mean(losses[:20]):.2f} -> "
          f"{np.mean(losses[-20:]):.2f} ({time.time()-t0:.0f}s)")

    # Skill check: single operations with held-out aliases and one in-context example.
    rng = random.Random(SEED + 1)
    skill = {}
    for f in ("A", "B"):
        ok = 0
        for _ in range(30):
            x = "".join(rng.choice("0123456789") for _ in range(4))
            t = alias_task([ops[f]], [pools[f][-1]], x, rng)
            ids = tok(t["prompt"] + "Answer: ", return_tensors="pt").input_ids.to(dev)
            gen = model.generate(ids, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            out = tok.decode(gen[0, ids.shape[1]:]).strip()
            ok += out.startswith(t["answer"])
        skill[f] = ok / 30
    print("single-op accuracy (held-out alias):", skill)

    sketch = Sketch(model, SKETCH_DIM, dev)
    gen_idx = random.Random(SEED).sample(range(len(generic)), N_GENERIC_SCORED)
    docs = [(d["family"], d["text"]) for d in planted] + [("generic", generic[i]) for i in gen_idx]
    fam = np.array([f for f, _ in docs])
    D = torch.zeros((len(docs), SKETCH_DIM), dtype=torch.float16, device=dev)
    norms = np.zeros(len(docs))
    for i, (_, text) in enumerate(docs):
        g = example_grad(model, tok, sketch, "", text, dev)
        norms[i] = float(g.norm())
        D[i] = (g / (g.norm() + 1e-12)).half()
        if i % 500 == 0:
            print(f"  doc grads {i}/{len(docs)} ({time.time()-t0:.0f}s)")

    bm25 = BM25([t for _, t in docs])
    rows = []
    for cond in ("alias", "named"):
        names = ([pools["A"][-1], pools["B"][-1]] if cond == "alias"
                 else [pools["A"][0], pools["B"][0]])
        qrng = random.Random(SEED + 7)
        cos_c, cos_w, bm = [], [], []
        for _ in range(N_QUERIES):
            x = "".join(qrng.choice("0123456789") for _ in range(4))
            t = alias_task([ops["A"], ops["B"]], names, x, qrng)
            wrong = "".join(qrng.choice("0123456789") for _ in range(len(t["answer"])))
            prompt = t["prompt"] + "Answer: "
            gc = example_grad(model, tok, sketch, prompt, t["answer"], dev)
            gw = example_grad(model, tok, sketch, prompt, wrong, dev)
            cos_c.append((D.float() @ (gc / gc.norm())).cpu().numpy())
            cos_w.append((D.float() @ (gw / gw.norm())).cpu().numpy())
            bm.append(bm25.scores(t["prompt"] + " " + t["answer"]))
        methods = {
            "BM25": np.mean(bm, 0),
            "gradient cosine": np.mean(cos_c, 0),
            "contrastive gradient": np.mean(cos_c, 0) - np.mean(cos_w, 0),
            "norm-only": norms,
        }
        for name, sc in methods.items():
            rows.append({"condition": cond, "method": name, **evaluate(sc, fam)})
        print(f"{cond}: done ({time.time()-t0:.0f}s)")

    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "precheck.csv", index=False)
    meta = {"model": MODEL, "planted_docs": len(planted), "generic_docs_cpt": len(generic),
            "repeats": REPEATS, "cpt_steps": len(losses),
            "cpt_loss_first20": float(np.mean(losses[:20])),
            "cpt_loss_last20": float(np.mean(losses[-20:])),
            "single_op_accuracy_heldout_alias": skill, "sketch_dim": SKETCH_DIM,
            "queries": N_QUERIES, "generic_docs_scored": N_GENERIC_SCORED,
            "device": str(dev), "seconds": time.time() - t0}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    cols = ["condition", "method", "recall@100", "auc_AB", "A_B_above_all_others", "z_A",
            "z_B", "z_C", "z_D", "z_F", "z_W", "z_generic"]
    print(res[cols].to_markdown(index=False, floatfmt=".3f"))


if __name__ == "__main__":
    main()

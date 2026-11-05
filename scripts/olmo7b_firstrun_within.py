"""Re-analysis of the first 7B run's saved TrackStar scores (scripts/olmo7b_midtrain.py):
does each problem's correct-minus-incorrect contrast replicate across halves of its own 8
rollouts, and how much of it is the finished-vs-cut-off contrast?

Reads $MID_ART (default artifacts/olmo7b_midtrain) with trackstar_p32/scores and
queries_meta.parquet; writes results/olmo7b_deep/first_run_within.json. Needs no GPU and no
bergson (the score file is read directly).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
MID = Path(os.environ.get("MID_ART", ROOT / "artifacts" / "olmo7b_midtrain")).resolve()
OUT = ROOT / "results" / "olmo7b_deep"


def main() -> None:
    path = MID / "trackstar_p32" / "scores"
    info = json.loads((path / "info.json").read_text())
    raw = np.memmap(path / "scores.bin", dtype=np.dtype(info["dtype"]), mode="r")
    S = np.stack([raw[f"score_{j}"].astype(np.float64) for j in range(info["num_scores"])], 1)
    qm = pd.read_parquet(MID / "queries_meta.parquet")
    cos = S * np.sign(qm.advantage.to_numpy())[None, :]  # columns were sign(A) cos
    correct = qm.correct.to_numpy()
    fin = (qm.finish_reason == "stop").to_numpy()
    pids = qm.problem_id.to_numpy()
    probs = sorted(set(pids))
    rng = np.random.default_rng(0)

    def contrast(idx, lab):
        return cos[:, idx[lab]].mean(1) - cos[:, idx[~lab]].mean(1)

    def within(lab_all, n=200):
        out = []
        for p in probs:
            idx = np.flatnonzero(pids == p)
            lab = lab_all[idx]
            if lab.sum() < 2 or (~lab).sum() < 2:
                continue
            r = []
            for _ in range(n):
                a = np.zeros(len(idx), bool)
                for v in (True, False):
                    j = rng.permutation(np.flatnonzero(lab == v))
                    a[j[: len(j) // 2]] = True
                r.append(spearmanr(contrast(idx[a], lab[a]), contrast(idx[~a], lab[~a]))[0])
            out.append(np.mean(r))
        return float(np.mean(out)), len(out)

    obs, n_p = within(correct)
    null = []
    for _ in range(20):
        lab = correct.copy()
        for p in probs:
            i = np.flatnonzero(pids == p)
            lab[i] = rng.permutation(lab[i])
        null.append(within(lab, n=20)[0])
    full = {p: contrast(np.flatnonzero(pids == p), correct[pids == p]) for p in probs}
    conf = [spearmanr(full[p], contrast(np.flatnonzero(pids == p), fin[pids == p]))[0]
            for p in probs if 0 < fin[pids == p].sum() < (pids == p).sum()]
    res = {"within_doc_rho": obs, "n_problems": n_p, "null_doc_rho": float(np.mean(null)),
           "null_doc_sd": float(np.std(null)),
           "cross_problem_doc_rho": float(np.mean([
               spearmanr(full[p], full[q])[0] for i, p in enumerate(probs)
               for q in probs[i + 1:]])),
           "contrast_vs_finished_contrast_rho": float(np.mean(conf)),
           "n_problems_finished_both": int(sum(
               correct[(pids == p) & fin].any() and not correct[(pids == p) & fin].all()
               for p in probs))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "first_run_within.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

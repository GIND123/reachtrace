"""`reachtrace <command>`: thin entry points over the library. Heavy imports stay lazy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import typer
import yaml

from .config import AttributionConfig, PairConfig, SyntheticConfig, load_config

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _sha(lock_path: Path, repo: str, revision: str) -> str:
    lock = yaml.safe_load(lock_path.read_text())
    try:
        return lock["repos"][repo][revision]
    except KeyError:
        raise typer.BadParameter(
            f"{repo}@{revision} not pinned; run `reachtrace pin` first"
        ) from None


@app.command()
def pin(
    config: Path = typer.Option(..., help="Model-pair YAML"),
    out: Path = typer.Option(Path("configs/pins.lock.yaml")),
    rl_steps: str = typer.Option("100,200,400,800,1600,2800", help="RL-Zero step branches to pin"),
):
    """Resolve every model revision (incl. stage finals, midtrain arms, RL steps) to a SHA."""
    from .registry import list_branches, midtrain_arms, pick_rl_steps, stage_finals
    from .registry import pin as do_pin

    cfg = load_config(config, PairConfig)
    base_br, rl_br = list_branches(cfg.base.id), list_branches(cfg.rl.id)
    base_revs = [
        cfg.base.revision,
        *stage_finals(base_br).values(),
        *midtrain_arms(base_br).values(),
    ]
    wanted = [int(s) for s in rl_steps.split(",") if s]
    rl_revs = [cfg.rl.revision, *pick_rl_steps(rl_br, wanted).values()]
    lock = do_pin({cfg.base.id: base_revs, cfg.rl.id: rl_revs}, out)
    typer.echo(yaml.safe_dump(lock, sort_keys=False))


@app.command()
def render(question: str, template: str = "rlzero31_math", cue: str = ""):
    """Print the exact prompt string both models receive."""
    from .prompts import get_template

    typer.echo(repr(get_template(template).render(question, cue=cue)))


@app.command()
def certify(k: int, gamma: float = 0.5, alpha: float = 0.05):
    """Samples with zero successes needed to certify pass@k < gamma."""
    from .reach.passk import n_to_certify

    typer.echo(n_to_certify(k, gamma, alpha))


@app.command()
def evals(
    sets: str = typer.Option("math500,math_l45,aime24,aime25,amc23,olympiadbench"),
    out: Path = typer.Option(Path("artifacts/evals/core.parquet")),
):
    """Download eval sets into one de-duplicated table (requires `reachtrace[data]`)."""
    from .data.evals import load_eval_sets

    df = load_eval_sets(sets.split(","))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    typer.echo(df.groupby("eval_set").size().to_string())


@app.command("decon-ngram")
def decon_ngram(
    evals_path: Path = typer.Option(Path("artifacts/evals/core.parquet"), "--evals"),
    out: Path = typer.Option(Path("artifacts/decon/ngram_report.parquet")),
    n: int = 13,
):
    """Exact n-gram scan of eval questions against Dolci and its upstream RL sources."""
    import pandas as pd
    from datasets import load_dataset

    from .data.dolci import DOLCI_MATH_ID, UPSTREAM_REFS
    from .decon.ngram import contamination_report

    refs = {"dolci": load_dataset(DOLCI_MATH_ID, split="train").to_pandas()}
    for name, (hf_id, col) in UPSTREAM_REFS.items():
        if hf_id is None:
            typer.echo(f"skipping {name}: Hub ID not verified yet", err=True)
            continue
        refs[name] = load_dataset(hf_id, split="train").to_pandas().rename(columns={col: "prompt"})
    refs = {
        k: pd.DataFrame(
            {"ref_id": [f"{k}/{i}" for i in range(len(v))], "text": v["prompt"].astype(str)}
        )
        for k, v in refs.items()
    }
    report = contamination_report(pd.read_parquet(evals_path), refs, n=n)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_parquet(out)
    typer.echo(f"flagged {int(report.any_flag.sum())} / {len(report)}")


@app.command()
def sample(
    config: Path = typer.Option(...),
    side: str = typer.Option(..., help="base | rl"),
    evals_path: Path = typer.Option(Path("artifacts/evals/clean.parquet"), "--evals"),
    lock: Path = typer.Option(Path("configs/pins.lock.yaml")),
    out_dir: Path = typer.Option(Path("artifacts/gen")),
    revision: str = typer.Option("", help="Override revision (e.g. stage2-step47684)"),
    cue: str = typer.Option("", help="Cue-forcing string appended after the prompt"),
    max_rounds: int = 32,
    tp: int = 1,
):
    """Adaptive vLLM sampling until every problem's reachability is determined or n_max."""
    import pandas as pd

    from .prompts import get_template
    from .sample.adaptive import plan_round
    from .sample.vllm_runner import load_counts, load_llm, run_jobs, write_shard

    cfg = load_config(config, PairConfig)
    ref = cfg.base if side == "base" else cfg.rl
    rev = revision or ref.revision
    sha = _sha(lock, ref.id, rev)
    problems = pd.read_parquet(evals_path).set_index("problem_id")
    tag = rev if not cue else f"{rev}+cue-{hashlib.sha1(cue.encode()).hexdigest()[:8]}"
    shard_dir = out_dir / cfg.pair / side / tag
    llm = load_llm(ref.id, sha, cfg.decode, tp)
    template = get_template(cfg.template_id)
    for r in range(max_rounds):
        counts = load_counts(shard_dir, list(problems.index))
        jobs = plan_round(counts, cfg.sampling, cfg.reach)
        typer.echo(f"round {r}: {len(jobs)} jobs")
        if not jobs:
            break
        df = run_jobs(
            llm,
            jobs,
            problems,
            model_id=ref.id,
            revision_sha=sha,
            template=template,
            decode=cfg.decode,
            base_seed=cfg.base_seed,
            batch=cfg.sampling.batch,
            cue=cue,
        )
        write_shard(df, shard_dir)


@app.command()
def partition(
    config: Path = typer.Option(...),
    evals_path: Path = typer.Option(Path("artifacts/evals/clean.parquet"), "--evals"),
    base_dir: Path = typer.Option(..., help="Generation shards for the base model"),
    rl_dir: Path = typer.Option(..., help="Generation shards for the RL model"),
    out: Path = typer.Option(Path("artifacts/partitions/partition.parquet")),
):
    """Build the R/U/L/N table over the configured K sweep."""
    import pandas as pd

    from .reach.partition import partition_counts, partition_table
    from .sample.vllm_runner import load_counts

    cfg = load_config(config, PairConfig)
    ids = list(pd.read_parquet(evals_path, columns=["problem_id"]).problem_id)
    b = load_counts(base_dir, ids).rename(columns={"n": "n_base", "c": "c_base"})
    r = load_counts(rl_dir, ids).rename(columns={"n": "n_rl", "c": "c_rl"})
    table = partition_table(
        b.merge(r, on="problem_id"),
        cfg.reach.ksweep,
        cfg.reach.mode,
        cfg.reach.gamma,
        cfg.reach.alpha,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out)
    typer.echo(pd.DataFrame({k: partition_counts(table, k) for k in cfg.reach.ksweep}).to_string())


@app.command("synth-corpus")
def synth_corpus(
    config: Path = typer.Option(Path("configs/synthetic/fragments_v1.yaml")),
    out_dir: Path = typer.Option(Path("artifacts/synthetic")),
):
    """Write the planted-fragment documents and the RL / eval / control task sets."""
    from .synthetic.tasks import make_ops, make_planted_corpus, task_suite

    cfg = load_config(config, SyntheticConfig)
    ops = make_ops(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = make_planted_corpus(ops, cfg.docs_per_family, cfg.seed)
    (out_dir / "planted_docs.jsonl").write_text("\n".join(json.dumps(d) for d in docs) + "\n")
    for name, tasks in task_suite(ops, cfg).items():
        (out_dir / f"{name}.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    typer.echo(f"{len(docs)} planted docs -> {out_dir}")


@app.command("bergson-cmds")
def bergson_cmds(
    config: Path = typer.Option(Path("configs/attribution/tierA.yaml")),
    run_root: Path = typer.Option(Path("runs")),
):
    """Print the Bergson commands for every index in an attribution config."""
    from .attrib.bergson_cmd import build_cmd

    cfg = load_config(config, AttributionConfig)
    for idx in [*cfg.query_index, *cfg.rl_item_index, *cfg.corpus_index]:
        typer.echo(" ".join(build_cmd(idx, run_root)))


if __name__ == "__main__":
    app()

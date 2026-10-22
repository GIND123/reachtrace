import pandas as pd

from reachtrace.cache import Manifest, gen_key, sample_seed, stable_hash
from reachtrace.config import AdaptiveConfig, ReachConfig
from reachtrace.data.dolci import build_rl_items
from reachtrace.decon.ngram import contamination_report
from reachtrace.prompts import get_template
from reachtrace.registry import midtrain_arms, pick_rl_steps, stage_finals
from reachtrace.sample.adaptive import next_draw, plan_round

BASE_BRANCHES = [
    "main",
    "stage1-step0",
    "stage1-step1000",
    "stage1-step1413814",
    "stage2-step1000",
    "stage2-step47684",
    "stage2-step47684-mix-round5-from-2T-ckpt",
    "stage2-step47684-mix-gen-mc-only-from-2T-ckpt",
    "stage3-step11921",
]
RL_BRANCHES = ["main"] + [f"step_{s:04d}" for s in range(100, 2900, 100)]


def test_stable_hash_is_order_invariant():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    kw = dict(
        model_id="m",
        revision_sha="s",
        prompt_sha="p",
        problem_hash="h",
        sample_idx=0,
        seed=1,
        temperature=1.0,
        top_p=1.0,
        max_tokens=10,
        stop=["x"],
        vllm_version="v",
    )
    assert gen_key(**kw) != gen_key(**{**kw, "sample_idx": 1})
    assert sample_seed(1, "p", 0) == sample_seed(1, "p", 0) != sample_seed(1, "p", 1)


def test_manifest_resume(tmp_path):
    m = Manifest(tmp_path / "m.jsonl")
    m.add(["a", "b", "a"])
    assert len(Manifest(tmp_path / "m.jsonl")) == 2 and "a" in m


def test_branch_parsing():
    assert stage_finals(BASE_BRANCHES) == {
        "stage1_final": "stage1-step1413814",
        "stage2_final": "stage2-step47684",
        "stage3_final": "stage3-step11921",
    }
    assert list(midtrain_arms(BASE_BRANCHES)) == [
        "midtrain_arm:gen-mc-only-from-2T-ckpt",
        "midtrain_arm:round5-from-2T-ckpt",
    ]
    assert pick_rl_steps(RL_BRANCHES, [100, 2800, 5000]) == {
        100: "step_0100",
        2800: "step_2800",
        5000: "step_2800",
    }


def test_adaptive_rule():
    cfg = AdaptiveConfig(min_n=32, batch=32, stop_correct=8, n_max=288)
    reach = ReachConfig(ksweep=[1, 4, 16, 64])
    assert next_draw(0, 0, cfg) == 32
    assert next_draw(32, 9, cfg) == 0
    assert next_draw(288, 0, cfg) == 0
    assert next_draw(64, 0, cfg, reach) == 32
    jobs = plan_round(pd.DataFrame({"problem_id": ["p", "q"], "n": [0, 32], "c": [0, 0]}), cfg)
    assert [(j.problem_id, j.batch_idx, j.n) for j in jobs] == [("p", 0, 32), ("q", 1, 32)]


def test_rl_items_dr_grpo_advantages():
    dolci = pd.DataFrame({"item_id": ["d0", "d1"], "prompt": ["1+1?", "2+2?"]})
    roll = pd.DataFrame(
        {
            "item_id": ["d0"] * 4 + ["d1"] * 2,
            "sample_idx": [0, 1, 2, 3, 0, 1],
            "completion": ["x"] * 6,
            "reward": [1, 0, 0, 0, 1, 1],
            "source_revision": ["main"] * 6,
        }
    )
    items = build_rl_items(roll, dolci, get_template("rlzero31_math"))
    assert items.advantage.tolist() == [0.75, -0.25, -0.25, -0.25, 0.0, 0.0]
    assert items.zero_adv_group.tolist() == [False] * 4 + [True] * 2
    assert items.rendered_prompt.iloc[0].endswith('after "Answer:"')


def test_ngram_contamination_flags_overlap():
    evals = pd.DataFrame(
        {
            "problem_id": ["e1", "e2"],
            "question": [
                "Let x be the largest real number such that x squared plus three x equals ten. Find x.",
                "How many primes are below one hundred and also congruent to one modulo four?",
            ],
        }
    )
    refs = {
        "dolci": pd.DataFrame(
            {
                "ref_id": ["r1"],
                "text": [
                    "Problem: let x be the largest real number such that x squared plus three x equals ten. Find x."
                ],
            }
        )
    }
    rep = contamination_report(evals, refs, n=8)
    assert rep.any_flag.tolist() == [True, False]

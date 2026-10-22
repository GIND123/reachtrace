import pandas as pd

from reachtrace.reach.partition import bootstrap_counts, partition_counts, partition_table
from reachtrace.reach.stage_ckpt import provenance_table


def _counts():
    return pd.DataFrame(
        {
            "problem_id": ["r", "u", "l", "n", "q"],
            "n_base": [288, 288, 288, 288, 40],
            "c_base": [200, 0, 150, 0, 0],
            "n_rl": [288, 288, 288, 288, 288],
            "c_rl": [250, 120, 0, 0, 100],
        }
    )


def test_labels_certified():
    t = partition_table(_counts(), [16])
    assert dict(zip(t.problem_id, t["label@16"])) == {
        "r": "R",
        "u": "U",
        "l": "L",
        "n": "N",
        "q": "?",
    }
    assert partition_counts(t, 16).to_dict() == {"R": 1, "U": 1, "L": 1, "N": 1, "?": 1}


def test_bootstrap_counts_bracket_point_estimate():
    t = partition_table(_counts(), [16])
    ci = bootstrap_counts(t, 16, n_boot=200)
    assert (ci.lo <= ci.point).all() and (ci.point <= ci.hi).all()


def test_provenance_first_reach():
    res = pd.DataFrame(
        {
            "problem_id": ["p"] * 3,
            "revision_name": ["stage1_final", "stage2_final", "rl_step_400"],
            "n": [288, 288, 288],
            "c": [0, 60, 200],
        }
    )
    t = provenance_table(res, ["stage1_final", "stage2_final", "rl_step_400"], k=16)
    assert t.first_reach.iloc[0] == "stage2_final"

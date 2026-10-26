"""End-to-end smoke test of the Modal jobs on a tiny model and a cheap GPU (cents).

    .venv/bin/python scripts/smoke.py
"""

from __future__ import annotations

import json

import modal

from reachtrace.prompts import get_template
from reachtrace.remote.app import app, download, generate, ledger_scores

MODEL, REV, GPU = "HuggingFaceTB/SmolLM2-135M", "main", "L4"


def main() -> None:
    t = get_template("rlzero31_math")
    prompt = t.render("What is $2+3$?")
    with modal.enable_output(), app.run():
        print(json.dumps(download.remote(MODEL, REV), indent=1))
        gen = generate.with_options(gpu=GPU)
        r1 = gen.remote(MODEL, REV, [{"key": "q1", "prompt": prompt, "n": 4, "seed": 1}],
                        "smoke/gen.parquet", max_tokens=64, max_model_len=2048)
        print(json.dumps(r1, indent=1))
        # Prefix continuation: the handoff map appends exact token IDs after the prompt.
        r2 = gen.remote(MODEL, REV,
                        [{"key": "q1", "prompt": prompt, "n": 2, "seed": 2,
                          "prefix_ids": [504, 3729, 314]}],
                        "smoke/prefix.parquet", max_tokens=32, max_model_len=2048)
        print(json.dumps(r2, indent=1))
        led = ledger_scores.with_options(gpu=GPU)
        r3 = led.remote(MODEL, REV, "smoke",
                        [{"trace_id": "t1", "prompt": prompt,
                          "completion_ids": [504, 3729, 314, 216, 34, 30, 33, 1550, 216, 37]}],
                        opening=2, max_model_len=2048)
        print(json.dumps(r3, indent=1))


if __name__ == "__main__":
    main()

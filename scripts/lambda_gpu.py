"""Run reachtrace stages on one Lambda Cloud GPU instance under a hard budget.

    export LAMBDA_API_KEY_FILE=~/.lambda_key LAMBDA_SSH_KEY_PATH=~/.ssh/id_ed25519
    python scripts/lambda_gpu.py types                      # prices and capacity
    python scripts/lambda_gpu.py up gpu_1x_a100_sxm4 --hours 6
    python scripts/lambda_gpu.py push                       # repo (tracked files) -> ~/reachtrace
    python scripts/lambda_gpu.py run "bash scripts/lambda/setup_remote.sh"
    python scripts/lambda_gpu.py run "cd reachtrace && ..." [--bg name]
    python scripts/lambda_gpu.py pull reachtrace/artifacts/gate_v3/scores artifacts/gate_v3
    python scripts/lambda_gpu.py down
    python scripts/lambda_gpu.py cost

`up` refuses to launch if the spend so far plus `--hours` at the instance price would pass
BUDGET_USD, and installs a watchdog on the instance that terminates it through the API at
the end of `--hours` (`extend` moves the deadline, still within the budget).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from reachtrace.remote import lambda_cloud as lc

BUDGET_USD = 50.0
# One state file per instance; LAMBDA_SLOT selects which (default slot 1).
SLOT = os.environ.get("LAMBDA_SLOT", "1")
STATE = Path("artifacts/lambda") / ("current.json" if SLOT == "1" else f"current_{SLOT}.json")


def _state() -> dict:
    if not STATE.exists():
        sys.exit("no instance recorded; run `up` first")
    return json.loads(STATE.read_text())


def cmd_types(_):
    for name, v in sorted(lc.instance_types().items(), key=lambda kv: kv[1]["price"]):
        if v["regions"] and v["gpus"]:
            print(f"{name:28s} ${v['price']:6.2f}/h  {v['regions']}")


def cmd_up(a):
    regions = lc.instance_types()[a.type]["regions"]
    if not regions:
        sys.exit(f"no capacity for {a.type}")
    inst = lc.launch(a.type, a.region or regions[0], BUDGET_USD, a.hours)
    print(f"launched {inst.id} ({a.type}, {inst.region}); waiting for it to boot")
    inst = lc.wait_active(inst.id)
    deadline = time.time() + a.hours * 3600
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"id": inst.id, "ip": inst.ip, "type": a.type,
                                 "deadline": deadline}))
    for _ in range(20):  # sshd comes up a little after the API says active
        if lc.run(inst.ip, "true", check=False, capture=True).returncode == 0:
            break
        time.sleep(15)
    lc.install_watchdog(inst.ip, inst.id, deadline)
    print(f"active at {inst.ip}; watchdog terminates it at "
          f"{time.strftime('%H:%M', time.localtime(deadline))}; spent ${lc.spent():.2f}")


def cmd_extend(a):
    s = _state()
    price = lc.instance_types()[s["type"]]["price"]
    new = s["deadline"] + a.hours * 3600
    if lc.spent() + price * max(0.0, (new - time.time()) / 3600) > BUDGET_USD:
        sys.exit("budget guard: extension would pass the budget")
    lc.run(s["ip"], "pkill -f wd_key || true", check=False)
    lc.install_watchdog(s["ip"], s["id"], new)
    s["deadline"] = new
    STATE.write_text(json.dumps(s))
    print(f"deadline now {time.strftime('%H:%M', time.localtime(new))}")


def cmd_push(_):
    s = _state()
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                           capture_output=True, text=True,
                           check=True).stdout.split()
    tar = subprocess.Popen(["tar", "czf", "-", *files], stdout=subprocess.PIPE)
    subprocess.run(lc.ssh_cmd(s["ip"], "mkdir -p reachtrace && tar xzf - -C reachtrace"),
                   stdin=tar.stdout, check=True)
    tar.wait()
    print(f"pushed {len(files)} files")


def cmd_pull(a):
    s = _state()
    Path(a.local).mkdir(parents=True, exist_ok=True)
    remote = a.remote.rstrip("/")
    parent, leaf = remote.rsplit("/", 1) if "/" in remote else (".", remote)
    ssh = subprocess.Popen(lc.ssh_cmd(s["ip"], f"tar czf - -C {parent} {leaf}"),
                           stdout=subprocess.PIPE)
    subprocess.run(["tar", "xzf", "-", "-C", a.local], stdin=ssh.stdout, check=True)
    ssh.wait()


def cmd_run(a):
    s = _state()
    command = a.command
    if a.bg:
        command = (f"mkdir -p ~/logs; nohup setsid bash -lc {json.dumps(command)} "
                   f"> ~/logs/{a.bg}.log 2>&1 < /dev/null & echo started {a.bg}")
    sys.exit(lc.run(s["ip"], command, check=False).returncode)


def cmd_down(_):
    s = _state()
    lc.terminate(s["id"])
    STATE.unlink()
    print(f"terminated {s['id']}; spent ${lc.spent():.2f}")


def cmd_cost(_):
    print(f"spent ${lc.spent():.2f} of ${BUDGET_USD}; open: {lc.open_instances()}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("types")
    up = sub.add_parser("up")
    up.add_argument("type")
    up.add_argument("--hours", type=float, required=True)
    up.add_argument("--region", default="")
    ex = sub.add_parser("extend")
    ex.add_argument("--hours", type=float, required=True)
    sub.add_parser("push")
    pl = sub.add_parser("pull")
    pl.add_argument("remote")
    pl.add_argument("local")
    r = sub.add_parser("run")
    r.add_argument("command")
    r.add_argument("--bg", default="")
    sub.add_parser("down")
    sub.add_parser("cost")
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()

"""Lambda Cloud GPU instances with a hard spend cap.

Everything goes through the public v1 API (https://cloud.lambda.ai/api/v1, HTTP basic auth
with the API key as user name). Nothing account-specific is stored in the repo:

  LAMBDA_API_KEY          API key (or LAMBDA_API_KEY_FILE pointing at a file holding it)
  LAMBDA_INSTANCE_NAME    name shown in the Lambda dashboard (default "reachtrace-gpu")
  LAMBDA_SSH_KEY_NAME     registered SSH key name (default: the instance name)
  LAMBDA_SSH_KEY_PATH     private key for that key (default ~/.ssh/id_ed25519)

Spend is tracked locally in `artifacts/lambda/ledger.jsonl`: one row per launch with the
hourly price, and one per termination. `spent()` bills every open instance up to now,
rounded up to the minute, so the guard also counts instances that are still running.

Instances also guard themselves: `watchdog_script` installs a background job that calls
the terminate endpoint on its own instance at a fixed deadline. Shutting the OS down does
not stop Lambda billing; only the API terminate call does.
"""

from __future__ import annotations

import base64
import json
import math
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

API = "https://cloud.lambda.ai/api/v1"
LEDGER = Path("artifacts/lambda/ledger.jsonl")


def _key() -> str:
    key = os.environ.get("LAMBDA_API_KEY", "")
    if not key and os.environ.get("LAMBDA_API_KEY_FILE"):
        key = Path(os.environ["LAMBDA_API_KEY_FILE"]).read_text().strip()
    if not key:
        raise RuntimeError("set LAMBDA_API_KEY or LAMBDA_API_KEY_FILE")
    return key


def instance_name() -> str:
    return os.environ.get("LAMBDA_INSTANCE_NAME", "reachtrace-gpu")


def ssh_key_name() -> str:
    return os.environ.get("LAMBDA_SSH_KEY_NAME", instance_name())


def ssh_key_path() -> Path:
    return Path(os.environ.get("LAMBDA_SSH_KEY_PATH", Path.home() / ".ssh" / "id_ed25519"))


def call(method: str, path: str, body: dict | None = None, timeout: int = 30) -> dict:
    auth = base64.b64encode(f"{_key()}:".encode()).decode()
    req = urllib.request.Request(
        API + path,
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        # The API sits behind Cloudflare, which rejects urllib's default User-Agent (1010).
        headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json",
                 "User-Agent": "reachtrace-lambda/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path}: HTTP {e.code} {e.read().decode()[:500]}") from e


def instance_types() -> dict[str, dict]:
    """{type name: {"price": USD/h, "gpus": n, "regions": [regions with capacity]}}"""
    out = {}
    for name, v in call("GET", "/instance-types")["data"].items():
        it = v["instance_type"]
        out[name] = {
            "price": it["price_cents_per_hour"] / 100,
            "gpus": it["specs"]["gpus"],
            "regions": [r["name"] for r in v["regions_with_capacity_available"]],
        }
    return out


def ensure_ssh_key() -> str:
    """Register the local public key under `ssh_key_name()` if no key has that name."""
    name = ssh_key_name()
    if name not in {k["name"] for k in call("GET", "/ssh-keys")["data"]}:
        pub = ssh_key_path().with_suffix(".pub").read_text().strip()
        call("POST", "/ssh-keys", {"name": name, "public_key": pub})
    return name


# ---------------------------------------------------------------------------- spend ledger


def _ledger_rows(ledger: Path = LEDGER) -> list[dict]:
    if not ledger.exists():
        return []
    return [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]


def _append(row: dict, ledger: Path = LEDGER) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a") as f:
        f.write(json.dumps(row) + "\n")


def spent(now: float | None = None, ledger: Path = LEDGER) -> float:
    """USD billed so far: each launch's price times its open minutes (rounded up)."""
    now = time.time() if now is None else now
    rows = _ledger_rows(ledger)
    ends = {r["id"]: r["t"] for r in rows if r["event"] == "terminate"}
    total = 0.0
    for r in rows:
        if r["event"] == "launch":
            minutes = math.ceil(max(0.0, ends.get(r["id"], now) - r["t"]) / 60)
            total += minutes / 60 * r["price"]
    return total + sum(r.get("usd", 0.0) for r in rows if r["event"] == "adjust")


def open_instances(ledger: Path = LEDGER) -> list[str]:
    rows = _ledger_rows(ledger)
    done = {r["id"] for r in rows if r["event"] == "terminate"}
    return [r["id"] for r in rows if r["event"] == "launch" and r["id"] not in done]


# ---------------------------------------------------------------------------- instances


@dataclass
class Instance:
    id: str
    ip: str | None
    status: str
    type: str
    region: str


def get(instance_id: str) -> Instance:
    d = call("GET", f"/instances/{instance_id}")["data"]
    return Instance(d["id"], d.get("ip"), d["status"], d["instance_type"]["name"],
                    d["region"]["name"])


def launch(
    type_name: str, region: str, budget_usd: float, hours: float, ledger: Path = LEDGER
) -> Instance:
    """Launch one instance unless `hours` at its price would push spend past the budget."""
    price = instance_types()[type_name]["price"]
    if spent(ledger=ledger) + price * hours > budget_usd:
        raise RuntimeError(
            f"budget guard: ${spent(ledger=ledger):.2f} spent + {hours} h x ${price}/h "
            f"> ${budget_usd}"
        )
    body = {
        "region_name": region,
        "instance_type_name": type_name,
        "ssh_key_names": [ensure_ssh_key()],
        "quantity": 1,
        "name": instance_name(),
    }
    iid = call("POST", "/instance-operations/launch", body)["data"]["instance_ids"][0]
    _append({"event": "launch", "id": iid, "type": type_name, "region": region,
             "price": price, "t": time.time()}, ledger)
    return get(iid)


def wait_active(instance_id: str, timeout_s: int = 1200) -> Instance:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        inst = get(instance_id)
        if inst.status == "active" and inst.ip:
            return inst
        if inst.status in ("terminated", "unhealthy"):
            raise RuntimeError(f"instance {instance_id} is {inst.status}")
        time.sleep(15)
    raise TimeoutError(f"instance {instance_id} not active after {timeout_s}s")


def terminate(instance_id: str, ledger: Path = LEDGER) -> None:
    call("POST", "/instance-operations/terminate", {"instance_ids": [instance_id]})
    _append({"event": "terminate", "id": instance_id, "t": time.time()}, ledger)


# ---------------------------------------------------------------------------- remote shell


def ssh_cmd(ip: str, command: str) -> list[str]:
    return [
        "ssh", "-i", str(ssh_key_path()), "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=20", f"ubuntu@{ip}", command,
    ]


def run(ip: str, command: str, check: bool = True, capture: bool = False):
    return subprocess.run(ssh_cmd(ip, command), check=check, text=True,
                          capture_output=capture)


def watchdog_script(instance_id: str, deadline_epoch: float) -> str:
    """Shell that terminates `instance_id` through the API at `deadline_epoch`.

    The key is written to a 0600 file in the home directory rather than put on the
    command line, so it does not show up in `ps` output.
    """
    return (
        "umask 077; cat > ~/.wd_key; "
        f"nohup bash -c 'while [ $(date +%s) -lt {int(deadline_epoch)} ]; do sleep 30; done; "
        f"curl -s -u $(cat ~/.wd_key): -X POST {API}/instance-operations/terminate "
        "-H \"Content-Type: application/json\" "
        f"-d \"{{\\\"instance_ids\\\": [\\\"{instance_id}\\\"]}}\"' "
        "> ~/.wd.log 2>&1 &"
    )


def install_watchdog(ip: str, instance_id: str, deadline_epoch: float) -> None:
    subprocess.run(ssh_cmd(ip, watchdog_script(instance_id, deadline_epoch)), input=_key(),
                   text=True, check=True)

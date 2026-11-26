#!/usr/bin/env python3
"""Run a repeated, order-rotated set of isolated pair validations.

With no --run this only performs each mode's read-only preflight. Every run is
an ordinary run_pair invocation with its own fresh result directory; this file
adds no GPU behaviour of its own. It exists because one run per mode in one
fixed order cannot separate a mode's speed from run order or run-to-run noise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time
import uuid

import cluster_config as cluster
import run_pair as pair


def schedule(modes, repeats):
    """(mode, repeat) pairs: each mode once per repeat, the first mode rotating.

    The first run is always the serial reference every later run is compared
    with. Rotation stops any one mode from always running first, when the page
    cache is coldest, or last, when the machines are warmest.
    """
    modes = list(modes)
    if not modes or modes[0] != "serial" or len(set(modes)) != len(modes) or set(modes) - set(pair.MODES):
        raise pair.Refusal("Modes must be distinct known modes starting with serial")
    if type(repeats) is not int or not 1 <= repeats <= 9:
        raise pair.Refusal("--repeats must be 1..9")
    order = []
    for repeat in range(repeats):
        shift = repeat % len(modes)
        order.extend((mode, repeat + 1) for mode in modes[shift:] + modes[:shift])
    return order


def wait_for_preflight(config, patience, sleep=time.sleep, clock=time.monotonic):
    """Preflight, waiting only for the previous run's port to leave TIME_WAIT."""
    deadline = clock() + patience
    while True:
        try:
            return pair.preflight_pair(config)
        except (OSError, pair.Refusal) as exc:
            if "Address already in use" not in str(exc) or clock() >= deadline:
                raise
        sleep(5)


def job(mode, requests):
    paths = pair.CLUSTER["paths"]
    return {"mode": mode, "kind": "lanes", "binary": paths["binary"], "requests": requests,
            "pack": paths["draft_pack"], "run_id": uuid.uuid4().hex,
            "config_identity": cluster.identity(pair.CLUSTER)}


def run_suite(modes, repeats, requests, output, timeout, patience):
    order = schedule(modes, repeats)
    for mode in modes:  # refuse before any GPU load if a later mode cannot start
        wait_for_preflight(job(mode, requests), patience)
    root = pair.new_output(output)
    manifest = {"schema": "dfly2-suite-v1", "modes": list(modes), "repeats": repeats,
                "launcher_sha256": pair.launcher_identity(), "requests_sha256": None,
                "complete": False, "runs": []}
    reference = None
    try:
        for index, (mode, repeat) in enumerate(order, 1):
            config = job(mode, requests)
            wait_for_preflight(config, patience)
            name = f"{index:02d}-{mode}-r{repeat}"
            result = pair.run_pair(config, root / name, timeout, reference)
            if manifest["requests_sha256"] is None:
                manifest["requests_sha256"] = result["preflight"]["nodes"][0]["requests_sha256"]
            manifest["runs"].append({"index": index, "mode": mode, "repeat": repeat, "directory": name,
                                     "run_id": result["run_id"], "success": result["success"]})
            if not result["success"]:
                manifest["error"] = f"{name}: {result.get('error') or 'run did not succeed'}"
                break
            if reference is None:
                reference = result
            print(f"{name}: ok", flush=True)
        else:
            manifest["complete"] = True
    finally:
        (root / "suite.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="private cluster JSON, identical on both nodes")
    parser.add_argument("--run", action="store_true", help="run the suite after every mode's preflight passes")
    parser.add_argument("--modes", nargs="+", default=list(pair.MODES), help="serial first; default all modes")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--requests", help="requests file inside experiment_root; default from the configuration")
    parser.add_argument("--output", type=Path, help="new suite directory inside experiment_root")
    parser.add_argument("--timeout", type=int, help="deadline for each pair run in seconds, 30..7200")
    parser.add_argument("--port-wait", type=int, default=240, help="seconds to wait for the previous run's port")
    args = parser.parse_args(argv)
    try:
        pair.configure(cluster.load(args.config))
        paths = pair.CLUSTER["paths"]
        requests = args.requests or paths["requests"]
        timeout = args.timeout if args.timeout is not None else pair.CLUSTER["execution"]["timeout_seconds"]
        if not 30 <= timeout <= 7200 or not 0 <= args.port_wait <= 900:
            raise pair.Refusal("--timeout must be 30..7200 and --port-wait 0..900 seconds")
        order = schedule(args.modes, args.repeats)
        if not args.run:
            checks = {mode: pair.preflight_pair(job(mode, requests)) for mode in args.modes}
            print(json.dumps({"ok": True, "planned_runs": [f"{m}-r{r}" for m, r in order], "preflight": checks}, indent=2))
            return 0
        output = args.output or Path(paths["results"]) / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-suite")

        def interrupted(signum, frame):
            raise InterruptedError(f"signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        manifest = run_suite(args.modes, args.repeats, requests, output, timeout, args.port_wait)
        print(json.dumps({"complete": manifest["complete"], "suite": str(output / "suite.json"),
                          "error": manifest.get("error")}, indent=2))
        return 0 if manifest["complete"] else 1
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

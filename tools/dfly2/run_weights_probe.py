#!/usr/bin/env python3
"""Preflight or explicitly run a bounded draft-weight upload check on one node."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time
import uuid

import cluster_config as cluster
import run_pair as pair


def verified(logs, rank):
    records = []
    for line in (logs["stdout"] + "\n" + logs["stderr"]).splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            records.append(row)
    return any(r.get("rank") == rank and r.get("ok") is True and r.get("all_uploaded_bytes_equal") is True
               and r.get("allocation_released") is True for r in records), records


def run(job, output, timeout):
    with pair.exclusive_lock():
        preflight = pair.node_action("preflight", job)
        if job["rank"] == 0:
            pair.stopped_dashboard()
        output = pair.new_output(output)
        report = {"run_id": job["run_id"], "rank": job["rank"], "kind": "weights", "success": False, "preflight": preflight}
        try:
            pair.node_action("create", job)
            pair.node_action("start", job)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                state = pair.node_action("state", job)
                report["state"] = state
                if state["Status"] in {"dead", "exited"}:
                    report["success"] = state["Status"] == "exited" and state["ExitCode"] == 0 and not state.get("OOMKilled")
                    break
                time.sleep(min(2, max(0, deadline - time.monotonic())))
            else:
                raise pair.Refusal("Weight probe deadline expired")
        except (Exception, KeyboardInterrupt) as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                logs = pair.node_action("logs", job)
                (output / "stdout.log").write_text(logs["stdout"], encoding="utf-8")
                (output / "stderr.log").write_text(logs["stderr"], encoding="utf-8")
                ok, report["records"] = verified(logs, job["rank"])
                report["success"] = report["success"] and ok
            except Exception as exc:
                report["log_error"] = str(exc)
                report["success"] = False
            try:
                report["cleanup"] = pair.node_action("cleanup", job)
            except Exception as exc:
                report["cleanup"] = {"error": str(exc)}
                report["success"] = False
            (output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True, help="run locally on this configured node")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--preflight", action="store_true", help="read-only checks; default")
    action.add_argument("--run", action="store_true", help="explicitly allocate draft weights, verify and release")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if not 30 <= args.timeout <= 1800:
            raise pair.Refusal("--timeout must be 30..1800 seconds")
        pair.configure(cluster.load(args.config))
        paths = pair.CLUSTER["paths"]
        job = {"rank": args.rank, "mode": "dfly2", "kind": "weights", "run_id": uuid.uuid4().hex,
               "config_identity": cluster.identity(pair.CLUSTER), "binary": paths["weights_probe"], "pack": paths["draft_pack"]}
        if not args.run:
            if args.rank == 0:
                pair.stopped_dashboard()
            print(json.dumps({"ok": True, "preflight": pair.node_action("preflight", job)}, indent=2))
            return 0
        output = args.output or Path(paths["results"]) / ("weights-r" + str(args.rank) + "-" + job["run_id"][:12])
        def interrupted(signum, frame):
            raise InterruptedError(f"signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        report = run(job, output, args.timeout)
        print(json.dumps({"success": report["success"], "result": str(output / "result.json")}, indent=2))
        return 0 if report["success"] else 1
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

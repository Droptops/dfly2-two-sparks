#!/usr/bin/env python3
"""Export a completed run_suite.py directory to a new public evidence folder.

Only the allowlisted fields in sanitize_evidence.py are copied. The request
file must be one of this repository's reviewed fixtures, and the export must
run from the same checkout that launched the suite, so the commit recorded
beside the receipts is the commit whose launcher bytes actually ran.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from sanitize_evidence import dump, sanitize_portable_run, validate_requests
from verify_evidence import require
from verify_portable_evidence import LAUNCHER_FILES, current_launcher_digest
from verify_suite_evidence import ROOT, reviewed_fixture_hash, verify


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True, help="private run_suite output containing suite.json")
    parser.add_argument("--requests", type=Path, required=True, help="the request file the suite ran")
    parser.add_argument("--pool-tokens", type=int, required=True, help="actual execution.pool from the reviewed configuration")
    parser.add_argument("--commit", required=True, help="40-hex Git commit of the checkout that ran the suite")
    parser.add_argument("--output", type=Path, required=True, help="new folder directly under the public evidence directory")
    args = parser.parse_args(argv)
    # Refuse unreviewed prompt text before creating any public output.
    fixture = args.requests.read_bytes()
    document = json.loads(fixture)
    validate_requests(document)
    request_hash = hashlib.sha256(fixture).hexdigest()
    require(request_hash == reviewed_fixture_hash(document["metadata"]["suite"], ROOT),
            "suite requests differ from the reviewed fixture")
    private = json.loads((args.suite_dir / "suite.json").read_text(encoding="utf-8"))
    require(private.get("schema") == "dfly2-suite-v1" and private.get("complete") is True
            and all(entry["success"] is True for entry in private["runs"]), "suite did not complete successfully")
    require(private["requests_sha256"] == request_hash, "suite ran a different request file")
    require(re.fullmatch(r"[0-9a-f]{40}", args.commit) is not None, "--commit must be a full 40-hex Git commit")
    require(private["launcher_sha256"] == current_launcher_digest(ROOT),
            "export from the checkout that ran the suite: this launcher differs from the one recorded")
    exported, entries = [], []
    for entry in private["runs"]:
        raw = json.loads((args.suite_dir / entry["directory"] / "result.json").read_text(encoding="utf-8"))
        require(raw["mode"] == entry["mode"] and raw["run_id"] == entry["run_id"], "suite manifest disagrees with a result")
        clean = sanitize_portable_run(raw, pool_tokens=args.pool_tokens)
        clean["suite"] = document["metadata"]["suite"]  # the reviewed fixture's own label
        require(all(node["launcher_sha256"] == private["launcher_sha256"] for node in clean["nodes"]),
                "a run used a different launcher than the suite recorded")
        name = f"run-{entry['index']:02d}-{entry['mode']}-r{entry['repeat']}.json"
        exported.append((name, clean))
        entries.append({"order": entry["index"], "file": name, "mode": entry["mode"], "repeat": entry["repeat"]})
    args.output.mkdir(parents=False, exist_ok=False)
    (args.output / "requests.json").write_bytes(fixture)
    for name, clean in exported:
        dump(args.output / name, clean)
    dump(args.output / "provenance.json", {"schema": "dfly2-evidence-provenance-v1",
                                           "launcher_sha256": private["launcher_sha256"],
                                           "launcher_files": list(LAUNCHER_FILES), "repository_commit": args.commit})
    dump(args.output / "suite.json", {"schema": "dfly2-public-suite-v1", "suite": document["metadata"]["suite"],
                                      "modes": private["modes"], "repeats": private["repeats"], "runs": entries})
    # Failed checks keep the staged receipts for diagnosis; no comparison is
    # written until every cross-run consistency check passes.
    dump(args.output / "comparison.json", verify(args.output))
    print("Exported and verified suite evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

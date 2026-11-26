#!/usr/bin/env python3
"""Export reviewed portable run_pair receipts to a new public evidence folder.

The request fixture must have been reviewed for publication. Infrastructure
fields are excluded by the explicit allowlist in sanitize_evidence.py.
"""
import argparse
import hashlib
import json
from pathlib import Path

from sanitize_evidence import dump, sanitize_portable_run, validate_requests
from verify_portable_evidence import MODES, require, sha256, verify


def validate_fixture(path, historical):
    fixture = Path(path).read_bytes()
    validate_requests(json.loads(fixture))
    require(hashlib.sha256(fixture).hexdigest() == sha256(Path(historical) / "chat-requests.json"),
            "portable request fixture differs from reviewed historical fixture")
    return fixture


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for mode in MODES:
        parser.add_argument("--" + mode, type=Path, required=True, help="private run_pair result.json")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--pool-tokens", type=int, required=True, help="actual execution.pool from reviewed configuration")
    parser.add_argument("--output", type=Path, required=True, help="new folder under the existing public evidence directory")
    args = parser.parse_args(argv)
    # Reject unknown prompt contents before creating any public output. Schema
    # validation alone does not determine whether text is safe to publish.
    fixture = validate_fixture(args.requests, args.output.parent)
    runs = {mode: sanitize_portable_run(json.loads(getattr(args, mode).read_text(encoding="utf-8")), pool_tokens=args.pool_tokens)
            for mode in MODES}
    if any(run["mode"] != mode for mode, run in runs.items()):
        raise ValueError("run mode does not match its input option")
    args.output.mkdir(parents=False, exist_ok=False)
    (args.output / "chat-requests.json").write_bytes(fixture)
    for mode, run in runs.items():
        dump(args.output / f"chat-{mode}.json", run)
    # Failed checks retain the staged numerical receipts for diagnosis. No
    # comparison is written until every cross-run consistency check passes.
    dump(args.output / "comparison.json", verify(args.output))
    print("Exported and verified portable GPU evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

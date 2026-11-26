#!/usr/bin/env python3
"""Compare complete token arrays, not short hashes or a model's success flag."""
import argparse
import json
from pathlib import Path


def replies(path):
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "reply" in record:
            if record["name"] in out:
                raise ValueError(f"Duplicate result: {record['name']}")
            out[record["name"]] = record
    if not out:
        raise ValueError(f"No replies in {path}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base, candidate = replies(args.serial), replies(args.candidate)
    if base.keys() != candidate.keys():
        raise ValueError("Request coverage differs")
    rows = []
    for name, reference in base.items():
        got = candidate[name]
        rows.append({"name": name, "equal": got["reply"] == reference["reply"] and got["reason"] == reference["reason"],
                     "tokens": len(got["reply"]), "drafted": got["drafted"], "accepted": got["accepted"],
                     "serial_seconds": reference["seconds"], "candidate_seconds": got["seconds"]})
    report = {"all_equal": all(r["equal"] for r in rows), "requests": rows,
              "timing_note": "Includes each request's prefill; this is not decode-only tok/s."}
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")
    raise SystemExit(0 if report["all_equal"] else 1)


if __name__ == "__main__":
    main()

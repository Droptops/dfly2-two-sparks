#!/usr/bin/env python3
"""Verify the pinned public AOT kit after CPU generation of its missing assets."""
import argparse
import hashlib
import json
from pathlib import Path

MANIFEST_SHA256 = "8c6dcb712a7557a8aeb794d68daf85b07befae7bb1bdfa470223e957718f2845"
BIAS = "vision/gate_bias_vl.safetensors"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(kit, allow_1017_bias=False):
    kit = kit.resolve()
    manifest = kit / "MANIFEST"
    if sha(manifest) != MANIFEST_SHA256:
        raise ValueError("kit/MANIFEST differs from the locked public kit revision")
    report = {"manifest_sha256": MANIFEST_SHA256, "checked_files": 0, "generated_bias": None}
    for line in manifest.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = (kit / relative).resolve()
        if not path.is_relative_to(kit):
            raise ValueError(f"kit path escapes directory: {relative}")
        actual = sha(path)
        if actual != expected:
            if relative != BIAS or not allow_1017_bias:
                raise ValueError(f"kit SHA256 mismatch: {relative}")
            # This explicit exception is the upstream recipe's 1017 bias rule.
            # It does not independently prove which checkpoint generated it.
            report["generated_bias"] = {"sha256": actual, "original_sha256": expected}
        report["checked_files"] += 1
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kit", type=Path)
    p.add_argument("--allow-1017-bias", action="store_true",
                   help="allow ONLY the 1017-specific vision routing bias to differ; record its hash")
    p.add_argument("--receipt", type=Path)
    a = p.parse_args()
    result = json.dumps(verify(a.kit, a.allow_1017_bias), indent=2) + "\n"
    if a.receipt:
        with a.receipt.open("x", encoding="utf-8") as out:
            out.write(result)
    print(result, end="")


if __name__ == "__main__":
    main()

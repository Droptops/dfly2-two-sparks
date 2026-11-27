#!/usr/bin/env python3
"""Prepare separately pinned experimental sources over the historical baseline.

Linux/WSL only. Fetches public Git sources; no Docker, models, SSH or GPU use.
Unfinalized locks fail before preparation. Failed destinations are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys

import prepare_source as baseline

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "dfly2-experimental-native-lock-v1"
RECEIPT_SCHEMA = "dfly2-experimental-source-v1"
BUILD_INPUTS = ("build/Dockerfile", "build/tool-requirements.txt", "build/verify_tools.py",
                "scripts/prepare_source.py", "scripts/prepare_experimental.py", "scripts/build_experimental.py")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_hash(value):
    # Same normalization as the historical source receipt.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def patch_file(root, record):
    require(set(record) == {"path", "sha256"}, "unknown experimental patch fields")
    name = record["path"]
    require(isinstance(name, str) and "\\" not in name, "invalid experimental patch path")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and relative.parts and
            all(p not in ("..", ".git") for p in relative.parts), "experimental patch path escapes repository")
    require(isinstance(record["sha256"], str) and HEX64.fullmatch(record["sha256"]),
            "experimental patch SHA256 is not finalized")
    path = baseline.checked_path(root, name)
    require(not (root/name).is_symlink(), "experimental patch must not be a symlink")
    return baseline.verify_file(root, record)


def locks(lock_path, root=ROOT):
    root = Path(root).resolve()
    exp = load(lock_path)
    require(set(exp) == {"schema", "status", "baseline_lock_sha256", "patches", "prepared_trees"},
            "unknown experimental lock fields")
    require(exp["schema"] == SCHEMA and exp["status"] == "finalized",
            "experimental lock is not finalized; no source preparation or build is allowed")
    base = load(root/"upstream.lock.json")
    require(exp["baseline_lock_sha256"] == canonical_hash(base), "historical baseline lock mismatch")
    require(set(exp["prepared_trees"]) == {"tensorfold", "inputs"}, "unexpected experimental trees")
    require(all(isinstance(v, str) and HEX40.fullmatch(v) for v in exp["prepared_trees"].values()),
            "experimental prepared trees are not finalized")
    require(exp["prepared_trees"]["inputs"] == base["inputs"]["prepared_tree"],
            "experimental workflow must preserve the historical inputs tree")
    require(exp["prepared_trees"]["tensorfold"] != base["tensorfold"]["prepared_tree"],
            "experimental native tree must differ from the historical tree")
    require(isinstance(exp["patches"], list) and exp["patches"], "experimental patch series is empty")
    paths = [str(patch_file(root, p)) for p in exp["patches"]]
    require(len(set(paths)) == len(paths), "duplicate experimental patch")
    for record in base["integration_patches"]:
        baseline.verify_file(root, record)
    return exp, base


def source_manifest(dest):
    for name in ("tensorfold", "inputs"):
        for path in (dest/name).rglob("*"):
            if ".git" not in path.relative_to(dest).parts:
                require(not path.is_symlink(), "unexpected source symlink: " + str(path))
    return baseline.source_manifest(dest)


def build_inputs(root):
    return {name: baseline.digest(root/name) for name in BUILD_INPUTS}


def baseline_receipt(base):
    return {"schema": 1, "lock_sha256": canonical_hash(base),
            "trees": {name: base[name]["prepared_tree"] for name in ("tensorfold", "inputs")},
            "native_patch_sha256": base["integration_patches"][0]["sha256"]}


def receipt(exp, base, manifest, root, baseline_manifest_sha256):
    return {"schema": RECEIPT_SCHEMA, "source_kind": "experimental",
            "historical_validation_applies": False,
            "experimental_lock_sha256": canonical_hash(exp), "experimental_lock": exp,
            "baseline_receipt": baseline_receipt(base), "prepared_trees": exp["prepared_trees"],
            "baseline_source_files_sha256": baseline_manifest_sha256,
            "source_files_sha256": hashlib.sha256(manifest).hexdigest(),
            "build_inputs_sha256": build_inputs(root)}


def verify_trees(dest, exp, base):
    for name in ("recipe", "tensorfold", "inputs"):
        require(baseline.run("git", "rev-parse", "HEAD", cwd=dest/name, capture=True) == base[name]["revision"],
                "changed upstream HEAD: " + name)
    for name in ("tensorfold", "inputs"):
        require(baseline.tree(dest/name) == exp["prepared_trees"][name], "changed experimental source tree: " + name)
    for record in base["recipe"]["serving_patches"] + base["recipe"]["inputs_patches"]:
        baseline.verify_file(dest/"recipe", record)


def check(dest, lock_path, root=ROOT):
    dest, root = Path(dest).resolve(), Path(root).resolve()
    exp, base = locks(lock_path, root)
    current = load(dest/"source-receipt.json")
    require(current.get("schema") == RECEIPT_SCHEMA and current.get("source_kind") == "experimental",
            "source is not an experimental prepared context")
    require(load(dest/"upstream.lock.json") == base, "prepared baseline lock changed")
    require(load(dest/"experimental.lock.json") == exp, "prepared experimental lock changed")
    require(load(dest/"baseline-source-receipt.json") == baseline_receipt(base), "baseline preparation provenance changed")
    verify_trees(dest, exp, base)
    manifest = source_manifest(dest)
    require((dest/"source-files.sha256").read_bytes() == manifest, "experimental source-file manifest mismatch")
    require(current == receipt(exp, base, manifest, root, baseline.digest(dest/"baseline-source-files.sha256")),
            "experimental source/build provenance mismatch")
    return current


def apply_experiment(dest, lock_path, root=ROOT):
    """Internal second stage; only a verified historical context may enter it."""
    dest, root = Path(dest).resolve(), Path(root).resolve()
    exp, base = locks(lock_path, root)
    original = baseline.check(dest, base)
    require(original == baseline_receipt(base), "unexpected historical source receipt")
    require(not (dest/"baseline-source-receipt.json").exists(), "experimental context already initialized")
    # Remove the historical attestation before the first source mutation, even
    # if patch application fails. The baseline receipt is retained as provenance.
    (dest/"source-receipt.json").rename(dest/"baseline-source-receipt.json")
    (dest/"source-files.sha256").rename(dest/"baseline-source-files.sha256")
    baseline.apply_series(dest/"tensorfold", root, exp["patches"])
    verify_trees(dest, exp, base)
    manifest = source_manifest(dest)
    (dest/"source-files.sha256").write_bytes(manifest)
    write_json(dest/"experimental.lock.json", exp)
    write_json(dest/"source-receipt.json", receipt(exp, base, manifest, root,
                                                baseline.digest(dest/"baseline-source-files.sha256")))
    return check(dest, lock_path, root)


def prepare(dest, lock_path, root=ROOT):
    dest, root = Path(dest).resolve(), Path(root).resolve()
    locks(lock_path, root)  # Fail closed before network/filesystem preparation.
    require(not dest.exists(), "destination exists; select a fresh experimental directory")
    baseline.run(sys.executable, str(root/"scripts/prepare_source.py"), "--destination", str(dest))
    return apply_experiment(dest, lock_path, root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, default=ROOT/"_build-experimental")
    p.add_argument("--lock", type=Path, default=ROOT/"experimental/native.lock.json")
    p.add_argument("--check", action="store_true", help="check prepared experimental sources without downloads")
    args = p.parse_args()
    if os.name == "nt":
        p.error("run on Linux/WSL: upstream filenames cannot be represented on Windows")
    result = (check if args.check else prepare)(args.destination, args.lock)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

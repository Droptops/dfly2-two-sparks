#!/usr/bin/env python3
"""Fetch exact public Git revisions, verify patch bytes, and prepare build inputs.

This performs no model downloads, Docker actions, SSH, or GPU operations. A new
destination is required; incomplete destinations are retained for inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, cwd: Path | None = None, capture: bool = False) -> str:
    result = subprocess.run(args, cwd=cwd, check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else ""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes root: {relative}")
    return path


def verify_file(root: Path, record: dict) -> Path:
    path = checked_path(root, record["path"])
    if digest(path) != record["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path}")
    return path


def clone(record: dict, dest: Path) -> None:
    run("git", "init", "-q", str(dest))
    run("git", "config", "core.autocrlf", "false", cwd=dest)
    run("git", "config", "core.filemode", "true", cwd=dest)
    run("git", "remote", "add", "origin", record["url"], cwd=dest)
    run("git", "fetch", "--depth", "1", "origin", record["revision"], cwd=dest)
    run("git", "checkout", "-q", "--detach", "FETCH_HEAD", cwd=dest)
    if run("git", "rev-parse", "HEAD", cwd=dest, capture=True) != record["revision"]:
        raise ValueError(f"Git revision mismatch: {dest}")


def apply_series(source: Path, root: Path, records: list[dict]) -> None:
    for record in records:
        patch = verify_file(root, record)
        print(f"Applying {record['path']}", flush=True)
        run("git", "apply", "--check", str(patch), cwd=source)
        run("git", "apply", str(patch), cwd=source)


def tree(source: Path) -> str:
    # Only these fresh generated checkouts are staged. No upstream commit is made.
    run("git", "add", "-A", cwd=source)
    return run("git", "write-tree", cwd=source, capture=True)


def source_manifest(dest: Path) -> bytes:
    lines = []
    for name in ("tensorfold", "inputs"):
        for path in sorted((dest / name).rglob("*")):
            relative = path.relative_to(dest)
            if ".git" in relative.parts or not path.is_file():
                continue
            if path.is_symlink():
                raise ValueError(f"unexpected source symlink: {relative}")
            lines.append(f"{digest(path)}  {relative.as_posix()}\n")
    return "".join(lines).encode()


def check(dest: Path, lock: dict) -> dict:
    receipt = json.loads((dest / "source-receipt.json").read_text())
    if receipt["lock_sha256"] != hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest():
        raise ValueError("lock changed since source preparation")
    for name, key in (("recipe", "recipe"), ("tensorfold", "tensorfold"), ("inputs", "inputs")):
        if run("git", "rev-parse", "HEAD", cwd=dest / name, capture=True) != lock[key]["revision"]:
            raise ValueError(f"changed upstream HEAD: {name}")
    for name in ("tensorfold", "inputs"):
        current = tree(dest / name)
        if current != receipt["trees"][name] or current != lock[name]["prepared_tree"]:
            raise ValueError(f"changed prepared source tree: {name}")
    for record in lock["recipe"]["serving_patches"] + lock["recipe"]["inputs_patches"]:
        verify_file(dest / "recipe", record)
    for record in lock["integration_patches"]:
        verify_file(ROOT, record)
    if (dest / "source-files.sha256").read_bytes() != source_manifest(dest):
        raise ValueError("source-file manifest mismatch")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "_build")
    parser.add_argument("--check", action="store_true", help="verify an already prepared directory; no downloads")
    args = parser.parse_args()
    if os.name == "nt":
        parser.error("run on Linux/WSL: upstream golden fixtures use filenames Windows cannot represent")
    dest = args.destination.resolve()
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    if args.check:
        print(json.dumps(check(dest, lock), indent=2))
        return
    for record in lock["integration_patches"]:
        verify_file(ROOT, record)
    if dest.exists():
        raise ValueError(f"destination exists; use --check or choose a fresh path: {dest}")
    dest.mkdir(parents=True)
    clone(lock["recipe"], dest / "recipe")
    for name in ("tensorfold", "inputs"):
        clone(lock[name], dest / name)
    actual_serving = sorted(p.relative_to(dest / "recipe").as_posix()
                            for p in (dest / "recipe/zig/patches").glob("*.patch"))
    actual_inputs = sorted(p.relative_to(dest / "recipe").as_posix()
                           for p in (dest / "recipe/zig/patches-inputs").glob("*.patch"))
    if actual_serving != [p["path"] for p in lock["recipe"]["serving_patches"]]:
        raise ValueError("recipe serving patch inventory differs from lock")
    if actual_inputs != [p["path"] for p in lock["recipe"]["inputs_patches"]]:
        raise ValueError("recipe input patch inventory differs from lock")
    apply_series(dest / "tensorfold", dest / "recipe", lock["recipe"]["serving_patches"])
    apply_series(dest / "inputs", dest / "recipe", lock["recipe"]["inputs_patches"])
    apply_series(dest / "tensorfold", ROOT, lock["integration_patches"])
    trees = {name: tree(dest / name) for name in ("tensorfold", "inputs")}
    for name, value in trees.items():
        if value != lock[name]["prepared_tree"]:
            raise ValueError(f"prepared tree mismatch for {name}: {value}")
    receipt = {"schema": 1, "lock_sha256": hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest(),
               "trees": trees, "native_patch_sha256": lock["integration_patches"][0]["sha256"]}
    (dest / "source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (dest / "upstream.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    (dest / "source-files.sha256").write_bytes(source_manifest(dest))
    (dest / ".dockerignore").write_text("**/.git\nrecipe\n**/.zig-cache\n**/zig-out\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

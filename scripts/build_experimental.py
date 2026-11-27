#!/usr/bin/env python3
"""Build an explicitly experimental ARM64 image from separately verified sources.

Uses the historical Dockerfile and toolchain without rewriting their pins.
No GPU or model use. This is not a historical source/build attestation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import uuid

import prepare_experimental as source

ROOT = Path(__file__).resolve().parents[1]


def command(dest, tag, jobs, exp, base, root=ROOT, iidfile=None):
    source.require(type(jobs) is int and 1 <= jobs <= 64, "build jobs must be 1..64")
    source.require(isinstance(tag, str) and tag and not tag.startswith("-"), "invalid image tag")
    cmd = ["docker", "build", "--platform", "linux/arm64", "--tag", tag,
           "--file", str(root/"build/Dockerfile"),
           "--label", "ai.dfly2.source-kind=experimental",
           "--label", "ai.dfly2.experimental-lock-sha256="+source.canonical_hash(exp),
           "--label", "ai.dfly2.native-tree="+exp["prepared_trees"]["tensorfold"],
           "--build-arg", "BASE_IMAGE="+base["toolchain"]["base_image"],
           "--build-arg", "ZIG_VERSION="+base["toolchain"]["zig_version"],
           "--build-arg", "ZIG_SHA256="+base["toolchain"]["zig_sha256"],
           "--build-arg", "BUILD_JOBS="+str(jobs)]
    if iidfile is not None:
        cmd += ["--iidfile", str(iidfile)]
    return cmd + [str(dest)]


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build(dest, lock_path, tag, jobs, receipt_path=None, print_command=False, root=ROOT):
    dest, root = Path(dest).resolve(), Path(root).resolve()
    verified = source.check(dest, lock_path, root)
    exp, base = source.locks(lock_path, root)
    for name in ("tool-requirements.txt", "verify_tools.py"):
        shutil.copyfile(root/"build"/name, dest/name)
    if print_command:
        return command(dest, tag, jobs, exp, base, root)
    receipt_path = Path(receipt_path).resolve() if receipt_path else dest/"experimental-build-receipt.json"
    source.require(not receipt_path.exists(), "build receipt already exists; select a fresh --receipt")
    source.require(not any(receipt_path.is_relative_to(dest/name) for name in ("tensorfold", "inputs", "recipe")),
                   "build receipt must be outside prepared source trees")
    iidfile = dest/(".experimental-image-"+uuid.uuid4().hex+".id")
    cmd = command(dest, tag, jobs, exp, base, root, iidfile)
    report = {"schema": "dfly2-experimental-build-v1", "source_kind": "experimental",
              "historical_validation_applies": False, "source_receipt": verified,
              "started_utc": now(), "finished_utc": None, "status": "running", "success": False,
              "command": cmd, "build_exit_code": None, "image": None}
    with receipt_path.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2); out.write("\n")
    try:
        result = subprocess.run(cmd, check=False)
        report["build_exit_code"] = result.returncode
        source.require(result.returncode == 0, "experimental Docker build failed; receipt retained")
        image_id = iidfile.read_text().strip()
        source.require(re.fullmatch(r"sha256:[0-9a-f]{64}", image_id), "Docker returned invalid image identity")
        inspected = subprocess.run(["docker", "image", "inspect", image_id], check=True, text=True,
                                   stdout=subprocess.PIPE)
        images = json.loads(inspected.stdout)
        source.require(len(images) == 1 and images[0]["Id"] == image_id, "built image identity mismatch")
        image = images[0]
        labels = image.get("Config", {}).get("Labels", {})
        source.require(image["Architecture"] == "arm64" and image["Os"] == "linux" and
                       labels.get("ai.dfly2.source-kind") == "experimental" and
                       labels.get("ai.dfly2.experimental-lock-sha256") == source.canonical_hash(exp) and
                       labels.get("ai.dfly2.native-tree") == exp["prepared_trees"]["tensorfold"],
                       "experimental image labels/platform mismatch")
        source.check(dest, lock_path, root)  # Detect source changes during the build.
        report["image"] = {"id": image_id, "architecture": image["Architecture"], "os": image["Os"],
                           "experimental_lock_sha256": source.canonical_hash(exp),
                           "native_tree": exp["prepared_trees"]["tensorfold"]}
        report.update(status="complete", success=True)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        report["finished_utc"] = now()
        source.write_json(receipt_path, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, default=ROOT/"_build-experimental")
    p.add_argument("--lock", type=Path, default=ROOT/"experimental/native.lock.json")
    p.add_argument("--tag", default="dfly2-two-sparks:parity-lab")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--receipt", type=Path)
    p.add_argument("--print-command", action="store_true")
    a = p.parse_args()
    result = build(a.source_dir, a.lock, a.tag, a.jobs, a.receipt, a.print_command)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

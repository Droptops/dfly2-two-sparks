#!/usr/bin/env python3
"""Build the locked ARM64 public image from verified prepared sources. No GPU use."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from prepare_source import ROOT, check


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, default=ROOT / "_build")
    p.add_argument("--tag", default="dfly2-two-sparks:experimental")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--print-command", action="store_true")
    a = p.parse_args()
    if not 1 <= a.jobs <= 64:
        p.error("--jobs must be 1..64")
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    check(a.source_dir.resolve(), lock)
    # Separate late Docker COPY keeps tool-only fixes from rebuilding the native
    # compiler layers. These files do not alter either prepared source tree.
    for name in ("tool-requirements.txt", "verify_tools.py"):
        shutil.copyfile(ROOT / "build" / name, a.source_dir / name)
    cmd = ["docker", "build", "--platform", "linux/arm64", "--tag", a.tag,
           "--file", str(ROOT / "build/Dockerfile"),
           "--build-arg", "BASE_IMAGE=" + lock["toolchain"]["base_image"],
           "--build-arg", "ZIG_VERSION=" + lock["toolchain"]["zig_version"],
           "--build-arg", "ZIG_SHA256=" + lock["toolchain"]["zig_sha256"],
           "--build-arg", "BUILD_JOBS=" + str(a.jobs),
           str(a.source_dir.resolve())]
    if a.print_command:
        print(json.dumps(cmd, indent=2))
    else:
        subprocess.run(cmd, check=True)
        subprocess.run(["docker", "image", "inspect", a.tag, "--format", "{{.Id}}"], check=True)


if __name__ == "__main__":
    main()

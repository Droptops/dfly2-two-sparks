#!/usr/bin/env python3
"""Run CPU-only release checks; no network, model load, Docker, or remote actions."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import unittest

from verify_experimental_build import verify as check_experimental_build


EXPERIMENTAL_BUNDLES = {
    "profile-v4-long": ("matrix", "profile"),
    "sweep-v4-chat": ("matrix", "benchmark"),
    "sweep-v5-chat": ("matrix", "benchmark"),
    "benchmark-chat": ("matrix", "benchmark"),
    "benchmark-long": ("matrix", "benchmark"),
    "heldout": ("matrix", "benchmark"),
    "reference-v3-original": ("reference", None),
    "reference-v4-rounding": ("reference", None),
}


def check_experimental_evidence(root):
    """Verify every staged experimental bundle; absence before staging is valid.

    Explicit bundle names prevent malformed or unexpected directories from
    silently disappearing from the release gate. Historical checks stay separate.
    """
    folder = root / "evidence/experimental"
    if not folder.exists():
        return 0
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("experimental evidence root must be a real directory")
    checked = 0
    for bundle in sorted(folder.iterdir()):
        if bundle.name == "README.md" and bundle.is_file() and not bundle.is_symlink():
            continue
        if bundle.is_symlink() or not bundle.is_dir() or bundle.name not in EXPERIMENTAL_BUNDLES:
            raise ValueError(f"unexpected experimental evidence entry: {bundle.name}")
        kind, purpose = EXPERIMENTAL_BUNDLES[bundle.name]
        marker = bundle / ("matrix.json" if kind == "matrix" else "manifest.json")
        if marker.is_symlink() or not marker.is_file():
            raise ValueError(f"missing experimental evidence manifest: {bundle.name}")
        record = json.loads(marker.read_text(encoding="utf-8"))
        if kind == "matrix":
            if record.get("schema") != "dfly2-lab-matrix-public-v1" or record.get("purpose") != purpose:
                raise ValueError(f"unexpected matrix schema/purpose: {bundle.name}")
            cmd = [sys.executable, str(root / "tools/dfly2/lab/analyze_matrix.py"),
                   "--verify-export", str(bundle)]
        else:
            if record.get("schema") != "dfly2-reference-public-manifest-v1":
                raise ValueError(f"unexpected reference schema: {bundle.name}")
            cmd = [sys.executable, str(root / "tools/dfly2/reference/export_reference.py"),
                   "--verify", str(bundle)]
        subprocess.run(cmd, cwd=root, check=True)
        checked += 1
    return checked


def check_experimental_selection(root):
    """A published candidate freeze requires all final evaluation evidence."""
    receipt=root/"evidence/experimental-selection-v5.json"
    if not receipt.exists(): return False
    if receipt.is_symlink() or not receipt.is_file():
        raise ValueError("experimental selection receipt must be a real file")
    subprocess.run([sys.executable,str(root/"tools/dfly2/lab/selection.py"),
                    "--verify",str(receipt),"--require-complete"],cwd=root,check=True)
    return True


def check_launcher_pins(root):
    """Where Git history is present, confirm each pinned commit holds the recorded launcher bytes.

    Receipts name the commit whose launcher ran. A shallow or exported checkout
    cannot read that commit; that is reported, not treated as a failure.
    """
    checked = skipped = 0
    for record_path in sorted((root / "evidence").glob("*/provenance.json")):
        record = json.loads(record_path.read_text())
        try:
            shown = [subprocess.run(["git", "show", f"{record['repository_commit']}:{name}"], cwd=root, capture_output=True)
                     for name in record["launcher_files"]]
        except OSError:  # no Git on this host, as in an exported archive
            shown = None
        if shown is None or any(p.returncode for p in shown):
            skipped += 1
            continue
        actual = hashlib.sha256(b"".join(p.stdout for p in shown)).hexdigest()
        assert actual == record["launcher_sha256"], f"{record_path}: pinned commit does not hold the recorded launcher"
        checked += 1
    return f"{checked} launcher pin(s) confirmed in Git history, {skipped} not readable in this checkout"


def main():
    root = Path(__file__).resolve().parents[1]
    tools = root / "tools/dfly2"
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    # test_primitives imports CUDA-enabled PyTorch and is an explicit GPU gate.
    for file in sorted(tools.glob("test_*.py")):
        if file.name != "test_primitives.py":
            suite.addTests(loader.discover(str(tools), pattern=file.name, top_level_dir=str(tools)))
    for test_dir in (root / "tests", root / "scripts"):
        if test_dir.is_dir():
            suite.addTests(loader.discover(str(test_dir), pattern="test_*.py", top_level_dir=str(test_dir)))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    for folder in (root / "tools", root / "scripts", root / "build"):
        for file in sorted(folder.rglob("*.py")):
            compile(file.read_bytes(), str(file.relative_to(root)), "exec")
    subprocess.run([sys.executable, str(tools / "verify_evidence.py")], cwd=root, check=True)
    subprocess.run([sys.executable, str(tools / "verify_portable_evidence.py")], cwd=root, check=True)
    for manifest in sorted((root / "evidence").glob("*/suite.json")):
        subprocess.run([sys.executable, str(tools / "verify_suite_evidence.py"), str(manifest.parent)], cwd=root, check=True)
    experimental_checked = check_experimental_evidence(root)
    experimental_build = check_experimental_build(root)
    selection_checked = check_experimental_selection(root)
    pinned = check_launcher_pins(root)
    build = json.loads((root / "evidence/public-build-validation.json").read_text())
    lock = json.loads((root / "upstream.lock.json").read_text())
    assert build["host_test_exit_code"] == build["public_docker_build_with_pinned_python_tools_exit_code"] == 0
    assert build["native_host_tests_passed"] == 582
    assert build["chat_request_payload_equal_to_published_fixture"]
    assert build["source_receipt"]["lock_sha256"] == hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()
    for name in ("tensorfold", "inputs"):
        assert build["source_receipt"]["trees"][name] == lock[name]["prepared_tree"]
    assert build["source_receipt"]["native_patch_sha256"] == hashlib.sha256((root / "patches/0001-dfly2-native.patch").read_bytes()).hexdigest()
    assert build["tool_requirements_sha256"] == hashlib.sha256((root / "build/tool-requirements.txt").read_bytes()).hexdigest()
    print(f"PASS: {result.testsRun} CPU tests, Python syntax, and published evidence; "
          f"{experimental_checked} experimental bundle(s) verified; {pinned}")
    if experimental_build is not None:
        print(f"Experimental v5 build consistency: {experimental_build}; execution is not authenticated by offline checks")
    if selection_checked:
        print("Experimental v5 selection and all final evaluations verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

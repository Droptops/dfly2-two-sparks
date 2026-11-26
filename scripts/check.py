#!/usr/bin/env python3
"""Run CPU-only release checks; no network, model load, Docker, or remote actions."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import unittest


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
    print(f"PASS: {result.testsRun} CPU tests, Python syntax, and published evidence; {pinned}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

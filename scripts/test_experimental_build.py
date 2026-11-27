"""Local synthetic Git fixtures; no network, Docker, GPUs or model measurements."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import build_experimental as build
import prepare_experimental as exp
import prepare_source as historical


class ExperimentalBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dfly-experimental-build-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/"repo"
        self.root.mkdir()
        self.dest = Path(self.temp.name)/"prepared"
        self.dest.mkdir()
        for name in exp.BUILD_INPUTS:
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(exp.ROOT/name, path)
        self.base = {"schema": 1, "toolchain": {"base_image": "base@sha256:"+"b"*64,
                     "zig_version": "0.17.0", "zig_sha256": "c"*64}}
        for name in ("recipe", "tensorfold", "inputs"):
            folder = self.dest/name
            self.git("init", "-q", str(folder), cwd=self.dest)
            self.git("config", "core.autocrlf", "false", cwd=folder)
            (folder/"sample.txt").write_bytes(b"baseline\n")
            self.git("add", "-A", cwd=folder)
            self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                     "commit", "-qm", "synthetic fixture", cwd=folder)
            self.base[name] = {"revision": self.git("rev-parse", "HEAD", cwd=folder),
                               "prepared_tree": self.git("write-tree", cwd=folder)}
        self.base["recipe"].update(serving_patches=[], inputs_patches=[])
        old_patch = self.root/"historical.patch"
        old_patch.write_bytes(b"historical identity fixture\n")
        self.base["integration_patches"] = [{"path": old_patch.name, "sha256": historical.digest(old_patch)}]
        exp.write_json(self.root/"upstream.lock.json", self.base)
        exp.write_json(self.dest/"upstream.lock.json", self.base)
        exp.write_json(self.dest/"source-receipt.json", exp.baseline_receipt(self.base))
        (self.dest/"source-files.sha256").write_bytes(historical.source_manifest(self.dest))
        (self.dest/".dockerignore").write_text("**/.git\nrecipe\n")
        native = self.dest/"tensorfold"
        (native/"sample.txt").write_bytes(b"experimental\n")
        (native/"new.txt").write_bytes(b"new code\n")
        self.git("add", "-A", cwd=native)
        native_tree = self.git("write-tree", cwd=native)
        delta = subprocess.run(["git", "diff", "--cached", "--binary", "HEAD"], cwd=native,
                               check=True, stdout=subprocess.PIPE).stdout
        (self.root/"experimental").mkdir()
        patch_path = self.root/"experimental/native.patch"
        patch_path.write_bytes(delta)
        self.git("reset", "--hard", "HEAD", cwd=native)
        self.lock = {"schema": exp.SCHEMA, "status": "finalized",
                     "baseline_lock_sha256": exp.canonical_hash(self.base),
                     "patches": [{"path": "experimental/native.patch", "sha256": historical.digest(patch_path)}],
                     "prepared_trees": {"tensorfold": native_tree, "inputs": self.base["inputs"]["prepared_tree"]}}
        self.lock_path = self.root/"experimental/native.lock.json"
        exp.write_json(self.lock_path, self.lock)
        self.addCleanup(patch.stopall)
        patch.object(historical, "ROOT", self.root).start()

    def git(self, *args, cwd):
        return subprocess.run(["git", *args], cwd=cwd, check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()

    def prepared(self):
        return exp.apply_experiment(self.dest, self.lock_path, self.root)

    def test_baseline_then_experimental_and_old_checker_refuses(self):
        historical.check(self.dest, self.base)
        receipt = self.prepared()
        self.assertEqual(receipt["schema"], exp.RECEIPT_SCHEMA)
        self.assertFalse(receipt["historical_validation_applies"])
        self.assertEqual(receipt, exp.check(self.dest, self.lock_path, self.root))
        self.assertEqual((self.dest/"tensorfold/new.txt").read_text(), "new code\n")
        self.assertEqual(receipt["baseline_receipt"], exp.baseline_receipt(self.base))
        with self.assertRaises((KeyError, ValueError)):
            historical.check(self.dest, self.base)

    def test_unfinalized_lock_refuses_before_subprocess(self):
        self.lock["status"] = "unfinalized"
        exp.write_json(self.lock_path, self.lock)
        with patch.object(historical, "run") as runner:
            with self.assertRaisesRegex(ValueError, "not finalized"):
                exp.prepare(self.dest/"fresh", self.lock_path, self.root)
            runner.assert_not_called()

    def test_prepare_rejects_existing_destination(self):
        with self.assertRaisesRegex(ValueError, "destination exists"):
            exp.prepare(self.dest, self.lock_path, self.root)

    def test_wrong_patch_hash_refuses(self):
        (self.root/"experimental/native.patch").write_bytes(b"changed\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.prepared()
        self.assertTrue((self.dest/"source-receipt.json").exists())

    def test_failed_patch_does_not_leave_historical_attestation(self):
        invalid = b"not a patch\n"
        path = self.root/"experimental/native.patch"
        path.write_bytes(invalid)
        self.lock["patches"][0]["sha256"] = historical.digest(path)
        exp.write_json(self.lock_path, self.lock)
        with self.assertRaises(subprocess.CalledProcessError):
            self.prepared()
        self.assertFalse((self.dest/"source-receipt.json").exists())
        self.assertTrue((self.dest/"baseline-source-receipt.json").exists())

    def test_wrong_prepared_tree_refuses(self):
        self.lock["prepared_trees"]["tensorfold"] = "f"*40
        exp.write_json(self.lock_path, self.lock)
        with self.assertRaisesRegex(ValueError, "source tree"):
            self.prepared()

    def test_altered_source_and_new_file_refused(self):
        self.prepared()
        (self.dest/"tensorfold/extra.txt").write_bytes(b"unreviewed")
        with self.assertRaisesRegex(ValueError, "source tree"):
            exp.check(self.dest, self.lock_path, self.root)

    def test_altered_baseline_provenance_refused(self):
        self.prepared()
        (self.dest/"baseline-source-files.sha256").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            exp.check(self.dest, self.lock_path, self.root)

    def test_changed_build_input_refused(self):
        self.prepared()
        (self.root/"build/Dockerfile").write_bytes(b"different build")
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            exp.check(self.dest, self.lock_path, self.root)

    def test_changed_baseline_lock_refused(self):
        changed = copy.deepcopy(self.base)
        changed["toolchain"]["zig_version"] = "other"
        exp.write_json(self.root/"upstream.lock.json", changed)
        with self.assertRaisesRegex(ValueError, "baseline lock mismatch"):
            exp.locks(self.lock_path, self.root)

    def test_patch_path_escape_refused(self):
        self.lock["patches"][0]["path"] = "../outside.patch"
        exp.write_json(self.lock_path, self.lock)
        with self.assertRaisesRegex(ValueError, "escapes repository"):
            exp.locks(self.lock_path, self.root)

    def test_changed_inputs_tree_not_permitted(self):
        self.lock["prepared_trees"]["inputs"] = "a"*40
        exp.write_json(self.lock_path, self.lock)
        with self.assertRaisesRegex(ValueError, "preserve the historical inputs"):
            exp.locks(self.lock_path, self.root)

    def test_print_command_checks_identity_without_docker(self):
        self.prepared()
        with patch.object(build, "subprocess") as docker:
            cmd = build.build(self.dest, self.lock_path, "candidate:test", 3, print_command=True, root=self.root)
            docker.run.assert_not_called()
        self.assertIn(str(self.root/"build/Dockerfile"), cmd)
        self.assertIn("ai.dfly2.source-kind=experimental", cmd)
        self.assertIn("BASE_IMAGE="+self.base["toolchain"]["base_image"], cmd)
        self.assertFalse((self.dest/"experimental-build-receipt.json").exists())

    def fake_docker(self, code=0, wrong_label=False, mutate=False):
        image_id = "sha256:"+"d"*64
        def run(cmd, **kwargs):
            if cmd[1] == "build":
                if code == 0:
                    Path(cmd[cmd.index("--iidfile")+1]).write_text(image_id)
                return SimpleNamespace(returncode=code)
            self.assertEqual(cmd, ["docker", "image", "inspect", image_id])
            if mutate:
                (self.dest/"tensorfold/sample.txt").write_bytes(b"changed during build")
            labels = {"ai.dfly2.source-kind": "wrong" if wrong_label else "experimental",
                      "ai.dfly2.experimental-lock-sha256": exp.canonical_hash(self.lock),
                      "ai.dfly2.native-tree": self.lock["prepared_trees"]["tensorfold"]}
            return SimpleNamespace(stdout=json.dumps([{"Id": image_id, "Architecture": "arm64", "Os": "linux",
                                                       "Config": {"Labels": labels}}]))
        return SimpleNamespace(run=run, PIPE=subprocess.PIPE)

    def test_build_records_immutable_image_identity_and_provenance(self):
        self.prepared()
        with patch.object(build, "subprocess", self.fake_docker()):
            result = build.build(self.dest, self.lock_path, "candidate:test", 2, root=self.root)
        self.assertTrue(result["success"])
        self.assertEqual(result["image"]["id"], "sha256:"+"d"*64)
        self.assertEqual(result["source_receipt"]["prepared_trees"], self.lock["prepared_trees"])
        self.assertEqual(result, exp.load(self.dest/"experimental-build-receipt.json"))

    def test_failed_build_retains_receipt_and_refuses_overwrite(self):
        self.prepared()
        with patch.object(build, "subprocess", self.fake_docker(code=7)):
            with self.assertRaisesRegex(ValueError, "Docker build failed"):
                build.build(self.dest, self.lock_path, "candidate:test", 2, root=self.root)
            with self.assertRaisesRegex(ValueError, "receipt already exists"):
                build.build(self.dest, self.lock_path, "candidate:test", 2, root=self.root)
        result = exp.load(self.dest/"experimental-build-receipt.json")
        self.assertFalse(result["success"])
        self.assertEqual(result["build_exit_code"], 7)
        self.assertEqual(result["status"], "failed")

    def test_wrong_image_label_is_failed_build_receipt(self):
        self.prepared()
        with patch.object(build, "subprocess", self.fake_docker(wrong_label=True)):
            with self.assertRaisesRegex(ValueError, "labels/platform"):
                build.build(self.dest, self.lock_path, "candidate:test", 2, root=self.root)
        self.assertFalse(exp.load(self.dest/"experimental-build-receipt.json")["success"])

    def test_source_change_during_build_is_failed_receipt(self):
        self.prepared()
        with patch.object(build, "subprocess", self.fake_docker(mutate=True)):
            with self.assertRaisesRegex(ValueError, "source tree"):
                build.build(self.dest, self.lock_path, "candidate:test", 2, root=self.root)
        self.assertEqual(exp.load(self.dest/"experimental-build-receipt.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()

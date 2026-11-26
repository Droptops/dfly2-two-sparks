import copy
import json
from pathlib import Path
import tempfile
import unittest

import export_portable_evidence as exporter
import sanitize_evidence as sanitizer
import verify_evidence as historical_verifier
import verify_portable_evidence as portable

EVIDENCE = Path(__file__).resolve().parents[2] / "evidence"
LAUNCHER = "1" * 64
TARGET = "2" * 64
IMAGE = "4" * 64


class PortableEvidenceTests(unittest.TestCase):
    def test_published_portable_measurements_recompute(self):
        result = portable.verify(EVIDENCE / "portable-validation")
        self.assertEqual(result, historical_verifier.load(EVIDENCE / "portable-validation/comparison.json"))
        self.assertEqual(result["output_tokens_per_run"], 959)
        self.assertTrue(result["matches_historical_serial_tokens_and_finish_reasons"])
        self.assertEqual([r["estimated_decode_tokens_per_second"] for r in result["runs"]], [38.49, 74.98, 45.06])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.binary = historical_verifier.load(EVIDENCE / "public-build-validation.json")["built_executable_sha256"]["/opt/tensorfold/bin/tf-dsv41-lanes"]
        (self.directory / "chat-requests.json").write_bytes((EVIDENCE / "chat-requests.json").read_bytes())
        for mode in portable.MODES:
            # Synthetic fixture based on historical numbers: this is never
            # exported as a real portable-run receipt.
            run = historical_verifier.load(EVIDENCE / f"chat-{mode}.json")
            run["validation"] = "portable-launcher"
            for node in run["nodes"]:
                node.update(binary_sha256=self.binary, launcher_sha256=LAUNCHER, target_config_sha256=TARGET,
                            image_content_sha256=IMAGE)
            self.write(mode, run)

    def write(self, mode, run):
        (self.directory / f"chat-{mode}.json").write_text(json.dumps(run), encoding="utf-8")

    def verify(self):
        return portable.verify(self.directory, historical=EVIDENCE, launcher_digest=LAUNCHER)

    def test_published_receipts_are_pinned_to_their_launcher_commit(self):
        record = portable.provenance(EVIDENCE / "portable-validation")
        for mode in portable.MODES:
            run = historical_verifier.load(EVIDENCE / f"portable-validation/chat-{mode}.json")
            self.assertEqual({n["launcher_sha256"] for n in run["nodes"]}, {record["launcher_sha256"]})
        result = portable.verify(EVIDENCE / "portable-validation")
        self.assertEqual(result["launcher_source_commit"], record["repository_commit"])
        # Later launcher edits must not re-label old measurements: the pin, not
        # the working tree, decides which launcher the receipts describe.
        (self.directory / "provenance.json").write_text(json.dumps({**record, "launcher_sha256": "9" * 64}))
        with self.assertRaisesRegex(ValueError, "published launcher"):
            portable.verify(self.directory, historical=EVIDENCE)
        (self.directory / "provenance.json").write_text(json.dumps({**record, "repository_commit": "main"}))
        with self.assertRaisesRegex(ValueError, "invalid provenance repository_commit"):
            portable.verify(self.directory, historical=EVIDENCE)

    def test_matched_fixture_recomputed_and_bound_to_build(self):
        result = self.verify()
        self.assertEqual(result["output_tokens_per_run"], 959)
        self.assertTrue(result["matches_historical_serial_tokens_and_finish_reasons"])
        self.assertEqual(result["binary_sha256"], self.binary)
        self.assertEqual([r["estimated_decode_tokens_per_second"] for r in result["runs"]], [38.46, 74.05, 45.84])

    def test_rejects_different_executable_even_if_both_ranks_match(self):
        run = historical_verifier.load(self.directory / "chat-dfly2.json")
        for node in run["nodes"]:
            node["binary_sha256"] = "3" * 64
        self.write("dfly2", run)
        with self.assertRaisesRegex(ValueError, "freshly built public binary"):
            self.verify()

    def test_rejects_changed_launcher_or_target_configuration(self):
        original = historical_verifier.load(self.directory / "chat-dfly2.json")
        for field, message in (("launcher_sha256", "published launcher"), ("target_config_sha256", "different target"),
                               ("image_content_sha256", "different image contents")):
            with self.subTest(field=field):
                run = copy.deepcopy(original)
                run["nodes"][0][field] = "3" * 64
                self.write("dfly2", run)
                with self.assertRaisesRegex(ValueError, message):
                    self.verify()

    def test_rejects_changed_pool_even_when_saved_equal_flag_is_true(self):
        run = historical_verifier.load(self.directory / "chat-dfly2.json")
        run["configuration"]["pool_tokens"] = 16384
        run["equal"] = True
        self.write("dfly2", run)
        with self.assertRaisesRegex(ValueError, "inference settings differ"):
            self.verify()

    def test_rejects_shared_unqualified_pool_or_changed_image_across_modes(self):
        for mode in portable.MODES:
            run = historical_verifier.load(self.directory / f"chat-{mode}.json")
            run["configuration"]["pool_tokens"] = 16384
            self.write(mode, run)
        with self.assertRaisesRegex(ValueError, "8192-token pool"):
            self.verify()
        for mode in portable.MODES:
            run = historical_verifier.load(self.directory / f"chat-{mode}.json")
            run["configuration"]["pool_tokens"] = 8192
            if mode == "dfly2":
                for node in run["nodes"]:
                    node["image_content_sha256"] = "5" * 64
            self.write(mode, run)
        with self.assertRaisesRegex(ValueError, "image contents differ across modes"):
            self.verify()

    def test_unreviewed_prompt_text_refused_before_export(self):
        request = historical_verifier.load(self.directory / "chat-requests.json")
        request["metadata"]["tasks"][0]["messages"][0]["content"] = "private prompt text"
        path = self.directory / "unreviewed.json"
        path.write_text(json.dumps(request), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reviewed historical fixture"):
            exporter.validate_fixture(path, EVIDENCE)
        output = self.directory / "portable-validation"
        argv = ["--requests", str(path), "--pool-tokens", "8192", "--output", str(output)]
        for mode in portable.MODES:
            argv.extend(["--" + mode, str(self.directory / f"chat-{mode}.json")])
        with self.assertRaisesRegex(ValueError, "reviewed historical fixture"):
            exporter.main(argv)
        self.assertFalse(output.exists(), "unreviewed text must not create public output")

    def test_portable_export_retains_only_additional_public_hashes(self):
        clean = historical_verifier.load(self.directory / "chat-dfly2.json")
        nodes = [{**node, "rank_cache": {"bytes": node["target_rank_cache_bytes"], "name": "private-cache"},
                  "config_identity": "private-config-identity", "image_id": "private-image-id"}
                 for node in clean["nodes"]]
        raw = {"mode": "dfly2", "success": True, "preflight": {"nodes": nodes},
               "started_utc": "2026-11-26T20:00:00Z", "finished_utc": "2026-11-26T20:01:00Z",
               "native": {"complete": True, "records": clean["records"], "summary": clean["summary"]}}
        exported = sanitizer.sanitize_portable_run(raw, pool_tokens=8192)
        self.assertNotIn("private-", json.dumps(exported))
        self.assertEqual(exported["nodes"][0]["launcher_sha256"], LAUNCHER)
        self.assertEqual(exported["nodes"][1]["target_config_sha256"], TARGET)
        self.assertEqual(exported["nodes"][1]["image_content_sha256"], IMAGE)
        self.assertEqual(exported["started_utc"], raw["started_utc"])
        for invalid in ("8192", True, 0, 16384):
            with self.subTest(pool=invalid), self.assertRaises(ValueError):
                sanitizer.sanitize_portable_run(raw, pool_tokens=invalid)
        raw["started_utc"] = "private timestamp note"
        with self.assertRaisesRegex(ValueError, "UTC timestamp"):
            sanitizer.sanitize_portable_run(raw, pool_tokens=8192)


if __name__ == "__main__":
    unittest.main()

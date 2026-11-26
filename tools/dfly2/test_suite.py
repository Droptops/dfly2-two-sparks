import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cluster_config as cluster
import export_suite_evidence as exporter
import run_pair as pair
import run_suite as suite
import sanitize_evidence as sanitizer
import verify_evidence as historical_verifier
import verify_portable_evidence as portable
import verify_suite_evidence as verifier

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "evidence"
EXAMPLE = ROOT / "examples/cluster.example.json"
COMMIT = "a" * 40


def raw_result(mode, source, launcher, minute):
    """A private run_pair result rebuilt from a published receipt.

    Synthetic input for these tests only: dspark_fixed5 has no measured receipt
    yet, so it borrows the dspark numbers. Nothing here is exported as evidence.
    """
    clean = historical_verifier.load(EVIDENCE / f"portable-validation/chat-{source}.json")
    nodes = [{**node, "launcher_sha256": launcher, "pack_manifest_sha256": node["pack_manifest_sha256"] if mode == "dfly2" else None,
              "rank_cache": {"name": "private-cache", "bytes": node["target_rank_cache_bytes"]},
              "host": "private-node", "image_id": "private-image-id"} for node in clean["nodes"]]
    return {"mode": mode, "success": True, "run_id": f"{minute:032x}", "preflight": {"nodes": nodes},
            "execution": {"world_size": 2, "streams": 1, "parallel": 1, "pool_tokens": 8192, "cuda_graphs": False,
                          "profiling": True, "drafting": mode != "serial",
                          "fixed_draft_depth": 5 if mode == "dspark_fixed5" else None},
            "started_utc": f"2026-11-27T10:{minute:02d}:00Z", "finished_utc": f"2026-11-27T10:{minute:02d}:50Z",
            "native": {"complete": True, "records": clean["records"], "summary": clean["summary"]}}


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        pair.configure(cluster.load(EXAMPLE))

    def test_every_mode_runs_once_per_repeat_and_the_first_mode_rotates(self):
        order = suite.schedule(pair.MODES, 3)
        self.assertEqual(order[0], ("serial", 1))
        self.assertEqual(len(order), 12)
        for repeat in (1, 2, 3):
            self.assertEqual(sorted(m for m, r in order if r == repeat), sorted(pair.MODES))
        self.assertEqual([order[i][0] for i in (0, 4, 8)], ["serial", "dspark", "dspark_fixed5"])
        for bad in (["dfly2", "serial"], ["serial", "serial"], ["serial", "unknown"], []):
            with self.assertRaises(pair.Refusal):
                suite.schedule(bad, 1)
        for bad in (0, 10, True):
            with self.assertRaises(pair.Refusal):
                suite.schedule(["serial"], bad)

    def test_port_wait_retries_only_the_time_wait_refusal(self):
        now = [0]
        calls = []

        def preflight(config):
            calls.append(now[0])
            if len(calls) < 3:
                raise OSError("[Errno 98] Address already in use")
            return {"ok": True}
        with patch.object(pair, "preflight_pair", side_effect=preflight):
            self.assertEqual(suite.wait_for_preflight({}, 60, sleep=lambda s: now.__setitem__(0, now[0] + s), clock=lambda: now[0]), {"ok": True})
        self.assertEqual(calls, [0, 5, 10])
        with patch.object(pair, "preflight_pair", side_effect=pair.Refusal("GPU compute processes remain")):
            with self.assertRaisesRegex(pair.Refusal, "GPU compute"):
                suite.wait_for_preflight({}, 60, sleep=lambda s: self.fail("must not wait"), clock=lambda: 0)
        with patch.object(pair, "preflight_pair", side_effect=pair.Refusal("worker: OSError: Address already in use")):
            with self.assertRaisesRegex(pair.Refusal, "Address already in use"):
                suite.wait_for_preflight({}, 10, sleep=lambda s: now.__setitem__(0, now[0] + s), clock=lambda: now[0])

    def test_suite_passes_the_first_serial_result_as_reference_and_stops_on_failure(self):
        seen = []

        def run(config, output, timeout, reference):
            seen.append((config["mode"], output.name, None if reference is None else reference["output"]))
            success = len(seen) != 3
            return {"success": success, "run_id": config["run_id"], "output": output.name, "error": None if success else "boom",
                    "preflight": {"nodes": [{"requests_sha256": "f" * 64}]}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(pair, "preflight_pair", return_value={}), \
                patch.object(pair, "run_pair", side_effect=run), patch.object(pair, "new_output", side_effect=lambda p: (p.mkdir(), p)[1]), \
                patch("builtins.print"):
            manifest = suite.run_suite(["serial", "dspark_fixed5", "dfly2"], 2, "/r.json", Path(tmp) / "out", 60, 0)
            saved = json.loads((Path(tmp) / "out/suite.json").read_text())
        self.assertEqual(seen, [("serial", "01-serial-r1", None), ("dspark_fixed5", "02-dspark_fixed5-r1", "01-serial-r1"),
                                ("dfly2", "03-dfly2-r1", "01-serial-r1")])
        self.assertFalse(manifest["complete"])
        self.assertEqual(saved, manifest)
        self.assertIn("03-dfly2-r1", manifest["error"])
        self.assertEqual(manifest["launcher_sha256"], pair.launcher_identity())

    def test_default_is_preflight_only(self):
        with patch.object(pair, "preflight_pair", return_value={}) as preflight, patch.object(suite, "run_suite") as run, patch("builtins.print"):
            self.assertEqual(suite.main(["--config", str(EXAMPLE), "--repeats", "2"]), 0)
        self.assertEqual(preflight.call_count, len(pair.MODES))
        run.assert_not_called()


class SuiteEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.private = Path(self.temporary.name) / "private"
        self.public = Path(self.temporary.name) / "public"
        self.launcher = portable.current_launcher_digest()
        self.requests = EVIDENCE / "chat-requests.json"
        order = suite.schedule(["serial", "dspark", "dspark_fixed5", "dfly2"], 2)
        runs = []
        for index, (mode, repeat) in enumerate(order, 1):
            name = f"{index:02d}-{mode}-r{repeat}"
            (self.private / name).mkdir(parents=True)
            raw = raw_result(mode, "dspark" if mode == "dspark_fixed5" else mode, self.launcher, index)
            (self.private / name / "result.json").write_text(json.dumps(raw))
            runs.append({"index": index, "mode": mode, "repeat": repeat, "directory": name, "run_id": raw["run_id"], "success": True})
        self.manifest = {"schema": "dfly2-suite-v1", "modes": ["serial", "dspark", "dspark_fixed5", "dfly2"], "repeats": 2,
                         "launcher_sha256": self.launcher, "requests_sha256": historical_verifier.sha256(self.requests),
                         "complete": True, "runs": runs}
        self.write_manifest()

    def write_manifest(self):
        (self.private / "suite.json").write_text(json.dumps(self.manifest))

    def export(self, requests=None):
        with patch("builtins.print"):
            return exporter.main(["--suite-dir", str(self.private), "--requests", str(requests or self.requests),
                                  "--pool-tokens", "8192", "--commit", COMMIT, "--output", str(self.public)])

    def test_export_then_verify_reports_medians_ranges_and_the_matched_pair(self):
        self.assertEqual(self.export(), 0)
        result = verifier.verify(self.public)
        self.assertEqual(result, historical_verifier.load(self.public / "comparison.json"))
        self.assertEqual(result["run_order"][:5], ["serial-r1", "dspark-r1", "dspark_fixed5-r1", "dfly2-r1", "dspark-r2"])
        self.assertEqual(result["launcher_source_commit"], COMMIT)
        self.assertEqual([m["mode"] for m in result["modes"]], self.manifest["modes"])
        serial, dspark, fixed, dfly = result["modes"]
        self.assertEqual(serial["estimated_decode_tokens_per_second"], {"median": 38.49, "min": 38.49, "max": 38.49})
        self.assertEqual(dfly["estimated_decode_tokens_per_second"]["median"], 45.06)
        self.assertEqual(dfly["runs"], 2)
        self.assertEqual(result["relative_to_serial"][-1]["requests_slower_than_serial_by_median"], ["chat_sampled_explain_cache"])
        self.assertEqual(result["matched_depth_comparison"]["modes"], ["dspark_fixed5", "dfly2"])
        self.assertEqual(result["matched_depth_comparison"]["dfly2_decode_rate_relative_to_dspark_fixed5"], 0.601)
        encoded = "".join(p.read_text() for p in self.public.iterdir())
        self.assertNotIn("private-", encoded)
        for run in (historical_verifier.load(self.public / e["file"]) for e in historical_verifier.load(self.public / "suite.json")["runs"]):
            self.assertIs(run["configuration"]["recorded_by_launcher"], True)

    def test_incomplete_failed_or_foreign_suites_are_refused_before_any_output(self):
        for change, message in ((lambda m: m.update(complete=False), "did not complete"),
                                (lambda m: m["runs"][3].update(success=False), "did not complete"),
                                (lambda m: m.update(requests_sha256="0" * 64), "different request file"),
                                (lambda m: m.update(launcher_sha256="0" * 64), "checkout that ran the suite"),
                                (lambda m: m["runs"][1].update(mode="dfly2"), "disagrees with a result")):
            with self.subTest(message=message):
                original = copy.deepcopy(self.manifest)
                change(self.manifest)
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, message):
                    self.export()
                self.assertFalse(self.public.exists())
                self.manifest = original
                self.write_manifest()
        unreviewed = Path(self.temporary.name) / "unreviewed.json"
        document = historical_verifier.load(self.requests)
        document["metadata"]["tasks"][0]["messages"][0]["content"] = "private prompt text"
        unreviewed.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "reviewed fixture"):
            self.export(unreviewed)
        self.assertFalse(self.public.exists())
        with self.assertRaisesRegex(ValueError, "40-hex"):
            with patch("builtins.print"):
                exporter.main(["--suite-dir", str(self.private), "--requests", str(self.requests), "--pool-tokens", "8192",
                               "--commit", "main", "--output", str(self.public)])

    def edit(self, name, change):
        run = historical_verifier.load(self.public / name)
        change(run)
        (self.public / name).write_text(json.dumps(run))

    def test_verifier_rejects_a_changed_token_a_wrong_depth_and_reordered_runs(self):
        self.export()
        pristine = {p.name: p.read_bytes() for p in self.public.iterdir()}

        def restore():
            for name, data in pristine.items():
                (self.public / name).write_bytes(data)
        cases = (("run-07-dfly2-r2.json", lambda r: [x for x in r["records"] if "reply" in x][0]["reply"].__setitem__(0, 1), "token digest mismatch"),
                 ("run-03-dspark_fixed5-r1.json", lambda r: r["configuration"].update(fixed_draft_depth=None), "contradict the mode"),
                 ("run-02-dspark-r1.json", lambda r: r["configuration"].update(fixed_draft_depth=5), "contradict the mode"),
                 ("run-04-dfly2-r1.json", lambda r: r["configuration"].pop("recorded_by_launcher"), "launcher-recorded settings"),
                 ("run-04-dfly2-r1.json", lambda r: r["configuration"].update(pool_tokens=16384), "unexpected inference settings"),
                 ("run-05-dspark-r2.json", lambda r: r.update(started_utc="2026-11-27T09:00:00Z"), "run times disagree"),
                 ("run-06-dspark_fixed5-r2.json", lambda r: [n.update(binary_sha256="3" * 64) for n in r["nodes"]], "freshly built public binary"),
                 ("run-01-serial-r1.json", lambda r: [n.update(pack_manifest_sha256="3" * 64) for n in r["nodes"]], "draft pack identity"),
                 ("run-02-dspark-r1.json", lambda r: r.update(suite="long"), "wrong suite receipt"))
        for name, change, message in cases:
            with self.subTest(name=name, message=message):
                self.edit(name, change)
                with self.assertRaisesRegex(ValueError, message):
                    verifier.verify(self.public)
                restore()
        manifest = historical_verifier.load(self.public / "suite.json")
        manifest["runs"] = manifest["runs"][:-1]
        (self.public / "suite.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "missing or duplicate repeat"):
            verifier.verify(self.public)
        restore()
        self.assertEqual(verifier.verify(self.public), json.loads(pristine["comparison.json"]))

    def test_recorded_execution_settings_must_be_complete_and_agree_with_the_mode(self):
        raw = raw_result("dspark_fixed5", "dspark", self.launcher, 1)
        self.assertEqual(sanitizer.sanitize_portable_run(raw, pool_tokens=8192)["configuration"]["fixed_draft_depth"], 5)
        for change in (lambda e: e.update(drafting=False), lambda e: e.update(fixed_draft_depth="5"),
                       lambda e: e.update(note="private"), lambda e: e.pop("streams"), lambda e: e.update(pool_tokens=True)):
            broken = copy.deepcopy(raw)
            change(broken["execution"])
            with self.assertRaises(ValueError):
                sanitizer.sanitize_portable_run(broken, pool_tokens=8192)
        broken = copy.deepcopy(raw)
        broken["execution"]["pool_tokens"] = 4096
        with self.assertRaisesRegex(ValueError, "pool differs"):
            sanitizer.sanitize_portable_run(broken, pool_tokens=8192)


if __name__ == "__main__":
    unittest.main()

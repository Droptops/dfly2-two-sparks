import json
from pathlib import Path
import tempfile
import unittest

import sanitize_evidence as sanitizer
import verify_evidence as verifier

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "evidence"


class EvidenceTests(unittest.TestCase):
    def test_public_claims_recomputed_from_arrays_and_times(self):
        result = verifier.verify(EVIDENCE)
        self.assertEqual(result["output_tokens_per_run"], 959)
        self.assertEqual([r["estimated_decode_tokens_per_second"] for r in result["runs"]], [38.46, 74.05, 45.84])
        self.assertEqual(result, verifier.load(EVIDENCE / "comparison.json"))

    def raw_fixture(self):
        clean = verifier.load(EVIDENCE / "chat-dfly2.json")
        nodes = []
        for n in clean["nodes"]:
            nodes.append({**n, "rank_cache": {"name": "private-cache-name", "bytes": n["target_rank_cache_bytes"]},
                          "host": "private-node.test", "fabric": {"address": "192.0.2.9"},
                          "image_id": "private-image-id", "image": "private-image:tag"})
        return {"mode": clean["mode"], "success": True, "run_id": "private-run-id",
                "preflight": {"nodes": nodes, "dashboard": {"secret": "private-dashboard-state"}},
                "states": {"Pid": 999999}, "cleanup": {"container_id": "private-container-id"},
                "native": {"complete": True, "records": clean["records"], "summary": clean["summary"]}}

    def test_allowlist_drops_unknown_envelope_node_and_record_fields(self):
        raw = self.raw_fixture()
        raw["native"]["records"][0]["private_note"] = "private-record-note"
        raw["native"]["records"].append({"rank": 0, "secret_metadata": "private-record-body"})
        encoded = json.dumps(sanitizer.sanitize_run(raw))
        self.assertNotIn("private-", encoded)
        self.assertNotIn("192.0.2.9", encoded)
        self.assertNotIn("999999", encoded)
        self.assertIn("token_sha", encoded)
        self.assertIn("prefill_s", encoded)

    def test_non_numeric_metric_payload_is_refused(self):
        raw = self.raw_fixture()
        raw["native"]["records"].append({"rank": 0, "profile": {"pass_ms": "unreviewed text"}})
        with self.assertRaises(ValueError):
            sanitizer.sanitize_run(raw)

    def test_non_numeric_reply_payload_is_refused(self):
        raw = self.raw_fixture()
        next(r for r in raw["native"]["records"] if "reply" in r)["seconds"] = "unreviewed text"
        with self.assertRaises(ValueError):
            sanitizer.sanitize_run(raw)

    def test_unknown_request_metadata_is_refused(self):
        request = verifier.load(EVIDENCE / "chat-requests.json")
        request["metadata"]["local_path"] = "unreviewed"
        with self.assertRaises(ValueError):
            sanitizer.validate_requests(request)

    def test_changed_token_is_detected_without_trusting_saved_equal_flag(self):
        run = verifier.load(EVIDENCE / "chat-dfly2.json")
        reply = next(r for r in run["records"] if "reply" in r)
        reply["reply"][0] += 1
        requests = verifier.load(EVIDENCE / "chat-requests.json")["requests"]
        with self.assertRaisesRegex(ValueError, "token digest mismatch"):
            verifier.summarize_run(run, requests, verifier.sha256(EVIDENCE / "chat-requests.json"), None)

    def summarize(self, run):
        requests = verifier.load(EVIDENCE / "chat-requests.json")["requests"]
        return verifier.summarize_run(run, requests, verifier.sha256(EVIDENCE / "chat-requests.json"), None)

    def test_changed_total_time_is_rejected_by_the_reply_clock(self):
        run = verifier.load(EVIDENCE / "chat-dfly2.json")
        run["summary"]["seconds"] -= 5
        with self.assertRaisesRegex(ValueError, "per-reply seconds disagree"):
            self.summarize(run)

    def test_consistently_shortened_times_are_rejected_by_the_step_clock(self):
        # Editing the total and a reply together still leaves the profiler's
        # independent step time, which a flattering edit would have to match.
        run = verifier.load(EVIDENCE / "chat-dfly2.json")
        run["summary"]["seconds"] = round(run["summary"]["seconds"] - 5, 2)
        reply = [r for r in run["records"] if "reply" in r][-1]
        reply["seconds"] = round(reply["seconds"] - 5, 2)
        with self.assertRaisesRegex(ValueError, "native step clock disagrees"):
            self.summarize(run)

    def test_missing_or_inconsistent_profile_is_rejected(self):
        run = verifier.load(EVIDENCE / "chat-dfly2.json")
        profile = next(r for r in run["records"] if "profile" in r)
        profile["profile"]["rounds"] += 1
        with self.assertRaisesRegex(ValueError, "profile rounds disagree"):
            self.summarize(run)
        run["records"] = [r for r in run["records"] if "profile" not in r]
        with self.assertRaisesRegex(ValueError, "exactly one native profile"):
            self.summarize(run)

    def test_per_request_and_end_to_end_figures_are_published(self):
        result = verifier.verify(EVIDENCE)
        dspark, dfly2 = result["relative_to_serial"]
        self.assertEqual((dspark["mode"], dfly2["mode"]), ("dspark", "dfly2"))
        self.assertEqual(dspark["requests_slower_than_serial"], [])
        self.assertEqual(dfly2["requests_slower_than_serial"], ["chat_sampled_explain_cache"])
        self.assertLess(dfly2["end_to_end_speedup"], dfly2["decode_speedup"])
        row = result["runs"][2]
        self.assertEqual([q["output_tokens"] for q in row["requests"]], [197, 133, 41, 256, 96, 236])
        self.assertEqual(sum(q["rounds"] for q in row["requests"]), row["rounds"])
        # The drafters land the same number of tokens a round; the gap is cost.
        self.assertAlmostEqual(row["tokens_per_round"], result["runs"][1]["tokens_per_round"], delta=0.1)
        self.assertLess(row["decode_tokens_per_second_if_draft_pass_were_free"],
                        result["runs"][1]["estimated_decode_tokens_per_second"])

    def test_custom_kernels_explain_only_a_minority_of_the_measured_draft_pass(self):
        result = verifier.verify(EVIDENCE)
        budget = result["microbenchmark_custom_kernel_ms_per_draft_pass"]
        self.assertEqual(budget, {"routed_expert_ms": [2.78, 3.71], "attention_ms": 2.3, "total_ms": [5.08, 6.01]})
        self.assertLess(budget["total_ms"][1], result["runs"][2]["native_profile"]["pass_ms"] / 4)

    def test_primitive_receipt_is_bound_to_the_patch_kernel(self):
        patch = (ROOT / "patches/0001-dfly2-native.patch").read_bytes()
        source = verifier.added_file_from_patch(ROOT / "patches/0001-dfly2-native.patch", verifier.KERNEL_PATH)
        self.assertIn(b"tf_dfly_gather_context", source)
        self.assertEqual(verifier.hashlib.sha256(source).hexdigest(),
                         verifier.load(EVIDENCE / "gpu-primitives.json")["source_sha256"])
        with tempfile.TemporaryDirectory() as tmp:
            changed = Path(tmp) / "changed.patch"
            changed.write_bytes(patch.replace(b"0.04419417382415922f", b"0.05f", 1))
            with self.assertRaisesRegex(ValueError, "published kernel source"):
                verifier.verify(EVIDENCE, patch=changed)


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
import unittest

import sanitize_evidence as sanitizer
import verify_evidence as verifier

EVIDENCE = Path(__file__).resolve().parents[2] / "evidence"


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

    def test_changed_time_recomputes_rate(self):
        run = verifier.load(EVIDENCE / "chat-dfly2.json")
        run["summary"]["seconds"] += 1
        requests = verifier.load(EVIDENCE / "chat-requests.json")["requests"]
        row, _ = verifier.summarize_run(run, requests, verifier.sha256(EVIDENCE / "chat-requests.json"), None)
        self.assertNotEqual(row["estimated_decode_tokens_per_second"], 45.84)


if __name__ == "__main__":
    unittest.main()

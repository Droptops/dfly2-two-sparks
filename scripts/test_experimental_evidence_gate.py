"""Refusal/dispatch checks for published experimental evidence discovery."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import check


class ExperimentalEvidenceGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="dfly-evidence-gate-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def marker(self, name, value, filename="matrix.json"):
        folder = self.root/"evidence/experimental"/name
        folder.mkdir(parents=True)
        (folder/filename).write_text(json.dumps(value), encoding="utf-8")
        return folder

    def test_absent_directory_permitted_without_validator_calls(self):
        with patch.object(check.subprocess, "run") as run:
            self.assertEqual(check.check_experimental_evidence(self.root), 0)
            run.assert_not_called()

    def test_valid_schema_dispatches_to_each_existing_verifier(self):
        self.marker("profile-v4-long", {"schema":"dfly2-lab-matrix-public-v1", "purpose":"profile"})
        self.marker("reference-v4-rounding", {"schema":"dfly2-reference-public-manifest-v1"}, "manifest.json")
        with patch.object(check.subprocess, "run") as run:
            self.assertEqual(check.check_experimental_evidence(self.root), 2)
            calls = run.call_args_list
        self.assertIn("--verify-export", calls[0].args[0])
        self.assertIn("--verify", calls[1].args[0])
        self.assertTrue(all(c.kwargs["check"] for c in calls))

    def test_present_malformed_manifest_refused(self):
        folder = self.marker("benchmark-chat", {})
        (folder/"matrix.json").write_text("{broken", encoding="utf-8")
        with patch.object(check.subprocess, "run") as run:
            with self.assertRaises(json.JSONDecodeError): check.check_experimental_evidence(self.root)
            run.assert_not_called()

    def test_v5_sweep_dispatches_to_matrix_verifier(self):
        folder=self.marker("sweep-v5-chat", {"schema":"dfly2-lab-matrix-public-v1", "purpose":"benchmark"})
        with patch.object(check.subprocess, "run") as run:
            self.assertEqual(check.check_experimental_evidence(self.root),1)
            self.assertEqual(run.call_args.args[0][-2:],["--verify-export",str(folder)])
            self.assertTrue(run.call_args.kwargs["check"])

    def test_wrong_purpose_cannot_disguise_profile_as_benchmark(self):
        self.marker("benchmark-chat", {"schema":"dfly2-lab-matrix-public-v1", "purpose":"profile"})
        with self.assertRaisesRegex(ValueError, "schema/purpose"):
            check.check_experimental_evidence(self.root)

    def test_incomplete_staged_directory_is_not_silently_skipped(self):
        (self.root/"evidence/experimental/heldout").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "missing.*manifest"):
            check.check_experimental_evidence(self.root)

    def test_unreviewed_directory_is_not_silently_skipped(self):
        self.marker("unreviewed-results", {})
        with self.assertRaisesRegex(ValueError, "unexpected.*entry"):
            check.check_experimental_evidence(self.root)

    def test_semantic_verifier_failure_propagates(self):
        self.marker("reference-v3-original", {"schema":"dfly2-reference-public-manifest-v1"}, "manifest.json")
        with patch.object(check.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["validator"])):
            with self.assertRaises(subprocess.CalledProcessError): check.check_experimental_evidence(self.root)

    def test_absent_selection_permitted_before_staging(self):
        with patch.object(check.subprocess,"run") as run:
            self.assertFalse(check.check_experimental_selection(self.root))
            run.assert_not_called()

    def test_published_selection_always_requires_complete_final_evidence(self):
        receipt=self.root/"evidence/experimental-selection-v5.json"
        receipt.parent.mkdir();receipt.write_text("{}",encoding="utf-8")
        # No final directories exist. The release gate still requests strict
        # completion, leaving the selection verifier to reject missing results.
        with patch.object(check.subprocess,"run") as run:
            self.assertTrue(check.check_experimental_selection(self.root))
            self.assertEqual(run.call_args.args[0][-3:],["--verify",str(receipt),"--require-complete"])
            self.assertTrue(run.call_args.kwargs["check"])
        with patch.object(check.subprocess,"run",side_effect=subprocess.CalledProcessError(1,["validator"])):
            with self.assertRaises(subprocess.CalledProcessError):check.check_experimental_selection(self.root)


if __name__ == "__main__":
    unittest.main()

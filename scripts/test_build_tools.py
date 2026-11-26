"""CPU-only integrity/refusal tests; no Git downloads, Docker or GPU actions."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import prepare_source
import verify_kit


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_patch_tampering_is_rejected(self):
        p = self.root / "integration.patch"
        p.write_bytes(b"reviewed source changes\n")
        record = {"path": p.name, "sha256": prepare_source.digest(p)}
        self.assertEqual(prepare_source.verify_file(self.root, record), p.resolve())
        p.write_bytes(b"different changes\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            prepare_source.verify_file(self.root, record)

    def test_patch_path_cannot_escape_root(self):
        with self.assertRaisesRegex(ValueError, "escapes root"):
            prepare_source.checked_path(self.root, "../outside.patch")

    def kit(self):
        (self.root / "vision").mkdir()
        (self.root / "vision/gate_bias_vl.safetensors").write_bytes(b"old bias")
        (self.root / "kernel.cubin").write_bytes(b"pinned kernel")
        text = "".join(f"{verify_kit.sha(self.root / n)}  {n}\n" for n in
                       ("kernel.cubin", "vision/gate_bias_vl.safetensors"))
        (self.root / "MANIFEST").write_text(text)
        return verify_kit.sha(self.root / "MANIFEST")

    def test_manifest_itself_is_pinned(self):
        self.kit()
        with self.assertRaisesRegex(ValueError, "MANIFEST differs"):
            verify_kit.verify(self.root)

    def test_bias_exception_requires_explicit_flag_and_records_bytes(self):
        expected = self.kit()
        bias = self.root / "vision/gate_bias_vl.safetensors"
        bias.write_bytes(b"1017 target bias")
        with patch.object(verify_kit, "MANIFEST_SHA256", expected):
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_kit.verify(self.root)
            result = verify_kit.verify(self.root, allow_1017_bias=True)
        self.assertEqual(result["checked_files"], 2)
        self.assertEqual(result["generated_bias"]["sha256"], hashlib.sha256(b"1017 target bias").hexdigest())

    def test_bias_exception_never_permits_a_changed_kernel(self):
        expected = self.kit()
        (self.root / "kernel.cubin").write_bytes(b"modified kernel")
        with patch.object(verify_kit, "MANIFEST_SHA256", expected):
            with self.assertRaisesRegex(ValueError, "kernel.cubin"):
                verify_kit.verify(self.root, allow_1017_bias=True)

    def test_manifest_path_cannot_escape_kit(self):
        text = "0" * 64 + "  ../other-file\n"
        (self.root / "MANIFEST").write_text(text)
        with patch.object(verify_kit, "MANIFEST_SHA256", verify_kit.sha(self.root / "MANIFEST")):
            with self.assertRaisesRegex(ValueError, "escapes directory"):
                verify_kit.verify(self.root)


if __name__ == "__main__":
    unittest.main()

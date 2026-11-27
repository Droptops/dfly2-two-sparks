"""Small byte fixtures only; no actual model files or GPU operations."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import extract_head_cache as e
from proposal_replay import RankCache, digest


def source_fixture(path, rank=1):
    lo, hi = rank * 2, (rank + 1) * 2
    rows = [("unrelated|None", "uint8", [3], b"xyz"),
            ("embed.weight|None", "bfloat16", [4, 2], bytes(range(16))),
            (f"head|({lo}, {hi})|None|tr", "int16", [1, 1, 1], b"ab"),
            (f"head|({lo}, {hi})|None|suh", "float16", [2], b"cdef"),
            (f"head|({lo}, {hi})|None|svh", "float16", [2], b"ghij"),
            ("layers.0.unrelated|None", "uint8", [7], b"ignored")]
    body, index = bytearray(), []
    for key, dtype, shape, data in rows:
        index.append([key, dtype, shape, len(body), len(data)])
        body.extend(data)
    at = len(body)
    body.extend(json.dumps(index).encode())
    body.extend(struct.pack("<Q", at) + b"TFDSRK01")
    path.write_bytes(body)
    return rows


class ExtractTests(unittest.TestCase):
    def test_copies_only_selected_bytes_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "original.bin", root / "compact.bin"
            rows = source_fixture(source)
            before = source.read_bytes()
            report = e.extract_head_cache(source, output, 1, hidden=2, vocab=4, chunk_bytes=3)
            compact = RankCache(output, hidden=2, vocab=4)
            self.assertTrue(report["complete"])
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(len(compact.entries), 4)
            self.assertEqual(report["output_sha256"], digest(output.read_bytes()))
            self.assertEqual(report["source"]["index_sha256"], RankCache(source, hidden=2, vocab=4).provenance["index_sha256"])
            wanted = rows[1:5]
            self.assertEqual([entry["key"] for entry in report["entries"]], [row[0] for row in wanted])
            for (_, dtype, shape, raw), rec in zip(wanted, report["entries"]):
                self.assertEqual(rec["sha256"], digest(raw))
                self.assertEqual((rec["dtype"], rec["shape"]), (dtype, shape))
                self.assertEqual(compact.read_range(rec["output_offset"], rec["bytes"]), raw)
                self.assertEqual(before[rec["source_offset"]:rec["source_end"]], raw)
            self.assertEqual(json.loads((root / "compact.bin.json").read_text()), report)

    def test_existing_outputs_or_sidecars_and_wrong_rank_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source.bin", root / "out.bin"
            source_fixture(source)
            with self.assertRaisesRegex(ValueError, "missing required"):
                e.extract_head_cache(source, output, 0, hidden=2, vocab=4)
            self.assertFalse(output.exists())
            output.write_bytes(b"preserve")
            with self.assertRaisesRegex(ValueError, "cache already exists"):
                e.extract_head_cache(source, output, 1, hidden=2, vocab=4)
            self.assertEqual(output.read_bytes(), b"preserve")
            output2 = root / "second.bin"
            sidecar = root / "second.bin.json"
            sidecar.write_text("preserve")
            with self.assertRaisesRegex(ValueError, "sidecar already exists"):
                e.extract_head_cache(source, output2, 1, hidden=2, vocab=4)
            self.assertEqual(sidecar.read_text(), "preserve")
            self.assertFalse(output2.exists())

    def test_failure_preserves_incomplete_evidence_and_cannot_retry_over_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source.bin", root / "out.bin"
            source_fixture(source)
            with patch.object(e, "sha256", return_value="0" * 64):
                with self.assertRaisesRegex(ValueError, "written cache bytes differ"):
                    e.extract_head_cache(source, output, 1, hidden=2, vocab=4, chunk_bytes=5)
            self.assertTrue(output.exists())
            report = json.loads((root / "out.bin.json").read_text())
            self.assertFalse(report["complete"])
            self.assertIn("written cache bytes differ", report["error"])
            self.assertEqual(report["partial_output_bytes"], output.stat().st_size)
            with self.assertRaisesRegex(ValueError, "cache already exists"):
                e.extract_head_cache(source, output, 1, hidden=2, vocab=4)


if __name__ == "__main__":
    unittest.main()

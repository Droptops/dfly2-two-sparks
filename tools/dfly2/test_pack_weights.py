#!/usr/bin/env python3
"""CPU tests for the raw TP2 checkpoint boundary; no model/GPU dependencies."""
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import pack_weights as pack


class PartitionTests(unittest.TestCase):
    def test_row_halves_reconstruct_exact_original(self):
        # Deliberately include signed-nibble and high E8M0 bit patterns.
        for itemsize in (1, 2, 4):
            raw = bytes((i * 73 + 127) % 256 for i in range(6 * 10 * itemsize))
            halves = [pack.split_bytes(raw, (6, 10), itemsize, 0, rank) for rank in range(2)]
            self.assertEqual(halves[0] + halves[1], raw)

    def test_column_halves_reconstruct_each_row(self):
        for itemsize in (1, 2, 4):
            raw = bytes((i * 13 + 231) % 256 for i in range(7 * 10 * itemsize))
            halves = [pack.split_bytes(raw, (7, 10), itemsize, 1, rank) for rank in range(2)]
            half_row = 5 * itemsize
            rebuilt = b"".join(halves[0][r * half_row:(r + 1) * half_row] +
                               halves[1][r * half_row:(r + 1) * half_row] for r in range(7))
            self.assertEqual(rebuilt, raw)

    def test_down_code_and_scale_splits_keep_quantization_blocks_aligned(self):
        # 2 rows, 128 dequantized channels: packed bytes and independent E8M0 bytes.
        codes = bytes((i * 23 + 7) % 256 for i in range(2 * 64))
        scales = bytes((120, 121, 122, 123, 124, 125, 126, 127))
        lut = (0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.)

        def dequant(code, scale, n, k):
            return [[lut[(code[r * (k // 2) + c // 2] >> (4 * (c % 2))) & 15] *
                     2.0 ** (scale[r * (k // 32) + c // 32] - 127) for c in range(k)] for r in range(n)]

        full = dequant(codes, scales, 2, 128)
        ranks = [dequant(pack.split_bytes(codes, (2, 64), 1, 1, r),
                         pack.split_bytes(scales, (2, 4), 1, 1, r), 2, 64) for r in range(2)]
        self.assertEqual([ranks[0][r] + ranks[1][r] for r in range(2)], full)

    def test_refuses_uneven_and_short_tensors(self):
        with self.assertRaises(ValueError):
            pack.split_bytes(bytes(12), (3, 4), 1, 0, 0)
        with self.assertRaises(ValueError):
            pack.split_bytes(bytes(11), (3, 4), 1, 1, 0)
        with self.assertRaises(ValueError):
            pack.split_bytes(bytes(12), (3, 4), 1, 1, 2)

    def test_packed_expert_strides_and_dense_shared_shapes(self):
        schema = {name: (dtype, shape) for name, dtype, shape, _ in pack.pack_plan()}
        self.assertEqual(len(schema), 106)
        self.assertEqual(schema["model.layers.1.ffn.experts.w1.weight"], ("U8", (128, 1152, 2560)))
        self.assertEqual(schema["model.layers.1.ffn.experts.w2.scale"], ("U8", (128, 5120, 36)))
        self.assertEqual(schema["model.layers.1.ffn.shared_experts.w2.weight"], ("BF16", (5120, 1152)))
        self.assertEqual(schema["model.hidden_correction.down_proj.weight"], ("BF16", (5120, 2560)))
        self.assertEqual(schema["model.mtp_fusion.eh_proj.weight"], ("BF16", (5120, 10240)))
        # 128 experts stay on each rank; intermediate dimensions are split.
        self.assertEqual(sum(math.prod(s) * pack.DTYPE_SIZE[d] for d, s in schema.values()), 4529862920)


class InputValidationTests(unittest.TestCase):
    def write_shard(self, path, header, data):
        encoded = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)

    def test_rejects_overlapping_source_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "model.safetensors", "b": "model.safetensors"}}))
            self.write_shard(root / "model.safetensors", {
                "a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
                "b": {"dtype": "U8", "shape": [4], "data_offsets": [2, 6]},
            }, bytes(6))
            with self.assertRaisesRegex(ValueError, "overlapping"):
                pack.Checkpoint(root)

    def test_rejects_index_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "../elsewhere.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "unsafe"):
                pack.Checkpoint(root)

    def test_preserves_signed_code_bytes_and_checks_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "model.safetensors"}}))
            raw = bytes((0xFF, 0x80, 0x7F, 0x01))
            self.write_shard(root / "model.safetensors", {"a": {"dtype": "I8", "shape": [2, 2], "data_offsets": [0, 4]}}, raw)
            with patch.object(pack, "expected_tensors", return_value={"a": ("I8", (2, 2))}):
                cp = pack.Checkpoint(root)
                try:
                    self.assertEqual(cp.read_slice("a", 1, 0), bytes((0xFF, 0x7F)))
                    self.assertEqual(cp.read_slice("a", 1, 1), bytes((0x80, 0x01)))
                finally:
                    cp.close()
            with patch.object(pack, "expected_tensors", return_value={"a": ("U8", (2, 2))}):
                with self.assertRaises(ValueError):
                    pack.Checkpoint(root)

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "packed"
            destination.mkdir()
            marker = destination / "valuable"
            marker.write_text("keep")
            with self.assertRaises(FileExistsError):
                pack.pack(root / "missing-source", destination)
            self.assertEqual(marker.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()

"""Independent CPU fixtures for the TP rounding diagnostic, not parity evidence."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import correction_tp_diagnostic as d


class CaptureSelectionTests(unittest.TestCase):
    def test_mixed_normalization_metadata_survives_sorted_batch_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fused = root / "v3/capture-000000"
            double = root / "v4/capture-000000"
            for directory in (double, fused):
                directory.mkdir(parents=True)
                (directory / "capture.json").write_text("{}")
            records = {fused.resolve(): ({"native": {"norm_rounding": "fused", "rms_norm_eps": 9.99999968e-21}}, {}, {"x": "a" * 64}),
                       double.resolve(): ({"native": {"norm_rounding": "double", "rms_norm_eps": 9.99999968e-21}}, {}, {"x": "b" * 64})}
            with patch.object(d.reference, "capture_metadata", side_effect=lambda path: records[path]):
                result = d.prepare_captures(capture_root=root)
                single = d.prepare_captures(capture=double)
            self.assertEqual([relative.as_posix() for _, relative, _ in result], ["v3/capture-000000", "v4/capture-000000"])
            self.assertEqual([prepared[0]["native"]["norm_rounding"] for _, _, prepared in result], ["fused", "double"])
            self.assertIs(result[0][2], records[fused.resolve()])
            self.assertIs(result[1][2], records[double.resolve()])
            self.assertEqual(single[0][1], Path("."))
            self.assertEqual(single[0][2], records[double.resolve()])


class CorrectionTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("correction TP diagnostic tests require PyTorch")
        cls.t = torch

    def make_linear(self, scale=1):
        t = self.t
        module = t.nn.Linear(4, 4, bias=False, dtype=t.bfloat16)
        with t.no_grad():
            module.weight.copy_(t.tensor([[128., .25, -128., .25]], dtype=t.bfloat16).expand(4, -1) * scale)
        return module

    def make_correction(self):
        t = self.t
        outer = self
        class TinyCorrection(t.nn.Module):
            def __init__(self):
                super().__init__()
                self.down_proj = outer.make_linear()
                self.attention = t.nn.Module()
                self.attention.wo_b = outer.make_linear(.5)
                self.raise_after_projection = False

            def forward(self, hidden, previous, context):
                dense = self.down_proj(hidden)
                if self.raise_after_projection:
                    raise RuntimeError("deliberate correction failure")
                attention = self.attention.wo_b(previous)
                return hidden + (dense + attention)
        return TinyCorrection()

    def captured(self):
        t = self.t
        return {"normalized": t.ones(6, 4, dtype=t.bfloat16),
                "previous_embeddings": t.ones(6, 4, dtype=t.bfloat16),
                "positions": t.arange(20, 26, dtype=t.int64),
                "context": t.zeros(3, 3, 2, dtype=t.bfloat16),
                "valid": t.tensor([[1, 0, 1]] * 3, dtype=t.int32),
                # The synthetic native target is deliberately the down-only variant.
                "corrected": t.full((5, 4), 1.25, dtype=t.bfloat16)}

    def test_real_bf16_linear_exposes_half_accumulation_rounding(self):
        t = self.t
        linear = self.make_linear()
        inputs = t.ones(2, 4, dtype=t.bfloat16)
        original = linear(inputs)
        split = d.half_k_forward(linear, inputs, t)
        self.assertEqual(split.dtype, t.bfloat16)
        self.assertTrue(t.equal(original, t.full((2, 4), .5, dtype=t.bfloat16)))
        self.assertTrue(t.equal(split, t.zeros(2, 4, dtype=t.bfloat16)))
        self.assertTrue(t.equal(linear(inputs), original))

    def test_four_variants_select_only_the_requested_projections(self):
        t = self.t
        correction, captured = self.make_correction(), self.captured()
        before = {name: value.detach().clone() for name, value in correction.state_dict().items()}
        result = d.evaluate_variants(correction, captured, t)
        expected = {"original": ((), 1.75), "bf16_half_k_down_proj": (("down_proj",), 1.25),
                    "bf16_half_k_wo_b": (("attention.wo_b",), 1.5),
                    "bf16_half_k_both": (("down_proj", "attention.wo_b"), 1.)}
        self.assertEqual(list(d.VARIANTS), list(expected))
        self.assertEqual(list(result), list(expected))
        for name, (modules, value) in expected.items():
            self.assertEqual(d.VARIANTS[name], modules)
            self.assertTrue(t.equal(result[name]["output"], t.full((5, 4), value, dtype=t.bfloat16)), name)
            self.assertEqual(result[name]["comparison"]["passed"], name == "bf16_half_k_down_proj")
            self.assertEqual(result[name]["comparison_vs_original"]["passed"], name == "original")
        self.assertEqual(dict(correction.down_proj._forward_hooks), {})
        self.assertEqual(dict(correction.attention.wo_b._forward_hooks), {})
        for name, value in correction.state_dict().items():
            self.assertTrue(t.equal(value, before[name]))
        baseline_after = correction(captured["normalized"].unsqueeze(0), captured["previous_embeddings"].unsqueeze(0), {})
        self.assertTrue(t.equal(baseline_after[0, 1:], result["original"]["output"]))

    def test_existing_hooks_and_forward_method_survive_success_and_exception(self):
        t = self.t
        correction, captured = self.make_correction(), self.captured()
        seen = []
        handle = correction.down_proj.register_forward_hook(lambda *args: seen.append("original hook"))
        before_hooks = dict(correction.down_proj._forward_hooks)
        original_function = correction.down_proj.forward.__func__
        original_has_instance_forward = "forward" in correction.down_proj.__dict__
        try:
            with d.half_k_hooks(correction, ("down_proj", "attention.wo_b"), t):
                output = correction(captured["normalized"].unsqueeze(0), captured["previous_embeddings"].unsqueeze(0), {})
                self.assertTrue(t.equal(output, t.ones(1, 6, 4, dtype=t.bfloat16)))
            self.assertEqual(dict(correction.down_proj._forward_hooks), before_hooks)
            correction.raise_after_projection = True
            with self.assertRaisesRegex(RuntimeError, "deliberate correction failure"):
                with d.half_k_hooks(correction, ("down_proj", "attention.wo_b"), t):
                    correction(captured["normalized"].unsqueeze(0), captured["previous_embeddings"].unsqueeze(0), {})
            self.assertEqual(dict(correction.down_proj._forward_hooks), before_hooks)
            self.assertEqual(dict(correction.attention.wo_b._forward_hooks), {})
            self.assertIs(correction.down_proj.forward.__func__, original_function)
            self.assertEqual("forward" in correction.down_proj.__dict__, original_has_instance_forward)
            self.assertEqual(seen, ["original hook", "original hook"])
        finally:
            handle.remove()

    def test_invalid_module_setup_leaves_no_partial_hooks(self):
        t = self.t
        correction = self.make_correction()
        correction.attention.wo_b = t.nn.Identity()
        with self.assertRaises((ValueError, TypeError)):
            with d.half_k_hooks(correction, ("down_proj", "attention.wo_b"), t):
                pass
        self.assertEqual(dict(correction.down_proj._forward_hooks), {})
        self.assertEqual(dict(correction.attention.wo_b._forward_hooks), {})

    def test_callback_failure_preserves_completed_variants_and_restores_math(self):
        t = self.t
        correction, captured = self.make_correction(), self.captured()
        observed = {}
        def preserve_then_fail(name, result):
            observed[name] = {"output": result["output"].detach().clone(),
                              "comparison": dict(result["comparison"])}
            if name == "bf16_half_k_down_proj":
                raise RuntimeError("deliberate evidence callback failure")
        with self.assertRaisesRegex(RuntimeError, "deliberate evidence callback failure"):
            d.evaluate_variants(correction, captured, t, on_variant=preserve_then_fail)
        self.assertEqual(list(observed), ["original", "bf16_half_k_down_proj"])
        self.assertTrue(t.equal(observed["original"]["output"], t.full((5, 4), 1.75, dtype=t.bfloat16)))
        self.assertTrue(t.equal(observed["bf16_half_k_down_proj"]["output"], t.full((5, 4), 1.25, dtype=t.bfloat16)))
        self.assertFalse(observed["original"]["comparison"]["passed"])
        self.assertTrue(observed["bf16_half_k_down_proj"]["comparison"]["passed"])
        self.assertEqual(dict(correction.down_proj._forward_hooks), {})
        self.assertEqual(dict(correction.attention.wo_b._forward_hooks), {})
        baseline_after = correction(captured["normalized"].unsqueeze(0), captured["previous_embeddings"].unsqueeze(0), {})
        self.assertTrue(t.equal(baseline_after[0, 1:], observed["original"]["output"]))

    def test_half_k_rejects_bias_odd_width_and_non_bf16_operands(self):
        t = self.t
        cases = [(t.nn.Linear(4, 2, bias=True, dtype=t.bfloat16), t.ones(1, 4, dtype=t.bfloat16)),
                 (t.nn.Linear(3, 2, bias=False, dtype=t.bfloat16), t.ones(1, 3, dtype=t.bfloat16)),
                 (t.nn.Linear(4, 2, bias=False, dtype=t.float32), t.ones(1, 4, dtype=t.bfloat16)),
                 (t.nn.Linear(4, 2, bias=False, dtype=t.bfloat16), t.ones(1, 4, dtype=t.float32))]
        for module, inputs in cases:
            with self.subTest(shape=tuple(module.weight.shape), bias=module.bias is not None,
                              weight_dtype=module.weight.dtype, input_dtype=inputs.dtype):
                with self.assertRaises((ValueError, TypeError)):
                    d.half_k_forward(module, inputs, t)


if __name__ == "__main__":
    unittest.main()

"""CPU tests for adapters and original-reference wiring; not GPU parity evidence."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import reference_replay as r

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = Path(os.environ["DFLY_REFERENCE_UPSTREAM"]) if os.environ.get("DFLY_REFERENCE_UPSTREAM") else None
PUBLIC = Path(os.environ.get("DFLY_REFERENCE_PUBLIC", ROOT))
CHECKPOINT = Path(os.environ["DFLY_REFERENCE_CHECKPOINT"]) if os.environ.get("DFLY_REFERENCE_CHECKPOINT") else None
CHECKPOINT_CONFIG = Path(os.environ["DFLY_REFERENCE_CHECKPOINT_CONFIG"]) if os.environ.get("DFLY_REFERENCE_CHECKPOINT_CONFIG") else (CHECKPOINT / "config.json" if CHECKPOINT else None)
_REFERENCE_READY = False


class DiscoveryTests(unittest.TestCase):
    def test_batch_rejects_empty_and_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "no completed"):
                r.discover_captures(capture_root=root)
            one = root / "capture-000000"
            one.mkdir()
            (one / "capture.started").touch()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                r.discover_captures(capture_root=root)

    def test_batch_sorting_preserves_nested_names_and_single_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("long/capture-000002", "chat/capture-000000", "long/capture-000001"):
                (root / name).mkdir(parents=True)
                (root / name / "capture.json").write_text("{}")
            cases = r.discover_captures(capture_root=root)
            self.assertEqual([p.as_posix() for _, p in cases], ["chat/capture-000000", "long/capture-000001", "long/capture-000002"])
            self.assertEqual(r.discover_captures(capture=root / "chat/capture-000000")[0][1], Path("."))
            with self.assertRaisesRegex(ValueError, "choose one"):
                r.discover_captures(capture=root, capture_root=root)


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("reference adapter tests require PyTorch")
        cls.torch = torch

    def ensure_reference(self):
        global _REFERENCE_READY
        if UPSTREAM is None:
            self.skipTest("set DFLY_REFERENCE_UPSTREAM to the pinned AngelSpec checkout")
        if not _REFERENCE_READY:
            r.verify_upstream(UPSTREAM)
            r.import_reference_namespaces(UPSTREAM)
            _REFERENCE_READY = True

    def test_fp4_nibble_order_and_scale(self):
        t = self.torch
        codes = t.arange(256, dtype=t.uint8).view(1, 256)
        scales = t.full((1, 16), 127, dtype=t.uint8)
        out = r.decode_fp4(codes, scales, t).float()
        expected = [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6]
        for byte in range(256):
            self.assertEqual(out[0, 2 * byte], expected[byte % 16])
            self.assertEqual(out[0, 2 * byte + 1], expected[byte // 16])
        doubled = r.decode_fp4(codes, scales + 1, t).float()
        self.assertTrue(t.equal(doubled, out * 2))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            r.decode_fp4(codes, t.full_like(scales, 255), t)

    def test_fp4_signed_zero_preserved(self):
        t = self.torch
        out = r.decode_fp4(t.full((1, 16), 0x80, dtype=t.uint8), t.full((1, 1), 127, dtype=t.uint8), t)
        self.assertEqual(out.view(t.int16)[0, 0], 0)
        self.assertEqual(out.view(t.int16)[0, 1], -32768)

    def test_comparison_rejects_nonfinite_and_large_drift(self):
        t = self.torch
        kwargs = dict(torch=t, rtol=.02, atol=.02, relative_rms=.01)
        a = t.ones(16)
        self.assertTrue(r.compare(a, a, **kwargs)["passed"])
        self.assertFalse(r.compare(a * 1.2, a, **kwargs)["passed"])
        self.assertFalse(r.compare(a * float("nan"), a, **kwargs)["finite"])

    def test_serving_to_original_config_and_parameter_shapes(self):
        self.ensure_reference()
        if CHECKPOINT_CONFIG is None:
            self.skipTest("set DFLY_REFERENCE_CHECKPOINT or DFLY_REFERENCE_CHECKPOINT_CONFIG")
        t = self.torch
        from angelspec.models.draft.dfly2 import DFly2DraftModel
        served = json.loads(CHECKPOINT_CONFIG.read_text())
        cfg = r.build_config(served, UPSTREAM, t)
        self.assertEqual(cfg.rms_norm_eps, 1e-20)
        self.assertEqual(cfg.hc_eps, 1e-6)
        self.assertEqual(cfg.num_hidden_layers, 3)
        self.assertEqual(cfg.num_experts, 128)
        self.assertFalse(cfg.enable_confidence_head)
        with t.device("meta"):
            model = DFly2DraftModel(cfg)
        params = dict(model.named_parameters())
        schema = r.load_pack_module(PUBLIC).expected_tensors()
        mapped = set()
        for src, (_, shape) in schema.items():
            if ".ffn.experts." in src or src.endswith(".ffn.gate.bias_vl"):
                continue
            dst = src.removeprefix("model.")
            if dst.startswith("layers."):
                dst = dst.replace(".attn.", ".self_attn.")
                dst = dst.replace(".attn_norm.", ".input_layernorm.").replace(".ffn_norm.", ".post_attention_layernorm.")
                dst = dst.replace(".ffn.gate.weight", ".mlp.router.gate.weight").replace(".ffn.gate.bias", ".mlp.expert_bias")
                dst = dst.replace(".ffn.shared_experts.w1.", ".mlp.shared_mlp.gate_proj.")
                dst = dst.replace(".ffn.shared_experts.w3.", ".mlp.shared_mlp.up_proj.")
                dst = dst.replace(".ffn.shared_experts.w2.", ".mlp.shared_mlp.down_proj.")
            self.assertIn(dst, params)
            self.assertEqual(tuple(params[dst].shape), shape, dst)
            mapped.add(dst)
        for layer in range(3):
            for w, name in (("w1", "gate"), ("w3", "up"), ("w2", "down")):
                shape = schema[f"model.layers.{layer}.ffn.experts.0.{w}.weight"][1]
                dst = f"layers.{layer}.mlp.experts_{name}_proj.weight"
                self.assertEqual(tuple(params[dst].shape), (128, shape[1] * 2, shape[0]))
                mapped.add(dst)
        self.assertEqual(mapped | {"embed_tokens.weight"}, set(params))

    def test_original_forward_and_causal_correction(self):
        self.ensure_reference()
        import os
        os.environ["ANGELSPEC_MTP_MOE_REFERENCE"] = "1"
        t = self.torch
        from angelspec.models.draft.dfly2 import DFly2Config, DFly2DraftModel
        t.manual_seed(13)
        cfg = DFly2Config(hidden_size=16, vocab_size=32, num_hidden_layers=1, model_arch="dflash",
            num_attention_heads=2, num_key_value_heads=1, head_dim=512, qk_rope_head_dim=64,
            q_lora_rank=16, o_lora_rank=8, o_groups=2, attention_type="dsv41_mla",
            draft_mlp_type="moe", num_experts=4, num_experts_per_tok=2,
            num_shared_experts=1, moe_intermediate_size=16, moe_router_score_func="sqrtsoftplus",
            moe_router_enable_expert_bias=True, moe_aux_free_bias_update=False,
            route_norm=True, router_scaling_factor=1.5, swiglu_limit=10,
            shared_expert_swiglu_clamp=True, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
            rms_norm_eps=1e-20, target_kv_layers=[28], target_index_layer=[28],
            target_layer_ids=[27], num_target_layers=1, target_num_hidden_layers=40, target_hidden_size=16,
            target_hc_init_layers=[28],
            target_index_topk=512, context_local_window=128, dsa_backend="torch",
            mtp_fusion="slot0", enable_hidden_correction=True, attentive_correction=True,
            attentive_correction_block_keys=True, attentive_correction_block_mode="gated",
            attentive_correction_block_kv_in="hidden_embed", enable_confidence_head=False,
            hidden_correction_intermediate_size=16, rope_theta=160000,
            original_seq_len=65536, rope_factor=16)
        model = DFly2DraftModel(cfg).eval()
        with t.no_grad():
            for name, p in model.named_parameters():
                if "norm.weight" in name:
                    p.fill_(1)
                else:
                    p.normal_(0, .05)
        captured = {"positions": t.arange(1200, 1206),
                    "context": t.randn(1, 640, 512) * .1,
                    "valid": t.cat((t.ones(600), t.zeros(40))).view(1, 640).int(),
                    "anchor_mask_embeddings": t.randn(2, 16),
                    "target_hidden": t.randn(16),
                    "previous_embeddings": t.randn(6, 16),
                    "normalized": t.randn(6, 16)}
        with t.no_grad():
            first = r.replay(model, captured, t)
            changed = dict(captured)
            changed["previous_embeddings"] = captured["previous_embeddings"].clone()
            changed["previous_embeddings"][5] += 9
            second = r.replay(model, changed, t)
        self.assertEqual(tuple(first["normalized"].shape), (6, 16))
        self.assertTrue(t.isfinite(first["corrected"]).all())
        self.assertTrue(t.equal(first["corrected"][:4], second["corrected"][:4]))
        self.assertFalse(t.equal(first["corrected"][4], second["corrected"][4]))


if __name__ == "__main__":
    unittest.main()

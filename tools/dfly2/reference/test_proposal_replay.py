"""CPU tests of diagnostic plumbing; these are not measured proposal parity."""
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import proposal_replay as p


def write_cache(path, rows, *, footer_magic=b"TFDSRK01"):
    body, index = bytearray(), []
    for key, dtype, shape, data in rows:
        index.append([key, dtype, shape, len(body), len(data)])
        body.extend(data)
    at = len(body)
    body.extend(json.dumps(index).encode())
    body.extend(struct.pack("<Q", at) + footer_magic)
    path.write_bytes(body)


class RankCacheTests(unittest.TestCase):
    def test_compact_sidecar_is_verified_and_links_original_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.bin"
            write_cache(path, [("embed.weight|None", "bfloat16", [4, 2], bytes(16))])
            original = p.RankCache(path, hidden=2, vocab=4)
            manifest = {"schema": "dfly2-head-cache-extract-v1", "complete": True,
                        "output_bytes": path.stat().st_size, "output_sha256": p.reference.sha256(path),
                        "output_index_sha256": original.provenance["index_sha256"],
                        "output_footer_sha256": original.provenance["footer_sha256"],
                        "source": {"file": "original.bin", "index_sha256": "a" * 64},
                        "entries": [{"key": "embed.weight|None", "source_offset": 100, "bytes": 16}]}
            sidecar = path.with_name(path.name + ".json")
            sidecar.write_text(json.dumps(manifest))
            cache = p.RankCache(path, hidden=2, vocab=4)
            evidence = cache.provenance["extraction"]
            self.assertEqual(evidence["sidecar_sha256"], p.digest(sidecar.read_bytes()))
            self.assertEqual(evidence["manifest"]["source"], manifest["source"])
            self.assertEqual(evidence["manifest"]["entries"], manifest["entries"])
            for field, value, message in (("complete", False, "incomplete"),
                                           ("output_bytes", 1, "output bytes differ"),
                                           ("output_index_sha256", "b" * 64, "index hash differs"),
                                           ("output_footer_sha256", "c" * 64, "footer hash differs"),
                                           ("output_sha256", "d" * 64, "output hash differs")):
                with self.subTest(field=field):
                    sidecar.write_text(json.dumps({**manifest, field: value}))
                    with self.assertRaisesRegex(ValueError, message):
                        p.RankCache(path, hidden=2, vocab=4)

    def test_original_or_unknown_sidecar_never_hashes_whole_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "original.bin"
            write_cache(path, [("embed.weight|None", "bfloat16", [4, 2], bytes(16))])
            with patch.object(p.reference, "sha256", side_effect=AssertionError("unexpected full-cache hash")):
                self.assertNotIn("extraction", p.RankCache(path, hidden=2, vocab=4).provenance)
                path.with_name(path.name + ".json").write_text(json.dumps({"schema": "different-format", "value": 1}))
                self.assertNotIn("extraction", p.RankCache(path, hidden=2, vocab=4).provenance)

    def test_probe_timeout_requires_positive_finite_value(self):
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.assertRaisesRegex(ValueError, "positive and finite"):
                p.ProbeConfig(*(Path("fixture") for _ in range(5)), timeout=value)

    def test_selected_rows_and_head_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rank0of2-fixture.bin"
            embed = bytes(range(16))
            write_cache(path, [("embed.weight|None", "bfloat16", [4, 2], embed),
                               ("head|(0, 2)|None|tr", "int16", [1, 1, 1], b"ab"),
                               ("head|(0, 2)|None|suh", "float16", [2], b"cdef"),
                               ("head|(0, 2)|None|svh", "float16", [2], b"ghij")])
            cache = p.RankCache(path, hidden=2, vocab=4)
            self.assertEqual(cache.embedding_bytes(2), embed[8:12])
            self.assertEqual(cache.provenance["embedding_rows"], {"2": p.digest(embed[8:12])})
            self.assertEqual(len(cache.hash_head(0)["head_entries"]), 3)
            self.assertEqual(cache.provenance["head_entries"]["head|(0, 2)|None|tr"]["sha256"], p.digest(b"ab"))
            with self.assertRaisesRegex(ValueError, "outside vocabulary"):
                cache.embedding_bytes(4)
            with self.assertRaisesRegex(ValueError, "missing"):
                cache.hash_head(1)

    def test_invalid_footer_and_index_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.bin"
            write_cache(path, [("embed.weight|None", "bfloat16", [4, 2], bytes(16))], footer_magic=b"badmagic")
            with self.assertRaisesRegex(ValueError, "magic"):
                p.RankCache(path, hidden=2, vocab=4)
            write_cache(path, [("embed.weight|None", "bfloat16", [4, 2], bytes(14))])
            with self.assertRaisesRegex(ValueError, "shape or bounds"):
                p.RankCache(path, hidden=2, vocab=4)
            body = bytes(18)
            index = json.dumps([["embed.weight|None", "bfloat16", [4, 2], 2, 16]]).encode()
            path.write_bytes(body + index + struct.pack("<Q", 18) + b"TFDSRK01")
            with self.assertRaisesRegex(ValueError, "overlap or leave gaps"):
                p.RankCache(path, hidden=2, vocab=4)


class ProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("proposal plumbing tests require PyTorch")
        cls.t = torch

    def test_top_two_uses_lower_token_on_ties_and_rejects_nan(self):
        t = self.t
        result = p.summarize_logits(t.tensor([[2., 3., 3., 1.], [1., -2., -2., 0.]]), t)
        self.assertEqual(result[0], {"token": 1, "score": 3., "runnerup_token": 2,
                                     "runnerup_score": 3., "margin": 0.})
        self.assertEqual(result[1]["token"], 0)
        self.assertEqual(result[1]["runnerup_token"], 3)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            p.summarize_logits(t.tensor([[float("nan"), 0.]]), t)

    def test_free_running_uses_chosen_predecessors_and_keeps_future_zero(self):
        t = self.t
        seen = []
        class Model:
            def hidden_correction(self, normalized, previous, context):
                seen.append(previous.clone())
                self_test.assertEqual(tuple(context["indices"].shape), (1, 6, 3))
                self_test.assertEqual(context["indices"][0, 0].tolist(), [0, -1, 2])
                return normalized + previous
        self_test = self
        captured = {"positions": t.arange(8, 14), "valid": t.tensor([[1, 0, 1]]),
                    "context": t.ones(1, 3, 2), "anchor_mask_embeddings": t.tensor([[7., 8.], [1., 1.]]),
                    # These native predecessors must never influence free-running state.
                    "previous_embeddings": t.full((6, 2), 999.)}
        normalized = t.zeros(6, 2)
        scores = []
        def score(slot, vector):
            scores.append(vector.clone())
            return {"results": [{"token": slot + 10}]}
        result = p.run_free_running(Model(), captured, normalized,
                                    lambda token: t.tensor([float(token), -float(token)]), score, t)
        self.assertEqual(result["tokens"], [11, 12, 13, 14, 15])
        self.assertTrue(t.equal(scores[0], t.tensor([7., 8.])))
        self.assertTrue(t.equal(scores[1], t.tensor([11., -11.])))
        for slot, previous in enumerate(seen, start=1):
            self.assertTrue(t.equal(previous[0, 0], t.tensor([7., 8.])))
            self.assertEqual(int(t.count_nonzero(previous[0, slot + 1:])), 0)
        self.assertTrue(t.equal(result["previous_embeddings"][5], t.tensor([14., -14.])))

    def test_native_probe_summary_must_agree_with_logits(self):
        t = self.t
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root / "input.bin"
            inp.write_bytes(bytes(p.HIDDEN * 2))
            config = p.ProbeConfig(root / "probe", root / "target", root / "rank0", root / "rank1", root / "cubin")
            config.grouped_cubin.write_bytes(b"synthetic-cubin")
            def fake_run(cmd, **kwargs):
                out = Path(cmd[-1])
                out.mkdir()
                logits = t.zeros(1, p.VOCAB)
                logits[0, 17] = 2
                (out / "logits.bin").write_bytes(logits.numpy().tobytes())
                (out / "top1.json").write_text(json.dumps({"schema": "dfly2-head-probe-v1", "rows": 1,
                    "m_per_launch": 1, "input_sha256": p.reference.sha256(inp),
                    "cubin_sha256": p.reference.sha256(config.grouped_cubin),
                    "logits_sha256": p.reference.sha256(out / "logits.bin"),
                    "vocab": p.VOCAB, "results": [{"token": 17, "score": 2., "runnerup_token": 0,
                                                  "runnerup_score": 0., "margin": 2.}]}))
                return SimpleNamespace(returncode=0)
            with patch.object(p.subprocess, "run", side_effect=fake_run):
                _, report = p.score(config, inp, root / "accepted", 1, t)
                self.assertEqual(report["results"][0]["token"], 17)
            def wrong_summary(cmd, **kwargs):
                result = fake_run(cmd, **kwargs)
                out = Path(cmd[-1]) / "top1.json"
                data = json.loads(out.read_text())
                data["results"][0]["token"] = 18
                out.write_text(json.dumps(data))
                return result
            with patch.object(p.subprocess, "run", side_effect=wrong_summary):
                with self.assertRaisesRegex(ValueError, "token differs"):
                    p.score(config, inp, root / "rejected", 1, t)
            self.assertTrue((root / "rejected/logits.bin").exists())
            self.assertTrue((root / "rejected.subprocess.log").exists())
            def wrong_hash(cmd, **kwargs):
                result = fake_run(cmd, **kwargs)
                out = Path(cmd[-1]) / "top1.json"
                data = json.loads(out.read_text())
                data["logits_sha256"] = "0" * 64
                out.write_text(json.dumps(data))
                return result
            with patch.object(p.subprocess, "run", side_effect=wrong_hash):
                with self.assertRaisesRegex(ValueError, "logits_sha256 differs"):
                    p.score(config, inp, root / "rejected-hash", 1, t)

    def test_failed_native_gate_preserves_report_and_never_corrects(self):
        t = self.t
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured = {"corrected": t.zeros(5, p.HIDDEN), "proposals": t.arange(5, dtype=t.int64),
                        "anchor_mask_embeddings": t.zeros(2, p.HIDDEN, dtype=t.bfloat16),
                        "previous_embeddings": t.zeros(6, p.HIDDEN, dtype=t.bfloat16)}
            config = p.ProbeConfig(root / "probe", root / "target", root / "rank0", root / "rank1", root / "cubin")
            class FakeCache:
                def __init__(self, *args):
                    self.provenance = {}
                def hash_head(self, rank):
                    pass
                def embedding_bytes(self, token):
                    return bytes(p.HIDDEN * 2)
            class Model:
                config = SimpleNamespace(mask_token_id=127)
                def hidden_correction(self, *args):
                    raise AssertionError("correction called after failed native gate")
            gate = {"results": [{"token": 99} for _ in range(5)]}
            with patch.object(p, "RankCache", FakeCache), patch.object(p.reference, "sha256", return_value="0" * 64), \
                 patch.object(p.reference, "capture_metadata", return_value=({"anchor_token": 4}, {}, {})), \
                 patch.object(p, "score", return_value=(None, gate)):
                with self.assertRaisesRegex(ValueError, "gate failed"):
                    p.run_proposal_parity(Model(), captured, t.zeros(6, p.HIDDEN), root / "capture",
                                          root / "result", config, t)
            report = json.loads((root / "result/report.json").read_text())
            self.assertFalse(report["head_gate_passed"])
            self.assertFalse(report["complete"])
            self.assertEqual(report["steps_completed"], [])
            self.assertIn("gate failed", report["error"])


if __name__ == "__main__":
    unittest.main()

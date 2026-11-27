#!/usr/bin/env python3
"""Replay native captures through pinned original AngelSpec using released weights.

This is a diagnostic reference, not an inference backend or speed benchmark.
It does not download code/weights, load the target, or start/stop services.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import types

UPSTREAM_COMMIT = "e02ea36e65d44e6e165c4ba73c139d500e12f2d9"
SCHEMA = "dfly2-reference-capture-v1"
CAPTURE_TENSORS = {
    "target_hidden": ("BF16", (5120,)),
    "anchor_mask_embeddings": ("BF16", (2, 5120)),
    "positions": ("I64", (6,)),
    "context": ("BF16", (3, 640, 512)),
    "valid": ("I32", (3, 640)),
    "initial_block": ("BF16", (6, 4, 5120)),
    "normalized": ("BF16", (6, 5120)),
    "previous_embeddings": ("BF16", (6, 5120)),
    "corrected": ("BF16", (5, 5120)),
    "proposals": ("I64", (5,)),
}
DTYPE_BYTES = {"BF16": 2, "F32": 4, "I64": 8, "I32": 4, "I8": 1, "U8": 1, "F8_E8M0": 1}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_upstream(path):
    path = Path(path).resolve()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    require(actual == UPSTREAM_COMMIT, f"expected AngelSpec {UPSTREAM_COMMIT}, got {actual}")
    changes = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--", "angelspec"], cwd=path, text=True)
    require(not changes.strip(), "AngelSpec source has modifications or untracked files")
    sys.path.insert(0, str(path))
    return actual


def import_reference_namespaces(path):
    """Skip package aggregators that import unrelated trainer dependencies.

    Original forward modules still load unmodified from the pinned checkout.
    This does not replace any mathematical operation or module implementation.
    """
    for name in ("angelspec", "angelspec.models", "angelspec.models.draft", "angelspec.models.ops", "angelspec.utils"):
        require(name not in sys.modules, f"reference namespace already imported: {name}")
        module = types.ModuleType(name)
        module.__path__ = [str(Path(path).resolve().joinpath(*name.split(".")))]
        module.__package__ = name
        sys.modules[name] = module


def capture_metadata(directory):
    directory = Path(directory).resolve()
    m = json.loads((directory / "capture.json").read_text())
    require(m.get("schema") == SCHEMA, "wrong capture schema")
    require(type(m.get("q0")) is int and m["q0"] >= 1, "invalid q0")
    require(type(m.get("anchor_token")) is int and 0 <= m["anchor_token"] < 129280, "invalid anchor")
    require(m.get("steps") == 5 and m.get("world_size") == 2, "capture must be a full TP2 B6 pass")
    native = m.get("native", {})
    for field in ("binary_sha256", "checkpoint_manifest_sha256", "requests_sha256"):
        require(bool(re.fullmatch(r"[0-9a-f]{64}", native.get(field, ""))), f"missing native {field}")
    require(isinstance(native.get("rms_norm_eps"), (int, float)) and math.isclose(native["rms_norm_eps"], 1e-20, rel_tol=1e-6, abs_tol=0), "unexpected deployed normalization epsilon")
    tensors = dict(CAPTURE_TENSORS)
    if (directory / "layers.bin").exists():
        tensors["layers"] = ("BF16", (3, 6, 5120))
    if (directory / "layers_residual.bin").exists():
        tensors["layers_residual"] = ("BF16", (3, 6, 4, 5120))
    hashes = {}
    for name, (dtype, shape) in tensors.items():
        p = directory / (name + ".bin")
        require(p.is_file() and not p.is_symlink(), f"missing or linked tensor {name}")
        require(p.stat().st_size == math.prod(shape) * DTYPE_BYTES[dtype], f"wrong tensor bytes: {name}")
        hashes[name] = sha256(p)
        recorded = m.get("tensors", {}).get(name)
        if isinstance(recorded, dict):
            require(recorded.get("file", name + ".bin") == name + ".bin", "unexpected tensor filename")
            require(recorded.get("shape", list(shape)) == list(shape), "metadata tensor shape mismatch")
            require(recorded.get("dtype", dtype) == dtype, "metadata tensor dtype mismatch")
            recorded = recorded.get("sha256")
        if recorded is not None:
            require(recorded == hashes[name], f"capture hash mismatch: {name}")
    return m, tensors, hashes


def read_tensor(path, dtype, shape, torch, device="cpu"):
    dt = {"BF16": torch.bfloat16, "F32": torch.float32, "I64": torch.int64,
          "I32": torch.int32, "I8": torch.int8, "U8": torch.uint8, "F8_E8M0": torch.uint8}[dtype]
    raw = bytearray(Path(path).read_bytes())
    require(len(raw) == math.prod(shape) * DTYPE_BYTES[dtype], "tensor byte count mismatch")
    return torch.frombuffer(raw, dtype=dt).reshape(shape).clone().to(device)


def write_bf16(path, tensor):
    import torch
    data = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()
    Path(path).write_bytes(data)


def decode_fp4(codes, scales, torch):
    """Exact E2M1 values times E8M0 powers of two; low nibble first.

    No source values are requantized. Nonfinite E8M0 is rejected.
    """
    require(codes.ndim == 2 and scales.shape == (codes.shape[0], codes.shape[1] // 16), "invalid FP4 layout")
    require(not bool((scales == 255).any()), "nonfinite expert E8M0 scale")
    codes = codes.to(torch.uint8)
    nibbles = torch.stack((codes & 15, codes >> 4), dim=-1).flatten(-2).long()
    table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32, device=codes.device)
    powers = torch.exp2(scales.to(torch.float32) - 127).repeat_interleave(32, dim=-1)
    decoded = table[nibbles] * powers
    require(bool(torch.isfinite(decoded).all()), "expert dequantization produced nonfinite values")
    bf = decoded.to(torch.bfloat16)
    require(bool(torch.isfinite(bf).all()), "expert values overflow BF16")
    require(torch.equal(decoded, bf.float()), "expert values are not exactly representable in BF16")
    return bf


def load_pack_module(public_repo):
    path = Path(public_repo) / "tools/dfly2/pack_weights.py"
    spec = importlib.util.spec_from_file_location("dfly_pack_schema", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_tensor(cp, name, torch):
    dtype, shape, mm, lo, hi = cp.tensors[name]
    dt = {"BF16": torch.bfloat16, "F32": torch.float32, "I8": torch.uint8,
          "U8": torch.uint8, "F8_E8M0": torch.uint8}[dtype]
    return torch.frombuffer(bytearray(mm[lo:hi]), dtype=dt).reshape(shape).clone()


def build_config(served, upstream, torch):
    from angelspec.models.draft.dfly2 import DFly2Config
    c = json.loads((Path(upstream) / "angelspec/config/dfly2_dsv41_flash_draft_config.json").read_text())
    t, d = served["text_config"], served["dfly_config"]
    for key in ("hidden_size", "vocab_size", "num_attention_heads", "head_dim", "qk_rope_head_dim", "q_lora_rank", "o_lora_rank", "o_groups", "moe_intermediate_size", "hc_mult", "hc_sinkhorn_iters", "hc_eps", "rms_norm_eps", "swiglu_limit"):
        c[key] = t[key]
    c.update(num_hidden_layers=len(d["target_kv_layers"]), target_kv_layers=d["target_kv_layers"],
             target_index_layer=d["target_index_layers"], target_index_topk=d["index_topk"],
             mask_token_id=d["mask_token_id"], mtp_fusion=d["mtp_fusion"],
             hidden_correction_intermediate_size=d["hidden_correction"]["intermediate_size"],
             attentive_correction=True, attentive_correction_block_keys=True,
             attentive_correction_block_mode="gated", attentive_correction_block_kv_in="hidden_embed",
             dsa_backend="torch", attentive_correction_backend=None,
             enable_confidence_head=False, markov_rank=0, moe_aux_free_bias_update=False,
             rope_theta=d["rope"]["theta"], original_seq_len=d["rope"]["original_seq_len"],
             rope_factor=d["rope"]["factor"], beta_fast=d["rope"]["beta_fast"], beta_slow=d["rope"]["beta_slow"])
    return DFly2Config(**c)


def load_reference(checkpoint, public_repo, upstream, device, torch):
    """Load released serving-layout weights into original AngelSpec modules."""
    from angelspec.models.draft.dfly2 import DFly2DraftModel
    pack = load_pack_module(public_repo)
    directory = Path(checkpoint)
    served = json.loads((directory / "config.json").read_text())
    pack.validate_config(served)
    provenance = json.loads((Path(public_repo) / "evidence/draft-pack-manifest.json").read_text())
    require(sha256(directory / "config.json") == provenance["source_config_sha256"], "released checkpoint config hash differs")
    for rec in provenance["source_files"]:
        p = directory / rec["file"]
        require(p.stat().st_size == rec["bytes"] and sha256(p) == rec["sha256"], f"released source shard differs: {rec['file']}")
    config = build_config(served, upstream, torch)
    with torch.device("meta"):
        model = DFly2DraftModel(config)
    # Every call supplies captured embeddings. This prevents an uninitialized target
    # embedding from silently participating without loading another giant model.
    class CapturedEmbeddingOnly(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise RuntimeError("reference replay must supply captured target embeddings")
    model.embed_tokens = CapturedEmbeddingOnly()
    expected = dict(model.named_parameters())
    loaded, used, ignored = set(), set(), []

    def assign(name, value):
        require(name in expected, f"unexpected reference parameter {name}")
        require(tuple(value.shape) == tuple(expected[name].shape), f"shape mismatch for {name}")
        parent, _, leaf = name.rpartition(".")
        module = model.get_submodule(parent) if parent else model
        setattr(module, leaf, torch.nn.Parameter(value.to(device), requires_grad=False))
        loaded.add(name)

    cp = pack.Checkpoint(directory)
    try:
        for src in sorted(cp.tensors):
            if ".ffn.experts." in src:
                continue
            if src.endswith(".ffn.gate.bias_vl"):
                original = checkpoint_tensor(cp, src.removesuffix("_vl"), torch)
                vision_bias = checkpoint_tensor(cp, src, torch)
                # TensorFold weights.zig/prompt.zig define this as the image-span
                # router bias. Both this replay and the native DFly path are text-only.
                ignored.append({"tensor": src, "reason": "vision-only router bias; text replay and native DFly use gate.bias",
                                "equals_text_bias": bool(torch.equal(original, vision_bias))})
                used.add(src)
                continue
            dst = src.removeprefix("model.")
            if dst.startswith("layers."):
                dst = dst.replace(".attn.", ".self_attn.")
                dst = dst.replace(".attn_norm.", ".input_layernorm.").replace(".ffn_norm.", ".post_attention_layernorm.")
                dst = dst.replace(".ffn.gate.weight", ".mlp.router.gate.weight")
                dst = dst.replace(".ffn.gate.bias", ".mlp.expert_bias")
                dst = dst.replace(".ffn.shared_experts.w1.", ".mlp.shared_mlp.gate_proj.")
                dst = dst.replace(".ffn.shared_experts.w3.", ".mlp.shared_mlp.up_proj.")
                dst = dst.replace(".ffn.shared_experts.w2.", ".mlp.shared_mlp.down_proj.")
            assign(dst, checkpoint_tensor(cp, src, torch))
            used.add(src)
        for layer in range(3):
            for w, name in (("w1", "gate"), ("w3", "up"), ("w2", "down")):
                dst = f"layers.{layer}.mlp.experts_{name}_proj.weight"
                fused = torch.empty(expected[dst].shape, dtype=torch.bfloat16, device=device)
                for expert in range(128):
                    base = f"model.layers.{layer}.ffn.experts.{expert}.{w}"
                    codes = checkpoint_tensor(cp, base + ".weight", torch)
                    scales = checkpoint_tensor(cp, base + ".scale", torch)
                    decoded = decode_fp4(codes, scales, torch)
                    fused[expert].copy_(decoded.t().to(device))
                    used.update((base + ".weight", base + ".scale"))
                assign(dst, fused)
                print(json.dumps({"loaded": dst}), flush=True)
        require(loaded == set(expected), f"unloaded model parameters: {set(expected) - loaded}")
        require(used == set(cp.tensors), f"unused checkpoint tensors: {set(cp.tensors) - used}")
    finally:
        cp.close()
    require(not any(p.is_meta for p in model.parameters()), "meta parameters remain")
    return model.eval(), {"source_config_sha256": provenance["source_config_sha256"],
                         "source_files": provenance["source_files"], "parameters_loaded": len(loaded),
                         "source_tensors_accounted_for": len(used), "unused_vision_biases": ignored,
                         "expert_loading": "exact E2M1/E8M0 dequantization to BF16, transposed into original fused expert layout",
                         "config": config.to_dict()}


def compare(actual, reference, *, torch, rtol, atol, relative_rms):
    require(actual.shape == reference.shape, "comparison shape mismatch")
    a, b = actual.float(), reference.float()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {"passed": False, "finite": False, "shape": list(a.shape)}
    diff = a - b
    rms = diff.square().mean().sqrt().item()
    scale = b.square().mean().sqrt().item()
    rel = rms / max(scale, 1e-30)
    close = torch.isclose(a, b, rtol=rtol, atol=atol)
    return {"passed": bool(close.all()) and rel <= relative_rms, "finite": True,
            "shape": list(a.shape), "max_abs": diff.abs().max().item(), "rms_abs": rms,
            "reference_rms": scale, "relative_rms": rel,
            "cosine_similarity": torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
            "different_elements": int((a != b).sum()), "outside_tolerance": int((~close).sum()),
            "elements": a.numel()}


def replay(model, captured, torch):
    pos = captured["positions"].unsqueeze(0)
    contexts = [x.unsqueeze(0) for x in captured["context"]]
    picks = []
    for valid in captured["valid"]:
        indices = torch.arange(640, device=pos.device, dtype=torch.int32).masked_fill(valid == 0, -1)
        block = torch.arange(640, 646, device=pos.device, dtype=torch.int32)
        picks.append(torch.cat((indices, block)).view(1, 1, -1).expand(1, 6, -1))
    embeds = captured["anchor_mask_embeddings"]
    base = torch.cat((embeds[0:1], embeds[1:2].expand(5, -1))).unsqueeze(0)
    injected = model.mtp_fusion(embeds[0:1], captured["target_hidden"].unsqueeze(0), 6)
    initial = base + injected
    layer_out, residual_out = [], []
    def hook(_m, _i, out):
        layer_out.append(out[1].detach().clone())
        residual_out.append(out[0].detach().clone())
    handles = [layer.register_forward_hook(hook) for layer in model.layers]
    try:
        normalized = model(draft_input_ids=None, context_feature=None,
                           draft_position_ids=pos,
                           context_position_ids=torch.arange(640, device=pos.device).unsqueeze(0),
                           noise_embedding=initial, sparse_kv_indices=picks, context_latents=contexts)
    finally:
        for h in handles:
            h.remove()
    # All six previous embeddings come from the native pass. Causal block masking
    # in Tencent's correction means each slot only sees already-supplied predecessors.
    correction_context = {"position_ids": pos, "latent": contexts[-1],
                          "indices": picks[-1][..., :640], "block_size": 6}
    corrected = model.hidden_correction(normalized, captured["previous_embeddings"].unsqueeze(0), correction_context)
    # Isolate correction implementation from upstream backbone drift as a second check.
    correction_given_native = model.hidden_correction(captured["normalized"].unsqueeze(0),
                                                      captured["previous_embeddings"].unsqueeze(0), correction_context)
    return {"initial_block": initial[0], "normalized": normalized[0],
            "layers": torch.stack(layer_out)[:, 0], "corrected": corrected[0, 1:],
            "layers_residual": torch.stack(residual_out)[:, 0],
            "correction_given_native_hidden": correction_given_native[0, 1:]}


def discover_captures(*, capture=None, capture_root=None):
    require((capture is None) != (capture_root is None), "choose one capture or capture root")
    if capture is not None:
        p = Path(capture).resolve()
        require((p / "capture.json").is_file(), "capture has no completed manifest")
        return [(p, Path("."))]
    root = Path(capture_root).resolve()
    require(root.is_dir(), "capture root is not a directory")
    # Do not silently turn a partially written native pass into absent evidence.
    for marker in root.rglob("capture.started"):
        require((marker.parent / "capture.json").is_file(), f"incomplete capture: {marker.parent.relative_to(root)}")
    manifests = sorted(root.rglob("capture.json"), key=lambda p: p.relative_to(root).as_posix())
    require(bool(manifests), "capture root contains no completed captures")
    out = []
    for manifest in manifests:
        require(not manifest.is_symlink() and manifest.parent.resolve().is_relative_to(root), "capture escapes root")
        relative = manifest.parent.relative_to(root)
        require(relative != Path("."), "batch root must contain capture subdirectories")
        out.append((manifest.parent, relative))
    return out


def case_report(meta, hashes, commit, args, torch):
    return {"schema": "dfly2-reference-replay-v1", "complete": False, "passed": False,
              "reference_commit": commit, "capture": meta, "capture_tensor_sha256": hashes,
              "torch": str(torch.__version__), "device": args.device,
              "thresholds": {"rtol": args.rtol, "atol": args.atol, "max_relative_rms": args.max_relative_rms},
              "scope": "original Tencent forward; exact released FP4 weights dequantized to BF16; teacher-forced native predecessor tokens",
              "import_mode": "namespace packages skip unrelated package aggregators; original forward modules are unmodified",
              "does_not_establish": ["target capture/index-selection correctness", "shared EXL3 head equality", "free-running proposal parity", "speed parity"]}


def run_capture(model, weights, capture, output, prepared, commit, args, torch):
    meta, tensors, hashes = prepared
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    report = case_report(meta, hashes, commit, args, torch)
    try:
        report["weights"] = weights
        captured = {name: read_tensor(capture / (name + ".bin"), dtype, shape, torch, args.device)
                    for name, (dtype, shape) in tensors.items()}
        require(torch.equal(captured["positions"].cpu(), torch.arange(meta["q0"], meta["q0"] + 6)), "capture positions differ from anchor")
        require(bool(((captured["valid"] == 0) | (captured["valid"] == 1)).all()), "invalid context mask")
        require(torch.equal(captured["previous_embeddings"][0], captured["previous_embeddings"][1]), "slot0 and slot1 must use anchor embedding")
        require(torch.equal(captured["previous_embeddings"][0], captured["anchor_mask_embeddings"][0]), "captured anchor embedding differs")
        for i in range(1, 4):
            require(torch.equal(captured["initial_block"][:, 0], captured["initial_block"][:, i]), "initial mHC copies are unequal")
        outputs = replay(model, captured, torch)
        comparisons = {}
        for key, ref in outputs.items():
            if key in ("layers", "layers_residual") and key not in captured:
                continue
            native = captured["corrected"] if key == "correction_given_native_hidden" else captured[key]
            if key == "initial_block":
                native = native[:, 0]
            comparisons[key] = compare(native, ref, torch=torch, rtol=args.rtol, atol=args.atol, relative_rms=args.max_relative_rms)
            write_bf16(output / ("reference_" + key + ".bin"), ref)
        report["comparisons"] = comparisons
        report["numeric_passed"] = all(x["passed"] for x in comparisons.values())
        report["passed"] = report["numeric_passed"]
        if getattr(args, "head_probe", None) is not None:
            from proposal_replay import run_proposal_parity
            report["proposal_parity"] = {"report": "proposal-parity/report.json", "complete": False, "passed": False}
            proposal = run_proposal_parity(model, captured, outputs["normalized"], capture,
                                           output / "proposal-parity", args, torch)
            report["proposal_parity"].update({key: proposal[key] for key in
                                               ("complete", "passed", "head_gate_passed", "native_proposals",
                                                "reference_proposals", "slot_matches", "matching_prefix_length")})
            report["passed"] = report["numeric_passed"] and proposal["passed"]
            report["scope"] += "; original reference free-running greedy proposals through gated deployed EXL3 head"
            report["does_not_establish"] = ["target capture/index-selection correctness",
                                            "independent EXL3 head mathematical correctness",
                                            "original native logit or margin equality",
                                            "sampling distribution parity", "speed parity"]
        report["complete"] = True
        report["native_proposals"] = captured["proposals"].tolist()
        report["output_sha256"] = {x.name: sha256(x) for x in output.glob("reference_*.bin")}
    except Exception as e:
        report["passed"] = False
        report["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        report["wall_seconds"] = round(time.time() - started, 3)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("complete", "passed", "comparisons", "wall_seconds")}, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--capture", type=Path, help="one capture; original output layout")
    selection.add_argument("--capture-root", type=Path, help="replay every completed capture below this root, sorted by relative path, loading weights once")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--angelspec", type=Path, required=True)
    p.add_argument("--public-repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--rtol", type=float, default=0.02)
    p.add_argument("--atol", type=float, default=0.02)
    p.add_argument("--max-relative-rms", type=float, default=0.01)
    p.add_argument("--head-probe", type=Path, help="optional native shared EXL3 head probe; requires all --head-* paths")
    p.add_argument("--head-target-model", type=Path)
    p.add_argument("--head-rank0-cache", type=Path)
    p.add_argument("--head-rank1-cache", type=Path)
    p.add_argument("--head-grouped-cubin", type=Path)
    p.add_argument("--head-timeout", type=float, default=300, help="per diagnostic subprocess timeout, seconds")
    args = p.parse_args()
    require(not args.output.exists(), "output directory already exists")
    require(all(math.isfinite(x) and x >= 0 for x in (args.rtol, args.atol, args.max_relative_rms)), "invalid tolerances")
    head_paths = (args.head_probe, args.head_target_model, args.head_rank0_cache, args.head_rank1_cache, args.head_grouped_cubin)
    require(not any(x is not None for x in head_paths) or all(x is not None for x in head_paths), "provide all five --head-* paths or none")
    require(math.isfinite(args.head_timeout) and args.head_timeout > 0, "invalid head subprocess timeout")
    cases = discover_captures(capture=args.capture, capture_root=args.capture_root)
    prepared = [(capture, relative, capture_metadata(capture)) for capture, relative in cases]
    commit = verify_upstream(args.angelspec)
    import_reference_namespaces(args.angelspec)
    os.environ["ANGELSPEC_MTP_MOE_REFERENCE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    started = time.time()
    batch = {"schema": "dfly2-reference-batch-v1", "complete": False, "passed": False,
             "reference_commit": commit, "captures_requested": [r.as_posix() for _, r in cases],
             "weights_load_count": 0, "cases": []}
    try:
        model, weights = load_reference(args.checkpoint, args.public_repo, args.angelspec, args.device, torch)
        batch["weights_load_count"] = 1
        batch["weights_load_seconds"] = round(time.time() - started, 3)
        for capture, relative, metadata in prepared:
            try:
                report = run_capture(model, weights, capture, args.output / relative, metadata, commit, args, torch)
            except Exception as e:
                batch["cases"].append({"capture": relative.as_posix(), "report": (relative / "report.json").as_posix(),
                                       "complete": False, "passed": False, "error": f"{type(e).__name__}: {e}"})
                raise
            batch["cases"].append({"capture": relative.as_posix(), "report": (relative / "report.json").as_posix(),
                                   "complete": report["complete"], "passed": report["passed"],
                                   "capture_tensor_sha256": report["capture_tensor_sha256"]})
        batch["complete"] = True
        batch["passed"] = all(x["passed"] for x in batch["cases"])
    except Exception as e:
        batch["error"] = f"{type(e).__name__}: {e}"
        if args.capture is not None and not (args.output / "report.json").exists():
            meta, _, hashes = prepared[0][2]
            failure = case_report(meta, hashes, commit, args, torch)
            failure["error"] = batch["error"]
            (args.output / "report.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    finally:
        batch["wall_seconds"] = round(time.time() - started, 3)
        if args.capture_root is not None:
            (args.output / "report.json").write_text(json.dumps(batch, indent=2) + "\n")
    return 0 if batch["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

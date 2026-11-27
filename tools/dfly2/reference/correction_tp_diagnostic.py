#!/usr/bin/env python3
"""Isolate two native BF16 TP boundaries using original Tencent correction.

This is a causal diagnostic, NOT original-reference parity evidence. The
unchanged original correction is retained as a separate baseline. No native
source, checkpoint tensor, captured tensor, or previous report is modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import time

import reference_replay as reference
from proposal_replay import correction_context


VARIANTS = {
    "original": (),
    "bf16_half_k_down_proj": ("down_proj",),
    "bf16_half_k_wo_b": ("attention.wo_b",),
    "bf16_half_k_both": ("down_proj", "attention.wo_b"),
}
PARTITIONS = {
    "down_proj": {"weight_shape": [5120, 5120], "axis": 1,
                  "rank_k_ranges": [[0, 2560], [2560, 5120]]},
    "attention.wo_b": {"weight_shape": [5120, 8192], "axis": 1,
                       "rank_k_ranges": [[0, 4096], [4096, 8192]]},
}
CAPTURE_INPUTS = ("normalized", "previous_embeddings", "positions", "context", "valid", "corrected")
LIMITS = [
    "Half-K variants intentionally replace two original outputs and cannot establish original-reference parity.",
    "This isolates correction conditioned on captured native hidden states and native predecessor embeddings.",
    "It does not run the backbone, shared EXL3 head, free-running proposals, or target verification.",
    "The original six-slot PyTorch batching is retained; native one-row GEMM algorithm choices are not reproduced.",
    "The original full-width linear runs before its hooked replacement, so timings are not serving performance.",
    "Reduced error supports these boundaries as contributors, not as the sole cause of all native differences.",
]


def half_k_forward(module, inputs, torch):
    """BF16(F32(BF16(left dot)) + F32(BF16(right dot))), native rank order.

    The two contiguous K halves are exactly the packer's axis-1 slices. No
    bias, activation, weight conversion, or residual operation is introduced.
    """
    reference.require(isinstance(module, torch.nn.Linear) and module.bias is None,
                      "half-K diagnostic requires a bias-free Linear")
    reference.require(isinstance(inputs, torch.Tensor) and inputs.dtype == torch.bfloat16,
                      "half-K input must be BF16")
    reference.require(module.weight.dtype == torch.bfloat16 and inputs.shape[-1] == module.in_features,
                      "half-K weight/input shape or dtype differs")
    width = module.in_features
    reference.require(width > 0 and width % 2 == 0, "half-K width must be positive and even")
    split = width // 2
    left = torch.nn.functional.linear(inputs[..., :split].contiguous(), module.weight[:, :split].contiguous())
    right = torch.nn.functional.linear(inputs[..., split:].contiguous(), module.weight[:, split:].contiguous())
    reference.require(left.dtype == torch.bfloat16 and right.dtype == torch.bfloat16,
                      "half-K GEMM did not materialize BF16")
    return (left.float() + right.float()).to(torch.bfloat16)


@contextmanager
def half_k_hooks(correction, names, torch):
    """Temporarily replace selected module outputs; preserve methods and other hooks."""
    reference.require(len(names) == len(set(names)), "duplicate diagnostic projection")
    modules = []
    for name in names:
        reference.require(name in PARTITIONS, f"unsupported diagnostic projection: {name}")
        module = correction.get_submodule(name)
        reference.require(isinstance(module, torch.nn.Linear) and module.bias is None,
                          "diagnostic projection must be a bias-free Linear")
        modules.append(module)
    handles = []

    def replace(module, args, _original_output):
        reference.require(len(args) == 1, "unexpected diagnostic Linear arguments")
        return half_k_forward(module, args[0], torch)

    try:
        for module in modules:
            handles.append(module.register_forward_hook(replace))
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()


def validate_native_partitions(correction, torch):
    for name, partition in PARTITIONS.items():
        module = correction.get_submodule(name)
        reference.require(isinstance(module, torch.nn.Linear) and module.bias is None,
                          f"unexpected released projection: {name}")
        reference.require(list(module.weight.shape) == partition["weight_shape"] and module.weight.dtype == torch.bfloat16,
                          f"released projection shape/dtype differs: {name}")


def validate_pack_partitions(public_repo):
    """Check the live packer maps these real tensors to contiguous K halves."""
    pack = reference.load_pack_module(public_repo)
    schema = pack.expected_tensors()
    for name, partition in PARTITIONS.items():
        tensor = "model.hidden_correction." + name + ".weight"
        dtype, shape = schema[tensor]
        reference.require(dtype == "BF16" and list(shape) == partition["weight_shape"] and
                          pack.split_axis(tensor) == 1, f"native packer partition differs: {tensor}")
    # Exercise the packer's generic byte slicer without allocating model tensors.
    probe = bytes(range(16))
    for rank, expected in ((0, bytes([0, 1, 2, 3, 8, 9, 10, 11])),
                           (1, bytes([4, 5, 6, 7, 12, 13, 14, 15]))):
        reference.require(pack.split_bytes(probe, (2, 8), 1, 1, rank) == expected,
                          "native packer does not use contiguous K halves")
    return {"file": "tools/dfly2/pack_weights.py",
            "sha256": reference.sha256(Path(public_repo) / "tools/dfly2/pack_weights.py"),
            "split_axis": 1, "rank_order": [0, 1], "contiguous_half_k_verified": True}


def evaluate_variants(correction, captured, torch, *, rtol=0.02, atol=0.02, max_relative_rms=0.01, on_variant=None):
    """Evaluate the unmodified baseline, then three temporary output replacements."""
    reference.require(captured["normalized"].dtype == torch.bfloat16 and
                      captured["previous_embeddings"].dtype == torch.bfloat16,
                      "captured correction inputs must be BF16")
    reference.require(captured["normalized"].shape == captured["previous_embeddings"].shape and
                      captured["normalized"].shape[0] == 6, "correction requires six aligned input slots")
    context = correction_context(captured, torch)
    outputs = {}
    baseline = None
    metrics = dict(torch=torch, rtol=rtol, atol=atol, relative_rms=max_relative_rms)
    with torch.no_grad():
        for name, projections in VARIANTS.items():
            with half_k_hooks(correction, projections, torch):
                value = correction(captured["normalized"].unsqueeze(0),
                                   captured["previous_embeddings"].unsqueeze(0), context)
            reference.require(value.dtype == torch.bfloat16 and
                              tuple(value.shape) == (1, *captured["normalized"].shape),
                              "original correction output shape/dtype differs")
            value = value[0, 1:].detach().clone()
            if baseline is None:
                baseline = value
            outputs[name] = {
                "output": value,
                "replaced_projections": list(projections),
                "comparison": reference.compare(captured["corrected"], value, **metrics),
                "comparison_vs_original": reference.compare(value, baseline, **metrics),
            }
            if on_variant is not None:
                on_variant(name, outputs[name])
    return outputs


def prepare_captures(*, capture=None, capture_root=None):
    return [(path, relative, reference.capture_metadata(path)) for path, relative in
            reference.discover_captures(capture=capture, capture_root=capture_root)]


def run_capture(correction, capture, output, prepared, args, torch):
    meta, tensors, hashes = prepared
    output.mkdir(parents=True, exist_ok=True)
    reference.require(not any(output.iterdir()), "diagnostic case output directory is not empty")
    report = {
        "schema": "dfly2-correction-tp-diagnostic-v1", "complete": False,
        "classification": "diagnostic-not-original-reference-parity", "capture": meta,
        "capture_tensor_sha256": hashes, "device": args.device, "torch": str(torch.__version__),
        "partitions": PARTITIONS, "limits": LIMITS,
        "thresholds": {"rtol": args.rtol, "atol": args.atol, "max_relative_rms": args.max_relative_rms},
        "input_scope": "captured native normalized hidden and predecessor embeddings; last captured target context",
        "baseline": "original", "variants": {},
    }
    started = time.time()
    try:
        captured = {name: reference.read_tensor(capture / (name + ".bin"), *tensors[name], torch, args.device)
                    for name in CAPTURE_INPUTS}
        reference.require(torch.equal(captured["positions"].cpu(), torch.arange(meta["q0"], meta["q0"] + 6)),
                          "capture positions differ from anchor")
        reference.require(bool(((captured["valid"] == 0) | (captured["valid"] == 1)).all()), "invalid context mask")
        reference.require(torch.equal(captured["previous_embeddings"][0], captured["previous_embeddings"][1]),
                          "slot0 and slot1 must use the anchor embedding")
        def save_variant(name, result):
            file = output / (name + ".bf16.bin")
            reference.write_bf16(file, result["output"])
            report["variants"][name] = {key: value for key, value in result.items() if key != "output"}
            report["variants"][name].update(output_file=file.name, output_sha256=reference.sha256(file))
            original_error = report["variants"]["original"]["comparison"].get("rms_abs")
            error = result["comparison"].get("rms_abs")
            report["variants"][name]["rms_error_ratio_to_original"] = error / original_error if error is not None and original_error else None
        evaluate_variants(correction, captured, torch, rtol=args.rtol, atol=args.atol,
                          max_relative_rms=args.max_relative_rms, on_variant=save_variant)
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["wall_seconds"] = round(time.time() - started, 3)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"capture": str(capture), "complete": True,
                      "relative_rms": {name: row["comparison"].get("relative_rms") for name, row in report["variants"].items()}}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--capture", type=Path)
    selection.add_argument("--capture-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--angelspec", type=Path, required=True)
    parser.add_argument("--public-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--max-relative-rms", type=float, default=0.01)
    args = parser.parse_args()
    reference.require(not args.output.exists(), "diagnostic output directory already exists")
    reference.require(all(math.isfinite(x) and x >= 0 for x in (args.rtol, args.atol, args.max_relative_rms)),
                      "invalid diagnostic thresholds")
    prepared = prepare_captures(capture=args.capture, capture_root=args.capture_root)
    commit = reference.verify_upstream(args.angelspec)
    reference.import_reference_namespaces(args.angelspec)
    os.environ["ANGELSPEC_MTP_MOE_REFERENCE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    report = {"schema": "dfly2-correction-tp-diagnostic-run-v1", "complete": False,
              "classification": "diagnostic-not-original-reference-parity", "limits": LIMITS,
              "reference_commit": commit, "weights_load_count": 0, "cases": [],
              "captures_requested": [relative.as_posix() for _, relative, _ in prepared],
              "diagnostic_script_sha256": reference.sha256(Path(__file__)),
              "reference_loader_sha256": reference.sha256(Path(reference.__file__))}
    started = time.time()
    try:
        report["packer"] = validate_pack_partitions(args.public_repo)
        model, provenance = reference.load_reference(args.checkpoint, args.public_repo, args.angelspec, args.device, torch)
        report["weights_load_count"] = 1
        report["weights"] = provenance
        report["weights_load_seconds"] = round(time.time() - started, 3)
        validate_native_partitions(model.hidden_correction, torch)
        for capture, relative, metadata in prepared:
            row = {"capture": relative.as_posix(), "report": (relative / "report.json").as_posix(), "complete": False}
            report["cases"].append(row)
            case = run_capture(model.hidden_correction, capture, args.output / relative, metadata, args, torch)
            row.update(complete=case["complete"], native_norm_rounding=metadata[0]["native"].get("norm_rounding"),
                       variants={name: {"relative_rms": variant["comparison"].get("relative_rms"),
                                        "rms_abs": variant["comparison"].get("rms_abs"),
                                        "within_recorded_thresholds": variant["comparison"]["passed"]}
                                 for name, variant in case["variants"].items()})
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["wall_seconds"] = round(time.time() - started, 3)
        (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    # Successful execution does not imply that any variant met the thresholds.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Numerical tests for dsv41_dfly2.cu, without loading a model or TensorFold.

Requires PyTorch with CUDA and nvcc (or --ptx pointing to an existing PTX).
Uses the CUDA driver through ctypes; no compiled Python extension is needed.
The reference calculations run on CPU, mostly in FP64, so they do not share
the CUDA kernels under test. Production parameter layouts are exercised.

Example (on an otherwise idle Spark):
  python tools/dfly2/test_primitives.py --report /tmp/dfly2-primitives.json

This proves primitive arithmetic for these fixtures, not end-to-end model
parity, checkpoint compatibility, sampling correctness, or throughput.
"""
from __future__ import annotations

import argparse
import ctypes as C
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

import torch


I64 = C.c_int64
BF = torch.bfloat16
F32 = torch.float32


class Driver:
    def __init__(self, ptx: Path, device: int):
        torch.cuda.set_device(device)
        # Ensure a current primary context, shared with Torch allocations.
        self.context_anchor = torch.empty(1, device="cuda")
        self.lib = C.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
        self.lib.cuModuleLoad.argtypes = [C.POINTER(C.c_void_p), C.c_char_p]
        self.lib.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        self.lib.cuLaunchKernel.argtypes = [
            C.c_void_p, C.c_uint, C.c_uint, C.c_uint,
            C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_void_p,
            C.POINTER(C.c_void_p), C.POINTER(C.c_void_p),
        ]
        self.lib.cuModuleUnload.argtypes = [C.c_void_p]
        self.module = C.c_void_p()
        self.check(self.lib.cuModuleLoad(C.byref(self.module), os.fsencode(ptx)))
        self.functions = {}
        self.stream = torch.cuda.current_stream(device)

    def check(self, code):
        if code:
            message = C.c_char_p()
            self.lib.cuGetErrorString(code, C.byref(message))
            raise RuntimeError(f"CUDA driver {code}: {message.value!r}")

    def launch(self, name, grid, threads, *args):
        name = "tf_dfly_" + name
        if name not in self.functions:
            fn = C.c_void_p()
            self.check(self.lib.cuModuleGetFunction(C.byref(fn), self.module, name.encode()))
            self.functions[name] = fn
        values = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                assert arg.is_cuda and arg.is_contiguous(), (name, arg.shape)
                values.append(C.c_uint64(arg.data_ptr()))
            elif isinstance(arg, C._SimpleCData):
                values.append(arg)
            elif isinstance(arg, float):
                values.append(C.c_float(arg))
            elif isinstance(arg, int):
                values.append(C.c_int(arg))
            else:
                raise TypeError(type(arg))
        params = (C.c_void_p * len(values))(*[C.addressof(v) for v in values])
        dims = (grid, 1, 1) if isinstance(grid, int) else tuple(grid)
        self.check(self.lib.cuLaunchKernel(
            self.functions[name], *dims, threads, 1, 1, 0,
            C.c_void_p(self.stream.cuda_stream), params, None,
        ))
        self.stream.synchronize()

    def close(self):
        self.check(self.lib.cuModuleUnload(self.module))


def gpu(x):
    return x.contiguous().cuda()


def randn(*shape, scale=1.0, dtype=BF):
    return (torch.randn(*shape) * scale).to(dtype)


def assert_tensor(actual, expected, *, exact=False, rtol=0.009, atol=0.0003):
    actual = actual.detach().cpu()
    expected = expected.detach().cpu().to(actual.dtype)
    a, b = actual.double(), expected.double()
    # Identical infinities (the all--infinity argmax fixture) have zero error.
    delta = torch.where(a == b, torch.zeros_like(a), (a - b).abs())
    stats = {
        "shape": list(actual.shape),
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "rms_abs": float(delta.square().mean().sqrt()) if delta.numel() else 0.0,
        "different_elements": int((actual != expected).sum()),
        "elements": actual.numel(), "exact_required": exact,
        "rtol": 0 if exact else rtol, "atol": 0 if exact else atol,
    }
    if exact:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return stats


def fp4_values(codes, scales, *, scale_kind, packing="adjacent"):
    """Decode E2M1: expert bytes pair adjacent columns; target cache pairs halves."""
    table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                          0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)
    low, high = codes.long() & 15, codes.long() >> 4
    if packing == "adjacent":
        nib = torch.stack((low, high), dim=-1).flatten(-2)
    elif packing == "halves":
        nib = torch.cat((low, high), dim=-1)
    else:
        raise ValueError(packing)
    if scale_kind == "e8m0":
        factors = torch.pow(2., scales.double() - 127)
        group = 32
    elif scale_kind == "e4m3":
        # Independent decode includes signed/subnormal E4M3FN, no NaNs in fixtures.
        s = scales.long()
        exponent, mantissa = (s >> 3) & 15, s & 7
        factors = torch.where(exponent == 0, mantissa.double() * 2**-9,
                              (1 + mantissa.double() / 8) * torch.pow(2., exponent - 7))
        factors *= torch.where(s & 128 != 0, -1., 1.)
        group = 16
    else:
        raise ValueError(scale_kind)
    return table[nib] * factors.repeat_interleave(group, dim=-1)


class Tests:
    def __init__(self, driver):
        self.d = driver

    def norm_concat(self):
        records = []
        for d, eps in [(5120, 1e-20), (97, 1e-6)]:
            x, y = randn(d), randn(d, scale=.125)
            wx, wy = randn(d, scale=.4) + 1, randn(d, scale=.4) + 1
            wx, wy = wx.to(BF), wy.to(BF)
            out = torch.empty(2 * d, device="cuda", dtype=BF)
            self.d.launch("norm_concat", 1, 256, gpu(x), gpu(y), gpu(wx), gpu(wy), out, d, eps)
            def norm(a, w):
                h = a.double() * torch.rsqrt(a.double().square().mean() + eps)
                return (h * w.double()).to(BF)
            expected = torch.cat((norm(x, wx), norm(y, wy)))
            stat = assert_tensor(out, expected)
            # AngelSpec DFlashRMSNorm rounds BEFORE multiplying its weight;
            # vLLM's fused RMSNorm and this kernel round only after multiplication.
            def training_norm(a, w):
                h = a.float() * torch.rsqrt(a.float().square().mean() + eps)
                return h.to(BF) * w
            training = torch.cat((training_norm(x, wx), training_norm(y, wy)))
            stat["angelspec_double_round_differences"] = int((training != out.cpu()).sum())
            records.append(stat)
        return records

    def init_block(self):
        d, rows = 97, 6
        embed, fusion, gate = randn(17, d), randn(d), randn(d)
        ids = torch.tensor([3, 16, 16, 16, 16, 16], dtype=torch.int64)
        out = torch.empty(rows, 4, d, device="cuda", dtype=BF)
        pre = torch.empty(rows, 4, device="cuda", dtype=F32)
        self.d.launch("init_block", rows, 256, gpu(embed), gpu(ids), gpu(fusion), gpu(gate), out, pre, d, rows)
        h = embed[ids].clone()
        h[0] = h[0] + fusion * gate
        expected_pre = torch.zeros(rows, 4)
        expected_pre[:, 0] = 1
        return [assert_tensor(out, h[:, None].expand(-1, 4, -1), exact=True),
                assert_tensor(pre, expected_pre, exact=True)]

    def add(self):
        a, b = randn(513), randn(513)
        out = torch.empty_like(gpu(a))
        self.d.launch("add", 3, 256, gpu(a), gpu(b), out, a.numel())
        return assert_tensor(out, a + b, exact=True)

    def silu(self):
        records = []
        # Negative gates remain unclamped; only the upper gate bound is clipped.
        gate = torch.cat((torch.tensor([-100, -20, -10.1, -10, -1, 0, 1, 10, 10.1, 20, 100]),
                          torch.randn(502) * 5)).to(BF)
        up = torch.linspace(-30, 30, gate.numel()).to(BF)
        for limit in (0., 10.):
            out = torch.empty_like(gpu(gate))
            self.d.launch("silu", 3, 256, gpu(gate), gpu(up), out, gate.numel(), limit)
            g = gate.float().clamp(max=limit) if limit else gate.float()
            u = up.float().clamp(-limit, limit) if limit else up.float()
            expected = (torch.nn.functional.silu(g).to(BF).float() * u).to(BF)
            records.append(assert_tensor(out, expected, atol=1e-5))
        return records

    def route(self):
        rows, d = 6, 5120
        # Exactly representable dot products keep a routing regression distinct
        # from accumulation-order noise; large bias verifies rank-vs-weight split.
        x = (torch.randint(-4, 5, (rows, d)).float() * .125).to(BF)
        gate = (torch.randint(-4, 5, (128, d)).float() * .125).to(BF)
        bias = torch.linspace(-.15, .15, 128)
        bias[7] = 20.
        logits = x.double() @ gate.double().T
        scores = torch.nn.functional.softplus(logits, beta=1, threshold=20).sqrt()
        chosen = (scores + bias.double()).topk(3, dim=-1).indices
        w = scores.gather(1, chosen)
        expected = (w / (w.sum(-1, keepdim=True) + 1e-20) * 1.5).float()
        ids = torch.empty(rows, 3, device="cuda", dtype=torch.int32)
        weights = torch.empty(rows, 3, device="cuda")
        self.d.launch("route", rows, 256, gpu(x), gpu(gate), gpu(bias), ids, weights, d)
        records = [assert_tensor(ids, chosen.int(), exact=True),
                   assert_tensor(weights, expected, rtol=3e-6, atol=3e-7)]
        # Equal unbiased scores: deterministic lowest-ID selection, bias is not
        # included in normalized selected weights.
        self.d.launch("route", rows, 256, gpu(torch.zeros_like(x)), gpu(gate),
                      gpu(torch.zeros_like(bias)), ids, weights, d)
        records += [assert_tensor(ids, torch.tensor([0, 1, 2]).expand(rows, -1).int(), exact=True),
                    # Equal softplus scores normalize through FP32 addition and
                    # division, so 0.5 can differ by one representable step.
                    assert_tensor(weights, torch.full((rows, 3), .5), rtol=1e-7, atol=0.)]
        # Finite very-negative logits underflow softplus to zero in FP32.
        # The +1e-20 normalization denominator must yield finite zero weights.
        tiny_x, negative_gate = torch.zeros_like(x), torch.zeros_like(gate)
        tiny_x[:, 0] = 1
        negative_gate[:, 0] = -1000
        self.d.launch("route", rows, 256, gpu(tiny_x), gpu(negative_gate),
                      gpu(torch.zeros_like(bias)), ids, weights, d)
        records += [assert_tensor(ids, torch.tensor([0, 1, 2]).expand(rows, -1).int(), exact=True),
                    assert_tensor(weights, torch.zeros(rows, 3), exact=True)]
        return records

    def fp4(self):
        records = []
        for n, k, expert_input in [(13, 96, 0), (17, 128, 1), (8, 5120, 0), (8, 1152, 1)]:
            experts, rows = 8, 2
            codes = torch.randint(0, 256, (experts, n, k // 2), dtype=torch.uint8)
            scales = torch.randint(124, 131, (experts, n, k // 32), dtype=torch.uint8)
            xrows = rows * 3 if expert_input else rows
            x = (torch.randint(-8, 9, (xrows, k)).float() * .125).to(BF)
            ids = torch.tensor([[7, 0, 3], [1, 7, 2]], dtype=torch.int32)
            decoded = fp4_values(codes, scales, scale_kind="e8m0")
            expected = torch.stack([
                decoded[int(ids.flatten()[j])] @ x[j if expert_input else j // 3].double()
                for j in range(rows * 3)
            ]).to(BF)
            out = torch.empty(rows * 3, n, device="cuda", dtype=BF)
            self.d.launch("fp4", ((n + 3) // 4, rows * 3, 1), 128,
                          gpu(codes), gpu(scales), gpu(x), gpu(ids), out, n, k, rows, expert_input)
            records.append(assert_tensor(out, expected, exact=True))
        return records

    def mix_experts(self):
        rows, d = 6, 97
        routed, shared = randn(rows, 3, d), randn(rows, d)
        weights = torch.rand(rows, 3)
        weights *= 1.5 / weights.sum(-1, keepdim=True)
        expected = shared.double() + (routed.double() * weights.double()[..., None]).sum(1)
        out = torch.empty(rows, d, device="cuda")
        self.d.launch("mix_experts", (rows * d + 255) // 256, 256,
                      gpu(routed), gpu(shared), gpu(weights), out, d, rows)
        return assert_tensor(out, expected, rtol=5e-6, atol=1e-6)

    def gather_context(self):
        records = []
        count, compressed_base, ring_rows, ring_base = 300, 5, 134, 134
        codes = torch.randint(0, 256, (count + compressed_base, 256), dtype=torch.uint8)
        # E4M3 includes subnormal and sign cases; unlike expert E8M0, zero is zero.
        scales = torch.randint(0, 127, (count + compressed_base, 32), dtype=torch.uint8)
        scales[::2] |= 128
        # A target-cache byte pairs columns j and j+256, unlike expert bytes
        # which pair 2*j and 2*j+1. This sentinel must distinguish the formats.
        codes[compressed_base, 0] = 0x21
        codes[compressed_base, 1] = 0x43
        scales[compressed_base] = 0x38  # E4M3 value 1, not E8M0 exponent 127.
        ring = randn(ring_base + ring_rows, 512)
        for anchor, floor in [(200, 73), (4, 0), (200, 180)]:
            indices = torch.randint(0, anchor + 1, (512,), dtype=torch.int64)
            indices[::7] = -1
            # Rejected/future rows must not leak from a captured target batch.
            indices[1::11] = anchor + 1
            indices[2::11] = anchor + 20
            indices[3] = 0
            out = torch.empty(640, 512, device="cuda", dtype=BF)
            valid = torch.empty(640, device="cuda", dtype=torch.int32)
            self.d.launch("gather_context", 640, 256, gpu(codes), gpu(scales), gpu(indices),
                          gpu(ring), out, valid, I64(compressed_base), I64(ring_base), ring_rows,
                          I64(anchor), I64(floor))
            expected = torch.zeros(640, 512, dtype=BF)
            mask = torch.zeros(640, dtype=torch.int32)
            decoded = fp4_values(codes, scales, scale_kind="e4m3", packing="halves").to(BF)
            assert decoded[compressed_base, 0] == .5
            assert decoded[compressed_base, 1] == 1.5
            assert decoded[compressed_base, 256] == 1.
            adjacent = fp4_values(codes[compressed_base:compressed_base + 1],
                                  scales[compressed_base:compressed_base + 1], scale_kind="e4m3")
            assert adjacent[0, 1] == 1. and adjacent[0, 1] != decoded[compressed_base, 1]
            selected = (indices >= 0) & (indices <= anchor)
            expected[:512][selected] = decoded[compressed_base + indices[selected]]
            mask[:512] = selected.int()
            for j, pos in enumerate(range(anchor - 127, anchor + 1)):
                if pos >= floor and pos >= 0:
                    mask[512 + j] = 1
                    expected[512 + j] = ring[ring_base + pos % ring_rows]
            records += [assert_tensor(out, expected, exact=True), assert_tensor(valid, mask, exact=True)]
        return records

    def attention(self):
        records = []
        heads, rows = 4, 6
        q, context, block = randn(rows, heads, 512, scale=.25), randn(640, 512), randn(6, 512)
        sink = torch.tensor([-3., 0., 2., 8.])
        for block_rows, valid_mode in [(6, "mixed"), (0, "mixed"), (0, "none"), (6, "none")]:
            valid = (torch.arange(640) % 3 != 0).int() if valid_mode == "mixed" else torch.zeros(640, dtype=torch.int32)
            out = torch.empty(rows, heads, 512, device="cuda", dtype=BF)
            self.d.launch("attention", rows * heads, 256, gpu(q), gpu(context), gpu(valid),
                          gpu(block), gpu(sink), out, heads, rows, block_rows)
            kv = torch.cat((context[valid.bool()], block[:block_rows])).double()
            logits = torch.einsum("rhd,kd->rhk", q.double(), kv) / math.sqrt(512)
            denom = torch.logsumexp(torch.cat((logits, sink.double()[None, :, None].expand(rows, -1, 1)), -1), -1)
            expected = torch.einsum("rhk,kd->rhd", (logits - denom[..., None]).exp(), kv).to(BF)
            records.append(assert_tensor(out, expected, atol=0.0001))
        return records

    def block_attention(self):
        records = []
        heads = 4
        q, keys, gate, initial = randn(heads, 512, scale=.25), randn(6, 512), randn(heads), randn(heads, 512)
        sink = torch.tensor([-2., 0., 1., 5.])
        rel_bias = torch.randn(heads, 16) * .5
        gate_bias = torch.randn(heads, 16) * .5
        for slot in (0, 1, 5):
            out = gpu(initial.clone())
            self.d.launch("block_attention", heads, 256, gpu(q), gpu(keys), gpu(gate),
                          gpu(sink), gpu(rel_bias), gpu(gate_bias), out, heads, slot)
            logits = q.double() @ keys[:slot + 1].double().T / math.sqrt(512)
            logits += rel_bias[:, :slot + 1].double().flip(-1)
            total = torch.logsumexp(torch.cat((logits, sink.double()[:, None]), -1), -1)
            read = (logits - total[:, None]).exp() @ keys[:slot + 1].double()
            g = torch.tanh(gate.double() + gate_bias[:, slot].double())
            expected = (initial.double() + g[:, None] * read).to(BF)
            records.append(assert_tensor(out, expected))
        # A zero gate must leave context attention exactly unchanged.
        out = gpu(initial.clone())
        self.d.launch("block_attention", heads, 256, gpu(q), gpu(keys), gpu(torch.zeros_like(gate)),
                      gpu(sink), gpu(rel_bias), gpu(torch.zeros_like(gate_bias)), out, heads, 5)
        records.append(assert_tensor(out, initial, exact=True))
        return records

    def best(self):
        records = []
        for n in (1, 97, 1025):
            x = torch.full((n,), -float("inf"))
            if n > 1:
                x[n // 3], x[-1] = 17., 17.
            out = torch.empty(2, device="cuda")
            self.d.launch("best", 1, 256, gpu(x), out, n, 64640)
            index = int(x.argmax())
            expected = torch.tensor([float(x[index]), float(index + 64640)])
            records.append(assert_tensor(out, expected, exact=True))
        return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_source = Path(__file__).resolve().parents[2] / "_build/tensorfold/zig/kernels/cuda/dsv41_dfly2.cu"
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--ptx", type=Path)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--arch", help="e.g. compute_121; defaults to selected GPU capability")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=262611)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--only", nargs="+", help="Test method names")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if not torch.cuda.is_available():
        parser.error("PyTorch CUDA is unavailable; run this on the Spark, not the Windows control host")
    torch.cuda.set_device(args.device)
    major, minor = torch.cuda.get_device_capability(args.device)
    report = {
        "time_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(args.device), "capability": [major, minor],
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "limitations": [
            "Synthetic primitive tests only; no model/checkpoint or whole-decoder parity.",
            "FP4 exact fixtures use binary-exact values; arbitrary accumulations need tolerance.",
            "Norm primary reference is fused vLLM math; AngelSpec DFlashRMSNorm double-round differences are recorded.",
            "Gather anchor means last accepted target position, one before the draft block anchor.",
            "Gather rejects future indices; captured target top-k must still refer to live compressed-cache rows.",
        ], "tests": [],
    }
    with tempfile.TemporaryDirectory(prefix="dfly2-primitives-") as tmp:
        ptx = args.ptx
        if ptx is None:
            ptx = Path(tmp) / "dfly2.ptx"
            command = [args.nvcc, "-ptx", "-O3", "--std=c++17", "--fmad=false", "--ftz=false",
                       "-arch=" + (args.arch or f"compute_{major}{minor}"), str(args.source), "-o", str(ptx)]
            report["compile_command"] = command
            subprocess.run(command, check=True)
        report["ptx_sha256"] = hashlib.sha256(ptx.read_bytes()).hexdigest()
        driver = Driver(ptx, args.device)
        tests = Tests(driver)
        names = args.only or [name for name, value in Tests.__dict__.items()
                              if callable(value) and not name.startswith("_")]
        try:
            for name in names:
                result = {"name": name}
                try:
                    result["measurements"] = getattr(tests, name)()
                    result["passed"] = True
                    print(f"PASS {name}", flush=True)
                except Exception as exc:
                    result.update(passed=False, error=str(exc), traceback=traceback.format_exc())
                    print(f"FAIL {name}: {exc}", flush=True)
                report["tests"].append(result)
        finally:
            driver.close()
    report["passed"] = all(row["passed"] for row in report["tests"])
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Report: {args.report}")
    print(f"{sum(row['passed'] for row in report['tests'])}/{len(report['tests'])} tests passed")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Bounded CUDA-event microbenchmarks for the experimental DFly2 primitives.

Compiles the current production source and temporary bit-decoder variants;
never edits the production source or starts a model server. CUDA graph batches
remove Python/ctypes launch gaps from kernel measurements. Results concern a
single GPU's warm synthetic kernels, not model throughput or two-Spark speed.
"""
from __future__ import annotations

import argparse
import ctypes as C
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import time

import torch

from test_primitives import Driver, BF, F32


E8M0_BITS = """__device__ float e8m0(unsigned c) {
    // E8M0 code0 is 2^-127, a FLOAT32 subnormal, not zero.
    return __uint_as_float(c == 0 ? 0x00400000u : c == 255 ? 0x7fc00000u : c << 23);
}"""

E2M1_BITS = """__device__ float e2m1(unsigned c) {
    unsigned m = c & 7;
    unsigned bits = m < 2 ? (m ? 0x3f000000u : 0u)
                         : ((126u + (m >> 1)) << 23) | ((m & 1) << 22);
    return __uint_as_float(bits | ((c & 8) << 28));
}"""

DECODE_PROBE = """
extern "C" __global__ void tf_dfly_bench_decode(float* scales, float* codes) {
    int i = threadIdx.x;
    if (i < 256) scales[i] = e8m0(i);
    if (i < 16) codes[i] = e2m1(i);
}
"""


def variant_source(source, variant):
    result = source
    if variant in ("e8m0_bits", "e8m0_e2m1_bits"):
        result, count = re.subn(r"__device__ float e8m0\(unsigned c\) \{.*?\n\}", E8M0_BITS, result, flags=re.S)
        if count != 1:
            raise ValueError("expected exactly one e8m0 helper; source changed")
    if variant == "e8m0_e2m1_bits":
        result, count = re.subn(r"__device__ float e2m1\(unsigned c\) \{.*?\n\}", E2M1_BITS, result, flags=re.S)
        if count != 1:
            raise ValueError("expected exactly one e2m1 helper; source changed")
    if variant not in ("baseline", "e8m0_bits", "e8m0_e2m1_bits"):
        raise ValueError(variant)
    return result + DECODE_PROBE


class PreparedLaunch:
    """Resolve function/parameters once; enqueue without Driver.launch's sync."""
    def __init__(self, driver, name, grid, threads, args):
        self.driver, self.args = driver, args  # keep tensors alive
        self.fn = C.c_void_p()
        driver.check(driver.lib.cuModuleGetFunction(C.byref(self.fn), driver.module, ("tf_dfly_" + name).encode()))
        self.values = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                if not arg.is_cuda or not arg.is_contiguous():
                    raise ValueError("CUDA tensors must be contiguous")
                self.values.append(C.c_uint64(arg.data_ptr()))
            elif isinstance(arg, C._SimpleCData):
                self.values.append(arg)
            elif isinstance(arg, int):
                self.values.append(C.c_int(arg))
            elif isinstance(arg, float):
                self.values.append(C.c_float(arg))
            else:
                raise TypeError(type(arg))
        self.params = (C.c_void_p * len(self.values))(*[C.addressof(x) for x in self.values])
        self.grid, self.threads = tuple(grid), threads

    def __call__(self):
        self.driver.check(self.driver.lib.cuLaunchKernel(
            self.fn, *self.grid, self.threads, 1, 1, 0,
            C.c_void_p(self.driver.stream.cuda_stream), self.params, None,
        ))


def time_kernel(launch, *, launches, samples, warmup):
    stream = launch.driver.stream
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            launch()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(launches):
                launch()
        graph.replay()
        stream.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        durations = []
        for _ in range(samples):
            start.record(stream)
            graph.replay()
            end.record(stream)
            end.synchronize()
            durations.append(start.elapsed_time(end) * 1000 / launches)
        del graph
    return {"median_us": statistics.median(durations), "min_us": min(durations),
            "max_us": max(durations), "samples_us": durations,
            "launches_per_graph": launches, "samples": samples, "warmup_launches": warmup}


def decode_check(driver):
    scales = torch.empty(256, device="cuda", dtype=F32)
    codes = torch.empty(16, device="cuda", dtype=F32)
    driver.launch("bench_decode", 1, 256, scales, codes)
    scales, codes = scales.cpu(), codes.cpu()
    want_scales = torch.tensor([math.ldexp(1., c - 127) for c in range(255)] + [float("nan")], dtype=F32)
    want_codes = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                               -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=F32)
    if not torch.equal(scales[:255].view(torch.int32), want_scales[:255].view(torch.int32)):
        raise AssertionError("E8M0 finite/subnormal decode differs bitwise")
    if not bool(torch.isnan(scales[255])):
        raise AssertionError("E8M0 code255 must be NaN")
    if not torch.equal(codes.view(torch.int32), want_codes.view(torch.int32)):
        raise AssertionError("E2M1 decode/sign-zero differs bitwise")
    return {"e8m0_finite_codes_exact": 255, "e8m0_nan_code": True,
            "e8m0_code0_bits": hex(int(scales[0].view(torch.int32))),
            "e2m1_all_16_bitwise_exact_including_negative_zero": True}


def fp4_fixture(n, k, expert_input, selection):
    # All 128 expert strides are present, as in the production rank pack.
    codes = torch.randint(0, 256, (128, n, k // 2), dtype=torch.uint8, device="cuda")
    scales = torch.randint(124, 131, (128, n, k // 32), dtype=torch.uint8, device="cuda")
    xrows = 18 if expert_input else 6
    x = (torch.randint(-8, 9, (xrows, k), device="cuda").float() * .125).to(BF)
    ids = (torch.tensor([1, 64, 127], device="cuda", dtype=torch.int32).repeat(6, 1)
           if selection == "same_three" else torch.arange(18, device="cuda", dtype=torch.int32).reshape(6, 3) * 7)
    out = torch.empty(18, n, device="cuda", dtype=BF)
    return {
        "kernel": "fp4", "grid": ((n + 3) // 4, 18, 1), "threads": 128,
        "args": (codes, scales, x, ids, out, n, k, 6, expert_input), "out": out,
        "shape": {"rows": 6, "top_k": 3, "n": n, "k": k, "expert_input": bool(expert_input),
                  "selected_experts": selection, "distinct_experts": 3 if selection == "same_three" else 18},
        "logical_code_scale_bytes_per_launch": 18 * n * (k // 2 + k // 32),
    }


def attention_fixture(rows, block_rows):
    q = (torch.randn(rows, 32, 512, device="cuda") * .25).to(BF)
    context = torch.randn(640, 512, device="cuda").to(BF)
    block = torch.randn(6, 512, device="cuda").to(BF)
    valid = torch.ones(640, device="cuda", dtype=torch.int32)
    sink = torch.linspace(-3., 3., 32, device="cuda")
    out = torch.empty(rows, 32, 512, device="cuda", dtype=BF)
    return {"kernel": "attention", "grid": (rows * 32, 1, 1), "threads": 256,
            "args": (q, context, valid, block, sink, out, 32, rows, block_rows), "out": out,
            "shape": {"heads_per_rank": 32, "rows": rows, "context_keys": 640, "block_keys": block_rows}}


def block_fixture(slot):
    q = (torch.randn(32, 512, device="cuda") * .25).to(BF)
    keys = torch.randn(6, 512, device="cuda").to(BF)
    gate = (torch.randn(32, device="cuda") * .1).to(BF)
    sink = torch.randn(32, device="cuda")
    rel_bias = torch.randn(32, 16, device="cuda") * .1
    gate_bias = torch.randn(32, 16, device="cuda") * .1
    out = torch.randn(32, 512, device="cuda").to(BF)
    return {"kernel": "block_attention", "grid": (32, 1, 1), "threads": 256,
            "args": (q, keys, gate, sink, rel_bias, gate_bias, out, 32, slot), "out": out,
            "initial": out.clone(), "shape": {"heads_per_rank": 32, "slot": slot, "causal_keys": slot + 1}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2] / "_build/tensorfold/zig/kernels/cuda/dsv41_dfly2.cu")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--arch", help="Default compute capability of the selected GPU")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--launches", type=int, default=16)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not (1 <= args.launches <= 64 and 1 <= args.samples <= 15 and 0 <= args.warmup <= 16):
        parser.error("bounded settings: launches1..64, samples1..15, warmup0..16")
    if not torch.cuda.is_available():
        parser.error("run on an idle CUDA host; this benchmark does not stop services")
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    major, minor = torch.cuda.get_device_capability(args.device)
    source = args.source.read_text()
    start = time.monotonic()
    report = {
        "time_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
        "device": torch.cuda.get_device_name(args.device), "capability": [major, minor],
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "source": str(args.source), "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "methodology": "CUDA events around a CUDA graph containing repeated prepared driver-kernel launches; warmup excluded",
        "limitations": ["Kernel-only, one GPU, synthetic inputs; not end-to-end or TP2 serving speed.",
                        "Repeated fixed buffers/IDs have warm caches; no claim of cold-weight bandwidth.",
                        "Candidate outputs are compared with current kernels, not a new whole-model reference.",
                        "Block-attention output is reset before each variant; repeated timing launches accumulate in-place.",
                        "No production sources are changed, and no serving process is started/stopped."],
        "variants": {}, "benchmarks": [],
    }
    with tempfile.TemporaryDirectory(prefix="dfly2-kernel-bench-") as temp:
        drivers = {}
        try:
            for variant in ("baseline", "e8m0_bits", "e8m0_e2m1_bits"):
                path, ptx = Path(temp) / (variant + ".cu"), Path(temp) / (variant + ".ptx")
                path.write_text(variant_source(source, variant))
                command = [args.nvcc, "-ptx", "-O3", "--std=c++17", "--fmad=false", "--ftz=false",
                           "-arch=" + (args.arch or f"compute_{major}{minor}"), str(path), "-o", str(ptx)]
                subprocess.run(command, check=True, timeout=180)
                driver = Driver(ptx, args.device)
                driver.stream = torch.cuda.Stream(device=args.device)
                drivers[variant] = driver
                report["variants"][variant] = {"compile_command": command,
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "ptx_sha256": hashlib.sha256(ptx.read_bytes()).hexdigest(), "decode_check": decode_check(driver)}
            cases = [
                (f"fp4_{n}x{k}_{selection}", lambda n=n, k=k, expert_input=expert_input, selection=selection:
                 fp4_fixture(n, k, expert_input, selection))
                for n, k, expert_input in [(1152, 5120, 0), (5120, 1152, 1)]
                for selection in ("same_three", "eighteen_distinct")
            ]
            cases += [("attention_draft", lambda: attention_fixture(6, 6)),
                      ("attention_correction", lambda: attention_fixture(1, 0))]
            cases += [(f"block_attention_slot{slot}", lambda slot=slot: block_fixture(slot)) for slot in range(1, 6)]
            for name, factory in cases:
                fixture = factory()
                torch.cuda.synchronize()
                record = {"name": name, "shape": fixture["shape"], "timings": {}}
                if "logical_code_scale_bytes_per_launch" in fixture:
                    record["logical_code_scale_bytes_per_launch"] = fixture["logical_code_scale_bytes_per_launch"]
                baseline = None
                for variant, driver in drivers.items():
                    if "initial" in fixture:
                        fixture["out"].copy_(fixture["initial"])
                    torch.cuda.synchronize()
                    launch = PreparedLaunch(driver, fixture["kernel"], fixture["grid"], fixture["threads"], fixture["args"])
                    launch()
                    driver.stream.synchronize()
                    actual = fixture["out"].cpu()
                    if baseline is None:
                        baseline = actual
                    else:
                        if not torch.equal(actual, baseline):
                            raise AssertionError(f"{name}/{variant} differs from baseline")
                    timing = time_kernel(launch, launches=args.launches, samples=args.samples, warmup=args.warmup)
                    timing["baseline_output_exact"] = True
                    record["timings"][variant] = timing
                    del launch
                base = record["timings"]["baseline"]["median_us"]
                for timing in record["timings"].values():
                    timing["speedup_vs_baseline"] = base / timing["median_us"]
                report["benchmarks"].append(record)
                print(name, {v: round(t["median_us"], 3) for v, t in record["timings"].items()}, flush=True)
                del fixture
                torch.cuda.empty_cache()
        finally:
            torch.cuda.synchronize()
            for driver in drivers.values():
                driver.close()
    report["elapsed_seconds"] = time.monotonic() - start
    report["passed"] = True
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Report: {args.report}", flush=True)


if __name__ == "__main__":
    main()

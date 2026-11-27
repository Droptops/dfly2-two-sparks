#!/usr/bin/env python3
"""Synthetic GPU checks for the opt-in DFlash RMSNorm materialization only.

The CPU FP64 oracle explicitly rounds before the weight multiply. Fixtures
must distinguish the fused and two-round results, so a no-op precision flag
cannot pass. Existing fused kernel entry points retain their original ABI.
This does not establish full-model or Tencent proposal parity or throughput.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time
import traceback


def fixture(torch, width):
    """Exact BF16 inputs and nonunit weights, with nontrivial RMS scales."""
    at = torch.arange(width)
    x = torch.where(at % 2 == 0, 1., -3.).to(torch.bfloat16)
    y = torch.where(at % 3 == 0, -2., 5.).to(torch.bfloat16)
    weights = torch.tensor([0., .5, 1., 1.0078125, 1.5, -2.75, .75390625], dtype=torch.bfloat16)
    wx, wy = weights[at % len(weights)], weights[(at + 3) % len(weights)]
    return x, y, wx, wy


def oracle(torch, x, weight, *, double, eps=1e-20, dtype=None):
    value = x.to(dtype or torch.float64)
    scaled = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    if double:
        scaled = scaled.to(torch.bfloat16).to(value.dtype)
    return (scaled * weight.to(value.dtype)).to(torch.bfloat16)


def check_fixture_oracles(torch):
    counts = {}
    for width in (97, 1280, 5120):
        x, y, wx, wy = fixture(torch, width)
        for double in (False, True):
            for vector, weight in ((x, wx), (y, wy)):
                # These fixtures avoid BF16 boundaries sensitive to differences
                # between CPU FP64 and FP32 reductions, independently of CUDA.
                assert torch.equal(oracle(torch, vector, weight, double=double),
                                   oracle(torch, vector, weight, double=double, dtype=torch.float32))
        fused = torch.cat([oracle(torch, x, wx, double=False), oracle(torch, y, wy, double=False)])
        reference = torch.cat([oracle(torch, x, wx, double=True), oracle(torch, y, wy, double=True)])
        counts[width] = int((fused != reference).sum())
        assert counts[width] > 0, 'fixture must detect missing intermediate BF16 rounding'
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--primitive-helper', type=Path, default=Path(__file__).with_name('test_primitives.py'))
    parser.add_argument('--cubin', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError('refusing to overwrite an earlier GPU receipt')
    spec = importlib.util.spec_from_file_location('dfly_primitives', args.primitive_helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    torch = helper.torch
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(schema='dfly2-reference-norms-gpu-checks-v1', passed=False, tests=[],
                  source_sha256=sha(args.source), helper_sha256=sha(args.primitive_helper),
                  scope='Synthetic DFlash norm materialization checks; no full-model/proposal parity or speed claim.')
    start = time.monotonic()
    try:
        counts = check_fixture_oracles(torch)
        report['cpu_oracle_distinguishing_elements'] = counts
        with tempfile.TemporaryDirectory(prefix='dfly-reference-norms-') as tmp:
            module = args.cubin or Path(tmp) / 'reference-norms-sm121.cubin'
            flags = ['-cubin', '-O3', '--std=c++17', '--fmad=false', '--ftz=false', '-arch=sm_121']
            if args.cubin is None:
                subprocess.run(['nvcc', *flags, str(args.source), '-o', str(module)], check=True)
            assert module.read_bytes().startswith(b'\x7fELF'), 'expected native cubin, not PTX'
            report.update(artifact_kind='cubin', artifact_sha256=sha(module),
                          compiled_here=args.cubin is None, compile_flags=flags if args.cubin is None else None)
            driver = helper.Driver(module, args.device)
            report['device'] = torch.cuda.get_device_name(args.device)
            try:
                for width in (97, 1280, 5120):
                    cpu = fixture(torch, width)
                    x, y, wx, wy = [v.cuda() for v in cpu]
                    old = torch.empty(2 * width, device='cuda', dtype=torch.bfloat16)
                    new = torch.empty_like(old)
                    driver.launch('norm_concat', 1, 256, x, y, wx, wy, old, width, 1e-20)
                    driver.launch('norm_concat_reference', 1, 256, x, y, wx, wy, new, width, 1e-20)
                    for output, double in ((old, False), (new, True)):
                        expected = torch.cat([oracle(torch, cpu[0], cpu[2], double=double),
                                              oracle(torch, cpu[1], cpu[3], double=double)])
                        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
                    assert int((old != new).sum()) == counts[width]
                    report['tests'].append(dict(name='concat_cpu_oracle_and_fused_control', width=width,
                                                differing_elements=counts[width], passed=True))

                    # Unit-weight norm followed by an in-place BF16 multiply is
                    # the strategy used around unchanged target norm kernels.
                    unit = torch.ones_like(x)
                    staged = torch.empty_like(old)
                    driver.launch('norm_concat', 1, 256, x, y, unit, unit, staged, width, 1e-20)
                    driver.launch('norm_weight', (width + 255) // 256, 256, staged[:width], wx, width, width)
                    driver.launch('norm_weight', (width + 255) // 256, 256, staged[width:], wy, width, width)
                    torch.testing.assert_close(staged, new, rtol=0, atol=0)
                    report['tests'].append(dict(name='unit_norm_then_bf16_weight', width=width, passed=True))

                width, rows = 97, 6
                x, y, wx, wy = [v.cuda() for v in fixture(torch, width)]
                embed = torch.stack([y, x, -y, -x])
                for token in (0, 1, 2, 3, -1, len(embed)):
                    ids = torch.tensor([2, token, 1], device='cuda', dtype=torch.int64)
                    output = torch.empty(2 * width, device='cuda', dtype=torch.bfloat16)
                    expected = torch.empty_like(output)
                    driver.launch('norm_concat_indexed_reference', 1, 256,
                                  x, embed, ids, wx, wy, output, width, 1, len(embed), 1e-20)
                    driver.launch('norm_concat_reference', 1, 256,
                                  x, embed[token if 0 <= token < len(embed) else 0], wx, wy, expected, width, 1e-20)
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
                    assert ids.cpu().tolist() == [2, token, 1]
                report['tests'].append(dict(name='indexed_predecessor_and_invalid_read', cases=6, passed=True))

                # Multirow stride, tail threads, in-place operation and untouched
                # sentinels around the output all matter to the six-row mHC path.
                values = torch.arange(rows * width, dtype=torch.float32).reshape(rows, width)
                values = ((values % 37 - 18) / 13).to(torch.bfloat16)
                weights = fixture(torch, width)[2]
                expected = (values.float() * weights.float()).to(torch.bfloat16)
                storage = torch.full((rows * width + 32,), -17., device='cuda', dtype=torch.bfloat16)
                middle = storage[16:-16].view(rows, width)
                middle.copy_(values)
                driver.launch('norm_weight', (rows * width + 255) // 256, 256,
                              middle, weights.cuda(), rows * width, width)
                torch.testing.assert_close(middle.cpu(), expected, rtol=0, atol=0)
                assert bool((storage[:16] == -17).all()) and bool((storage[-16:] == -17).all())
                report['tests'].append(dict(name='inplace_multirow_weight_guards', rows=rows, width=width, passed=True))
                report['passed'] = True
            finally:
                driver.stream.synchronize()
                driver.close()
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['wall_seconds'] = time.monotonic() - start
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x') as out:
            json.dump(report, out, indent=2, allow_nan=False)
            out.write('\n')
    print(json.dumps(dict(passed=report['passed'], checks=len(report['tests']), report=str(args.report))))


if __name__ == '__main__':
    main()

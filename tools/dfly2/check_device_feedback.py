#!/usr/bin/env python3
"""GPU-only checks for device token feedback and padded rank-winner transport.

The sequential fixture uses the old kernel/host path as a byte-level control,
and a deliberately constructed cyclic vocabulary head as an independent token
oracle. This checks dependency preservation, ties and failure sentinels. It
does not establish Tencent parity, model acceptance or end-to-end speed.
"""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--primitive-helper', type=Path, default=Path(__file__).with_name('test_primitives.py'))
    p.add_argument('--cubin', type=Path, help='optional precompiled SM121 native cubin (no driver PTX JIT)')
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--device', type=int, default=0)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('dfly_primitive_helper', args.primitive_helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    torch = helper.torch
    torch.manual_seed(48291)
    records = []
    with tempfile.TemporaryDirectory(prefix='dfly-feedback-') as tmp:
        module = args.cubin or Path(tmp)/'feedback-sm121.cubin'
        compile_flags = ['-cubin', '-O3', '--std=c++17', '--fmad=false', '--ftz=false', '-arch=sm_121']
        if args.cubin is None:
            # Native code avoids requiring a driver PTX JIT newer than the
            # installed host driver. This does not alter or update that driver.
            subprocess.run(['nvcc', *compile_flags, str(args.source), '-o', str(module)], check=True)
        assert module.read_bytes().startswith(b'\x7fELF'), 'expected a native cubin, not PTX text'
        driver = helper.Driver(module, args.device)
        real_stream = driver.stream

        class QueuedStream:
            cuda_stream = real_stream.cuda_stream
            def synchronize(self):
                pass

        def queued():
            driver.stream = QueuedStream()

        def synchronized():
            real_stream.synchronize()
            driver.stream = real_stream

        try:
            # Cross-rank ties use the lowest global ID, including -inf scores.
            for stride in (2, 4):
                fixtures = [
                    ((2., 3., 1., 5.), 3), ((1., 3., 2., 5.), 5),
                    ((2., 7., 2., 3.), 3), ((2., 3., 2., 7.), 3),
                    ((float('-inf'), 7., float('-inf'), 3.), 3),
                    ((float('nan'), 3., 2., 7.), 3),
                    ((1., 3., float('nan'), 7.), 3),
                    ((2., -1., 1., 3.), -1), ((2., 19., 1., 3.), -1),
                    ((2., float('nan'), 1., 3.), -1),
                    ((2., float('inf'), 1., 3.), -1),
                    ((2., 3.5, 1., 3.), -1),
                ]
                for values, expected in fixtures:
                    scores = torch.full((2, stride), float('nan'), device='cuda', dtype=torch.float32)
                    scores[0,:2] = torch.tensor(values[:2], device='cuda')
                    scores[1,:2] = torch.tensor(values[2:], device='cuda')
                    ids = torch.full((6,), -99, device='cuda', dtype=torch.int64)
                    driver.launch('select_best', 1, 1, scores, ids, 4, 19, stride)
                    got = ids.cpu().tolist()
                    assert got == [-99,-99,-99,-99,expected,-99], (values, got)
                records.append(dict(name='rank_winner', stride=stride, cases=len(fixtures), passed=True))

            d, vocab, split, anchor = 97, 19, 9, 16
            embed = torch.zeros((vocab,d), device='cuda', dtype=torch.bfloat16)
            embed[torch.arange(vocab, device='cuda'),torch.arange(vocab, device='cuda')] = 1
            x = torch.zeros(d, device='cuda', dtype=torch.bfloat16)
            wx = torch.ones_like(x)
            wy = torch.ones_like(x)
            head = torch.zeros((2*d,vocab), device='cuda', dtype=torch.float32)
            for token in range(vocab):
                head[d+token, (token+1)%vocab] = 1

            for stride in (2,4):
                for steps in range(1,6):
                    old_out = torch.empty((steps,2*d), device='cuda', dtype=torch.bfloat16)
                    previous = anchor
                    reference = []
                    for step in range(steps):
                        driver.launch('norm_concat', 1, 256, x, embed[previous], wx, wy, old_out[step], d, 1e-6)
                        logits = old_out[step].float() @ head
                        previous = int(logits.argmax().item())
                        reference.append(previous)
                    expected = [(anchor+step+1)%vocab for step in range(steps)]
                    assert reference == expected

                    ids = torch.full((6,), -99, device='cuda', dtype=torch.int64)
                    ids[0] = anchor
                    out = torch.empty_like(old_out)
                    rank_winners = []  # retain allocations until queued kernels finish
                    all_logits = []
                    queued()
                    for step in range(steps):
                        driver.launch('norm_concat_indexed', 1, 256, x, embed, ids, wx, wy, out[step], d, step, vocab, 1e-6)
                        logits = out[step].float() @ head
                        all_logits.append(logits)
                        best = torch.full((2,stride), float('nan'), device='cuda', dtype=torch.float32)
                        rank_winners.append(best)
                        driver.launch('best', 1, 256, logits[:split], best[0], split, 0)
                        driver.launch('best', 1, 256, logits[split:], best[1], vocab-split, split)
                        if stride == 4:
                            driver.launch('pad_best', 1, 1, best[0])
                            driver.launch('pad_best', 1, 1, best[1])
                        driver.launch('select_best', 1, 1, best, ids, step+1, vocab, stride)
                    synchronized()
                    assert ids.cpu().tolist() == [anchor]+expected+[-99]*(5-steps)
                    torch.testing.assert_close(out, old_out, rtol=0, atol=0)
                    if stride == 4:
                        for best in rank_winners:
                            assert torch.count_nonzero(best[:,2:]).item() == 0
                    records.append(dict(name='sequential_device_feedback', stride=stride, steps=steps,
                                        exact_tokens=True, exact_normalized_vectors=True, passed=True))

            # Failure propagation must never read outside the embedding table.
            for bad in (-1, vocab):
                ids = torch.tensor([bad], device='cuda', dtype=torch.int64)
                out = torch.empty(2*d, device='cuda', dtype=torch.bfloat16)
                expected = torch.empty_like(out)
                driver.launch('norm_concat_indexed', 1, 256, x, embed, ids, wx, wy, out, d, 0, vocab, 1e-6)
                driver.launch('norm_concat', 1, 256, x, embed[0], wx, wy, expected, d, 1e-6)
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                assert ids.item() == bad
            records.append(dict(name='invalid_embedding_safe_read', cases=2, passed=True))
        finally:
            synchronized()
            driver.close()
        report = dict(schema='dfly2-device-feedback-gpu-checks-v1', passed=True,
                      device=torch.cuda.get_device_name(args.device),
                      source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                      artifact_kind='cubin', artifact_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                      compiled_here=args.cubin is None, compile_flags=compile_flags if args.cubin is None else None,
                      tests=records,
                      scope='Synthetic GPU dependency/selection tests; no full drafter parity or throughput claim.')
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(passed=True, checks=len(records), report=str(args.report))))


if __name__ == '__main__':
    main()

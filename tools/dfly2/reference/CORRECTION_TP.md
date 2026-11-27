# Correction TP rounding diagnostic

`correction_tp_diagnostic.py` isolates two premature BF16 materializations in
the native correction path. It is a separate causal diagnostic, **not an
original-reference parity run**, and never changes the original reference's
math or reports.

The native packer partitions both projections along contiguous input columns:

| Original projection weight | Full shape [N,K] | Rank 0 K | Rank 1 K |
| --- | --- | --- | --- |
| `hidden_correction.down_proj` | 5120,5120 | [0,2560) | [2560,5120) |
| `hidden_correction.attention.wo_b` | 5120,8192 | [0,4096) | [4096,8192) |

`pack_weights.py:split_axis` selects axis 1, and `split_bytes` copies the first
or second half of each row unchanged. The diagnostic checks the packer's live
schema, axis, and byte-slicing behavior and retains its SHA-256. The native
MLP gate/up partitions produce the corresponding 2560 activation values;
each rank's four contiguous attention output groups produce 4096 values.

Native `linearBf16` rounds each rank's projection result before converting it
back to F32 for gathering. `sumPartial` then adds rank 0 and rank 1 and rounds
to BF16. The unchanged Tencent module instead has one full-width projection
with one BF16 output. The isolated replacement is therefore:

```text
left  = F.linear(BF16(input first half), BF16(weight first half)) -> BF16
right = F.linear(BF16(input second half), BF16(weight second half)) -> BF16
replacement = BF16(left.float() + right.float())
```

Exactly four variants run in order:

1. `original`: Tencent correction unchanged.
2. `bf16_half_k_down_proj`: replace only the MLP down projection output.
3. `bf16_half_k_wo_b`: replace only the attention output projection output.
4. `bf16_half_k_both`: replace both outputs.

Temporary forward hooks replace selected outputs; the original methods and
weights are untouched. Handles are removed in `finally`, including on errors,
and existing hooks are retained. The original full projection is evaluated
before its replacement, so execution timings are diagnostic overhead.

The existing verified loader checks the pinned Tencent source and all released
checkpoint files, then loads weights once (approximately 29 GB). The diagnostic
calls only the original correction module with captured native normalized hidden
states, captured native predecessor embeddings, positions, and the last target
context. It retains the original six-slot PyTorch batch shape. It does not run
the backbone, target head, free-running proposals, or native one-row GEMM
algorithm. Those are separate comparisons.

```sh
python3 correction_tp_diagnostic.py \
  --capture-root CAPTURE_ROOT \
  --checkpoint RELEASED_DFLY_CHECKPOINT --angelspec PINNED_ANGELSPEC \
  --public-repo PUBLIC_REPO --output NEW_DIAGNOSTIC_OUTPUT --device cuda
```

Use `--capture DIR` for one pass. Batch selection preserves sorted relative
paths and accepts both `native.norm_rounding: fused` and `double` without
changing or discarding that metadata. A new output directory is required;
existing original-reference reports cannot be overwritten.

Each case writes four private BF16 vector files and a `report.json` under
schema `dfly2-correction-tp-diagnostic-v1`. The report includes hashes, metrics
against native corrected vectors, metrics against the unchanged original
baseline, and each variant's RMS-error ratio to baseline. The recorded defaults
remain `rtol=0.02`, `atol=0.02`, and relative RMS at most `0.01`. No thresholds
are relaxed to favor a variant.
Each completed variant is saved immediately, so a later failure retains the
earlier baseline and variant artifacts in the incomplete case report.

`run.json` records checkpoint/source/packer provenance, the one model load,
all requested cases, and incomplete failures under schema
`dfly2-correction-tp-diagnostic-run-v1`. Exit code 0 means the diagnostic
completed, even if every variant failed its numeric thresholds. There is no
overall parity-pass field. Failed original results are preserved independently.

A lower error in a replacement variant supports these specific rounding
boundaries as contributors. It does not turn the replacement into Tencent's
original reference, prove that all remaining differences are explained, or
establish faster serving. Captures, vectors, and hashes remain private; only
reviewed numeric summaries should be published separately.

CPU tests use small explicit BF16 tensors and test selection/restoration:

```sh
python -m unittest discover -s tools/dfly2/reference \
  -p test_correction_tp_diagnostic.py -v
```

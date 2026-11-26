# Recorded two-Spark results

## Fresh public-launcher GPU validation

The portable public launcher and newly built public image completed serial,
DFly2 and DSpark runs on two NVIDIA GB10 systems. All three modes used the same
executable, image contents, launcher source, target configuration and six
request payloads. All **959 output token IDs and all six finish reasons
matched**, including the historical serial reference.

| Mode | Estimated decode tokens/s | Total seconds including prefill | Accepted / proposed draft tokens |
| --- | ---: | ---: | ---: |
| Drafting disabled | 38.49 | 27.14 | 0 / 0 |
| Existing DSpark | 74.98 | 15.16 | 690 / 916 |
| Custom DFly2 | 45.06 | 24.56 | 685 / 1,351 |

DFly2 was about 17% faster than drafting disabled and about 40% slower than
DSpark on this estimated decode rate. The public path reproduced target output
equivalence; DSpark remained faster on this workload. These were one run per
mode, in serial/DFly2/DSpark order, with the same eager, one-stream, 8,192-token
configuration and profiling enabled. They are not a statistical speed study.

The complete sanitized receipts are
[serial](../evidence/portable-validation/chat-serial.json),
[DFly2](../evidence/portable-validation/chat-dfly2.json), and
[DSpark](../evidence/portable-validation/chat-dspark.json). The
[derived comparison](../evidence/portable-validation/comparison.json) includes
the common executable SHA-256
`0a3831df751158a6cada9b4efb266c958ed7ebfe7606803d7386abd52e248029`,
launcher and image-content identities, the draft-pack identity, and the timing
arithmetic. Prompt contents, settings and seeds are unchanged from the reviewed
[request fixture](../evidence/portable-validation/chat-requests.json).

## Historical integration run

The native DFly2 port ran across two NVIDIA GB10 systems with the
DeepSeek-V4.1-Flash-1017 EXL3 EAQ v1 2.9 bpw target. On six fixed chat-template
requests, all **959 output token IDs and all six finish reasons matched** the
target with drafting disabled. Three requests were greedy; three used
temperature 0.7, top-p 1, top-k 20, and fixed seeds.

| Mode | Estimated decode tokens/s | Total seconds including prefill | Accepted / proposed draft tokens |
| --- | ---: | ---: | ---: |
| Drafting disabled | 38.46 | 27.21 | 0 / 0 |
| Existing DSpark | 74.05 | 15.28 | 690 / 918 |
| Custom DFly2 | 45.84 | 23.73 | 685 / 1,351 |

DFly2 was about 19% faster than drafting disabled on this measured decode-rate
estimate, and about 38% slower than DSpark. These results support keeping it an
experimental option. They do not reproduce the publisher's advertised speedups.

## Public release build validation

The public source preparation was run on ARM64 Linux and produced the locked
TensorFold and inputs tree hashes. The image then built successfully from the
pinned public NVIDIA base, including the native server, lane runner, weight
checker, and rank-cache builder. All **582 native host tests** passed with GPU
access disabled. The public Python checks passed **44 tests** on both Windows
and Linux.

The image's isolated, hash-pinned chat-tool environment was tested offline
against the actual target tokenizer and template. All six generated request
payloads, including input tokens and sampling settings, matched the published
fixture. The native executable hashes stayed unchanged when this Python layer
was added. The fresh build receipt is
[public-build-validation.json](../evidence/public-build-validation.json).

This build receipt records build, CPU-test and tokenization checks. The fresh
two-node GPU comparison above is recorded separately under
`evidence/portable-validation/`. The historical integration measurements and
their original executable hashes remain unchanged.

After adding the portable GPU receipts and image-store compatibility fix, all
**55 public CPU checks** passed on Windows and ARM64 Linux. The Linux check ran
from the exported release archive and verified both recorded comparisons.

## Recompute the comparison

From the repository root, using only the Python standard library:

```sh
python3 tools/dfly2/verify_evidence.py
python3 tools/dfly2/verify_portable_evidence.py
python3 -m unittest discover -s tools/dfly2 -p test_evidence.py -v
python3 -m unittest discover -s tools/dfly2 -p test_portable_evidence.py -v
```

The verifier recomputes each reply's token digest, checks actual arrays and
finish reasons against the serial run, checks request hashes on both ranks,
reconciles counters, and calculates the rates from recorded elapsed times.
It also checks both recorded weight-upload receipts, primitive-test receipts,
and the original microbenchmark kernel's source hash. It checks the saved
evidence; it does not rerun GPU inference.

The portable verifier additionally binds all three runs to the executable
recorded by the public build, checks the current published launcher bytes,
requires identical image contents and target metadata, and recomputes equality
against the fresh serial run. It also reports whether that serial output
matches the historical serial run. Both comparisons retain their own times.

The reviewed synthetic prompts, their complete tokenized inputs, sampling
settings, and seeds are in [chat-requests.json](../evidence/chat-requests.json).
Outputs and timings are in [chat-serial.json](../evidence/chat-serial.json),
[chat-dspark.json](../evidence/chat-dspark.json), and
[chat-dfly2.json](../evidence/chat-dfly2.json). The generated calculation is
[comparison.json](../evidence/comparison.json).

## Timing scope and limits

These were one-stream eager runs with CUDA graphs disabled and profiling
enabled, an 8,192-token pool, and fixed requests. Prompt lengths were 82–142
tokens. Each output budget was 256 tokens; the combined output was 959 tokens.
One response reached its output budget. Matching token IDs establishes that the
verified speculative path retained these target outputs; it does not grade the
answers or establish correctness for all inputs and sampling settings.

Estimated decode seconds equal the native recorder's total request time minus
the sum of six recorded prefill times. The rate counts 953 tokens, excluding the
first token generated during each prefill. Native elapsed times are rounded to
0.01 seconds. This is not a production throughput benchmark or a statistical
estimate with confidence intervals. Long-context serving, including 64K, was
not measured in this chat comparison.

In the historical integration test, the DSpark baseline was repeated after
builds finished. Its initial run was
73.20 estimated tokens/s; the repeated run shown above was 74.05. Both receipts
are retained, including [the initial run](../evidence/chat-dspark-initial.json).
The final DSpark and DFly2 runs used the same executable. The serial run used
the previous executable; subsequent changes formatted new DFly2 files and
optimized DFly2's private FP4 helpers, which serial mode does not invoke.
Executable hashes remain in each receipt so this distinction is visible.

## Recorded component checks

Both rank-weight checks loaded the actual packed drafter, compared every GPU
byte against its packed file, and verified allocation release. Each rank holds
**4,529,862,920 draft-weight bytes**, plus runtime scratch. This is additional
DFly2 weight storage, not total model RAM. The experimental mode replaces the
loaded DSpark drafter. The matching
[pack manifest](../evidence/draft-pack-manifest.json) retains source and rank-file
SHA-256 digests; model weights are not bundled.

All eleven recorded independent CUDA primitive checks passed on SM121. Their
full numerical measurements, including reported normalization differences from
the AngelSpec training reference, remain in
[gpu-primitives.json](../evidence/gpu-primitives.json). Passing the selected
primitive tolerances is distinct from full drafter/reference parity.

The FP4 decoder optimization reduced the measured time of four representative
FP4 kernel fixtures by a factor of **1.72–1.80**, retaining identical outputs.
All 256 E8M0 scale codes and all 16 E2M1 values were checked, including the E8M0
subnormal, NaN handling, and signed zero. These are warm synthetic kernel timings
on one GPU, measured by CUDA events around graph batches. The kernel gain is
not a whole-model speedup. Complete samples are in
[primitive-benchmark.json](../evidence/primitive-benchmark.json).

To reproduce the old-versus-new decoder microbenchmark on an idle CUDA host:

```sh
python3 tools/dfly2/benchmark_primitives.py \
  --source evidence/kernels/dsv41_dfly2_before_optimization.cu \
  --report /tmp/dfly2-kernel-benchmark.json
```

The historical baseline CUDA bytes are preserved with SHA-256
`9fd2910be8d0725416008d3b2fa1f6c65acbd0927fa98c1a69d39e71ca27fa1e`.
Use that source explicitly: benchmarking an already optimized source would
make the named decoder variants equivalent.

## Evidence export

The public receipts are schema-filtered derivatives of private run receipts.
They retain numerical inputs, outputs, sampling, timings, counters, and relevant
code/checkpoint hashes. Infrastructure and session fields are omitted. The
request fixture and original benchmark CUDA file are byte-identical copies.
[EVIDENCE.md](EVIDENCE.md) documents the field mapping and excluded artifacts.

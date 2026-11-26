# Recorded two-Spark results

The repeated validation completed **24 GPU runs**, twelve per fixture. Each
fixture runs serial, DFly2, fixed-five DSpark and ordinary DSpark three times
with the starting mode rotating. All runs match their fixture's first serial
reference in every output token and finish reason. Both fixtures use the same
native executable, image contents, target configuration, 8,192-token pool and
launcher commit `bfd3a816fabbb749146a9eadededad5052e3e43c`.

## Repeated short-chat GPU suite

The six short prompts contain **82 to 142 tokens**, with a 256-token output
budget each. Every one of the twelve runs produced the same **959 output token
IDs and all six finish reasons**. Three requests are greedy; three use
temperature 0.7, top-p 1, top-k 20 and fixed seeds.

| Mode | Estimated decode tok/s, median (min–max) | End-to-end tok/s, median (min–max) | Median tokens landed per round |
| --- | ---: | ---: | ---: |
| No drafting | 38.76 (38.32–38.85) | 35.53 (35.09–35.56) | 1.00 |
| DSpark | 73.65 (73.20–74.28) | 62.35 (61.95–62.68) | 3.56 |
| DSpark, fixed five drafts | 66.23 (65.01–67.78) | 55.72 (55.50–58.09) | 3.71 |
| DFly2 | 45.34 (44.47–45.47) | 39.37 (39.21–39.96) | 3.52 |

DFly2's median decode gain over no drafting was 17.0%; including prefill, 10.8%.
Its median decode rate remained 31.5% below fixed-five DSpark. The short prose
request remains a negative result: DFly2's median estimated rate was 27.61
tok/s against 38.52 without drafting, about 28.3% slower. The respective rates
were 43.93 for fixed-five DSpark and 53.65 for ordinary DSpark. DFly2 was the
only mode with a request slower than no drafting by median decode time.

The complete [suite manifest](../evidence/suite-chat/suite.json),
[request bytes](../evidence/suite-chat/requests.json),
[provenance](../evidence/suite-chat/provenance.json), and
[recomputed comparison](../evidence/suite-chat/comparison.json) retain all
twelve runs and per-request measurements. No run was dropped or retried within
this suite. The earlier single-run chat measurements below are preserved
separately; they are not pooled into these medians.

## Repeated longer-input GPU suite

The merged suite launcher completed all **12 runs** on two NVIDIA GB10 systems:
four modes, three repeats each, with the starting mode rotating. Every run
matched all **1,169 output token IDs and all six finish reasons** of the first
serial run. The six reviewed prompts contain **1,210 to 6,892 tokens**, with a
256-token output budget each. The native executable and image contents were
unchanged from the earlier public build; this run used launcher commit
`bfd3a816fabbb749146a9eadededad5052e3e43c`.

| Mode | Estimated decode tok/s, median (min–max) | End-to-end tok/s, median (min–max) | Median tokens landed per round |
| --- | ---: | ---: | ---: |
| No drafting | 38.21 (38.18–38.50) | 25.61 (25.34–25.65) | 1.00 |
| DSpark | 81.39 (80.88–81.96) | 39.53 (39.28–39.55) | 4.21 |
| DSpark, fixed five drafts | 76.66 (75.86–77.69) | 38.00 (37.50–38.09) | 4.32 |
| DFly2 | 55.75 (54.86–56.10) | 32.04 (29.68–32.28) | 4.31 |

On this fixture, DFly2's median decode rate was 45.9% above no drafting and its
median prefill-inclusive rate was 25.1% above it. DSpark remained faster:
DFly2's median decode rate was 27.3% below fixed-five DSpark. Fixed-five DSpark
and DFly2 landed almost the same number of tokens per round, while their median
draft-pass times were 3.71 ms and 27.01 ms respectively. The fixed-depth mode
matches draft depth only; DSpark still has suffix-copy proposals and DFly2 does
not. This does not isolate a general difference between the two algorithms.

The complete [suite manifest](../evidence/suite-long/suite.json),
[request bytes](../evidence/suite-long/requests.json),
[provenance](../evidence/suite-long/provenance.json), and
[recomputed comparison](../evidence/suite-long/comparison.json) retain every
run, per-request timing and draft count. DFly2's third repeat had 18.19 seconds
of reported prefill and the lowest end-to-end rate, 29.68 tok/s; it is retained
in the range above. No run was dropped or retried within this suite.

These results extend observed output preservation to the longer-input capture
path. They do not establish AngelSpec hidden-state/proposal parity, 64K
context support, concurrent serving or CUDA-graph correctness. Three repeats
provide a measured range, not a confidence interval. The long and short
fixtures contain different tasks, so their rate differences cannot be
attributed to context length alone.

## What this comparison controls and what it does not

The earlier single-run chat comparison below motivated the repeated suites.
Its limitations remain relevant when interpreting either set of results as a
ranking of two drafters.

**Matching tokens test the hooks, not the drafter.** Target verification and
sampling are exact: a proposed token is kept only when it equals the token the
target itself produces. Output therefore matches the serial run whatever the
drafter proposes. Matching tokens show that capturing the target's
hidden state and selections, and discarding rejected rows, leave the target
undisturbed. They say nothing about whether this port computes what the
AngelSpec reference computes. No test here runs the released checkpoint through
the reference and compares hidden states or proposals; the primitive checks
compare each kernel with a reference written for this port. The only drafter
signal is acceptance, discussed below.

**The two drafting lanes use different proposal policies.** The native patch
gives the DFly2 lane a fixed policy and leaves DSpark its usual one:

| | DSpark lane | DFly2 lane |
| --- | --- | --- |
| Draft depth | Trimmed each round by the drafter's confidences | Always five; every draft reported as certain |
| Suffix-lookup copy proposals | Enabled (the lane runner has no switch) | Disabled by the patch |
| Scheduler's draft cost | 3.5 ms constant, close to the measured 3.59 ms | The same 3.5 ms constant; measured 28.92 ms |

The table's measured costs and row counts describe the earlier single-run chat
comparison. The patch also forces draft-survival calibration off for DFly2; the lane runner
leaves it off for DSpark as well, so that is not a difference in these runs.
The released DFly2 checkpoint carries no confidence head, so the lane has no
acceptance estimate to give the scheduler. The result is that DFly2 verified
5.99 rows a round against DSpark's 4.42, and could never choose not to draft.
`dspark_fixed5` in the launcher gives DSpark the same fixed five-draft policy
and removes the first row of that table as a difference. Copies cannot be
switched off for DSpark without a native change, so that row remains.

**The earlier chat prompts were short.** Their longest request reaches
398 positions including its reply. The target's sparse layers select the top
512 positions, so at this length they select every position, and the selection
DFly2 shares with the target is never a real choice. DFly2's published results
are for 2K to 128K contexts. The subsequent
[long-context fixture](../evidence/fixtures/long-requests.json) (1,210 to 6,892
prompt tokens) now has twelve completed GPU receipts above. Coverage still
stops at these six longer prompts and the configured 8,192-token pool.

In the earlier comparison each mode was run once, in the order serial, DFly2, DSpark, on six prompts
of which five ask for code or structured output.

## Earlier single-run public-launcher GPU validation

The portable public launcher and newly built public image completed serial,
DFly2 and DSpark runs on two NVIDIA GB10 systems. All three modes used the same
executable, image contents, launcher source, target configuration and six
request payloads. All **959 output token IDs and all six finish reasons
matched**, including the historical serial reference.

| Mode | Estimated decode tokens/s | End-to-end tokens/s | Total seconds including prefill | Accepted / proposed draft tokens |
| --- | ---: | ---: | ---: | ---: |
| Drafting disabled | 38.49 | 35.34 | 27.14 | 0 / 0 |
| Existing DSpark | 74.98 | 63.26 | 15.16 | 690 / 916 |
| Custom DFly2 | 45.06 | 39.05 | 24.56 | 685 / 1,351 |

On the estimated decode rate DFly2 was about 17% faster than drafting disabled
and about 40% slower than DSpark. End to end, with prefill included, DFly2 was
10.5% faster than drafting disabled and DSpark 79%. The public path reproduced
target output equivalence; DSpark remained faster on this workload. These were
one run per mode, in serial/DFly2/DSpark order, with the same eager, one-stream,
8,192-token configuration and profiling enabled. They are not a statistical
speed study.

The decode-only figure leaves out a prefill cost that DFly2 adds. Its first
request's prefill took 1.36 s against 0.51 s without drafting, and later
requests' prefills were 0.03 to 0.05 s longer. The native profiler's own
prefill time explains about 0.2 s of the 1.0 s total difference; the rest, almost all of it in
the first request, is not accounted for by any published counter.

### Per request

| Request | Output tokens | No drafting tok/s | DSpark tok/s | DFly2 tok/s | DSpark tokens/round | DFly2 tokens/round |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| greedy, merge intervals (code) | 197 | 38.28 | 94.69 | 59.57 | 5.30 | 5.03 |
| greedy, latest-row SQL | 133 | 38.71 | 80.98 | 53.66 | 4.00 | 4.12 |
| greedy, extract JSON | 41 | 38.10 | 86.96 | 65.57 | 5.00 | 5.00 |
| sampled, binary search (code) | 256 | 38.64 | 77.27 | 50.40 | 3.70 | 3.81 |
| sampled, multi-turn edit (code) | 96 | 38.78 | 103.26 | 77.87 | 5.59 | 5.94 |
| sampled, explain caching (prose) | 236 | 38.34 | 54.15 | **27.65** | 2.26 | 2.16 |

Rates are estimated decode tokens per second for that request; with elapsed
times rounded to 0.01 s, the shortest request's rate is good to about 2%. On
the prose request DFly2 was 28% slower than not drafting at all: 23% of its
proposals were accepted, and the fixed policy kept paying for five drafts and
six verification rows regardless. DSpark was faster than no drafting on every
request.

Tokens landed per round are close to equal for the two drafters on every
request, and DFly2 is ahead on three of the six. The drafters are not far apart
in what they get right. They are far apart in what a round costs.

### Where a round's time goes

| | No drafting | DSpark | DFly2 |
| --- | ---: | ---: | ---: |
| Mean round, ms | 25.98 | 47.47 | 78.08 |
| Target forward, ms | 25.89 | 43.37 | 51.68 |
| Rows verified per round | 1.00 | 4.42 | 5.99 |
| Draft pass, ms | 0 | 3.59 | 28.92 |
| Proposed / accepted drafts per round | 0 / 0 | 3.42 / 2.57 | 4.99 / 2.53 |
| Decode tok/s if the draft pass were free | n/a | 81.04 | 71.53 |

These are the native profiler's means from the same receipts. The mean round
time multiplied by the number of rounds reproduces the estimated decode seconds
to within 0.015 s in every receipt, which is the cross-check the verifier now
enforces. Each extra verification row cost about 5.1 ms on this engine.

Two conclusions follow. A DFly2 draft pass costs about eight times a DSpark
pass and slightly more than a whole undrafted target step. And the last row is
an optimistic bound: with the same rounds and a drafter that cost nothing,
DFly2 would reach about 71.5 tok/s, still under DSpark's measured 74.98,
because it verifies six rows where DSpark verifies 4.4. Faster DFly2 kernels
alone cannot close the gap on this workload. Closing it needs fewer wasted
verification rows, more accepted tokens per round, or both.

The kernel microbenchmark bounds how much of the pass the custom kernels can
explain. Summing its medians over the launches in one five-draft pass gives 2.8
to 3.7 ms for the routed FP4 experts and 2.3 ms for the attention kernels,
5.1 to 6.0 ms in all, against a measured pass of 27 to 29 ms. Those
are warm one-GPU timings beside a two-node eager pass, so the comparison is
rough, but it places roughly four fifths of the pass outside the kernels that
were benchmarked and optimized: dense projections, five evaluations of the
shared target head, mHC mixing, 21 cross-rank gathers and five blocking host
readbacks per pass. None of that has been profiled.

### Acceptance beside the publisher's figures

The checkpoint's [model card](https://huggingface.co/AngelSlim/DS-V4.1-Flash-DFly2-B6)
reports mean accepted length counting the bonus token, the same quantity as
tokens landed per round here. As read on 2026-11-26 it gives 4.28 and 4.70 on
two code benchmarks and 3.03 on MT-Bench at temperature 0, averaging 4.12 over
its general set, with decoding speedups averaging 3.14x on H20 GPUs at TP8.
This port landed 3.81 to 5.94 tokens per round on its five code and structured
prompts and 2.16 on its one prose prompt, 3.52 overall.

That is a sanity check and no more. The prompts, target revision, quantization
and sampling differ, and one prose prompt is not a benchmark. It does suggest
the port's proposals are broadly as good as intended on code, and that the
shortfall from the published speedups comes from the cost of a round here, 3.0
times an undrafted step, where the published speedup and accepted length imply
about 1.3.

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

The verifier cross-checks, matched-depth mode, suite tools and long-context
fixture were added after those runs. The merged PR passed **73 CPU checks** on
x86-64 Linux. The subsequent two-fixture validation above exercises the added
launcher paths in 24 runs on the two GB10 systems, including fixed-five DSpark.
After both suites were exported, **73 CPU checks** and all published-evidence
verifiers passed on Windows, with **three launcher pins confirmed from Git
history and none unreadable**. This checks the recorded evidence; it is
separate from the GPU runs themselves.

## Recompute the comparison

From the repository root, using only the Python standard library:

```sh
python3 tools/dfly2/verify_evidence.py
python3 tools/dfly2/verify_portable_evidence.py
python3 tools/dfly2/verify_suite_evidence.py evidence/suite-chat
python3 tools/dfly2/verify_suite_evidence.py evidence/suite-long
python3 -m unittest discover -s tools/dfly2 -p test_evidence.py -v
python3 -m unittest discover -s tools/dfly2 -p test_portable_evidence.py -v
```

The verifier recomputes each reply's token digest, checks actual arrays and
finish reasons against the serial run, checks request hashes on both ranks,
reconciles counters, and calculates the rates from recorded elapsed times.
It also checks both recorded weight-upload receipts, primitive-test receipts,
and the original microbenchmark kernel's source hash. It checks the saved
evidence; it does not rerun GPU inference.

The verifier also requires two further native clocks to agree with each
estimate: the per-reply times must sum to the total, and the profiler's mean
round time multiplied by the number of rounds must reproduce the estimated
decode seconds. It requires the passing primitive receipt to carry the SHA-256
of the kernel source this repository's patch creates. The derived comparison
holds the per-request rows, end-to-end rates and profile figures quoted above,
so none of them is typed by hand.

All of this checks that the records agree with each other. The receipts are
not signed or attested by anything outside this repository, and a consistent
set of invented numbers would pass.

The portable verifier additionally binds all three runs to the executable
recorded by the public build, checks them against the launcher digest pinned
in `provenance.json` together with the commit that holds those bytes,
requires identical image contents and target metadata, and recomputes equality
against the fresh serial run. It also reports whether that serial output
matches the historical serial run. Both comparisons retain their own times.

The reviewed synthetic prompts, their complete tokenized inputs, sampling
settings, and seeds are in [chat-requests.json](../evidence/chat-requests.json).
Outputs and timings are in [chat-serial.json](../evidence/chat-serial.json),
[chat-dspark.json](../evidence/chat-dspark.json), and
[chat-dfly2.json](../evidence/chat-dfly2.json). The generated calculation is
[comparison.json](../evidence/comparison.json).

## Repeated suite design

The repeated suite adds the following controls to the earlier single-run
comparison. Both fixtures above are completed applications of it.

- **Matched depth.** `--mode dspark_fixed5` runs DSpark with the lane runner's
  existing `--fix-k 5`: five drafts every round, each treated as certain, the
  policy the DFly2 lane uses. Comparing it with DFly2 separates the drafters
  from the depth policy. Copy proposals stay on for DSpark.
- **Long context.** `evidence/fixtures/long-requests.json` holds six reviewed
  synthetic requests of 1,210 to 6,892 prompt tokens, three greedy and three
  sampled, generated with the target's own tokenizer and template. Every one is
  more than twice the top-512 selection size, so the selection has to leave
  positions out and DFly2's prompt-capture path sees multi-chunk prompts.
- **Repeats and order.** `run_suite.py` runs every mode once per repeat and
  rotates which mode goes first, comparing every run, including repeated
  serial runs, with the first serial run. The suite verifier reports a median
  and range per mode.

[RUN.md](RUN.md#matched-repeated-and-long-context-suite) has the commands. The
first longer-input suite preserved the serial output in all twelve runs.
Future refusals or token mismatches remain findings about the port and should
be retained, rather than removed from a comparison.

## Timing scope and limits

All measurements here use one-stream eager runs with CUDA graphs disabled and
profiling enabled, an 8,192-token pool, and fixed requests. The earlier chat
prompts were 82–142 tokens; the longer-input prompts are 1,210–6,892 tokens.
Each output budget is 256 tokens. Combined output is 959 tokens for chat and
1,169 for the longer fixture. Matching token IDs establishes that the
verified speculative path retained these target outputs; it does not grade the
answers or establish correctness for all inputs and sampling settings.

Estimated decode seconds equal the native recorder's total request time minus
the sum of six recorded prefill times. The rate counts 953 tokens for chat and
1,163 for the longer fixture, excluding the first token generated during each
prefill. Native elapsed times are rounded to
0.01 seconds. This is not a production throughput benchmark or a statistical
estimate with confidence intervals. Serving at 64K was not measured by any
of these suites.

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

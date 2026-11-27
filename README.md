# DFly2 on Two Sparks

Experimental Tencent DFly2 speculative decoding for DeepSeek-V4.1-Flash-1017 on two NVIDIA GB10 nodes, using TensorFold's native Zig/CUDA runtime.

The latest, separately pinned v5 evaluation completed **39 GPU runs**: three rotated repeats across short chat, longer inputs and a frozen held-out fixture. Every run preserved its serial-reference output tokens and finish reasons. The new adaptive policy improved aggregate chat decode by **3.3%** over legacy DFly2 but regressed by **7.7%** on longer inputs, and was slower than legacy on **10 of their 12 individual requests**. It remains opt-in.

**Performance parity with DSpark was not reached.** Ordinary DSpark was faster than adaptive DFly2 on every measured request. The [complete v5 results](docs/EXPERIMENTAL_RESULTS.md) retain all ranges, request regressions and slower repeats. The [historical 24-run evaluation](docs/RESULTS.md) remains separate and unchanged. Exact final target outputs do not establish fidelity to Tencent's drafter: the [released-weight reference comparisons](docs/REFERENCE.md) still fail the full numerical gates.

This repository publishes the implementation, tests, setup recipe, and measured results so others can reproduce and improve it. It is an experimental port, with text, tensor parallelism across two nodes, one active stream, replicated KV, and eager execution. CUDA graphs, concurrent requests, vision, contexts beyond the measured fixtures including 64K, and production serving flows have not been qualified.

The separate [experimental build](docs/EXPERIMENTAL.md) adds GPU token feedback, padded RDMA winner exchange, fixed/adaptive draft-depth controls, copy-policy controls, and bounded reference captures. [Released-weight comparisons](docs/REFERENCE.md) now run the original Tencent methods on the same captured state: **full numerical parity still fails**, even though most proposal chains match. Its source pins and evidence remain separate from the historical release.

## What is included

- A native patch that captures the accepted target predecessor's hidden state and sparse KV selections, then runs the separate DFly2 B6 drafter while preserving target verification and sampling.
- A byte-preserving TP2 checkpoint converter and GPU upload checker.
- CUDA kernels compiled for GB10/SM121, numerical fixtures, and a kernel microbenchmark.
- A standalone, configurable two-node validation launcher with a read-only preflight default and ownership-checked cleanup, a matched-depth DSpark mode, and a suite runner for rotated, repeated runs.
- A reviewed long-context request fixture (1,210 to 6,892 prompt tokens) for the regime the short chat prompts cannot reach.
- A pinned public source/build recipe and sanitized benchmark evidence.

This is a patch-based integration repository. The build prepares pinned TensorFold and deployment-recipe sources rather than copying their full history into this repository. Model weights and generated model/kernel assets are downloaded or prepared separately. This is a TensorFold backend, not a TensorRT-LLM plugin.

## Setup

Use ARM64 Linux on the two GB10 nodes with working NVIDIA drivers, NVIDIA Container Toolkit, Docker, SSH, and the inter-node RDMA link. The initial validation launcher requires at least 112 GiB available host memory per node before loading the model. See the full disk, networking, model, and toolchain requirements in [Build and model preparation](docs/BUILD.md).

1. Follow [the prerequisites and model preparation](docs/BUILD.md). For the new controls, build the separately pinned [experimental image](docs/EXPERIMENTAL.md#separately-pinned-experimental-build); the historical build and results remain available separately.
2. Download the separately licensed [DFly2 B6 checkpoint](https://huggingface.co/AngelSlim/DS-V4.1-Flash-DFly2-B6), then pack it with `tools/dfly2/pack_weights.py`. The exact revision and layout are documented in [the weight guide](docs/WEIGHTS.md).
3. Copy [the example cluster configuration](examples/cluster.example.json) to `cluster.local.json`, fill in both nodes and paths, and follow [the run guide](docs/RUN.md).
4. Run the read-only preflight first. An explicit `--run` starts an isolated test pair after the checks pass. The launcher does not stop existing workloads.

The v5 public image has embedded SM121 kernels and completed the 39-run evaluation. [Build and component evidence](docs/EXPERIMENTAL.md#recorded-v5-build-and-component-validation) distinguishes the source-host tests, constructed GPU checks and actual model runs.

## Results and limits

Median estimated decode tokens/s over three repeats per measured mode and fixture:

| Backend | Chat: 82–142 prompt tokens | Longer inputs: 1,210–6,892 | Held out: 84–3,018 |
| --- | ---: | ---: | ---: |
| No drafting | 38.60 | 38.41 | 38.02 |
| Legacy DFly2 | 45.45 | 55.78 | Not measured |
| Adaptive DFly2 | 46.97 | 51.46 | 47.32 |
| DSpark, fixed five, copies off | 67.97 | 79.17 | Not measured |
| Ordinary DSpark | 74.57 | 81.61 | 71.67 |

The adaptive candidate enables GPU token feedback, padded RDMA winner exchange and observed-cost depth selection. Reference normalization rounding stays off. Ordinary DSpark retains its adaptive depth and suffix-copy proposals; the fixed-five control disables copying. All modes use the same v5 executable and image contents. These are measurements of the implemented policies on these fixtures, not a general algorithm ranking.

The small chat aggregate gain hides five request regressions; its cache-explanation request improves 29.4% versus legacy but remains about 7% slower than serial. Longer-input aggregate decode regresses 7.7% versus legacy, also with five request regressions. On held-out tasks, adaptive DFly2 gains 24.5% over serial but remains 34.0% below DSpark, and two prose requests remain slower than serial. No held-out legacy comparison was measured.

Including prefill, adaptive rates are 42.06, 31.17 and 36.02 tok/s for chat, long and held out respectively. The [full evaluation](docs/EXPERIMENTAL_RESULTS.md) includes every mode's range, request-level changes, prefill variation and evidence links. Three repeats provide observed ranges, not confidence intervals. The fixtures use different tasks; their differences cannot be attributed to context length alone.

These are two-node, single-stream eager runs with an 8,192-token pool. Detailed timing events and capture are off; ordinary native summary counters remain on. Each fixture is half greedy and half temperature 0.7/top-p 1/top-k 20 with fixed seeds. Decode rate subtracts recorded prefill time and excludes the first token generated during each prefill; end-to-end includes prefill. Model loading and launcher startup are excluded. Native timings and displayed rates are rounded.

The drafter was released for the unsuffixed DeepSeek-V4.1-Flash target; this experiment uses the 1017 target and verifies compatible geometry and RoPE settings. The reference work identifies remaining internal differences. Neither these output checks nor the longest measured prompt qualify 64K context, concurrent requests, CUDA graphs or production serving.

## Validation

Run the CPU checks and public-evidence verification from the repository root:

```sh
python3 scripts/check.py
```

The check recomputes every published rate from the receipts, cross-checks each against two further native clocks, and confirms the primitive receipt describes the kernel this patch builds. It establishes that the records agree with each other. It cannot establish that they came from a GPU.

GPU primitive checks, byte-upload checks, and full two-node comparisons are separate procedures in [the run guide](docs/RUN.md). They require the relevant hardware and prepared model assets. Small synthetic kernel tests are not end-to-end performance measurements.

Use the [experimental comparison instructions](docs/EXPERIMENTAL.md#validation-and-comparisons) for the new controls and frozen evaluation, or the [historical suite instructions](docs/RUN.md#matched-repeated-and-long-context-suite) for the earlier four-mode suites. Every published attempt and its qualification status is retained.

## Contributing

Please report the target/checkpoint revisions, hardware, context length, sampling settings, and whether a result includes prefill. Include matched DSpark and no-drafting comparisons when claiming a speed improvement. Preserve negative results and distinguish kernel-only timing from full serving throughput.

Useful next steps, most valuable first:

1. Resolve the remaining original-reference discrepancies using the released-weight replay and captured intermediate values. Keep numerical failures and proposal mismatches visible.
2. Reduce correction attention and repeated target-head costs identified by the new detailed GPU profiles, then measure complete serving requests with those extra timing events disabled.
3. Investigate the observed-cost scheduler's request regressions and calibration costs before changing defaults; use a new frozen evaluation for subsequent tuning.
4. Extend context and workload coverage with matched no-drafting and DSpark controls; qualify larger contexts and concurrent serving separately.

New execution modes need target-output checks and cache-lifetime tests before being enabled by default.

## Attribution and license

This work builds on [TensorFold](https://github.com/bertholomus/TensorFold), the [MiaAI-Lab two-Spark deployment recipe](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks), and [Tencent AngelSpec](https://github.com/Tencent/AngelSpec). See [ATTRIBUTION.md](ATTRIBUTION.md), [NOTICE](NOTICE), and [LICENSE](LICENSE) for provenance and retained notices. Model weights and external container images retain their respective licenses and terms.

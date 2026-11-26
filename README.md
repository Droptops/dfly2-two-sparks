# DFly2 on Two Sparks

Experimental Tencent DFly2 speculative decoding for DeepSeek-V4.1-Flash-1017 on two NVIDIA GB10 nodes, using TensorFold's native Zig/CUDA runtime.

The repeated suites completed **24 GPU runs** on two GB10 systems: four modes, three repeats each, on both short chat prompts and longer inputs, with the starting mode rotating. Every run matched its fixture's serial reference: **959 output tokens for chat**, **1,169 for longer inputs**, and all six finish reasons in each case. DFly2's median estimated decode rate was **45.34 tok/s on chat** and **55.75 tok/s on the longer fixture**. Both DSpark modes remained faster on both fixtures.

Target verification is exact, so matching tokens establish output preservation on these requests, not fidelity to Tencent's drafter implementation. The fixed-five comparison matches draft depth, but DSpark still has suffix-copy proposals while DFly2 does not. Ordinary DSpark also adjusts depth by confidence. These numbers describe the implemented lanes on these fixtures. [Results and limitations](docs/RESULTS.md) retain the full ranges, per-request evidence and earlier single-run measurements.

This repository publishes the implementation, tests, setup recipe, and measured results so others can reproduce and improve it. It is an experimental port, with text, tensor parallelism across two nodes, one active stream, replicated KV, and eager execution. CUDA graphs, concurrent requests, vision, contexts beyond the measured fixtures including 64K, and production serving flows have not been qualified.

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

1. Follow [the build instructions](docs/BUILD.md) to prepare the pinned public sources, build the image, and generate the target rank cache and kernel assets.
2. Download the separately licensed [DFly2 B6 checkpoint](https://huggingface.co/AngelSlim/DS-V4.1-Flash-DFly2-B6), then pack it with `tools/dfly2/pack_weights.py`. The exact revision and layout are documented in [the weight guide](docs/WEIGHTS.md).
3. Copy [the example cluster configuration](examples/cluster.example.json) to `cluster.local.json`, fill in both nodes and paths, and follow [the run guide](docs/RUN.md).
4. Run the read-only preflight first. An explicit `--run` starts an isolated test pair after the checks pass. The launcher does not stop existing workloads.

The public build and portable launcher have completed both repeated GPU suites. [Results and methodology](docs/RESULTS.md) separate those measurements from build/CPU checks and retain the earlier integration results.

## Results and limits

Median estimated decode tokens/s over three repeats per mode and fixture:

| Backend | Short chat: 82–142 prompt tokens | Longer inputs: 1,210–6,892 prompt tokens |
| --- | ---: | ---: |
| No drafting | 38.76 | 38.21 |
| DSpark | 73.65 | 81.39 |
| DSpark, fixed five drafts | 66.23 | 76.66 |
| DFly2 | 45.34 | 55.75 |

Compared with no drafting, DFly2's median decode gains were 17.0% on chat and 45.9% on the longer fixture. Including prefill, the gains were 10.8% and 25.1%. On the short prose request DFly2 was about 28% slower than no drafting (27.61 against 38.52 tok/s). It remained 31.5% below fixed-five DSpark on chat and 27.3% below it on longer-input median decode rate. [Full results](docs/RESULTS.md) include end-to-end rates, measured minimum/maximum ranges and the slower DFly2 prefill repeat. Different tasks are used in the two fixtures; their rate differences cannot be attributed to context length alone.

These are single-stream eager measurements with profiling enabled, the same native executable and image contents for all modes, and an 8,192-token pool. Each fixture has three greedy requests and three using temperature 0.7, top-p 1, top-k 20, and fixed seeds. Decode seconds are estimated by subtracting recorded prefill time from total request time; the rate's token count excludes the first output token produced during each prefill. Native timings are rounded to 0.01 seconds. The [chat](evidence/suite-chat/comparison.json) and [longer-input](evidence/suite-long/comparison.json) evidence records every run and the calculation. Three repeats give a measured range, not a confidence interval or a general performance ranking.

[Results and methodology](docs/RESULTS.md) explain the exact checks, numerical differences from the reference implementation, omissions from public receipts, and untested cases. Final target-token equality does not establish whole-drafter numerical parity or identical proposal acceptance: no test yet compares this port's hidden states or proposals with the AngelSpec reference on the released weights. The longer-input suite extends observed capture-path coverage to these prompts; it does not qualify 64K context or production serving. The drafter was released for the unsuffixed DeepSeek-V4.1-Flash target; this experiment uses the 1017 target and verifies compatible geometry and RoPE settings.

## Validation

Run the CPU checks and public-evidence verification from the repository root:

```sh
python3 scripts/check.py
```

The check recomputes every published rate from the receipts, cross-checks each against two further native clocks, and confirms the primitive receipt describes the kernel this patch builds. It establishes that the records agree with each other. It cannot establish that they came from a GPU.

GPU primitive checks, byte-upload checks, and full two-node comparisons are separate procedures in [the run guide](docs/RUN.md). They require the relevant hardware and prepared model assets. Small synthetic kernel tests are not end-to-end performance measurements.

Use the [suite instructions](docs/RUN.md#matched-repeated-and-long-context-suite) to reproduce all four modes with rotated repeats. The two published suites retain all 24 successful runs and the full ranges of timings.

## Contributing

Please report the target/checkpoint revisions, hardware, context length, sampling settings, and whether a result includes prefill. Include matched DSpark and no-drafting comparisons when claiming a speed improvement. Preserve negative results and distinguish kernel-only timing from full serving throughput.

Useful next steps, most valuable first:

1. A parity test that runs the released checkpoint through the AngelSpec reference and this port on the same captured context and compares hidden states and proposals. Nothing else can show the port is a faithful DFly2.
2. Repeating the comparison on additional independently chosen tasks and extending context coverage beyond the six measured longer prompts.
3. A runner switch for suffix-lookup copy proposals, so DSpark and DFly2 can be compared with copies off for both. This needs a native change and a rebuild.
4. An acceptance estimate for DFly2, for example from its own softmax, so the scheduler can trim depth instead of always verifying six rows.
5. Profiling the draft pass. The custom kernels account for about 5 to 6 ms of its 29 ms; the remainder is spread over dense projections, five evaluations of the shared target head, 21 cross-rank gathers and five blocking host readbacks.

New execution modes need target-output checks and cache-lifetime tests before being enabled by default.

## Attribution and license

This work builds on [TensorFold](https://github.com/bertholomus/TensorFold), the [MiaAI-Lab two-Spark deployment recipe](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks), and [Tencent AngelSpec](https://github.com/Tencent/AngelSpec). See [ATTRIBUTION.md](ATTRIBUTION.md), [NOTICE](NOTICE), and [LICENSE](LICENSE) for provenance and retained notices. Model weights and external container images retain their respective licenses and terms.

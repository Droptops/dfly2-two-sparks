# DFly2 on Two Sparks

Experimental Tencent DFly2 speculative decoding for DeepSeek-V4.1-Flash-1017 on two NVIDIA GB10 nodes, using TensorFold's native Zig/CUDA runtime.

The public launcher and freshly built image ran on two GB10 systems and matched all **959 reference output tokens and all six finish reasons**. In that small chat test, estimated decode speed was **45.06 tok/s with DFly2**, **74.98 tok/s with the existing DSpark drafter**, and **38.49 tok/s without drafting**. DSpark was faster on this workload.

Read those two results for what they are. Target verification is exact, so matching tokens show that the DFly2 hooks leave the target's output intact; any drafter, good or broken, would produce the same tokens. And the two drafting lanes did not run under the same proposal policy: DSpark trims its draft depth by confidence and may use suffix-lookup copy proposals, while this DFly2 lane always proposes five drafts and uses no copies. The numbers describe this integration, not the two drafters. [What the comparison controls](docs/RESULTS.md#what-this-comparison-controls-and-what-it-does-not) has the details, and a matched, repeated, long-context suite is ready to run but not yet measured.

This repository publishes the implementation, tests, setup recipe, and measured results so others can reproduce and improve it. It is an experimental port, with text, tensor parallelism across two nodes, one active stream, replicated KV, and eager execution. CUDA graphs, concurrent requests, vision, long contexts including 64K, and production serving flows have not been qualified.

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

The public build and portable launcher have completed a fresh two-node GPU comparison. [Results and methodology](docs/RESULTS.md) separate those measurements from build/CPU checks and retain the earlier integration results.

## Results and limits

| Backend | Estimated decode tok/s | End-to-end tok/s | Total seconds including prefill | Tokens landed per round |
| --- | ---: | ---: | ---: | ---: |
| No drafting | 38.49 | 35.34 | 27.14 | 1.00 |
| DSpark | 74.98 | 63.26 | 15.16 | 3.56 |
| DFly2 | 45.06 | 39.05 | 24.56 | 3.52 |

Decode-only, DFly2 was 17% faster than no drafting; end to end, including its slower prefill, 10.5%. On the one prose request it was slower than no drafting (27.65 against 38.34 tok/s). Both drafters landed about the same number of tokens per round, so the gap to DSpark is cost, not acceptance: a DFly2 draft pass took 28.92 ms against 3.59 ms, and its fixed depth verifies six rows every round.

These are single-stream eager measurements with profiling enabled, one run per mode in a fixed order, using the same freshly built executable and image contents for all three modes. Three requests were greedy; three used temperature 0.7, top-p 1, top-k 20, and fixed seeds. Prompts were 82–142 tokens; the combined output was 959 tokens. Decode seconds are estimated by subtracting recorded prefill time from total request time; the rate's token count excludes the first output token produced during each prefill. Native timings are rounded to 0.01 seconds. [Portable-run evidence](evidence/portable-validation/comparison.json) records the calculation. These results are neither a general performance ranking nor a claim to reproduce Tencent's published speedups.

[Results and methodology](docs/RESULTS.md) explain the exact checks, numerical differences from the reference implementation, omissions from public receipts, and untested cases. Final target-token equality does not establish whole-drafter numerical parity or identical proposal acceptance: no test yet compares this port's hidden states or proposals with the AngelSpec reference on the released weights. Every published prompt is under 400 tokens with its reply, so the target's top-512 selection keeps every position and DFly2's shared-selection path has only been exercised in that trivial form. The drafter was released for the unsuffixed DeepSeek-V4.1-Flash target; this experiment uses the 1017 target and verifies compatible geometry and RoPE settings.

## Validation

Run the CPU checks and public-evidence verification from the repository root:

```sh
python3 scripts/check.py
```

The check recomputes every published rate from the receipts, cross-checks each against two further native clocks, and confirms the primitive receipt describes the kernel this patch builds. It establishes that the records agree with each other. It cannot establish that they came from a GPU.

GPU primitive checks, byte-upload checks, and full two-node comparisons are separate procedures in [the run guide](docs/RUN.md). They require the relevant hardware and prepared model assets. Small synthetic kernel tests are not end-to-end performance measurements.

For a comparison worth citing, run the [matched suite](docs/RUN.md#matched-repeated-and-long-context-suite): all four modes, three rotated repeats, on both the chat fixture and the long-context fixture.

## Contributing

Please report the target/checkpoint revisions, hardware, context length, sampling settings, and whether a result includes prefill. Include matched DSpark and no-drafting comparisons when claiming a speed improvement. Preserve negative results and distinguish kernel-only timing from full serving throughput.

Useful next steps, most valuable first:

1. A parity test that runs the released checkpoint through the AngelSpec reference and this port on the same captured context and compares hidden states and proposals. Nothing else can show the port is a faithful DFly2.
2. Publishing the matched suite on the long-context fixture, which is where DFly2 is designed to help.
3. A runner switch for suffix-lookup copy proposals, so DSpark and DFly2 can be compared with copies off for both. This needs a native change and a rebuild.
4. An acceptance estimate for DFly2, for example from its own softmax, so the scheduler can trim depth instead of always verifying six rows.
5. Profiling the draft pass. The custom kernels account for about 5 to 6 ms of its 29 ms; the remainder is spread over dense projections, five evaluations of the shared target head, 21 cross-rank gathers and five blocking host readbacks.

New execution modes need target-output checks and cache-lifetime tests before being enabled by default.

## Attribution and license

This work builds on [TensorFold](https://github.com/bertholomus/TensorFold), the [MiaAI-Lab two-Spark deployment recipe](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks), and [Tencent AngelSpec](https://github.com/Tencent/AngelSpec). See [ATTRIBUTION.md](ATTRIBUTION.md), [NOTICE](NOTICE), and [LICENSE](LICENSE) for provenance and retained notices. Model weights and external container images retain their respective licenses and terms.

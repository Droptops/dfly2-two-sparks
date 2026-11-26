# DFly2 on Two Sparks

Experimental Tencent DFly2 speculative decoding for DeepSeek-V4.1-Flash-1017 on two NVIDIA GB10 nodes, using TensorFold's native Zig/CUDA runtime.

The integration runs on real hardware and matched all **959 reference output tokens and finish reasons** in a six-request chat test. In that small test, estimated decode speed was **45.84 tok/s with DFly2**, **74.05 tok/s with the existing DSpark drafter**, and **38.46 tok/s without drafting**. DSpark was faster on this workload.

This repository publishes the implementation, tests, setup recipe, and measured results so others can reproduce and improve it. It is an experimental port, with text, tensor parallelism across two nodes, one active stream, replicated KV, and eager execution. CUDA graphs, concurrent requests, vision, long contexts including 64K, and production serving flows have not been qualified.

## What is included

- A native patch that captures the accepted target predecessor's hidden state and sparse KV selections, then runs the separate DFly2 B6 drafter while preserving target verification and sampling.
- A byte-preserving TP2 checkpoint converter and GPU upload checker.
- CUDA kernels compiled for GB10/SM121, numerical fixtures, and a kernel microbenchmark.
- A standalone, configurable two-node validation launcher with a read-only preflight default and ownership-checked cleanup.
- A pinned public source/build recipe and sanitized benchmark evidence.

This is a patch-based integration repository. The build prepares pinned TensorFold and deployment-recipe sources rather than copying their full history into this repository. Model weights and generated model/kernel assets are downloaded or prepared separately. This is a TensorFold backend, not a TensorRT-LLM plugin.

## Setup

Use ARM64 Linux on the two GB10 nodes with working NVIDIA drivers, NVIDIA Container Toolkit, Docker, SSH, and the inter-node RDMA link. The initial validation launcher requires at least 112 GiB available host memory per node before loading the model. See the full disk, networking, model, and toolchain requirements in [Build and model preparation](docs/BUILD.md).

1. Follow [the build instructions](docs/BUILD.md) to prepare the pinned public sources, build the image, and generate the target rank cache and kernel assets.
2. Download the separately licensed [DFly2 B6 checkpoint](https://huggingface.co/AngelSlim/DS-V4.1-Flash-DFly2-B6), then pack it with `tools/dfly2/pack_weights.py`. The exact revision and layout are documented in [the weight guide](docs/WEIGHTS.md).
3. Copy [the example cluster configuration](examples/cluster.example.json) to `cluster.local.json`, fill in both nodes and paths, and follow [the run guide](docs/RUN.md).
4. Run the read-only preflight first. An explicit `--run` starts an isolated test pair after the checks pass. The launcher does not stop existing workloads.

The new launcher and public build path are release-preparation changes. Their validation status is listed in [the results](docs/RESULTS.md); the measured inference results describe the original implementation and harness. Do not infer new throughput results from source preparation or CPU tests.

## Results and limits

| Backend | Estimated decode tok/s | Total seconds including prefill |
| --- | ---: | ---: |
| No drafting | 38.46 | 27.21 |
| DSpark | 74.05 | 15.28 |
| DFly2 | 45.84 | 23.73 |

These are single-stream eager measurements with profiling enabled. Three requests were greedy; three used temperature 0.7, top-p 1, top-k 20, and fixed seeds. Prompts were 82–142 tokens; the combined output was 959 tokens. Decode time is estimated from rounded native timings and excludes the first token from each prefill. These results are neither a general performance ranking nor a claim to reproduce Tencent's published speedups.

[Results and methodology](docs/RESULTS.md) explain the exact checks, numerical differences from the reference implementation, omissions from public receipts, and untested cases. Final target-token equality does not establish whole-drafter numerical parity or identical proposal acceptance. The drafter was released for the unsuffixed DeepSeek-V4.1-Flash target; this experiment uses the 1017 target and verifies compatible geometry and RoPE settings.

## Validation

Run the CPU checks and public-evidence verification from the repository root:

```sh
python3 scripts/check.py
```

GPU primitive checks, byte-upload checks, and full two-node comparisons are separate procedures in [the run guide](docs/RUN.md). They require the relevant hardware and prepared model assets. Small synthetic kernel tests are not end-to-end performance measurements.

## Contributing

Please report the target/checkpoint revisions, hardware, context length, sampling settings, and whether a result includes prefill. Include matched DSpark and no-drafting comparisons when claiming a speed improvement. Preserve negative results and distinguish kernel-only timing from full serving throughput.

Useful next steps include profiling draft correction and host synchronization, qualifying long contexts, and investigating graph capture. New execution modes need target-output checks and cache-lifetime tests before being enabled by default.

## Attribution and license

This work builds on [TensorFold](https://github.com/bertholomus/TensorFold), the [MiaAI-Lab two-Spark deployment recipe](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks), and [Tencent AngelSpec](https://github.com/Tencent/AngelSpec). See [ATTRIBUTION.md](ATTRIBUTION.md), [NOTICE](NOTICE), and [LICENSE](LICENSE) for provenance and retained notices. Model weights and external container images retain their respective licenses and terms.

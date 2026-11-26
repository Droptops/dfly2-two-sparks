# Attribution and source revisions

This is a patch and validation repository for the native TensorFold DeepSeek
family. It builds on the projects below; their source notices are retained.
The newly written integration code is Apache-2.0, as stated in [LICENSE](LICENSE).

| Source | Revision used | Contribution |
| --- | --- | --- |
| [TensorFold](https://github.com/ashhart/TensorFold) and [BertholomusAI's DeepSeek fork](https://github.com/bertholomus/TensorFold) | `fbbc97132cd004237d9887b8d3185896efaedc5c` | Native server, target model, exact verifier/sampler, CUDA abstractions, TP2 transport, and existing DSpark implementation. The new DFly2 path modifies and calls this code. Apache-2.0; earlier MIT notices retained. |
| [MiaAI-Lab two-Spark DeepSeek recipe](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks) | `c7f2b85100f0a7b099c92a4e21825e30d93c9c51` | The deployed patch baseline and build/runtime recipe on which this integration was tested. Apache-2.0, Copyright 2026 MiaAI-Lab and credited contributors. |
| [Tencent AngelSpec](https://github.com/Tencent/AngelSpec) | `e02ea36e65d44e6e165c4ba73c139d500e12f2d9` | DFly2 architecture and arithmetic reference: draft attention, mHC, slot-0 fusion, hidden/attentive correction, and packed FP4 interpretation. Reimplemented in Zig/CUDA. Copyright 2026 Tencent; Apache-2.0 with the upstream third-party notices preserved. |
| [AngelSlim DFly2 B6 checkpoint](https://huggingface.co/AngelSlim/DS-V4.1-Flash-DFly2-B6) | `d08e55a6897b1974307ea9771038838de5986e66` | Tensor names, shapes and released drafter weights used for the measured test. The repository contains the converter and hash manifest, not the weights. |
| [Mia's AI Lab EXL3 target](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw) | Installed 1017 EAQ v1 2.9 bpw pack | Target checkpoint used for validation, with DeepSeek's Engram data and the recipe's tokenizer/template. No target weights are redistributed. |

The family and recipe notices identify the underlying DeepSeek, ExLlamaV3,
Triton, PyTorch, CUTLASS, Pillow, and other upstream contributions. Complete
copies are in [LICENSES](LICENSES). In particular, the recipe's actual license
at the pinned revision is Apache-2.0. `MiaAI-Lab-MIT.txt` is a separate notice
inherited from TensorFold for its Flash Next adaptation; it is not the license
of the DeepSeek recipe.

The historical upstream attribution documents are preserved verbatim. Their
descriptions of their own authors' development and validation refer to those
upstream projects. This integration's changes and tests are described in
[NOTICE](NOTICE), [docs/RESULTS.md](docs/RESULTS.md), and the public evidence.

The saved pre-optimization kernel in
`evidence/kernels/dsv41_dfly2_before_optimization.cu` is this integration's
original native implementation. Its original bytes and SHA-256 are retained
so the FP4 microbenchmark can be rerun against the optimized version.

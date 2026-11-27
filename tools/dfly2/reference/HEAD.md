# Shared EXL3 head diagnostic

The optional proposal replay uses the deployed target's actual TP2 EXL3 head
and BF16 embedding rows. It does not dequantize the target or invent a second
head implementation. The native probe calls the existing `rotMany` and
`glinear` kernels with the same BF16 input, FP16 rotation, F32 logits, m=1,
half-vocabulary plans, PDL flag, and L2 discard flag used by the serving drafter.

The native integration adds one selective weight-loader entry point and one
standalone executable in `weights.zig`, `zig/build/cuda.zig`, and the new
`dfly_head_main.zig`; it does not alter existing head math. From the patched
TensorFold source tree, build it with the project's pinned Zig compiler:

```sh
zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 \
  -Doptimize=ReleaseFast tf-dsv41-dfly-head
zig-out/bin/tf-dsv41-dfly-head \
  TARGET_MODEL_DIR RANK0_CACHE_FILE RANK1_CACHE_FILE \
  GROUPED_CUBIN INPUT_BF16 NEW_OUTPUT_DIR
```

Use these explicit CUDA flags when building the serving binaries too; their
native operations require embedded CUDA images. A full build uses the same
flags with `install native` in place of `tf-dsv41-dfly-head`. An unflagged
successful Zig build does not establish that the serving runtime has its GPU
kernels. The head-only probe obtains both rotation and grouped-linear kernels
from the supplied `GROUPED_CUBIN` (`linear_grouped.cubin`); it does not load the
full serving model or its embedded operations. The cubin must be the deployed
target's compatible kernel-kit artifact and its hash is retained in the report.

Input is raw little-endian BF16 `[rows,5120]`, for 1–128 rows. Each row uses a
separate m=1 launch. Both original cache files must be accessible locally; one
head half is loaded at a time. Only head trellis/sign/scale entries are uploaded,
and each half retains its original 64,640-column plan. The target's layers,
KV cache, embedding table, and model services are not loaded by this executable.

The probe refuses an existing output directory, initializes counters and
scratch, rejects nonfinite input/output, and writes completion metadata last:

- `logits.bin`: raw F32 `[rows,129280]`, rank0 vocabulary followed by rank1.
- `top1.json`: IDs, scores, runner-up IDs/scores, margins, kernel plans, and input,
  cubin, and output SHA-256 hashes.
- `head.started`: presence without `top1.json` means incomplete execution.

The Python extension first scores the five **captured native** corrected
vectors. It independently recomputes top1/top2 from every output logit and
requires all five IDs to equal the captured proposals. It also checks the
anchor, mask, and captured predecessor embedding bytes against both actual
rank caches. A failed gate stops the comparison and preserves its artifacts.

After that gate, each reference correction step uses Tencent's original module
and the original reference backbone output. Its chosen token selects the actual
target embedding row for the next step. The five resulting IDs are compared
with the native sequence. All corrected vectors, logits, margins, selected
embedding hashes, head-entry hashes, and subprocess logs are retained.

This diagnostic does not establish independent EXL3 mathematical correctness,
correct target index selection, unquantized-target equality, sampled-distribution
equality, or throughput. Original native pass logits were not part of capture
v1, so the gate proves captured proposal-ID reproduction and records probe
margins; it cannot establish original-pass logit or margin equality. Timing
includes subprocess startup, repeated head loads, synchronization, and output
I/O and must not be used as an inference benchmark.

Enable it alongside the usual `reference_replay.py` capture/checkpoint arguments:

```sh
python3 reference_replay.py --capture-root CAPTURES \
  --checkpoint RELEASED_DFLY_CHECKPOINT --angelspec PINNED_ANGELSPEC \
  --public-repo PUBLIC_REPO --output NEW_REPLAY_OUTPUT --device cuda \
  --head-probe /absolute/path/to/tf-dsv41-dfly-head \
  --head-target-model TARGET_MODEL_DIR \
  --head-rank0-cache RANK0_CACHE_FILE --head-rank1-cache RANK1_CACHE_FILE \
  --head-grouped-cubin GROUPED_CUBIN
```

All five head paths are required together. The root replay result passes only
when both its recorded numerical thresholds and all five free-running proposal
IDs pass. Numerical and proposal results remain separately visible, including
when one succeeds and the other fails.

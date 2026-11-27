# DFly2 real-checkpoint capture/replay v1

One directory per completed draft pass, recorded only by rank 0. Every `.bin`
is contiguous little-endian raw storage, with no header. Write files only after
the producing stream has synchronized. Keep captures private: they contain
hidden states, token IDs and potentially prompt-derived content.

Required files (native tensor dimensions, no batch axis):

| File | Dtype | Shape | Capture point |
| --- | --- | --- | --- |
| `target_hidden.bin` | BF16 | 5120 | `view.hidden` |
| `anchor_mask_embeddings.bin` | BF16 | 2,5120 | target embedding rows for anchor and MASK |
| `positions.bin` | I64 | 6 | `dr.positions`, q0 through q0+5 |
| `context.bin` | BF16 | 3,640,512 | each `dr.context[j]`, after gather_context |
| `valid.bin` | I32 | 3,640 | corresponding `dr.valid[j]` |
| `initial_block.bin` | BF16 | 6,4,5120 | `dr.h` immediately after init_block |
| `normalized.bin` | BF16 | 6,5120 | `dr.normalized` after final_norm |
| `previous_embeddings.bin` | BF16 | 6,5120 | previous-token embedding supplied to correctionInput for slots 0..5; slots 0 and 1 use anchor |
| `corrected.bin` | BF16 | 5,5120 | each slot 1..5 `dr.corrected` before native shared head |
| `proposals.bin` | I64 | 5 | returned native proposals |

Optional `layers.bin` BF16 [3,6,5120] records each completed layer's collapsed
hidden state; do not capture its stale asynchronous scratch before materializing
the pending post. The final layer's value precedes final_norm.
Optional `layers_residual.bin` BF16 [3,6,4,5120] records all four residual
streams at the same point. Either optional representation is compared when present.

`capture.json` is written last:

```json
{
  "schema": "dfly2-reference-capture-v1",
  "q0": 1210,
  "anchor_token": 123,
  "steps": 5,
  "world_size": 2,
  "native": {
    "binary_sha256": "64 lowercase hex characters",
    "checkpoint_manifest_sha256": "64 lowercase hex characters",
    "requests_sha256": "64 lowercase hex characters",
    "rms_norm_eps": 1e-20,
    "norm_rounding": "fused"
  }
}
```

The replay seals tensor files with hashes in its report; an optional `tensors`
dictionary can record each file's expected SHA-256 at capture time. Both are
integrity/provenance checks, not authentication of an untrusted capture.

The default comparison uses pinned original AngelSpec forward methods on the
released packed-FP4 checkpoint dequantized to exact BF16 values. It reports MTP
fusion, backbone and teacher-forced hidden correction independently. This does
not include the quantized target head and does not prove free-running proposal
parity or verify the target's index selection. Outputs include reference
corrected hidden vectors to enable an independent same-native-head check.

Use `--capture DIR` for one pass, or `--capture-root ROOT` to replay completed
passes in sorted relative-path order while loading the reference weights once.
Batch mode refuses empty roots and unfinished `capture.started` markers. Every
case retains its own report; the root report records all requested cases and
continues after numerical mismatches. Runtime failures stop the batch and are
preserved as incomplete evidence.

The optional shared-head extension is described in [HEAD.md](HEAD.md). Its
five-step loop feeds the original reference's own proposal embeddings back into
Tencent's correction module. It must first reproduce the captured native
proposal IDs from the captured native corrected vectors.

Tensor-parallel accumulation and BF16 rounding differ from an unsharded PyTorch
reference. Tolerances must be set before a run and are always recorded, along
with exact mismatch counts, maximum error, RMS error and relative RMS error.
Failure is preserved in the JSON report and returns exit code 2.

# Free-running proposal diagnostic

`proposal_replay.py` extends a completed original-Tencent replay with the actual
deployed TP2 EXL3 output head. It does not load the target transformer or run a
serving endpoint. The head probe reads only each rank cache's head tensors and
runs the existing grouped EXL3 kernels with the original half-vocabulary plans.
Each row runs separately with `m=1`, matching a native DFly proposal step.

The Python diagnostic requires both rank-cache files, the target model directory
containing its configuration, the deployed `linear_grouped.cubin`, and the new
`tf-dsv41-dfly-head` executable. These must describe the target that produced the
capture. It hashes the probe binary, cubin, target config, both rank-cache indexes,
selected head tensors, and every embedding row read. It also requires the anchor,
mask and captured native predecessor embeddings to equal the actual cache rows,
and requires those rows to agree between both rank caches.

Before any reference proposal is scored, the probe scores all five captured
native correction vectors. Every resulting greedy token must exactly equal its
captured native proposal. Failure preserves the report and subprocess files and
stops the reference proposal test. Passing this admission check reproduces those
five proposal IDs; it does not independently prove the head's mathematical
correctness or match original logits that were never captured.

The replay then reuses the original-Tencent backbone output. For each of the five
proposal slots it calls original `model.hidden_correction`, supplies the actual
target embedding of its own preceding choice, and scores that corrected vector
through the admitted EXL3 probe. Future predecessor slots remain zero until they
are reached; original correction's causal mask prevents those future values from
participating. The sixth slot is the last proposal, so its selected token does not
need an embedding lookup.

Raw FP32 logits, lower-token-ID tie breaking, runner-up tokens and margins are
retained for every probe call. Python independently validates the probe's JSON
summary against all raw logits. The final report records all five slot matches
and the matching prefix length. A completed mismatch is retained as
`complete: true, passed: false`; a failed admission or runtime error is incomplete.

The driver can call:

```python
from proposal_replay import run_proposal_parity

proposal_report = run_proposal_parity(
    model, captured, reference_outputs["normalized"], capture_dir,
    output_dir / "proposal-parity", args, torch,
)
```

Required argument attributes are `head_probe`, `head_target_model`,
`head_rank0_cache`, `head_rank1_cache`, and `head_grouped_cubin`.
`head_timeout` defaults to 300 seconds per subprocess. A `ProbeConfig` can be
passed instead of an argparse namespace. The outer report should include this
report and require its `passed` field in its overall pass condition.

This diagnostic checks five greedy draft proposals conditioned on captured target
state and using the deployed quantized target head. It does not independently
validate target capture/index selection, full target generation, unquantized-head
equivalence, sampling distributions, or speed. Repeated model subprocess startup
and complete logit transfers make its timings unsuitable as throughput evidence.

CPU wiring tests run with:

```console
python -m unittest discover -s tools/dfly2/reference -p test_proposal_replay.py -v
```

These tests use explicit synthetic fixtures and subprocess doubles. They are not
GPU parity evidence.

## Compact rank-cache transfer

If only one original rank cache is available on each machine, run
`extract_head_cache.py SOURCE OUTPUT --rank 0` (or `--rank 1`) on the corresponding
machine. It copies only the full target embedding table and that rank's three
EXL3 head tensors, byte for byte, into a valid compact TFDSRK01 cache. The probe
and Python replay accept these compact files without conversion. At the observed
6-bit head size this is approximately 1.57 GB per rank, instead of transferring
the full target cache. Future reference-selected embedding rows are unknown in
advance, so the complete embedding table is retained.

The CPU-only extractor refuses existing outputs and writes `OUTPUT.json` last,
including completion status, source index/footer hashes, original and new tensor
ranges, per-tensor hashes, and the complete output hash. Preserve and transfer
the sidecar with its cache; require `complete: true`. An extraction failure leaves
its partial output and incomplete report in place. The source cache is read-only.

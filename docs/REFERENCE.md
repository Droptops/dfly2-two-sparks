# Released-weight reference comparisons

**Full numerical parity with the original Tencent reference has not been
established.** Both recorded bundles completed 24 comparisons, and none passed
all numerical gates. The deployed output-head probe reproduced every captured
native proposal sequence, but the original reference's own five-token proposal
chain matched the native chain in only 22 of 24 comparisons in each bundle.

These are diagnostic results. They are separate from target-output checks and
serving-performance measurements in [RESULTS.md](RESULTS.md). A target verifier
can reject a different draft and still produce the same final target output;
matching final responses does not establish drafter/reference parity.

## What the reference uses

[reference_replay.py](../tools/dfly2/reference/reference_replay.py) verifies an
unmodified AngelSpec checkout at commit
`e02ea36e65d44e6e165c4ba73c139d500e12f2d9` and calls its original forward
methods. The configuration is derived from the released DFly2 B6 serving
checkpoint. The loader checks the source configuration and all four checkpoint
shards against [the pack manifest](../evidence/draft-pack-manifest.json).

The released expert E2M1 codes and E8M0 scales are decoded to exactly
representable BF16 values, with no requantization. The reference uses those
values in the original PyTorch modules; this does not reproduce a separate
publisher serving kernel. The recorded loader accounts for all 2,392 source
tensors and populates 94 parameters. Three vision-only router biases are
accounted for but unused in this text-only comparison; they are not assumed
equal to the text biases. The native TP2 pack continues to store the original
packed values, as described in [WEIGHTS.md](WEIGHTS.md).

Each capture records one completed native draft pass: target hidden state,
anchor/MASK embeddings, positions, gathered target KV context and validity
masks, native intermediate vectors, predecessor embeddings, and five draft
proposals. The backbone replay uses the captured context. The teacher-forced
correction comparison uses the native predecessor embeddings. A second
correction comparison also supplies the native normalized hidden vectors,
separating correction error from preceding backbone error.

The additional proposal test first scores the captured native corrected
vectors through the deployed TP2 EXL3 head. All five greedy IDs must match the
captured native proposals before the reference chain is evaluated. It then
feeds the original reference's own selected-token embeddings into subsequent
correction steps, using that same admitted head. Passing the head check means
the probe reproduced those captured IDs; original-pass logits were not captured
and cannot be compared.

## Recorded outcomes

| Recorded bundle | Completed reports | Distinct captured tensor sets | Full numerical passes | Native head checks passed | Exact five-token chains | Matching proposal slots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [v3 original](../evidence/experimental/reference-v3-original/reference.json) | 24 / 24 | 12 | 0 / 24 | 24 / 24 | 22 / 24 | 114 / 120 |
| [v4 normalization comparison](../evidence/experimental/reference-v4-rounding/reference.json) | 24 / 24 | 24 | 0 / 24 | 24 / 24 | 22 / 24 | 114 / 120 |

Each bundle contains six `chat` and six `long` captures for each of two native
variants. In v3, the `legacy` and `optimized` variants produced identical
captured tensor sets for the corresponding cases. Its 24 reports therefore
represent only 12 distinct captured states, not 24 independent observations.
In v4, the `optimized` fused-normalization baseline and the `candidate`
double-rounding variant produce 24 distinct complete tensor sets. Distinct
sets include intermediate outputs; they do not imply 24 unrelated input states.
The bundles also share cases and must not be pooled as 48 independent tests.

The fixed numerical gates require every element to satisfy `rtol=0.02` and
`atol=0.02`, and relative RMS error to be at most `0.01`. Relative RMS is the
RMS error divided by the reference RMS. All required comparisons must pass for
a case to pass. A small aggregate error can still fail because individual
elements exceed the recorded tolerance.

| Numerical comparison | v3 passing reports | v4 passing reports |
| --- | ---: | ---: |
| Initial fusion/block | 24 / 24 | 24 / 24 |
| Final normalized backbone output | 4 / 24 | 4 / 24 |
| Correction using reference backbone output | 0 / 24 | 0 / 24 |
| Correction given native normalized hidden state | 8 / 24 | 12 / 24 |

The correction output is a remaining failure in every report. The two failed
proposal chains in each bundle agree for the first two slots and diverge at
the remaining three. In v3 these are the duplicated variant observations of
the same longer-input captured state. The reports retain every mismatch,
including proposal IDs, slot matches, errors and head-probe margins.

## Normalization improves part of the discrepancy

The v4 candidate materializes the intermediate BF16 rounding used by the
original DFlash RMSNorm sites:

```text
BF16(weight * BF16(input * RMS_scale))
```

This is an opt-in native arithmetic change. The original reference's methods
and the recorded thresholds stay unchanged. It does not change every
attention normalization or tensor-parallel accumulation boundary.

Nine baseline/candidate pairs have identical captured target hidden state,
anchor/MASK embeddings, positions, gathered KV context and validity masks.
Their predecessor embedding hashes also match. Pairing these fields in the
[v4 report](../evidence/experimental/reference-v4-rounding/reference.json)
allows a comparison on the same inputs; the other three cases per variant
are not included in this paired calculation.

| Relative RMS error on the nine matched inputs | Fused median | Double-rounding median | Pairs with lower error |
| --- | ---: | ---: | ---: |
| Initial fusion/block | 0.00227226 | 0 | 9 / 9 |
| Final normalized backbone output | 0.00567929 | 0.00509160 | 9 / 9 |
| Correction using reference backbone output | 0.01954740 | 0.01753480 | 9 / 9 |
| Correction given native normalized hidden state | 0.01041530 | 0.00926160 | 9 / 9 |

Initial fusion is exact in all nine matched candidate cases. All four recorded
relative RMS metrics improve in each matched pair, supporting this rounding
boundary as one contributor. This is a partial improvement: the candidate
still fails the full numerical gates, and the overall proposal-match counts
do not improve. Comparisons across unmatched captured states cannot isolate
the effect of this change.

## Correction-only tensor-parallel diagnostic

The separate [correction TP diagnostic](../tools/dfly2/reference/CORRECTION_TP.md)
replaces selected reference projection outputs with BF16 half-width partial
sums to investigate native tensor-parallel rounding. The
[recorded summary](../evidence/correction-tp-v4-summary.json) contains 24 distinct
captured correction-input states and 96 finite comparisons across four variants.
Each substitution is compared with the unchanged original correction on the
same native normalized hidden state, predecessor embeddings and context. This
diagnostic does not run the backbone, head, proposal chain or target verifier.

| Correction variant | Cases within both recorded gates | Median paired absolute-RMS error reduction |
| --- | ---: | ---: |
| Original correction | 12 / 24 | 0% |
| BF16 partial sums for `down_proj` only | 13 / 24 | 3.13% |
| BF16 partial sums for attention `wo_b` only | 10 / 24 | 2.77% |
| BF16 partial sums for both projections | 11 / 24 | 5.36% |

The **12/24 original passes are correction-only checks given native hidden
states**. They do not revise the **0/24 full numerical passes** in either
original-reference bundle above. The substitutions deliberately change the
reference calculation and cannot count as original-reference parity passes.

Changing both projections reduces absolute RMS error in 22 cases and worsens
it in two. Despite the 5.36% median paired reduction, individual elements
outside tolerance increase from 6 to 16, and the strict pass count falls from
12 to 11: three previously passing cases fail while two failing cases pass.
This identifies a partial numerical contributor, but does not justify deploying
these substitutions as a native fix. No new native change or serving-speed
improvement is claimed from this diagnostic; the full-reference failures remain.

## Verify the public summaries or reproduce the diagnostic

The bundles contain numerical summaries and hashes, with separate
[v3](../evidence/experimental/reference-v3-original/manifest.json) and
[v4](../evidence/experimental/reference-v4-rounding/manifest.json) manifests.
From the repository root, verify their recorded consistency without loading
weights or using a GPU:

```sh
python3 tools/dfly2/reference/export_reference.py --verify evidence/experimental/reference-v3-original
python3 tools/dfly2/reference/export_reference.py --verify evidence/experimental/reference-v4-rounding
```

For a new numerical run, follow the [capture/replay format](../tools/dfly2/reference/CAPTURE.md),
[native head build and invocation](../tools/dfly2/reference/HEAD.md), and
[free-running proposal procedure](../tools/dfly2/reference/PROPOSALS.md).
They require the pinned source, released weights, target head/kernel assets
and newly generated capture state. Raw hidden-state captures, vectors and
logits are deliberately absent from the public bundles; their hashes do not
allow reconstruction. The public verifier checks consistency and integrity of
recorded summaries, not independent numerical recomputation or authentication
of GPU execution.

These tests do not independently validate target KV/index selection, EXL3
head mathematics, unquantized-target equivalence, sampling distributions or
general drafter quality. They examine five greedy proposals conditioned on
captured state, not arbitrary free-running target sessions. Recorded anchor
positions range from 93 to 7,018; this is not a 64K-context qualification.
Loading, repeated head subprocesses, synchronization and artifact I/O are
included in diagnostic wall time, so none of those times is a serving-speed
measurement.

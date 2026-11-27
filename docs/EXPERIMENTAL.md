# Controlled DFly2 performance experiments

The optional controls below support implementation comparisons and diagnostics.
The completed [39-run v5 evaluation](EXPERIMENTAL_RESULTS.md) reports mixed performance and request regressions. These controls remain opt-in; their presence alone is not a performance result. The historical results and their
source pins remain separate; they do not validate a newly built executable.
Build the candidate through the separate workflow below, stage the same image
and native executable on both nodes, then follow the identity, idle-host and
ownership checks in [RUN.md](RUN.md).

## Separately pinned experimental build

The historical `upstream.lock.json`, native patch, preparation/build scripts,
and published receipts remain unchanged. The additional native patch series is
pinned in `experimental/native.lock.json`. The checked-in lock is finalized
with reviewed patch SHA256 hashes and the resulting Git tree IDs. An
`unfinalized` lock or a missing hash is deliberately refused before any source
download or build. The lock's baseline hash uses the same sorted-JSON normalization as the
historical source receipt; each patch hash covers its exact file bytes.

The current series retains the original performance patch and adds the
initial-draft policy correction as a second patch. The retained
`experimental/v4.native.lock.json` identifies the earlier source used for the
v4 diagnostics. It is provided for provenance and reproducing those results;
its fixed/adaptive depth sweep is disqualified as described below. Use the
default current lock for new measurements.

Use the ARM64 Linux prerequisites in [BUILD.md](BUILD.md), then run from this
repository's root:

```bash
python3 scripts/prepare_experimental.py --destination _build-experimental
python3 scripts/prepare_experimental.py --destination _build-experimental --check
python3 scripts/build_experimental.py --source-dir _build-experimental \
  --tag dfly2-two-sparks:parity-lab --jobs 4 --print-command
python3 scripts/build_experimental.py --source-dir _build-experimental \
  --tag dfly2-two-sparks:parity-lab --jobs 4
```

Preparation first invokes the unchanged `prepare_source.py` for its pinned
baseline. After verifying that baseline, it applies the experimental patches
in order, checks the final native tree, and requires the inputs tree to remain
unchanged. Use a fresh destination for each preparation; incomplete directories
are retained. The source check also verifies upstream revisions, patch bytes,
the complete source-file manifest, and build-input/script hashes.

The original receipt and file manifest remain as `baseline-source-*` provenance.
The active `source-receipt.json` has the distinct
`dfly2-experimental-source-v1` schema, embeds both the original receipt and the
experimental lock, and explicitly says historical validation does not apply.
The historical source checker and `build_image.py` cannot certify this context.
Use the new experimental checker/build script for it.

The build reuses the same `build/Dockerfile`, base-image digest, compiler pins,
and Python tool inputs. Its image retains the experimental source receipt at
`/opt/tensorfold/share/build/source-receipt.json`. Explicit image labels record
the experimental source kind, lock hash and native tree. The separate
`experimental-build-receipt.json` records source provenance, the Docker exit
status and, on success, the immutable image ID and checked platform/labels.
The source identity is checked again after building. These receipts attest
source/build consistency, not model correctness or performance.

A failed build retains its receipt. Choose a fresh `--receipt` path for another
attempt; the helper refuses to overwrite an earlier one. `--print-command`
checks identity and stages the two Python tool inputs but does not invoke
Docker. Transfer the successful image and extract its binaries using the
procedures in [BUILD.md](BUILD.md), substituting the experimental image tag.
Keep those artifacts separate from previously measured binaries.

The new workflow has local Git-fixture integrity/refusal tests, requiring Git
but no network, Docker or GPU:

```bash
python3 -m unittest discover -s scripts -p test_experimental_build.py -v
```

## Recorded v5 build and component validation

The current two-patch source series completed a fresh ARM64 Linux Docker build.
All 15 changed native files matched the reviewed v5 source. The separate native
host suite passed **594/594 tests** with no GPU access requested; that suite ran
against mounted source in the earlier build environment, not inside the final
public image. The final image's CUDA information check reported embedded SM121
kernels. Its complete configuration and ordered filesystem-layer fingerprint
matched on both nodes, and all three staged executables matched across ranks.

In that public image, **13 device-feedback and 8 reference-normalization GPU
test groups passed** on constructed inputs. These are component checks, not
whole-drafter parity or throughput measurements. The reviewed hashes, source
pins and counts are in [experimental-build-v5.json](../evidence/experimental-build-v5.json).
The final image's lane runner starts with SHA256 `9469e5b23e08`; the earlier
mounted-source build starts with `078f89d13e63`. Those distinct artifacts must
not be substituted in receipts.

This evidence checks recorded build/source consistency. Offline hashes do not
independently authenticate GPU execution or establish model correctness. The
historical build and v4 reference receipts remain separate, and the v4 depth
sweep remains disqualified.

## Explicit controls

Add an `experimental` object to a private cluster configuration. Both hosts
must receive identical contents at their respective `nodes[].config_path`.
The launcher validates and records effective controls; it accepts no arbitrary
environment variables or shell fragments.

| Configuration field | Meaning |
| --- | --- |
| `device_feedback` | Boolean, default false. Keeps each selected proposal token on the GPU for the next correction step. The five sequential correction/head dependencies remain; proposals are checked after the final readback. |
| `rdma_best` | Boolean, default false. Pads each rank's score/token winner to four initialized floats so the engine can use its existing RDMA exchange. The unpadded two-float path remains available independently. |
| `reference_norms` | Boolean, default false. Enables the experimental reference normalization rounding path. Capture metadata distinguishes `double` from the original `fused` rounding. Compare it separately; older receipts may omit this field and retain the original path. |
| `detailed_profile` | Boolean, default false. Enables extra CUDA timing events and emits a `dfly2_profile` record on each rank. Use for diagnosis, with separate receipts from throughput comparisons. |
| `adaptive` | Boolean, default false. Enables the experimental observed-acceptance/depth policy. Its measurements and decisions are emitted as `dfly_policy` records for each request. This is an inference-time scheduling experiment, not a trained confidence head. |
| `copies` | Optional boolean. Controls suffix-copy proposals for DSpark comparisons. DFly2 rejects true; serial always records false. Omission retains the selected mode's ordinary copy policy. |
| `dfly_depth` | Optional integer 0–5. Caps DFly2 depth from the initial draft onward. Cannot be combined with `adaptive`. Zero must produce no draft passes or proposed tokens, although DFly2 assets can still be loaded; keep ordinary serial as the no-drafter baseline. |
| `capture` | Optional object with `directory`, `limit` and `every`. Enables bounded reference captures. Directory must be strictly inside the experiment root; defaults are limit 4 and every 1, with limits 1–64 and 1–4096. Capture requires fixed five-draft execution and excludes adaptive scheduling. |

For a fixed-five, copies-off candidate:

```json
"experimental": {
  "device_feedback": true,
  "rdma_best": true,
  "detailed_profile": false,
  "adaptive": false,
  "copies": false,
  "dfly_depth": 5
}
```

These two execution optimizations are separate switches so either can be tested
alone. Set both to false for the candidate build's legacy-path control. For an
adaptive experiment, remove `dfly_depth` and set `adaptive` to true. Keep
`detailed_profile` false and omit `capture` in speed comparisons. The ordinary
native summary counters remain enabled, as in the historical launcher.

## Validation and comparisons

First run a serial reference with the candidate executable. Compare the same
reviewed request bytes, model weights, target caches, sampling settings and
pool size across variants. Use fresh result directories and retain every
attempt, including failures. A later attempt must not continue a failed matrix.

Matching target outputs does not prove that execution controls were honored.
The retained [v4 sweep](../evidence/experimental/README.md) exposed an initial
draft that bypassed depth selection. Its comparisons are disqualified, and the
source correction applies the policy to the initial draft as well as later
rounds. Test control behavior explicitly before selecting a candidate.

Useful independent comparisons are:

1. Legacy feedback and transport versus device feedback only.
2. Legacy transport versus padded winner exchange, with feedback fixed.
3. Fixed depths 0–5 versus adaptive scheduling, with the implementation switches
   fixed and both decode and prefill-inclusive response time reported.
4. DFly2 versus fixed-five DSpark with suffix copies off for both. Ordinary
   DSpark remains a separate comparison of the implemented serving policies.

For each variant, install its private configuration on both hosts, perform
read-only preflight, and then run explicitly on otherwise idle nodes:

```bash
python3 tools/dfly2/run_pair.py --config /etc/dfly2/candidate.json \
  --mode dfly2 --requests /srv/models/dfly2-experiment/requests-chat.json

python3 tools/dfly2/run_pair.py --config /etc/dfly2/candidate.json \
  --mode dfly2 --run \
  --requests /srv/models/dfly2-experiment/requests-chat.json \
  --reference /srv/models/dfly2-experiment/results/serial/result.json \
  --output /srv/models/dfly2-experiment/results/candidate-r1
```

The existing suite runner rotates **modes within one configuration**. A matrix
with several configurations needs an outer coordinator that installs each
configuration, waits for preflight/port reuse, invokes `run_pair`, records the
attempt, and stops at the first failure. The offline lab tools below do not
perform these remote actions. For repeats, rotate which variant starts, while
keeping the very first run serial. Do not retry or drop a slower run to improve
the reported median.

Use the existing chat and longer-input fixtures while selecting a candidate.
Freeze that candidate before running a reviewed held-out fixture. Keep that
final evaluation distinct from the data used to tune the policy.

## Frozen v5 candidate selection

The corrected [v5 chat sweep](../evidence/experimental/sweep-v5-chat/analysis.json)
completed nine runs with identical final token arrays and finish reasons. Its
fixed-zero run passed the no-drafting gate. Selection used one exploratory
repeat of fixed depths 1–5, adaptive scheduling and reference normalization;
serial and depth zero were controls. The declared rule selects the highest
aggregate estimated decode rate, retaining the earlier variant in the frozen
sweep order for an exact tie. No held-out tasks were used for selection.

The rule selected **adaptive**, with an exploratory **48.72 estimated decode
tokens/s**, and froze its controls at `2026-11-27T01:16:06.174077+00:00`.
Device feedback and padded winner exchange are enabled; reference normalization,
suffix copies, detailed timing events and capture are disabled. This single
selection measurement can overfit and is not a final performance result.

The [public freeze receipt](../evidence/experimental-selection-v5.json) binds
every sweep file, the final v5 build identity, effective controls, and the
held-out task/request hashes. The final protocol requires three rotated repeats
of serial, legacy DFly2, the frozen candidate, fixed-five DSpark without copies,
and ordinary DSpark on each of chat and long. The held-out fixture then compares
serial, the same candidate and ordinary DSpark in three rotated repeats:
**15 + 15 + 9 = 39 runs**. The selection receipt does not assert their completion.

The offline exporter accepts the private frozen receipt only after strictly
verifying the corrected public sweep and build identities. It writes an
allowlisted public receipt and refuses replacement with a different freeze:

```bash
python3 tools/dfly2/lab/selection.py --freeze candidate-frozen-v5.json \
  --output evidence/experimental-selection-v5.json
python3 tools/dfly2/lab/selection.py \
  --verify evidence/experimental-selection-v5.json
python3 tools/dfly2/lab/selection.py \
  --verify evidence/experimental-selection-v5.json --require-complete
```

The middle command permits a staged freeze before final bundles exist. The last
command, also used by the release gate, requires all 39 runs. Verification
recomputes the winner and tie rule, checks unchanged controls and identities,
exact outputs, rotations, held-out hashes and recorded chronology from sweep
through freeze to final runs. These are offline consistency checks, not
independent authentication of GPU execution or timestamps.

### Reproduce the held-out request bytes

The published [task fixture](../evidence/fixtures/heldout-tasks.json) contains
12 synthetic tasks: three each for prose, structured output, code and
multilingual responses. Six use greedy decoding and six use temperature 0.7,
top-p 1, top-k 20 and fixed seeds; each permits up to 256 output tokens.
They are separate from the chat sweep used to select the candidate.

Use the pinned target tokenizer and chat-template assets prepared in
[BUILD.md](BUILD.md). Rendering uses no model weights or GPU:

```bash
python3 tools/dfly2/make_heldout.py \
  --frozen evidence/fixtures/heldout-tasks.json \
  --model /srv/models/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw \
  --public-repo . --output /private/requests-heldout.json
```

The output path must be new. For this experiment the exact request SHA256 is
`58c5d2c0bf4cce0d00db45a80783d48f48a94b9068e6da9091e1ea9e55856910`,
with input lengths from 84 to 3,018 tokens. A different tokenizer/template or
different request bytes constitute a new fixture. Preserve the published
tasks and frozen controls when reproducing this evaluation; do not use its
results to select another candidate and still call it held out.

## GPU feedback checks

On an idle GB10, the standalone helper checks rank winner selection, tie and
invalid-ID handling, initialized padding, and dependent proposal chains at
depths 1–5. It uses a deliberately constructed cyclic vocabulary head and an
exact legacy-path control. It does not load target weights or establish
Tencent reference parity or model throughput.

```bash
python3 tools/dfly2/check_device_feedback.py \
  --source /path/to/prepared/tensorfold/zig/kernels/cuda/dsv41_dfly2.cu \
  --report /path/to/new-results/feedback-checks.json
```

The helper compiles an SM121 native cubin, which avoids relying on a driver's
PTX JIT supporting the image's compiler version. `--cubin` selects a prebuilt
native cubin instead. The receipt records artifact kind/hash and whether the
helper compiled it. The original historical primitive test and receipts are
unchanged.

The separate normalization helper checks the new BF16 rounding primitives:

```bash
python3 tools/dfly2/check_reference_norms.py \
  --source /path/to/prepared/tensorfold/zig/kernels/cuda/dsv41_dfly2.cu \
  --report /path/to/new-results/reference-norm-checks.json
```

Those constructed inputs validate the primitive boundaries only. The
[released-weight replay](REFERENCE.md) still fails full numerical parity.

For a real-model implementation comparison, also compare captured proposal
IDs and intermediate values between the legacy and optimized paths on identical
target state. Exact final target outputs alone cannot establish that the
drafter computes the same proposals: target verification can reject a bad
proposal while still producing the expected final answer.

## Reading detailed profiles

`dfly2_profile` records cumulative pass count, the depth histogram, host
readback count, total elapsed pass milliseconds and named timing intervals.
Each category retains its call count and total elapsed milliseconds. Divide
by the corresponding count only when describing a mean for that category.

The events measure **elapsed stream intervals**, including GPU execution,
communication waits and gaps while the host enqueues work. They are not
exclusive kernel execution times. The total-pass interval contains the
category intervals; adding it to those categories would double-count time.
Additional timing events and their final collection affect execution, so
profile jobs are excluded from the lab's throughput comparisons. Compare
both ranks before attributing a delay to compute or network communication.

`dfly_policy` records chosen depths, observed prefix survival, calibration and
probe counts, and measured verify/draft cost estimates. Skipping a draft pass
means that mean draft-pass cost and mean draft cost **per target round** differ.
The lab analysis preserves both and uses the actual number of draft passes
when calculating the hypothetical free-draft bound. That bound is calculated,
not a measured speed.

## Offline matrix evidence

The [lab analyzer](../tools/dfly2/lab/analyze_matrix.py) reads local receipts
only. It uses distinct `dfly2-lab-*` schemas so experimental controls and nested
diagnostics cannot be silently discarded by the historical exporter.

Create a private plan with `make_manifest`; the complete effective execution
settings are derived from each private configuration rather than handwritten:

```python
import sys
from pathlib import Path
sys.path.insert(0, "tools/dfly2")
from lab.analyze_matrix import make_manifest, dump, schedule

variants = [
    ("serial", "serial", "/private/configs/serial.json"),
    ("legacy", "dfly2", "/private/configs/legacy.json"),
    ("feedback", "dfly2", "/private/configs/feedback.json"),
    ("feedback_rdma", "dfly2", "/private/configs/feedback-rdma.json"),
    ("dspark", "dspark", "/private/configs/dspark.json"),
]
plan = make_manifest(variants, "/private/requests-chat.json", Path.cwd(),
                     purpose="benchmark", repeats=3)
dump("/private/matrix.json", plan)
print(schedule([v[0] for v in variants], 3))
```

After each individually named run has been collected locally, append an entry
such as the following. `config` is optional when that attempt used its
variant's original config file; include it when the coordinator gave each
attempt a distinct config-file path.

```json
{"order": 1, "variant": "serial", "repeat": 1,
 "result": "/private/results/001-serial-r1/result.json",
 "config": "/private/configs/001-serial-r1.json"}
```

Keep both `rank*.stdout.jsonl` files beside `result.json`; detailed profiles
require the rank-1 summary. Leave `complete` false after a failure or while
work remains. Set it true only after all scheduled runs succeed. Keep a
launcher/preflight failure in its own failed attempt envelope if the pair
launcher did not reach the point of creating `result.json`; retain its logs
privately. Never invent missing numerical records.

```bash
python3 tools/dfly2/lab/analyze_matrix.py --manifest /private/matrix.json \
  --output /private/analysis.json --export evidence/new-lab-comparison

python3 tools/dfly2/lab/analyze_matrix.py \
  --verify-export evidence/new-lab-comparison
```

The exporter refuses an existing destination. It retains every listed attempt,
full reply token arrays, finish reasons, lifecycle results, effective controls,
and complete policy/profile records. Infrastructure addresses, hostnames,
private paths/config hashes, run UUIDs and container IDs are excluded.
Unexpected metric text or schema fields fail review rather than being silently
copied. The private check verifies each exact config identity; configuration
differences are limited to declared experimental controls and config-file
paths. Binary, image, launcher, request, target and rank-cache identities must
remain unchanged within a matrix.

Only a complete, successful `purpose: "benchmark"` matrix without declared
control defects receives aggregate throughput comparisons. The optional
`known_issues` list uses the strict `dfly2-known-issue-v1` schema, a supported
issue code and declared affected variant names. The retained v4 sweep records
`initial_draft_depth_bypass`. Any declared issue suppresses the whole matrix's
ranking while preserving its receipts. The public verifier also rejects a
fixed-zero run with draft passes or proposed/accepted tokens unless that
variant explicitly carries the known defect. Adding the issue to an unrelated
variant does not excuse the violation.

`profile` and `capture` matrices retain diagnostic
records without benchmark rankings. Decode and end-to-end rates include
medians and minimum/maximum ranges, including each request. A few repetitions
provide observed ranges, not confidence intervals. The known chat/long fixture
hashes are approved for export; a new reviewed fixture requires
`--approved-fixture-sha256`. This assertion does not replace prompt review or
permit changing the bytes of a completed experiment.

These checks establish consistency of the recorded evidence. They do not
independently attest its GPU origin, Tencent implementation fidelity, quality
on arbitrary prompts, or production serving behavior.

## Sanitized numerical replay reports

The separate [reference exporter](../tools/dfly2/reference/export_reference.py)
reads existing replay reports without loading weights, tensors, CUDA, or model
services. Its output contains numerical comparisons, exact proposal IDs,
slot matches and matching prefixes, native-head gate results and probe margins,
fixed tolerances, runtime versions, and file/source hashes. It copies no tensor
or embedding bytes, logs, hostnames, private paths, model config names/paths, or
raw error messages. A failed comparison remains a failed comparison.

One reference process can load the weights once and replay all capture groups.
In a private export plan, `case_prefix` selects the exact first component of
each relative capture path in that batch. It never selects an arbitrary subset
within the prefix. Every requested case in each source batch must be covered
exactly once across the plan, including failed and unattempted cases:

```json
{
  "schema": "dfly2-reference-export-plan-v1",
  "groups": [
    {"fixture": "chat", "variant": "legacy", "expected_cases": 6,
     "report": "/private/replay/report.json", "case_prefix": "capture-legacy-chat",
     "run_result": "/private/runs/legacy-chat/result.json"},
    {"fixture": "long", "variant": "legacy", "expected_cases": 6,
     "report": "/private/replay/report.json", "case_prefix": "capture-legacy-long",
     "run_result": "/private/runs/legacy-long/result.json"},
    {"fixture": "chat", "variant": "optimized", "expected_cases": 6,
     "report": "/private/replay/report.json", "case_prefix": "capture-optimized-chat",
     "run_result": "/private/runs/optimized-chat/result.json"},
    {"fixture": "long", "variant": "optimized", "expected_cases": 6,
     "report": "/private/replay/report.json", "case_prefix": "capture-optimized-long",
     "run_result": "/private/runs/optimized-long/result.json"}
  ]
}
```

`run_result` is optional. When supplied, the exporter requires a successful
capture run, matching binary/request/checkpoint/image/launcher identities on
both ranks, clean rank exits and cleanup, and recorded capture controls whose
limit/interval and total draft passes explain the captured pass names/count.
Capture binary, request and checkpoint hashes must match that run. The public
report retains the reviewed execution settings and hashes. `legacy` requires
both feedback/RDMA switches off; `optimized` requires both on; `candidate`
allows independently recorded ablations. Reference normalization metadata must
agree with its recorded opt-in. Without a run receipt,
`run_provenance_provided` is false: a group label does not attest execution
controls. Capture metadata v1 alone does not record feedback/RDMA settings.

```bash
python3 tools/dfly2/reference/export_reference.py \
  --plan /private/reference-plan.json --output evidence/new-reference-diagnostic
python3 tools/dfly2/reference/export_reference.py \
  --verify evidence/new-reference-diagnostic
```

The exporter refuses an existing destination. The public bundle contains only
`reference.json` and its checksum manifest. Group completion and whole-batch
completion are separate. Numerical failures, proposal mismatches, failed gates,
missing failed reports and cases not reached after a runtime failure remain
visible. Totals count reported comparisons; `unique_captured_tensor_sets`
separately identifies repeated byte-identical capture sets. Paired legacy and
optimized replays are not independent model states.

The fixed numerical thresholds are `rtol=0.02`, `atol=0.02`, and maximum relative
RMS error `0.01`. The verifier checks shapes/counts, finite metrics, RMS
relationships, pass flags, proposal-file hashes against the exact IDs, matching
prefixes, gate IDs, top-two margins, capture/run hash agreement, complete batch
coverage and the public file checksum. It does **not** recalculate numerical
differences from private tensors. Independent numerical replay requires
regenerating the captured state and reference results with the pinned code,
weights, target head and embeddings. Consistent hashes do not authenticate an
untrusted report or prove it came from a GPU.

Optional `source_files` in the private plan maps reviewed roles
(`reference_replay`, `proposal_replay`, `pack_weights`, `head_probe_source`) to
`{"path": "...", "sha256": "..."}` records. These bytes are checked at export
time and labeled `export_time_source_sha256`; this does not attest which source
was executed earlier. A later normalized/rebuilt native tree cannot attest the
bytes of an earlier development binary. Keep its original binary and report
hashes and build provenance distinct.

Diagnostic wall times include loading, subprocesses, synchronization and I/O.
They are not serving throughput. Head-gate margins describe the probe's
scores; original native pass logits/margins were not captured. Passing this
diagnostic alone does not establish target index-selection correctness,
independent EXL3 mathematical correctness, unquantized-target equality, or
sampled-distribution parity.

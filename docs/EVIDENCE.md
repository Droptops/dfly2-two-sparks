# Public evidence schema and export

`tools/dfly2/sanitize_evidence.py` exports an explicit field allowlist. It does
not copy an entire raw receipt and then attempt to remove recognizable secrets.
New unknown envelope, node, and measurement fields are excluded. The public
files are labeled `dfly2-public-evidence-v1`; they are derived records, not
byte-identical raw receipts.

| Input section | Public fields retained |
| --- | --- |
| Run | Mode, success/completion, fixed inference settings: TP2, one stream, 8,192-token pool, eager execution, profiling enabled. |
| Rank | Rank number, GPU model, executable/request/pack SHA-256, target rank-cache byte count. |
| Prefill | Request index/name, prompt-token count, cache/restore token counts, reported prefill seconds. |
| Reply | Request index/name, complete token ID array and count, finish reason, token digest prefix, rounds, draft/accept counts, minimum verification rows, elapsed seconds. |
| Native metrics | Explicit numeric profile, round-clock, Engram-prefill and Engram-decode measurements. |
| Summary | Request/token/round/draft/accept counts and total elapsed seconds. |
| Primitive checks | Test names, pass flags, tensor shapes, numerical errors and tolerances, special-reference differences, compiler flags, source/PTX hashes, hardware/software versions. |
| Microbenchmark | Shapes, exact-output flags, every timing sample, medians/ranges, repetitions, source/PTX hashes and exhaustive decoder checks. Temporary paths are replaced with repository paths; only compiler flags are retained from commands. |
| Weight checks | Rank, architecture, tensor count, allocated weight bytes, common manifest identity, full-byte readback and allocation-release flags. |

Personal hostnames, IP addresses, usernames, absolute workspace/deployment
paths, network interface/HCA names, private image names and IDs, container IDs,
process IDs, UUID run/session identifiers, dashboard state, unrelated model
state, service-unit names, and cleanup envelopes are not exported. Independent
rank exit envelopes are omitted; the retained success/completion fields and
native results record the inference outcome.

The six chat prompts are reviewed synthetic fixtures, including their example
names and addresses. `chat-requests.json` retains their complete text, tokenized
inputs, sampling and seeds byte for byte. Its SHA-256 therefore still matches
the value recorded on both ranks. Request-envelope and message-schema fields
are validated before copying. Future input fixtures need their own content
review; the exporter does not infer whether arbitrary prompt text is suitable
for publication.

`draft-pack-manifest.json` and the historical CUDA baseline are also
byte-identical copies. They contain model tensor-file names, content hashes,
and code rather than deployment identifiers. Sanitized receipt hashes naturally
differ from the private originals. The verifier uses preserved request, token,
source, and weight-manifest hashes for the relevant comparisons; it does not
pretend a cleaned receipt has its original file digest.

Raw infrastructure preflight/final-state receipts, SSH and container logs,
deployment manifests, the private launcher, the earlier integration archive,
the old unsanitized comparison reports, and the ambiguous historical host-test
log are omitted. Earlier 128-token smoke receipts are also omitted from this
public comparison: the retained six chat requests provide the complete input
and output evidence for the 959-token claim. No model weights, executable
binaries, kernel kits, or container images are included.

`public-build-validation.json` separately records release-preparation checks:
the public base digest, locked source trees, successful build/test exit codes,
native executable hashes, dependency versions and offline request equivalence.
Its source mount path is replaced with a placeholder. It contains no machine
identity or raw host logs. The historical inference and fresh build checks are
kept distinct; no new performance result is inferred from rebuilding the image.

`portable-validation/` contains the subsequent fresh serial, DSpark and DFly2
GPU receipts, a byte-identical copy of the reviewed request fixture, and a
separately recomputed comparison. All three modes used the newly built public
executable. The original top-level chat receipts and comparison remain the
historical integration measurements; their times have not been replaced or
combined with the new run times.

`tools/dfly2/export_portable_evidence.py` applies the same allowlist to runs
made with the public launcher. It also retains validated UTC start/end times,
the combined launcher/configuration-module source SHA-256, the target's
configuration-file SHA-256, and a runtime image-content SHA-256. The latter
identifies canonical architecture, OS, variant, image configuration and ordered
RootFS content. Docker storage backends can report different image IDs for the
same transferred image, so those private IDs are neither exported nor used as
the cross-node content identity. The private cluster configuration and its digest
remain excluded. For these earlier portable receipts, the pool size was supplied
explicitly from the reviewed configuration because the raw run receipt did not
embed that setting; this published suite requires exactly 8,192 tokens. Other
published execution settings are fixed by the launcher. The exporter requires
a new output directory and writes its derived comparison only after the
cross-run checks pass. It also requires the reviewed historical request-file
digest before creating any output directory, so unreviewed prompt text is not
written into public evidence.

`tools/dfly2/verify_portable_evidence.py` additionally binds all three modes to
the executable in the public build receipt, verifies the launcher digest
pinned in that folder's `provenance.json`, requires matching image contents, target metadata and execution settings across
runs, checks the DFly2 pack identity, and compares the fresh serial output with
the historical serial output. Historical run timings are never substituted
into a portable comparison. These checks require neither CUDA nor model files.

The standard-library verifier reconstructs the token equality and throughput
calculation without trusting the raw receipts' saved `equal` flags. Its tests
exercise exclusion of injected infrastructure fields, refusal of text in
numeric measurements, request-schema rejection, and detection of token and
timing changes. This establishes consistency of the retained records. Running
the model or the GPU checks is a separate action.

## What the verifiers establish, and what they cannot

The verifiers recompute; they do not attest. Token arrays are compared in
full, every rate is recalculated from elapsed times, and each estimate must
agree with two further native clocks in the same receipt: the per-reply times
must sum to the total, and the profiler's mean round time multiplied by the
number of rounds must reproduce the estimated decode seconds. The passing
primitive receipt must carry the SHA-256 of the kernel source that
`patches/0001-dfly2-native.patch` creates, and the earlier one the hash of the
preserved original kernel. A change to one recorded time, or to the kernel,
is caught.

None of that proves a measurement happened. The receipts are derived JSON
written by this repository's own tools, with nothing signed by the hardware or
by a third party, and a self-consistent invented set would pass. Reproduction
on other hardware is the only independent check.

Some retained fields are statements, not recordings. In the receipts published
before the launcher recorded its settings, the `configuration` block (TP2, one
stream, 8,192-token pool, eager, profiling) was written by the exporter from
the launcher's fixed arguments, and for the historical integration run from
the private launcher's. Checking those constants against each other verifies
nothing. Receipts exported from a `result.json` that has an `execution` block
carry `recorded_by_launcher: true` and the launcher's own values, including
whether drafting was on and any fixed draft depth; suite verification requires
that form. The native runner's `equal` flags are true whenever a request has
no expected digest, which is every request here, so no verifier reads them.

## Launcher provenance

A receipt's `launcher_sha256` identifies the launcher bytes that ran. Each
receipt folder made with the public launcher holds a `provenance.json` with
that digest and the Git commit containing those bytes; verification compares
receipts with the pin. It no longer compares them with whatever launcher the
working tree holds, which would either reject every old receipt after an edit
or tempt an edit to the receipts. `scripts/check.py` confirms each pinned
commit really holds the recorded bytes when Git history is available, and
reports when it is not.

The GitHub workflow fetches full history so CI can verify the historical and
suite launcher pins. An exported archive without Git history still reports
those commit checks as unreadable; it does not turn them into verified pins.

## Target identity

The recorded `target_config_sha256` equals the SHA-256 of `config.json` at the
pinned target revision `45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b`, and the
request fixtures' tokenizer and chat-template hashes equal that revision's
`tokenizer.json` and `chat_template.jinja` (checked 2026-11-26). Target weights
are still identified only by that revision and by rank-cache byte counts; no
receipt hashes them.

## Request fixtures and suites

`evidence/fixtures/long-requests.json` is generated by
`make_requests.py --suite long` from `tools/dfly2/long_tasks.py`, which builds
every character of its synthetic text with no outside data. A test requires
the published fixture's task text to equal that generator's output, so the
text published with a long-suite receipt is text reviewed in this repository.

`tools/dfly2/export_suite_evidence.py` applies the same field allowlist to a
`run_suite.py` directory and writes `suite.json`, one receipt per run,
`provenance.json` and a recomputed `comparison.json`. It accepts only request
files whose bytes match a reviewed fixture, only completed suites, and only
when run from a checkout whose launcher matches the one the suite recorded.
`tools/dfly2/verify_suite_evidence.py` requires every run, repeated serial
runs included, to match the first serial run's tokens and finish reasons, and
reports a median and range per mode. A median over three repeats is a better
summary than one run. It is not a confidence interval.

`evidence/suite-chat/` and `evidence/suite-long/` contain the two completed
twelve-run suites: four modes, three repeats each, with every output token and
all six finish reasons equal to that fixture's first serial run. The chat
fixture produces 959 output tokens and the longer fixture 1,169. All 24
individual receipts retain the full measured ranges, including the short
prose slowdown and the longer fixture's slower DFly2 prefill repeat. These
records are separate from all earlier single-run evidence.

In recorded execution settings, `fixed_draft_depth` describes a command-line
override. It is `5` for `dspark_fixed5` and `null` for DFly2 because no
`--fix-k` override is passed to that mode. The native DFly2 implementation
itself always proposes five drafts. A null override does not mean adaptive
DFly2 depth. The suite matches depth only: DSpark copy proposals remain enabled.

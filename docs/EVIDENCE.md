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
remain excluded. The pool size is supplied explicitly from the reviewed
configuration because the raw run receipt does not embed that setting; this
published suite requires exactly 8,192 tokens. Other
published execution settings are fixed by the launcher. The exporter requires
a new output directory and writes its derived comparison only after the
cross-run checks pass. It also requires the reviewed historical request-file
digest before creating any output directory, so unreviewed prompt text is not
written into public evidence.

`tools/dfly2/verify_portable_evidence.py` additionally binds all three modes to
the executable in the public build receipt, verifies the published launcher's
source digest, requires matching image contents, target metadata and execution settings across
runs, checks the DFly2 pack identity, and compares the fresh serial output with
the historical serial output. Historical run timings are never substituted
into a portable comparison. These checks require neither CUDA nor model files.

The standard-library verifier reconstructs the token equality and throughput
calculation without trusting the raw receipts' saved `equal` flags. Its tests
exercise exclusion of injected infrastructure fields, refusal of text in
numeric measurements, request-schema rejection, and detection of token and
timing changes. This establishes consistency of the retained records. Running
the model or the GPU checks is a separate action.

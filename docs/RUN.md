# Configure and validate two Sparks

The public two-node inference launcher and freshly built image completed serial, DSpark and DFly2 GPU runs with matching 959 output tokens and all six finish reasons. [RESULTS.md](RESULTS.md) separates this fresh portable validation from the earlier integration measurements. A successful preflight alone does not reproduce GPU results.

Complete [BUILD.md](BUILD.md) first. Runtime commands below are for Linux on the two configured GB10 nodes. Python 3.10+, Docker access, `ip`, `nvidia-smi`, existing SSH host keys and passwordless head-to-worker SSH are required. The launchers never stop services, change network configuration, download weights or restore a previously running model.

## Cluster configuration

Copy `examples/cluster.example.json` to `cluster.local.json`, replace every node and fabric placeholder, and keep this private file outside version control. The example IP addresses are documentation addresses and will not work on your fabric. Install identical configuration contents on both hosts at each node's configured `config_path`. Put identical repository contents at each configured `repo_dir`; these directories may differ between nodes. Run the coordinator from the configured head.

The configuration rejects unknown fields and accepts no shell snippets, extra Docker arguments, arbitrary environment variables or additional mounts. The schema is:

| Field | Meaning |
| --- | --- |
| `version` | Exactly `1`. |
| `image` | Same public image on both hosts; defaults to `dfly2-two-sparks:experimental`. Build once and transfer it; the preflight compares canonical image contents across Docker storage backends. |
| `nodes` | Exactly two entries, rank 0/head first. Each has short `hostname`, direct-fabric IPv4 `address`, absolute Linux `repo_dir`, `config_path`, and `fabric`. |
| `nodes[1].ssh_target` | Existing `user@host` or `user@SSH-alias`. `ssh_port` defaults to 22. SSH options cannot be embedded here. |
| `fabric` | Explicit `interface`, list of `hcas`, and common `gid_index`. The live route, active HCA ports, IPv4-mapped RoCE v2 GIDs and local addresses must agree. |
| `paths.model_root` | Same absolute root on both hosts, mounted read-only inside containers. Local disks may contain different corresponding rank caches below this path. |
| `paths.experiment_root` | Dedicated strict child of `model_root`; binaries, requests and results must live inside it. |
| Other `paths` | `target_model`, `engram`, `rank_cache`, `kit`, `token_map`, `binary`, `weights_probe`, `requests`, `draft_pack`, `results`. All must be under `model_root`. Example paths match BUILD.md. |
| `execution` | `port` defaults to 4460, `pool` must be 8192, `min_available_gib` is at least 112, and `timeout_seconds` is 30–7200 (default 1800). |
| `stopped` | Optional `dashboard_url`, `systemd_user_units`, `systemd_units`, `container_labels` (`key=value`) and `container_names` to check before loading. Defaults are null/empty. Checks apply on both hosts, except the dashboard is queried by the head. |

Use your system's read-only discovery tools to establish the actual fabric values:

```bash
hostname -s
ip -4 -j addr show
ip -j route get PEER_FABRIC_IP
rdma link show
```

Replace `PEER_FABRIC_IP` with the other node's direct-link address. Inspect the selected HCA's `/sys/class/infiniband/HCA/ports/1/gids/INDEX` and `gid_attrs/types/INDEX`; it must be an IPv4-mapped GID with type `RoCE v2`. The launcher verifies that the GID belongs to an interface associated with that HCA, and that at least one selected HCA matches the configured primary interface/address. It does not guess or repair fabric settings.

The optional dashboard URL must be a head/loopback HTTP(S) URL without credentials or query parameters. Its JSON must contain `phase`, `active_model`, `desired_model`, `inflight` and `loaded`. Readiness requires phase `idle`, no active/desired model, zero inflight requests and empty loaded-model lists. Set it to null for another dashboard schema, and configure the relevant systemd units or container names/labels instead. GPU-idle checks remain mandatory regardless of these optional guards. Stopped benchmark processes are not killed or resumed.

Host permissions must allow the launcher to read all configured assets and execute the staged binaries. The repository must be writable for `.state/` lock and container-ID receipts. The host running a command must be able to create new result directories below `experiment_root`; containers receive only a read-only model-root mount. If a worker's data share is read-only, provide a local writable results subdirectory at the configured path before running its weight probe. The launcher refuses an existing result directory and never writes into model-weight directories outside `experiment_root`.

## CPU checks and request generation

From the repository root, this check is CPU-only and makes no Docker, SSH or model calls:

```bash
python3 scripts/check.py
```

Generate the original smoke inputs or six actual chat/code tasks with the target's tokenizer and `chat_template.jinja`. The image supplies the Python dependencies. For the example paths, run on the head:

```bash
mkdir -p /srv/models/dfly2-experiment
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$PWD:/repo:ro" -v /srv/models/target:/target:ro \
  -v /srv/models/dfly2-experiment:/out -w /repo \
  --entrypoint python3 dfly2-two-sparks:experimental \
  tools/dfly2/make_requests.py /target /out/requests.json

docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$PWD:/repo:ro" -v /srv/models/target:/target:ro \
  -v /srv/models/dfly2-experiment:/out -w /repo \
  --entrypoint python3 dfly2-two-sparks:experimental \
  tools/dfly2/make_requests.py /target /out/requests-chat.json --suite chat
```

No `--gpus` is requested by these generation commands. The chat suite contains three greedy and three temperature-0.7/top-p-1 requests, each with a 256-token output budget. It records the tokenizer/template hashes. Make the identical generated request file available at the same configured absolute path on the worker; a shared filesystem is optional. `--requests` selects an alternate file inside `experiment_root`. Every request must fit the 8192-token pool with a 16-token margin; vision and spill/restore requests are refused.

## Read-only preflight

Supply your installed configuration path, for example:

```bash
python3 tools/dfly2/run_pair.py --config /etc/dfly2-two-sparks/cluster.json --mode serial
python3 tools/dfly2/run_pair.py --config /etc/dfly2-two-sparks/cluster.json --mode dfly2 \
  --requests /srv/models/dfly2-experiment/requests-chat.json
```

Omitting `--run` performs only read-only checks, including remote checks on the worker. It starts no container and allocates no GPU weights. The checks cover hostname, repository location, canonical configuration and launcher hashes, actual fabric, memory, absence of GPU compute processes, one GB10 per host, configured stopped services, available port, expected cache files and accessible assets. Both hosts must agree on the canonical image-content fingerprint, native binary hash, target config hash, request hash and DFly pack-manifest hash. The image fingerprint includes architecture, OS, variant, complete image configuration and ordered RootFS layers; Docker storage-backend image IDs may differ. Baseline modes refuse an image that inherits `TF_DS_DFLY2_DIR`, including an empty value.

The pack check validates its format, two rank entries, local file size and digest syntax. It does not rehash multiple gigabytes on every preflight. Use the converter's recorded SHA256 checks and the upload probe to verify weights. Similarly, the rank-cache filename is not a hash of every target weight byte; preserve the model provenance checks from BUILD.md. Preflight does not reserve the GPU, so keep other model launchers stopped while validation runs.

## Explicit GPU gates

These commands use the GPU. Run them only on otherwise idle nodes after reviewing preflight. The primitive test compiles and checks the DFly kernels without target weights. Its default CUDA source is the prepared `_build/tensorfold` tree:

```bash
mkdir -p /srv/models/dfly2-experiment/results/primitives
docker run --rm --network none --gpus all \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/repo:ro" \
  -v /srv/models/dfly2-experiment/results/primitives:/results -w /repo \
  --entrypoint python3 dfly2-two-sparks:experimental \
  tools/dfly2/test_primitives.py --report /results/result.json
```

The separate native upload probe checks the configured draft rank's uploaded bytes and allocation release. Run rank 0 locally on the head and rank 1 locally on the worker. These also default to preflight unless `--run` is present:

```bash
python3 tools/dfly2/run_weights_probe.py --config /etc/dfly2-two-sparks/cluster.json --rank 0
python3 tools/dfly2/run_weights_probe.py --config /etc/dfly2-two-sparks/cluster.json --rank 0 --run
# On the worker, use its configured config_path:
python3 tools/dfly2/run_weights_probe.py --config /etc/dfly2-two-sparks/cluster.json --rank 1 --run
```

Probe containers have no network and a 12 GiB host-memory limit. The success receipt must report the requested rank, byte equality and released allocation. This checks loading; it does not establish model quality or decoding speed.

## Isolated serial, DSpark and DFly2 runs

Run from the head. Choose fresh result directories for every invocation:

```bash
python3 tools/dfly2/run_pair.py --config /etc/dfly2-two-sparks/cluster.json \
  --run --mode serial --requests /srv/models/dfly2-experiment/requests-chat.json \
  --output /srv/models/dfly2-experiment/results/chat-serial

python3 tools/dfly2/run_pair.py --config /etc/dfly2-two-sparks/cluster.json \
  --run --mode dspark --requests /srv/models/dfly2-experiment/requests-chat.json \
  --output /srv/models/dfly2-experiment/results/chat-dspark \
  --reference /srv/models/dfly2-experiment/results/chat-serial/result.json

python3 tools/dfly2/run_pair.py --config /etc/dfly2-two-sparks/cluster.json \
  --run --mode dfly2 --requests /srv/models/dfly2-experiment/requests-chat.json \
  --output /srv/models/dfly2-experiment/results/chat-dfly2 \
  --reference /srv/models/dfly2-experiment/results/chat-serial/result.json
```

The bounded configuration is TP2, one text stream, eager execution, replicated KV, pool 8192, no side stream or prefetch, AIO enabled and replay floor 1. Serial uses zero drafts; DSpark uses the original drafter; DFly2 alone receives `TF_DS_DFLY2_DIR`. Native profiling is enabled. Each `result.json` records these settings in an `execution` block, written from the same values the container arguments are built from.

A fourth mode, `--mode dspark_fixed5`, runs the original drafter with the lane runner's `--fix-k 5`: five drafts every round, each treated as certain. That is the proposal-depth policy the DFly2 lane always uses, where plain `dspark` trims its depth by confidence. Use it when comparing the two drafters rather than the two lanes. Suffix-lookup copy proposals remain enabled for both DSpark modes and disabled for DFly2; this runner has no switch for them. `--reference` compares every reply token and finish reason against a successful serial result using the same request-file hash. It does not compare abbreviated token hashes alone.

The launcher captures both ranks' stdout/stderr and writes `result.json` with preflight identities, completion state, cleanup and equivalence. Inspect native timing fields and readable outputs before making speed or quality claims. These small suites do not establish general quality equivalence, long-context correctness, serving compatibility or production readiness.

An immediate next run can fail the port check with `Address already in use`
while the previous connection is in TCP `TIME_WAIT`, even after both containers
have exited. Check the configured port locally, wait for normal TCP expiry, and
retry preflight when it is free. Do not stop an unrelated listener to clear the
port. This was encountered after a successful run during public validation.

## Matched, repeated and long-context suite

One run per mode in one fixed order cannot separate a mode from run order or run-to-run noise, and the six chat prompts never exceed 398 positions. `run_suite.py` runs every mode once per repeat, rotates which mode goes first, waits for the previous run's port to leave `TIME_WAIT`, and passes the first serial result as the reference for every later run, repeated serial runs included. It starts nothing itself: each run is an ordinary `run_pair` run with its own fresh directory, lock and cleanup.

Stage the reviewed long-context fixture at the same path on both nodes. It was generated with the target's own tokenizer and template, so regenerating it inside the image with `make_requests.py --suite long` must give the same bytes:

```bash
cp evidence/fixtures/long-requests.json /srv/models/dfly2-experiment/requests-long.json
sha256sum /srv/models/dfly2-experiment/requests-long.json
# 856411dc8409168f65ad1e2bb6640ae2a5d3b526b36b8101f0ffb22715cae539
```

Without `--run` the suite only performs each mode's read-only preflight and prints the planned order:

```bash
python3 tools/dfly2/run_suite.py --config /etc/dfly2-two-sparks/cluster.json \
  --requests /srv/models/dfly2-experiment/requests-chat.json
```

Then run both fixtures from the head, on otherwise idle nodes:

```bash
python3 tools/dfly2/run_suite.py --config /etc/dfly2-two-sparks/cluster.json --run --repeats 3 \
  --requests /srv/models/dfly2-experiment/requests-chat.json \
  --output /srv/models/dfly2-experiment/results/suite-chat

python3 tools/dfly2/run_suite.py --config /etc/dfly2-two-sparks/cluster.json --run --repeats 3 \
  --requests /srv/models/dfly2-experiment/requests-long.json \
  --output /srv/models/dfly2-experiment/results/suite-long
```

The default is all four modes, twelve runs per fixture. Every run reloads the model: the published chat runs took 77 to 96 seconds each, and the port wait adds about a minute between runs, so allow roughly half an hour for the chat fixture and longer for the long one. A run that fails stops the suite and is recorded in `suite.json`; nothing is retried.

The long fixture is the first time DFly2's prompt capture sees multi-chunk prompts and a selection that leaves positions out. If a DFly2 run refuses or its tokens differ from the serial run, keep the result directory. That is a finding about the port, and worth more than a clean number.

Export from the same checkout that launched the suite, so the recorded commit is the one whose launcher bytes ran, then re-run the checks:

```bash
python3 tools/dfly2/export_suite_evidence.py --suite-dir /srv/models/dfly2-experiment/results/suite-long \
  --requests /srv/models/dfly2-experiment/requests-long.json --pool-tokens 8192 \
  --commit "$(git rev-parse HEAD)" --output evidence/suite-long
python3 scripts/check.py
```

The export refuses an incomplete suite, a request file that is not one of the repository's reviewed fixtures, and a launcher that differs from the one the suite recorded. `comparison.json` then holds a median and range per mode, per-request medians, and the `dspark_fixed5` against DFly2 comparison.

## Cleanup and interrupted runs

The pair uses fixed container names `dfly2-validation-r0` and `dfly2-validation-r1`; upload probes use `dfly2-weights-r0/r1`. Any preexisting container with either tool's name is refused, including stopped containers. Each successful Docker create writes its exact container ID into `.state/node-runs/RUN_UUID/rankN.cid`. Subsequent actions require that ID, expected name, rank, experimental label and invocation UUID to match. Cleanup removes only those recorded containers. Lost SSH create responses can be recovered through the same journal, and ordinary errors, Ctrl-C or SIGTERM trigger bounded cleanup attempts. A lost host, SIGKILL or inaccessible Docker daemon can leave a container behind.

If cleanup fails, inspect `result.json`, the node's journal and `docker inspect` locally. Confirm the exact recorded ID and its `ai.spark-dfly2.experimental=1`, `ai.spark-dfly2.run` and `ai.spark-dfly2.rank` labels before manually removing that specific container. Do not use broad Docker prune commands or delete another run's journal to bypass the collision guard. The tools do not adopt or stop unrelated containers.

Raw logs and configuration may contain private paths, addresses or generated text. Keep them private; use the evidence sanitizer and review its output before publishing a new result.

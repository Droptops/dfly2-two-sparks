#!/usr/bin/env python3
"""Isolated two-Spark lane validation. With no --run this only performs preflight.

The head invokes this same file on the worker using its private `node` JSON RPC.
It never stops dashboard models. Cleanup requires both the recorded creation ID
and this invocation's UUID labels; existing containers are never adopted.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
import cluster_config as cluster

CLUSTER = None
LABEL = "ai.spark-dfly2.experimental"
RUN_LABEL = "ai.spark-dfly2.run"
RANK_LABEL = "ai.spark-dfly2.rank"
MODES = ("serial", "dspark", "dfly2")


def configure(value):
    global CLUSTER, HEAD, WORKER, HEAD_IP, WORKER_IP, ROOT, EXPERIMENT, MODEL, ENGRAM, PORT, POOL, MIN_AVAILABLE
    CLUSTER = cluster.validate(value)
    HEAD, WORKER = [n["hostname"] for n in CLUSTER["nodes"]]
    HEAD_IP, WORKER_IP = [n["address"] for n in CLUSTER["nodes"]]
    paths, execution = CLUSTER["paths"], CLUSTER["execution"]
    ROOT, EXPERIMENT = Path(paths["model_root"]), Path(paths["experiment_root"])
    MODEL, ENGRAM = Path(paths["target_model"]), Path(paths["engram"])
    PORT, POOL = execution["port"], execution["pool"]
    MIN_AVAILABLE = execution["min_available_gib"] * 1024**3


def container_name(job):
    kind = "weights" if job.get("kind") == "weights" else "validation"
    return f"dfly2-{kind}-r{job['rank']}"


def launcher_identity():
    return hashlib.sha256(Path(__file__).read_bytes() + Path(cluster.__file__).read_bytes()).hexdigest()


class Refusal(RuntimeError):
    pass


def command(args, *, timeout=30, check=True, input_text=None):
    p = subprocess.run([str(x) for x in args], input=input_text, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=timeout, check=False)
    if check and p.returncode:
        raise Refusal(f"{args[0]} exited {p.returncode}: {p.stderr.strip()[-1500:]}")
    return p


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hostname():
    return socket.gethostname().split(".")[0]


def contained(path, root, *, file=True):
    p = Path(path).resolve(strict=True)
    if not p.is_relative_to(root.resolve(strict=True)):
        raise Refusal(f"Path is outside {root}: {p}")
    if not (p.is_file() if file else p.is_dir()):
        raise Refusal(f"Missing {'file' if file else 'directory'}: {p}")
    return p



def read_requests(path):
    path = contained(path, EXPERIMENT)
    if path.stat().st_size > 4 * 1024**2:
        raise Refusal("Requests file exceeds 4 MiB")
    reqs = json.loads(path.read_text())["requests"]
    if not isinstance(reqs, list) or not 1 <= len(reqs) <= 64:
        raise Refusal("Expected 1..64 validation requests")
    names = set()
    for r in reqs:
        name, prompt, tokens = r.get("name", ""), r.get("prompt"), r.get("max_tokens")
        # The native recorder interpolates names into JSON without escaping.
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", name) or name in names:
            raise Refusal("Each request needs a unique plain name")
        names.add(name)
        if not isinstance(prompt, list) or not prompt or any(type(t) is not int or not 0 <= t < 129280 for t in prompt):
            raise Refusal(f"Invalid tokenized prompt: {name}")
        if type(tokens) is not int or tokens <= 0 or len(prompt) + tokens + 16 > POOL:
            raise Refusal(f"Request exceeds the {POOL}-token pool: {name}")
        if r.get("images") or r.get("drop_memory"):
            raise Refusal("Initial pair validation supports text requests without spill/restore")
    return path, reqs


def stopped_dashboard():
    url = CLUSTER["stopped"]["dashboard_url"]
    if url is None:
        return {"configured": False}
    with urllib.request.urlopen(url, timeout=5) as response:
        data = json.load(response)
    keys = ("phase", "active_model", "desired_model", "inflight", "loaded")
    if any(k not in data for k in keys):
        raise Refusal("Dashboard status schema was not recognized")
    state = {k: data[k] for k in keys}
    if (state["phase"] != "idle" or state["active_model"] is not None
            or state["desired_model"] is not None or state["inflight"] != 0
            or not isinstance(state["loaded"], dict) or any(state["loaded"].values())):
        raise Refusal("Dashboard models must be stopped; this launcher will not stop them")
    return state


def check_pack(pack, rank):
    pack = contained(pack, ROOT, file=False)
    manifest = contained(pack / "dfly2-pack.json", pack)
    if manifest.stat().st_size > 1024 * 1024:
        raise Refusal("Draft pack manifest exceeds 1 MiB")
    data = json.loads(manifest.read_text())
    if data.get("format") != "tensorfold-dfly2-tp2-v1" or data.get("world_size") != 2:
        raise Refusal("Unsupported draft pack format")
    ranks = data.get("ranks", [])
    if len(ranks) != 2 or {r.get("file") for r in ranks} != {"rank0.safetensors", "rank1.safetensors"}:
        raise Refusal("Draft pack must describe exactly two rank files")
    weights = contained(pack / f"rank{rank}.safetensors", pack)
    record = next(r for r in ranks if r["file"] == weights.name)
    if (type(record.get("bytes")) is not int or record["bytes"] != weights.stat().st_size
            or not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", ""))):
        raise Refusal("Draft pack rank size or recorded digest is invalid")
    # Full multi-GB weight hashes belong to pack conversion/upload verification.
    # Preflight verifies shape metadata, file sizes and manifest consistency only.
    contained(pack / "config.json", pack)
    return digest(manifest)


def image_content_fingerprint(info):
    """Identify runnable image content across Docker image-store backends.

    Docker's .Id can name an OCI index or the image config after save/load.
    Platform, full runtime config and ordered uncompressed layers must agree.
    Names, tags, store metadata and the store-specific .Id are excluded.
    """
    if not isinstance(info, dict) or info.get("Architecture") != "arm64" or info.get("Os") != "linux":
        raise Refusal("Runtime image must identify Linux ARM64 content")
    variant = info.get("Variant", "")
    if not isinstance(variant, str) or not re.fullmatch(r"[A-Za-z0-9_.-]*", variant):
        raise Refusal("Runtime image has an invalid platform variant")
    config, rootfs = info.get("Config"), info.get("RootFS")
    if not isinstance(config, dict) or not isinstance(rootfs, dict) or rootfs.get("Type") != "layers":
        raise Refusal("Runtime image configuration or root filesystem is missing")
    layers = rootfs.get("Layers")
    if (not isinstance(layers, list) or not layers
            or any(not isinstance(x, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", x) for x in layers)):
        raise Refusal("Runtime image must expose ordered SHA-256 filesystem layers")
    inherited = config.get("Env")
    if inherited is None:
        inherited = []
    if not isinstance(inherited, list) or any(not isinstance(x, str) for x in inherited):
        raise Refusal("Runtime image has invalid environment metadata")
    content = {"Architecture": info["Architecture"], "Os": info["Os"], "Variant": variant,
               "Config": config, "RootFS": {"Type": "layers", "Layers": layers}}
    try:
        encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Refusal("Runtime image metadata is not valid canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def check_image(mode):
    image = CLUSTER["image"]
    info = json.loads(command(["docker", "image", "inspect", "--format", "{{json .}}", image]).stdout)
    fingerprint = image_content_fingerprint(info)
    image_id = info.get("Id")
    if not isinstance(image_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise Refusal("Runtime image has an invalid Docker image ID")
    inherited = info["Config"].get("Env") or []
    if mode != "dfly2" and any(x.startswith("TF_DS_DFLY2_DIR=") for x in inherited):
        raise Refusal("Baseline image unexpectedly inherits a DFly2 directory")
    return image_id, fingerprint


def check_container_names(config, allow_id=None):
    # Refuse either tool's preexisting container, including stopped ones.
    for kind in ("lanes", "weights"):
        name = container_name({**config, "kind": kind})
        existing = command(["docker", "ps", "-aq", "--no-trunc", "--filter", "name=^/" + re.escape(name) + "$"]).stdout.strip()
        if existing and existing != allow_id:
            raise Refusal("A validation container name is already in use")


def node_preflight(config, *, allow_id=None):
    rank = config["rank"]
    expected = CLUSTER["nodes"][rank]
    if hostname() != expected["hostname"]:
        raise Refusal(f"Expected {expected['hostname']}, reached {hostname()}")
    if Path(__file__).resolve().parents[2] != Path(expected["repo_dir"]).resolve():
        raise Refusal("Launcher repository location differs from cluster configuration")
    if config["config_identity"] != cluster.identity(CLUSTER):
        raise Refusal("Node cluster configurations disagree")
    fabric = cluster.fabric(CLUSTER, rank)
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(mem["MemAvailable"].split()[0]) * 1024
    if available < MIN_AVAILABLE:
        raise Refusal(f"Need {MIN_AVAILABLE / 1024**3:g} GiB MemAvailable, found {available / 1024**3:.2f}")
    stopped = CLUSTER["stopped"]
    for label in stopped["container_labels"]:
        if command(["docker", "ps", "-q", "--filter", "label=" + label]).stdout.strip():
            raise Refusal("A configured inference container label is running")
    for name in stopped["container_names"]:
        if command(["docker", "ps", "-q", "--filter", "name=^/" + re.escape(name) + "$"]).stdout.strip():
            raise Refusal("A configured inference container name is running")
    for user, units in ((True, stopped["systemd_user_units"]), (False, stopped["systemd_units"])):
        for unit in units:
            args = ["systemctl", *(["--user"] if user else []), "is-active", unit]
            state = command(args, check=False).stdout.strip()
            if state not in {"inactive", "failed", "unknown"}:
                raise Refusal(f"{unit} state is {state or 'unreadable'}")
    if command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]).stdout.strip():
        raise Refusal("GPU compute processes remain")
    gpu = command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).stdout.strip()
    if gpu.splitlines() != ["NVIDIA GB10"]:
        raise Refusal("Expected one NVIDIA GB10")
    check_container_names(config, allow_id)
    with socket.socket() as probe:
        probe.bind((expected["address"], PORT))
    if not Path("/dev/infiniband").is_dir():
        raise Refusal("InfiniBand device directory is unavailable")
    binary = contained(config["binary"], EXPERIMENT)
    if not os.access(binary, os.X_OK):
        raise Refusal("Staged native binary is not executable")
    contained(MODEL / "config.json", ROOT)
    paths = CLUSTER["paths"]
    request_sha, count, cache_record = None, 0, None
    if config.get("kind") != "weights":
        requests, reqs = read_requests(config["requests"])
        request_sha, count = digest(requests), len(reqs)
        for path in (ENGRAM / "config.json", Path(paths["kit"]) / "aot/aot.json", Path(paths["token_map"])):
            contained(path, ROOT)
        cache_dir = contained(paths["rank_cache"], ROOT, file=False)
        caches = list(cache_dir.glob(f"rank{rank}of2-*.bin"))
        if len(caches) != 1 or caches[0].stat().st_size == 0:
            raise Refusal(f"Expected one nonempty rank{rank} target cache")
        cache = contained(caches[0], ROOT)
        cache_record = {"name": cache.name, "bytes": cache.stat().st_size}
    pack_sha = None
    if config["mode"] == "dfly2":
        pack_sha = check_pack(config["pack"], rank)
    image = CLUSTER["image"]
    image_id, image_content_sha256 = check_image(config["mode"])
    return {"host": expected["hostname"], "rank": rank, "mem_available_bytes": available,
            "fabric": fabric, "gpu": gpu, "image": image, "image_id": image_id,
            "image_content_sha256": image_content_sha256,
            "config_identity": cluster.identity(CLUSTER), "launcher_sha256": launcher_identity(),
            "target_config_sha256": digest(MODEL / "config.json"),
            "binary_sha256": digest(binary), "requests_sha256": request_sha,
            "request_count": count, "pack_manifest_sha256": pack_sha, "rank_cache": cache_record}


def create_argv(config, preflight, cidfile):
    rank, mode = config["rank"], config["mode"]
    image = CLUSTER["image"]
    env = cluster.gpu_environment(preflight["fabric"])
    env = {k: str(v) for k, v in env.items()}
    env.update({"TF_RDMA_GID_INDEX": preflight["fabric"]["gid"], "TF_RDMA_IDLE_S": "2",
                "TF_RDMA_NAP_US": "500", "TF_DS_STREAMS": "1", "TF_DS_REPLAY_FLOOR": "1",
                "TF_DS_KV_SHARD": "0", "TF_DS_GRAPHS": "0", "TF_DS_HC_SIDE": "0",
                "TF_DS_L2_PREFETCH": "0", "TF_DS_ENGRAM_AIO": "1", "TF_DS_WINDOW": str(POOL)})
    env.update({"TF_DS_KIT": CLUSTER["paths"]["kit"], "TENSORFOLD_CUDA_KERNELS": CLUSTER["paths"]["kit"] + "/aot",
                "TF_DS_TOKEN_MAP": CLUSTER["paths"]["token_map"], "TF_DS_RANK_CACHE": CLUSTER["paths"]["rank_cache"],
                "TF_DS_ENGRAM": CLUSTER["paths"]["engram"]})
    # envOf() distinguishes unset from empty: an empty string would still select
    # DFly2 and break the DSpark baseline. Reviewed base images have no such ENV.
    if mode == "dfly2":
        env["TF_DS_DFLY2_DIR"] = config["pack"]
    # No inherited NCCL log path or managed service label.
    env.pop("NCCL_DEBUG_FILE", None)
    args = ["docker", "create", "--cidfile", str(cidfile), "--name", container_name(config),
            "--label", f"{LABEL}=1", "--label", f"{RUN_LABEL}={config['run_id']}",
            "--label", f"{RANK_LABEL}={rank}", "--network=host", "--ipc=host", "--gpus", "all",
            "--ulimit", "memlock=-1", "--device", "/dev/infiniband", "--cap-add", "IPC_LOCK",
            "--stop-timeout", "15", "--mount", f"type=bind,src={ROOT},dst={ROOT},readonly"]
    for key, value in sorted(env.items()):
        args.extend(["-e", f"{key}={value}"])
    if config.get("kind") == "weights":
        args[args.index("--network=host")] = "--network=none"
        args.extend(["--memory", "12g", "--memory-swap", "12g", "--entrypoint", config["binary"],
                     image, config["pack"], str(rank), str(MODEL)])
        return args
    args.extend(["--entrypoint", config["binary"], image, str(MODEL), CLUSTER["paths"]["rank_cache"],
                 str(rank), "2", HEAD_IP, str(PORT), CLUSTER["paths"]["kit"],
                 "--requests", config["requests"], "--engram", str(ENGRAM), "--token-map", CLUSTER["paths"]["token_map"],
                 "--pool", str(POOL), "--parallel", "1", "--streams", "1", "--graphs", "0",
                 "--side", "0", "--prefetch", "0", "--aio", "1", "--profile", "1", "--rdma", preflight["fabric"]["hcas"],
                 "--drafts", "0" if mode == "serial" else "1", "--serial", "1" if mode == "serial" else "0"])
    return args


def journal(config):
    if not re.fullmatch(r"[0-9a-f]{32}", config["run_id"]):
        raise Refusal("Invalid run UUID")
    return Path(__file__).resolve().parents[2] / ".state" / "node-runs" / config["run_id"] / f"rank{config['rank']}.cid"


def inspect_created(config, *, missing_ok=False):
    path = journal(config)
    if not path.is_file():
        if missing_ok:
            return None
        raise Refusal("No recorded container creation ID")
    cid = path.read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", cid):
        raise Refusal("Invalid recorded container ID")
    fmt = '{"Id":{{json .Id}},"Name":{{json .Name}},"Labels":{{json .Config.Labels}},"State":{{json .State}}}'
    p = command(["docker", "inspect", "--format", fmt, cid], check=False)
    if p.returncode:
        # Only confirmed absence is harmless; daemon/access errors are reported.
        if missing_ok and ("No such object" in p.stderr or "No such container" in p.stderr):
            return None
        raise Refusal("Could not inspect recorded validation container")
    info = json.loads(p.stdout)
    if (info["Id"] != cid or info["Name"] != "/" + container_name(config)
            or info["Labels"].get(LABEL) != "1" or info["Labels"].get(RUN_LABEL) != config["run_id"]
            or info["Labels"].get(RANK_LABEL) != str(config["rank"])):
        raise Refusal("Recorded ID/experimental labels disagree; refusing container action")
    return info


def node_action(action, config):
    if type(config.get("rank")) is not int or config["rank"] not in (0, 1) or config.get("mode") not in MODES:
        raise Refusal("Invalid node configuration")
    if config.get("kind", "lanes") not in {"lanes", "weights"}:
        raise Refusal("Unknown validation kind")
    if config.get("config_identity") != cluster.identity(CLUSTER):
        raise Refusal("Node cluster configurations disagree")
    if hostname() != (HEAD, WORKER)[config["rank"]]:
        raise Refusal("Node identity mismatch")
    if action == "preflight":
        return node_preflight(config)
    if action == "create":
        check = node_preflight(config)
        path = journal(config)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
        # Docker itself writes the creation ID, recoverable even if SSH drops
        # before the controller receives the result. Never create with --rm.
        command(create_argv(config, check, path), timeout=60)
        return {"created_id": inspect_created(config)["Id"]}
    if action == "cleanup":
        info = inspect_created(config, missing_ok=True)
        if info is None:
            return {"removed": False}
        command(["docker", "rm", "-f", info["Id"]], timeout=45)
        return {"removed": True, "id": info["Id"]}
    info = inspect_created(config)
    if action == "start":
        if info["State"]["Status"] != "created":
            raise Refusal("Only this run's unstarted container may be started")
        node_preflight(config, allow_id=info["Id"])
        command(["docker", "start", info["Id"]], timeout=45)
        return {"started_id": info["Id"]}
    if action == "state":
        return info["State"]
    if action == "logs":
        p = command(["docker", "logs", "--tail", "50000", info["Id"]], check=False)
        return {"stdout": p.stdout, "stderr": p.stderr, "returncode": p.returncode}
    raise Refusal("Unknown node action")


def rpc(rank, action, config):
    payload = {**config, "rank": rank}
    if rank == 0:
        return node_action(action, payload)
    node = CLUSTER["nodes"][rank]
    ssh = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=8",
           "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2", "-p", str(node["ssh_port"]), node["ssh_target"]]
    remote = ["python3", node["repo_dir"] + "/tools/dfly2/run_pair.py", "node", "--config", node["config_path"]]
    p = command([*ssh, shlex.join(remote)], timeout=90,
                input_text=json.dumps({"action": action, "config": payload}))
    response = json.loads(p.stdout)
    if not response.get("ok"):
        raise Refusal(f"{node['hostname']}: {response.get('error', 'node operation failed')}")
    return response["result"]


def preflight_pair(config):
    if hostname() != HEAD:
        raise Refusal(f"Run the coordinator on {HEAD}")
    dashboard = stopped_dashboard()
    nodes = [rpc(rank, "preflight", config) for rank in (0, 1)]
    for key in ("binary_sha256", "requests_sha256", "pack_manifest_sha256", "launcher_sha256", "config_identity", "image_content_sha256", "target_config_sha256"):
        if nodes[0][key] != nodes[1][key]:
            raise Refusal(f"The ranks disagree on {key}")
    return {"dashboard": dashboard, "nodes": nodes}


def parse_native(stdout, request_count):
    records = []
    for line in stdout.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            records.append(item)
    replies = [r for r in records if r.get("rank") == 0 and "token_sha" in r]
    summaries = [r for r in records if r.get("rank") == 0 and "requests" in r and "all_equal" in r]
    errors = [r for r in records if "error" in r]
    complete = (len(replies) == request_count and {r.get("request") for r in replies} == set(range(request_count))
                and len(summaries) == 1 and summaries[0]["requests"] == request_count and not errors
                and summaries[0]["all_equal"] is True and all(r.get("equal") is True for r in replies)
                and all(isinstance(r.get("reply"), list) and len(r["reply"]) == r.get("tokens")
                        and isinstance(r.get("reason"), str) for r in replies))
    return {"complete": complete, "records": records, "replies": replies,
            "summary": summaries[0] if len(summaries) == 1 else None}


def compare_reference(result, reference):
    if result["preflight"]["nodes"][0]["requests_sha256"] != reference["preflight"]["nodes"][0]["requests_sha256"]:
        raise Refusal("Reference used a different requests file")
    if not reference.get("success") or reference["mode"] != "serial":
        raise Refusal("Reference must be a completed serial run")
    # The native token_sha is abbreviated. Compare every token and the finish
    # reason, not only that digest, to establish exact behavioral equivalence.
    expected = {r["name"]: (r["reply"], r["reason"]) for r in reference["native"]["replies"]}
    actual = {r["name"]: (r["reply"], r["reason"]) for r in result["native"]["replies"]}
    return {"equal": actual == expected, "reference_run_id": reference["run_id"],
            "mismatches": sorted(k for k in actual.keys() | expected.keys() if actual.get(k) != expected.get(k))}


@contextmanager
def exclusive_lock():
    # Linux-only lock loaded here so --help and local unit tests also work on Windows.
    import fcntl
    lock_path = Path(__file__).resolve().parents[2] / ".state" / "pair-validation.lock"
    lock_path.parent.mkdir(mode=0o700, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Refusal("Another pair validation is running") from exc
        yield


def run_pair(config, output, timeout, reference=None):
    with exclusive_lock():
        return _run_pair(config, output, timeout, reference)


def new_output(output):
    root = EXPERIMENT.resolve(strict=True)
    output = output.resolve()
    if output == root or not output.is_relative_to(root):
        raise Refusal("Evidence output must be a new directory inside experiment_root")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    return output


def _run_pair(config, output, timeout, reference):
    report = {"run_id": config["run_id"], "mode": config["mode"], "success": False,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "preflight": preflight_pair(config), "states": {}, "cleanup": {}, "equivalence": None}
    output = new_output(output)
    deadline = time.monotonic() + timeout
    try:
        for rank in (0, 1):
            rpc(rank, "create", config)
        for rank in (0, 1):
            stopped_dashboard()  # recheck just before each GPU load
            rpc(rank, "start", config)
        while time.monotonic() < deadline:
            states = {str(rank): rpc(rank, "state", config) for rank in (0, 1)}
            report["states"] = states
            if any(s["Status"] in {"dead", "exited"} and (s.get("ExitCode") != 0 or s.get("OOMKilled")) for s in states.values()):
                raise Refusal("A validation rank failed; inspect captured logs")
            if all(s["Status"] == "exited" for s in states.values()):
                break
            time.sleep(min(2, max(0, deadline - time.monotonic())))
        else:
            raise Refusal(f"Validation exceeded {timeout} seconds")
        report["success"] = True
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # Each node's cidfile survives a lost create response. Every action still
        # verifies the exact recorded ID, run UUID, rank and experimental label.
        for rank in (0, 1):
            try:
                logs = rpc(rank, "logs", config)
                (output / f"rank{rank}.stdout.jsonl").write_text(logs["stdout"])
                (output / f"rank{rank}.stderr.log").write_text(logs["stderr"])
                if rank == 0:
                    report["native"] = parse_native(logs["stdout"], report["preflight"]["nodes"][0]["request_count"])
            except Exception as exc:
                report[f"rank{rank}_log_error"] = str(exc)
            try:
                report["cleanup"][str(rank)] = rpc(rank, "cleanup", config)
            except Exception as exc:
                report["cleanup"][str(rank)] = {"error": str(exc)}
                report["success"] = False
        report["success"] = report["success"] and report.get("native", {}).get("complete", False)
        if reference is not None:
            try:
                report["equivalence"] = compare_reference(report, reference)
                report["success"] = report["success"] and report["equivalence"]["equal"]
            except Exception as exc:
                report["equivalence"] = {"error": str(exc)}
                report["success"] = False
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "node":
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=Path, required=True)
        args = parser.parse_args(argv[1:])
        try:
            configure(cluster.load(args.config))
            request = json.load(sys.stdin)
            result = node_action(request["action"], request["config"])
            print(json.dumps({"ok": True, "result": result}))
        except Exception as exc:
            print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="private cluster JSON, identical on both nodes")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--preflight", action="store_true", help="read-only checks (the default)")
    action.add_argument("--run", action="store_true", help="create/start the isolated pair after checks")
    parser.add_argument("--mode", choices=MODES, default="dfly2")
    parser.add_argument("--requests", help="alternate requests file inside experiment_root")
    parser.add_argument("--timeout", type=int, help="pair deadline in seconds, 30..7200")
    parser.add_argument("--output", type=Path, help="new evidence directory inside experiment_root")
    parser.add_argument("--reference", type=Path, help="serial result.json for full-token/finish-reason comparison")
    args = parser.parse_args(argv)
    try:
        configure(cluster.load(args.config))
        paths = CLUSTER["paths"]
        timeout = args.timeout if args.timeout is not None else CLUSTER["execution"]["timeout_seconds"]
        if not 30 <= timeout <= 7200:
            raise Refusal("--timeout must be 30..7200 seconds")
        config = {"mode": args.mode, "kind": "lanes", "binary": paths["binary"],
                  "requests": args.requests or paths["requests"], "pack": paths["draft_pack"],
                  "run_id": uuid.uuid4().hex, "config_identity": cluster.identity(CLUSTER)}
        if not args.run:
            print(json.dumps({"ok": True, "preflight": preflight_pair(config)}, indent=2))
            return 0
        reference = json.loads(args.reference.read_text()) if args.reference else None
        output = args.output or Path(paths["results"]) / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + args.mode + "-" + config["run_id"][:8])
        def interrupted(signum, frame):
            raise InterruptedError(f"signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        result = run_pair(config, output, timeout, reference)
        print(json.dumps({"success": result["success"], "result": str(output / "result.json"),
                          "error": result.get("error"), "equivalence": result["equivalence"]}, indent=2))
        return 0 if result["success"] else 1
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

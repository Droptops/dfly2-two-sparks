#!/usr/bin/env python3
"""Export allowlisted numerical evidence from private experimental receipts.

This exporter copies only the fields below, rather than recursively redacting a
raw receipt. Unknown envelope/node/record keys never enter the public output.
Chat request text is an explicit exception: it is a reviewed synthetic fixture,
copied byte-for-byte to keep its published request digest reproducible.
"""
import argparse
import json
from pathlib import Path
import re

SCHEMA = "dfly2-public-evidence-v1"
HASH = re.compile(r"[0-9a-f]{64}\Z")
REQUEST_KEYS = {"name", "prompt", "max_tokens", "temperature", "top_p", "top_k", "seed"}
REPLY_KEYS = ("rank", "request", "name", "tokens", "reason", "token_sha", "rounds", "drafted", "accepted", "min_rows", "seconds", "reply")
PREFILL_KEYS = ("rank", "request", "name", "prompt_tokens", "cached", "restored", "prefill_s")
SUMMARY_KEYS = ("rank", "requests", "tokens", "steps", "shared_rounds", "drafted", "accepted", "seconds")
METRIC_KEYS = {
    "profile": ("rounds", "rows_a_round", "step_ms", "send_ms", "enqueue_ms", "forward_ms", "absorb_ms", "logits_ms", "sample_ms", "passes", "streams_a_pass", "pass_ms", "prefills", "prefill_ms"),
    "round_clock_us": ("rounds", "wait", "tail"),
    "prefill_engram_ms": ("calls", "rows", "upload", "mm", "exchange", "gate"),
    "engram": ("calls_a_round", "hash_ms", "read_ms", "decode_ms", "upload_ms"),
}
MEASUREMENT_KEYS = ("shape", "max_abs", "rms_abs", "different_elements", "elements", "exact_required", "rtol", "atol", "angelspec_double_round_differences")
TIMING_KEYS = ("median_us", "min_us", "max_us", "samples_us", "launches_per_graph", "samples", "warmup_launches", "baseline_output_exact", "speedup_vs_baseline")
SHAPE_KEYS = ("rows", "top_k", "n", "k", "expert_input", "selected_experts", "distinct_experts", "heads_per_rank", "context_keys", "block_keys", "slot", "causal_keys")


def pick(obj, fields):
    return {key: obj[key] for key in fields if key in obj}


def digest(value):
    if not isinstance(value, str) or not HASH.fullmatch(value):
        raise ValueError("invalid SHA-256")
    return value


def numeric(value):
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [numeric(x) for x in value]
    raise ValueError("non-numeric value in numerical evidence")


def named_record(record):
    if not re.fullmatch(r"chat_(greedy|sampled)_[a-z_]+", record["name"]):
        raise ValueError("unexpected request name")
    return {k: v if k in ("name", "reason", "token_sha") else numeric(v) for k, v in record.items()}


def sanitize_run(raw):
    if raw["mode"] not in ("serial", "dspark", "dfly2"):
        raise ValueError("unknown mode")
    nodes = []
    for n in raw["preflight"]["nodes"]:
        if n["rank"] not in (0, 1) or n["gpu"] != "NVIDIA GB10":
            raise ValueError("unexpected hardware")
        nodes.append({"rank": n["rank"], "gpu": n["gpu"],
                      "binary_sha256": digest(n["binary_sha256"]),
                      "requests_sha256": digest(n["requests_sha256"]),
                      "pack_manifest_sha256": digest(n["pack_manifest_sha256"]) if n["pack_manifest_sha256"] else None,
                      "target_rank_cache_bytes": numeric(n["rank_cache"]["bytes"])})
    native = raw["native"]
    records = []
    for r in native["records"]:
        if "reply" in r:
            record = pick(r, REPLY_KEYS)
            if record["reason"] not in ("stop", "length") or not re.fullmatch(r"[0-9a-f]{12}", record["token_sha"]):
                raise ValueError("unexpected reply metadata")
            records.append(named_record(record))
        elif "prefill_s" in r:
            records.append(named_record(pick(r, PREFILL_KEYS)))
        else:
            for name, keys in METRIC_KEYS.items():
                if name in r:
                    records.append({"rank": numeric(r["rank"]), name: {k: numeric(v) for k, v in pick(r[name], keys).items()}})
    return {"schema": SCHEMA, "suite": "chat", "mode": raw["mode"],
            "success": raw["success"] is True, "complete": native["complete"] is True,
            "configuration": {"world_size": 2, "streams": 1, "parallel": 1,
                              "pool_tokens": 8192, "cuda_graphs": False, "profiling": True},
            "nodes": nodes, "records": records,
            "summary": {k: numeric(v) for k, v in pick(native["summary"], SUMMARY_KEYS).items()}}


def validate_requests(obj):
    if set(obj) != {"requests", "metadata"}:
        raise ValueError("unreviewed request envelope")
    for request in obj["requests"]:
        if set(request) != REQUEST_KEYS:
            raise ValueError("unreviewed request fields")
        numeric(request["prompt"])
        if not re.fullmatch(r"chat_(greedy|sampled)_[a-z_]+", request["name"]):
            raise ValueError("unexpected request name")
    metadata = obj["metadata"]
    if set(metadata) != {"suite", "thinking_mode", "chat_template_sha256", "tokenizer_sha256", "tasks"}:
        raise ValueError("unreviewed request metadata")
    digest(metadata["chat_template_sha256"])
    digest(metadata["tokenizer_sha256"])
    for task in metadata["tasks"]:
        if set(task) != {"name", "messages"}:
            raise ValueError("unreviewed task metadata")
        for message in task["messages"]:
            if set(message) != {"role", "content"} or message["role"] not in ("system", "user", "assistant"):
                raise ValueError("unreviewed message fields")


def compiler_flags(command):
    return [x for x in command if re.fullmatch(r"(?:-ptx|-O[0-3]|--std=c\+\+\d+|--fmad=(?:true|false)|--ftz=(?:true|false)|-arch=compute_\d+)", x)]


def sanitize_primitives(raw):
    result = pick(raw, ("seed", "device", "capability", "torch", "torch_cuda", "source_sha256", "ptx_sha256", "passed"))
    result.update(schema=SCHEMA, source_path="zig/kernels/cuda/dsv41_dfly2.cu", compiler_flags=compiler_flags(raw["compile_command"]))
    tests = []
    for t in raw["tests"]:
        m = t["measurements"]
        rows = m if isinstance(m, list) else [m]
        tests.append({"name": t["name"], "passed": t["passed"] is True,
                      "measurements": [{k: numeric(v) for k, v in pick(row, MEASUREMENT_KEYS).items()} for row in rows]})
    result["tests"] = tests
    return result


def sanitize_benchmark(raw):
    result = pick(raw, ("seed", "device", "capability", "torch", "torch_cuda", "source_sha256", "methodology", "elapsed_seconds", "passed"))
    result.update(schema=SCHEMA, source_path="evidence/kernels/dsv41_dfly2_before_optimization.cu")
    result["variants"] = {}
    for name in ("baseline", "e8m0_bits", "e8m0_e2m1_bits"):
        v = raw["variants"][name]
        dec = pick(v["decode_check"], ("e8m0_finite_codes_exact", "e8m0_nan_code", "e8m0_code0_bits", "e2m1_all_16_bitwise_exact_including_negative_zero"))
        if dec["e8m0_code0_bits"] != "0x400000":
            raise ValueError("unexpected E8M0 subnormal encoding")
        result["variants"][name] = {"source_sha256": digest(v["source_sha256"]), "ptx_sha256": digest(v["ptx_sha256"]),
                                     "compiler_flags": compiler_flags(v["compile_command"]), "decode_check": dec}
    result["benchmarks"] = []
    for b in raw["benchmarks"]:
        result["benchmarks"].append({"name": b["name"], "shape": pick(b["shape"], SHAPE_KEYS),
            **pick(b, ("logical_code_scale_bytes_per_launch",)),
            "timings": {name: {k: numeric(v) for k, v in pick(b["timings"][name], TIMING_KEYS).items()}
                        for name in result["variants"]}})
    return result


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def export(input_dir, output_dir):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requests = (input_dir / "DFly2-chat-requests.json").read_bytes()
    validate_requests(json.loads(requests))
    (output_dir / "chat-requests.json").write_bytes(requests)
    mapping = {"DFly2-chat-serial.json": "chat-serial.json", "DFly2-chat-dspark.json": "chat-dspark-initial.json",
               "DFly2-chat-dspark-repeat.json": "chat-dspark.json", "DFly2-chat-dfly2.json": "chat-dfly2.json"}
    for source, destination in mapping.items():
        dump(output_dir / destination, sanitize_run(json.loads((input_dir / source).read_text())))
    for suffix, destination in (("", "gpu-primitives-initial.json"), ("-final", "gpu-primitives.json")):
        dump(output_dir / destination, sanitize_primitives(json.loads((input_dir / f"DFly2-Spark-GPU-primitives{suffix}.json").read_text())))
    dump(output_dir / "primitive-benchmark.json", sanitize_benchmark(json.loads((input_dir / "DFly2-primitive-benchmark.json").read_text())))
    for rank in (0, 1):
        raw = json.loads((input_dir / f"DFly2-rank{rank}-weight-check.json").read_text())
        dump(output_dir / f"rank{rank}-weight-check.json", pick(raw, ("ok", "rank", "sm", "tensors", "device_bytes", "identity", "all_uploaded_bytes_equal", "allocation_released")))
    return mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.input, args.output)

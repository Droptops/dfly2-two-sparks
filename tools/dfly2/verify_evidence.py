#!/usr/bin/env python3
"""Recompute the public 959-token comparison and its estimated decode rates.

No CUDA, model weights, network, or third-party Python packages are required.
This checks the consistency of recorded measurements; it does not rerun them.
"""
import argparse
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rounded(value, places=2):
    return float(value.quantize(Decimal(1).scaleb(-places)))


def summarize_run(run, requests, request_hash, reference):
    require(run["schema"] == "dfly2-public-evidence-v1" and run["success"] and run["complete"], "incomplete receipt")
    nodes = run["nodes"]
    require(sorted(n["rank"] for n in nodes) == [0, 1], "need exactly both TP2 ranks")
    require(all(n["requests_sha256"] == request_hash for n in nodes), "request bytes differ")
    require(len({n["binary_sha256"] for n in nodes}) == 1, "ranks used different executables")
    replies = [r for r in run["records"] if "reply" in r]
    prefills = [r for r in run["records"] if "prefill_s" in r]
    require(len(replies) == len(prefills) == len(requests), "missing request/reply/prefill")
    require([r["request"] for r in replies] == list(range(len(requests))), "replies not complete and ordered")
    require([r["request"] for r in prefills] == list(range(len(requests))), "prefills not complete and ordered")
    expected = {}
    for request, reply, prefill in zip(requests, replies, prefills):
        name = request["name"]
        require(reply["name"] == prefill["name"] == name, "unmatched request names")
        require(prefill["prompt_tokens"] == len(request["prompt"]), f"{name}: prompt count mismatch")
        tokens = reply["reply"]
        require(all(type(token) is int and 0 <= token < 129280 for token in tokens), "invalid token ID")
        require(reply["tokens"] == len(tokens) <= request["max_tokens"], f"{name}: output count mismatch")
        require(reply["reason"] in ("stop", "length"), "unexpected finish reason")
        token_hash = hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()[:12]
        require(token_hash == reply["token_sha"], f"{name}: token digest mismatch")
        require(0 <= reply["accepted"] <= reply["drafted"], "invalid draft counts")
        expected[name] = (tokens, reply["reason"])
    if reference is not None:
        require(expected == reference, f"{run['mode']}: token arrays or finish reasons differ from serial reference")
    summary = run["summary"]
    output_tokens = sum(len(r["reply"]) for r in replies)
    require(summary["requests"] == len(requests) and summary["tokens"] == output_tokens, "summary counts disagree")
    for aggregate, per_request in (("steps", "rounds"), ("drafted", "drafted"), ("accepted", "accepted")):
        require(summary[aggregate] == sum(r[per_request] for r in replies), f"summary {aggregate} disagrees")
    prefill = sum((Decimal(str(r["prefill_s"])) for r in prefills), Decimal(0))
    total = Decimal(str(summary["seconds"]))
    decode = total - prefill
    decode_tokens = sum(max(0, len(r["reply"]) - 1) for r in replies)
    require(decode > 0 and all(r["prefill_s"] >= 0 for r in prefills), "invalid elapsed time")
    result = {"mode": run["mode"], "binary_sha256": nodes[0]["binary_sha256"],
              "exact_reference_tokens_and_finish_reasons": True,
              "output_tokens": output_tokens, "decode_tokens": decode_tokens,
              "rounds": summary["steps"], "drafted": summary["drafted"], "accepted": summary["accepted"],
              "total_seconds_including_prefill": rounded(total), "summed_reported_prefill_seconds": rounded(prefill),
              "estimated_decode_seconds": rounded(decode),
              "estimated_decode_tokens_per_second": rounded(Decimal(decode_tokens) / decode)}
    return result, expected


def verify(evidence):
    evidence = Path(evidence)
    request_path = evidence / "chat-requests.json"
    request_doc = load(request_path)
    requests = request_doc["requests"]
    require(len(requests) == 6, "published chat suite contains six requests")
    require(sum(r["temperature"] == 0 for r in requests) == 3, "expected three greedy requests")
    require(sum(r["temperature"] == .7 and r["top_p"] == 1 and r["top_k"] == 20 for r in requests) == 3,
            "sampled request settings changed")
    request_hash = sha256(request_path)
    reference = None
    rows = []
    for filename, expected_mode in (("chat-serial.json", "serial"), ("chat-dspark.json", "dspark"), ("chat-dfly2.json", "dfly2")):
        run = load(evidence / filename)
        require(run["mode"] == expected_mode, "wrong receipt mode")
        row, actual = summarize_run(run, requests, request_hash, reference)
        reference = actual if reference is None else reference
        row["receipt"] = filename
        rows.append(row)
    require(all(r["output_tokens"] == 959 and r["decode_tokens"] == 953 for r in rows), "959-token published claim changed")
    initial, _ = summarize_run(load(evidence / "chat-dspark-initial.json"), requests, request_hash, reference)
    require(initial["mode"] == "dspark", "wrong repeated baseline")
    pack_hash = sha256(evidence / "draft-pack-manifest.json")
    weights = [load(evidence / f"rank{rank}-weight-check.json") for rank in (0, 1)]
    for rank, w in enumerate(weights):
        require(w["rank"] == rank and w["ok"] and w["sm"] == 121, "weight probe failed")
        require(w["tensors"] == 106 and w["device_bytes"] == 4529862920, "weight storage claim changed")
        require(w["identity"] == pack_hash and w["all_uploaded_bytes_equal"] and w["allocation_released"], "weight evidence mismatch")
    dfly_run = load(evidence / "chat-dfly2.json")
    require(all(n["pack_manifest_sha256"] == pack_hash for n in dfly_run["nodes"]), "chat drafter pack differs from checked pack")
    primitives = load(evidence / "gpu-primitives.json")
    require(primitives["passed"] and len(primitives["tests"]) == 11 and all(t["passed"] for t in primitives["tests"]), "primitive receipt not passing")
    bench = load(evidence / "primitive-benchmark.json")
    require(sha256(evidence / "kernels/dsv41_dfly2_before_optimization.cu") == bench["source_sha256"], "benchmark original source differs")
    require(bench["passed"] and len(bench["benchmarks"]) == 11, "benchmark receipt not passing")
    fp4_speedups = []
    for case in bench["benchmarks"]:
        for timing in case["timings"].values():
            require(timing["baseline_output_exact"], "benchmark candidate output differed")
            require(math.isclose(statistics.median(timing["samples_us"]), timing["median_us"], rel_tol=1e-12), "median timing differs")
        if case["name"].startswith("fp4_"):
            fp4_speedups.append(case["timings"]["baseline"]["median_us"] / case["timings"]["e8m0_e2m1_bits"]["median_us"])
    return {"schema": "dfly2-public-comparison-v1", "request_count": len(requests), "requests_sha256": request_hash,
            "all_token_arrays_and_finish_reasons_equal": True, "output_tokens_per_run": 959,
            "prompt_token_range": [min(len(r["prompt"]) for r in requests), max(len(r["prompt"]) for r in requests)],
            "runs": rows, "dspark_initial_run": initial,
            "draft_weights_bytes_per_rank": weights[0]["device_bytes"], "draft_pack_manifest_sha256": pack_hash,
            "recorded_primitive_checks_passed": 11,
            "recorded_fp4_kernel_speedup_range": [min(fp4_speedups), max(fp4_speedups)],
            "timing_method": "953 post-first-token outputs divided by (native total seconds minus six recorded prefill times); native times rounded to 0.01 seconds",
            "scope": "Recorded one-stream eager runs, profiling enabled, short prompts. Output equivalence is not a quality evaluation; kernel speedup is not model speedup."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path(__file__).resolve().parents[2] / "evidence")
    parser.add_argument("--write-summary", action="store_true", help="write the derived comparison instead of checking the stored one")
    args = parser.parse_args()
    result = verify(args.evidence)
    comparison = args.evidence / "comparison.json"
    if args.write_summary:
        comparison.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        require(load(comparison) == result, "stored comparison differs from recomputed evidence")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

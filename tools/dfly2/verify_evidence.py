#!/usr/bin/env python3
"""Recompute the public 959-token comparison and its estimated decode rates.

No CUDA, model weights, network, or third-party Python packages are required.
This checks the consistency of recorded measurements; it does not rerun them.
Consistent records are not independently attested records: nothing here proves
the saved numbers came from a GPU, only that they agree with each other.
"""
import argparse
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
KERNEL_PATH = "zig/kernels/cuda/dsv41_dfly2.cu"
# Seven recorded elapsed times are each rounded to 0.01 s, and the step clock
# excludes request admission between rounds. The published receipts agree
# within 0.015 s; a larger gap means two native clocks tell different stories.
CLOCK_TOLERANCE_SECONDS = Decimal("0.05")
CLOCK_TOLERANCE_FRACTION = Decimal("0.005")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rounded(value, places=2):
    return float(value.quantize(Decimal(1).scaleb(-places)))


def decimal(value):
    return Decimal(str(value))


def added_file_from_patch(patch, path):
    """Bytes of a file a Git patch creates, without applying the patch."""
    data = Path(patch).read_bytes()
    marker = f"diff --git a/{path} b/{path}\n".encode()
    require(data.count(marker) == 1, f"patch does not create {path} exactly once")
    start = data.index(marker)
    end = data.find(b"\ndiff --git ", start)
    lines = data[start:len(data) if end < 0 else end].split(b"\n")
    require(b"--- /dev/null" in lines[:6], f"{path} is not a new file in the patch")
    body = lines[next(i for i, line in enumerate(lines) if line.startswith(b"@@")) + 1:]
    require(body and all(line.startswith(b"+") for line in body), f"unexpected patch body for {path}")
    return b"".join(line[1:] + b"\n" for line in body)


def native_profile(run):
    profiles = [r["profile"] for r in run["records"] if "profile" in r]
    require(len(profiles) == 1, "expected exactly one native profile record")
    return profiles[0]


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
    request_rows = []
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
        # A reply's seconds run from admission, so they include its prefill.
        request_prefill, request_total = decimal(prefill["prefill_s"]), decimal(reply["seconds"])
        request_decode = request_total - request_prefill
        require(request_prefill >= 0 and request_decode >= 0, f"{name}: invalid elapsed time")
        after_first = max(0, len(tokens) - 1)
        request_rows.append({
            "name": name, "prompt_tokens": prefill["prompt_tokens"], "output_tokens": len(tokens),
            "rounds": reply["rounds"], "drafted": reply["drafted"], "accepted": reply["accepted"],
            "prefill_seconds": rounded(request_prefill), "total_seconds": rounded(request_total),
            "estimated_decode_seconds": rounded(request_decode),
            "estimated_decode_tokens_per_second": rounded(Decimal(after_first) / request_decode) if request_decode > 0 and after_first else None,
            "tokens_per_round": rounded(Decimal(after_first) / reply["rounds"]) if reply["rounds"] else None})
    if reference is not None:
        require(expected == reference, f"{run['mode']}: token arrays or finish reasons differ from serial reference")
    summary = run["summary"]
    output_tokens = sum(len(r["reply"]) for r in replies)
    require(summary["requests"] == len(requests) and summary["tokens"] == output_tokens, "summary counts disagree")
    for aggregate, per_request in (("steps", "rounds"), ("drafted", "drafted"), ("accepted", "accepted")):
        require(summary[aggregate] == sum(r[per_request] for r in replies), f"summary {aggregate} disagrees")
    prefill = sum((decimal(r["prefill_s"]) for r in prefills), Decimal(0))
    total = decimal(summary["seconds"])
    decode = total - prefill
    decode_tokens = sum(max(0, len(r["reply"]) - 1) for r in replies)
    require(decode > 0 and all(r["prefill_s"] >= 0 for r in prefills), "invalid elapsed time")
    # Two further native clocks must tell the same story as the estimate: the
    # per-reply times, and the profiler's mean step time over every round.
    tolerance = max(CLOCK_TOLERANCE_SECONDS, CLOCK_TOLERANCE_FRACTION * total)
    reply_seconds = sum((decimal(r["seconds"]) for r in replies), Decimal(0))
    require(abs(reply_seconds - total) <= tolerance, "per-reply seconds disagree with the total request time")
    profile = native_profile(run)
    rounds = summary["steps"]
    require(profile["rounds"] == rounds and rounds > 0, "native profile rounds disagree with the summary")
    step_ms, pass_ms = decimal(profile["step_ms"]), decimal(profile["pass_ms"])
    step_seconds = step_ms * rounds / 1000
    require(abs(step_seconds - decode) <= tolerance, "native step clock disagrees with the estimated decode time")
    require(0 <= pass_ms < step_ms, "invalid native draft-pass time")
    drafted, accepted = summary["drafted"], summary["accepted"]
    result = {"mode": run["mode"], "binary_sha256": nodes[0]["binary_sha256"],
              "exact_reference_tokens_and_finish_reasons": True,
              "output_tokens": output_tokens, "decode_tokens": decode_tokens,
              "rounds": rounds, "drafted": drafted, "accepted": accepted,
              "total_seconds_including_prefill": rounded(total), "summed_reported_prefill_seconds": rounded(prefill),
              "estimated_decode_seconds": rounded(decode),
              "estimated_decode_tokens_per_second": rounded(Decimal(decode_tokens) / decode),
              "end_to_end_tokens_per_second": rounded(Decimal(output_tokens) / total),
              "tokens_per_round": rounded(Decimal(decode_tokens) / rounds),
              "proposed_per_round": rounded(Decimal(drafted) / rounds),
              "accepted_per_round": rounded(Decimal(accepted) / rounds),
              "acceptance_rate": rounded(Decimal(accepted) / drafted, 3) if drafted else None,
              "native_profile": {"step_ms": float(step_ms), "forward_ms": profile["forward_ms"], "pass_ms": float(pass_ms),
                                 "rows_per_round": profile["rows_a_round"], "step_seconds": rounded(step_seconds)},
              # An optimistic bound, not a measurement: the same rounds and
              # accepted tokens with the drafter pass costing nothing.
              "decode_tokens_per_second_if_draft_pass_were_free":
                  rounded(Decimal(decode_tokens) / ((step_ms - pass_ms) * rounds / 1000)) if drafted else None,
              "requests": request_rows}
    return result, expected


def relative_to_serial(rows):
    """Compare each drafting row with the first (serial) row over identical outputs."""
    serial = rows[0]
    require(serial["mode"] == "serial", "first row must be the serial reference")
    out = []
    for row in rows[1:]:
        require(row["decode_tokens"] == serial["decode_tokens"], "runs produced different token counts")
        slower = [mine["name"] for base, mine in zip(serial["requests"], row["requests"])
                  if decimal(mine["estimated_decode_seconds"]) > decimal(base["estimated_decode_seconds"])]
        out.append({"mode": row["mode"], "receipt": row.get("receipt"),
                    "decode_speedup": rounded(decimal(serial["estimated_decode_seconds"]) / decimal(row["estimated_decode_seconds"]), 3),
                    "end_to_end_speedup": rounded(decimal(serial["total_seconds_including_prefill"]) / decimal(row["total_seconds_including_prefill"]), 3),
                    "first_request_prefill_seconds": row["requests"][0]["prefill_seconds"],
                    "serial_first_request_prefill_seconds": serial["requests"][0]["prefill_seconds"],
                    "requests_slower_than_serial": slower})
    return out


def draft_pass_kernel_budget(bench):
    """Sum the microbenchmark medians over the custom-kernel launches of one draft pass.

    Launch counts follow dfly.zig's five-draft pass: each of three layers runs
    the routed experts' w1 and w3 (1152x5120) and w2 (5120x1152) and one draft
    attention; each of five correction slots runs one context attention and
    one gated block read. Dense cuBLAS projections, the shared target head,
    mHC mixing, cross-rank gathers and host readbacks are not in this figure.
    One-GPU warm timings are set beside a two-node eager pass only to bound
    how much of that pass these kernels can explain.
    """
    median = {case["name"]: case["timings"]["e8m0_e2m1_bits"]["median_us"] for case in bench["benchmarks"]}
    experts = [3 * (2 * median[f"fp4_1152x5120_{kind}"] + median[f"fp4_5120x1152_{kind}"]) for kind in ("same_three", "eighteen_distinct")]
    attention = (3 * median["attention_draft"] + 5 * median["attention_correction"]
                 + sum(median[f"block_attention_slot{slot}"] for slot in range(1, 6)))
    return {"routed_expert_ms": [round(v / 1000, 2) for v in experts], "attention_ms": round(attention / 1000, 2),
            "total_ms": [round((v + attention) / 1000, 2) for v in experts]}


def verify(evidence, patch=None):
    evidence = Path(evidence)
    patch = ROOT / "patches/0001-dfly2-native.patch" if patch is None else Path(patch)
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
    # The passing receipt must describe the kernel this repository's patch
    # actually builds, and the earlier one the preserved original kernel.
    kernel_hash = hashlib.sha256(added_file_from_patch(patch, KERNEL_PATH)).hexdigest()
    require(primitives["source_sha256"] == kernel_hash, "primitive receipt was not measured on the published kernel source")
    original_kernel_hash = sha256(evidence / "kernels/dsv41_dfly2_before_optimization.cu")
    require(load(evidence / "gpu-primitives-initial.json")["source_sha256"] == original_kernel_hash, "initial primitive receipt differs from the preserved kernel")
    bench = load(evidence / "primitive-benchmark.json")
    require(original_kernel_hash == bench["source_sha256"], "benchmark original source differs")
    require(bench["passed"] and len(bench["benchmarks"]) == 11, "benchmark receipt not passing")
    fp4_speedups = []
    for case in bench["benchmarks"]:
        for timing in case["timings"].values():
            require(timing["baseline_output_exact"], "benchmark candidate output differed")
            require(math.isclose(statistics.median(timing["samples_us"]), timing["median_us"], rel_tol=1e-12), "median timing differs")
        if case["name"].startswith("fp4_"):
            fp4_speedups.append(case["timings"]["baseline"]["median_us"] / case["timings"]["e8m0_e2m1_bits"]["median_us"])
    return {"schema": "dfly2-public-comparison-v2", "request_count": len(requests), "requests_sha256": request_hash,
            "all_token_arrays_and_finish_reasons_equal": True, "output_tokens_per_run": 959,
            "prompt_token_range": [min(len(r["prompt"]) for r in requests), max(len(r["prompt"]) for r in requests)],
            "runs": rows, "relative_to_serial": relative_to_serial(rows), "dspark_initial_run": initial,
            "draft_weights_bytes_per_rank": weights[0]["device_bytes"], "draft_pack_manifest_sha256": pack_hash,
            "recorded_primitive_checks_passed": 11, "primitive_receipt_kernel_source_sha256": kernel_hash,
            "recorded_fp4_kernel_speedup_range": [min(fp4_speedups), max(fp4_speedups)],
            "microbenchmark_custom_kernel_ms_per_draft_pass": draft_pass_kernel_budget(bench),
            "timing_method": "953 post-first-token outputs divided by (native total seconds minus six recorded prefill times); native times rounded to 0.01 seconds; cross-checked against per-reply seconds and the native step clock",
            "scope": "Recorded one-stream eager runs, profiling enabled, short prompts. Output equivalence is not a quality evaluation; kernel speedup is not model speedup. The drafting modes do not share a proposal policy; see docs/RESULTS.md."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "evidence")
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

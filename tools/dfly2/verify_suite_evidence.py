#!/usr/bin/env python3
"""Verify an exported suite of repeated public-launcher runs.

Standard library only; this reads evidence and starts nothing on a GPU. A suite
is a set of run_suite.py runs: every mode repeated, the starting mode rotated,
each run compared token for token with the first serial run. Unlike the earlier
one-run-per-mode receipts, a suite reports a median and a range for each mode,
and can hold the matched-depth DSpark mode beside DFly2.
"""
import argparse
import json
from pathlib import Path
import statistics

from verify_evidence import decimal, load, require, rounded, sha256, summarize_run
from verify_portable_evidence import provenance

ROOT = Path(__file__).resolve().parents[2]
MODES = ("serial", "dspark", "dspark_fixed5", "dfly2")
FIXED_DEPTH = {"dspark_fixed5": 5}
SHARED = {"world_size": 2, "streams": 1, "parallel": 1, "pool_tokens": 8192, "cuda_graphs": False, "profiling": True}
# Request text is published with the receipts, so only fixtures whose text has
# been reviewed in this repository may be exported or verified.
REVIEWED_FIXTURES = {"chat": "evidence/chat-requests.json", "long": "evidence/fixtures/long-requests.json"}


def reviewed_fixture_hash(suite, root=ROOT):
    require(suite in REVIEWED_FIXTURES, "request suite has no reviewed fixture")
    return sha256(Path(root) / REVIEWED_FIXTURES[suite])


def spread(values):
    values = [decimal(v) for v in values]
    return {"median": rounded(statistics.median(values)), "min": rounded(min(values)), "max": rounded(max(values))}


def summarize_mode(mode, rows):
    names = [q["name"] for q in rows[0]["requests"]]
    per_request = []
    for index, name in enumerate(names):
        seconds = [decimal(r["requests"][index]["estimated_decode_seconds"]) for r in rows]
        tokens = max(0, rows[0]["requests"][index]["output_tokens"] - 1)
        median = statistics.median(seconds)
        per_request.append({"name": name, "prompt_tokens": rows[0]["requests"][index]["prompt_tokens"],
                            "output_tokens": rows[0]["requests"][index]["output_tokens"],
                            "median_estimated_decode_seconds": rounded(median),
                            "median_estimated_decode_tokens_per_second": rounded(tokens / median) if median > 0 and tokens else None,
                            "median_prefill_seconds": rounded(statistics.median([decimal(r["requests"][index]["prefill_seconds"]) for r in rows])),
                            "tokens_per_round": spread([r["requests"][index]["tokens_per_round"] or 0 for r in rows])})
    return {"mode": mode, "runs": len(rows),
            "estimated_decode_tokens_per_second": spread([r["estimated_decode_tokens_per_second"] for r in rows]),
            "end_to_end_tokens_per_second": spread([r["end_to_end_tokens_per_second"] for r in rows]),
            "estimated_decode_seconds": spread([r["estimated_decode_seconds"] for r in rows]),
            "total_seconds_including_prefill": spread([r["total_seconds_including_prefill"] for r in rows]),
            "tokens_per_round": spread([r["tokens_per_round"] for r in rows]),
            "proposed_per_round": spread([r["proposed_per_round"] for r in rows]),
            "accepted_per_round": spread([r["accepted_per_round"] for r in rows]),
            "draft_pass_ms": spread([r["native_profile"]["pass_ms"] for r in rows]),
            "verify_forward_ms": spread([r["native_profile"]["forward_ms"] for r in rows]),
            "rows_per_round": spread([r["native_profile"]["rows_per_round"] for r in rows]),
            "rounds_identical_across_repeats": len({r["rounds"] for r in rows}) == 1,
            "requests": per_request}


def ratio(numerator, denominator):
    return rounded(decimal(numerator) / decimal(denominator), 3)


def verify(evidence, *, root=ROOT):
    evidence, root = Path(evidence), Path(root)
    manifest = load(evidence / "suite.json")
    require(manifest.get("schema") == "dfly2-public-suite-v1", "unknown suite manifest")
    request_path = evidence / "requests.json"
    request_doc = load(request_path)
    requests, suite = request_doc["requests"], request_doc["metadata"]["suite"]
    request_hash = sha256(request_path)
    require(manifest.get("suite") == suite and request_hash == reviewed_fixture_hash(suite, root),
            "suite requests are not a reviewed fixture")
    binary_hash = load(root / "evidence/public-build-validation.json")["built_executable_sha256"]["/opt/tensorfold/bin/tf-dsv41-lanes"]
    pack_hash = sha256(root / "evidence/draft-pack-manifest.json")
    record = provenance(evidence)
    entries = manifest["runs"]
    require(isinstance(entries, list) and entries, "suite has no runs")
    reference, target, image, cache = None, None, None, None
    rows, by_mode, started = [], {}, []
    for position, entry in enumerate(entries, 1):
        require(set(entry) == {"order", "file", "mode", "repeat"} and entry["order"] == position, "malformed suite entry")
        mode = entry["mode"]
        require(mode in MODES and Path(entry["file"]).name == entry["file"], "unexpected suite entry")
        require(position > 1 or (mode == "serial" and entry["repeat"] == 1), "the first run must be the serial reference")
        run = load(evidence / entry["file"])
        require(run["mode"] == mode and run.get("validation") == "portable-launcher" and run.get("suite") == suite,
                "wrong suite receipt")
        nodes = run["nodes"]
        require(all(n["binary_sha256"] == binary_hash for n in nodes), "run differs from freshly built public binary")
        require(all(n["launcher_sha256"] == record["launcher_sha256"] for n in nodes), "run differs from the recorded published launcher source")
        current = (nodes[0]["target_config_sha256"], nodes[0]["image_content_sha256"],
                   sorted((n["rank"], n["target_rank_cache_bytes"]) for n in nodes))
        require(all((n["target_config_sha256"], n["image_content_sha256"]) == current[:2] for n in nodes), "ranks disagree on target or image")
        if target is None:
            target, image, cache = current
        require(current == (target, image, cache), "target or image differs across runs")
        config = run["configuration"]
        require(config.get("recorded_by_launcher") is True, "suite receipts need launcher-recorded settings")
        require(all(type(config[k]) is type(v) and config[k] == v for k, v in SHARED.items()), "unexpected inference settings")
        require(config["drafting"] is (mode != "serial") and config["fixed_draft_depth"] == FIXED_DEPTH.get(mode),
                "recorded drafting settings contradict the mode")
        require(all(len(r["prompt"]) + r["max_tokens"] + 16 <= config["pool_tokens"] for r in requests), "request exceeds pool")
        require(all(n["pack_manifest_sha256"] == (pack_hash if mode == "dfly2" else None) for n in nodes), "unexpected draft pack identity")
        row, actual = summarize_run(run, requests, request_hash, reference)
        reference = actual if reference is None else reference
        row.update(receipt=entry["file"], repeat=entry["repeat"], order=position, started_utc=run["started_utc"])
        rows.append(row)
        by_mode.setdefault(mode, []).append(row)
        started.append((run["started_utc"], run["finished_utc"]))
    require(all(a[0] <= a[1] <= b[0] for a, b in zip(started, started[1:])) and started[-1][0] <= started[-1][1],
            "recorded run times disagree with the manifest order")
    repeats = manifest["repeats"]
    require(list(by_mode) == manifest["modes"] and list(by_mode)[0] == "serial", "manifest modes disagree with its runs")
    for mode, mode_rows in by_mode.items():
        require(sorted(r["repeat"] for r in mode_rows) == list(range(1, repeats + 1)), f"{mode}: missing or duplicate repeat")
    modes = [summarize_mode(mode, mode_rows) for mode, mode_rows in by_mode.items()]
    serial = modes[0]
    relative = []
    for summary in modes[1:]:
        slower = [mine["name"] for base, mine in zip(serial["requests"], summary["requests"])
                  if decimal(mine["median_estimated_decode_seconds"]) > decimal(base["median_estimated_decode_seconds"])]
        relative.append({"mode": summary["mode"],
                         "median_decode_speedup": ratio(serial["estimated_decode_seconds"]["median"], summary["estimated_decode_seconds"]["median"]),
                         "median_end_to_end_speedup": ratio(serial["total_seconds_including_prefill"]["median"], summary["total_seconds_including_prefill"]["median"]),
                         "requests_slower_than_serial_by_median": slower})
    matched = None
    if "dspark_fixed5" in by_mode and "dfly2" in by_mode:
        fixed, dfly = (next(m for m in modes if m["mode"] == name) for name in ("dspark_fixed5", "dfly2"))
        # Same fixed five-draft policy for both drafters. Copy proposals remain
        # enabled only for DSpark, so this isolates depth policy, not copies.
        matched = {"modes": ["dspark_fixed5", "dfly2"],
                   "dfly2_decode_rate_relative_to_dspark_fixed5": ratio(dfly["estimated_decode_tokens_per_second"]["median"], fixed["estimated_decode_tokens_per_second"]["median"]),
                   "tokens_per_round": {"dspark_fixed5": fixed["tokens_per_round"]["median"], "dfly2": dfly["tokens_per_round"]["median"]},
                   "draft_pass_ms": {"dspark_fixed5": fixed["draft_pass_ms"]["median"], "dfly2": dfly["draft_pass_ms"]["median"]},
                   "remaining_difference": "suffix-lookup copy proposals are enabled for DSpark and disabled for DFly2"}
    return {"schema": "dfly2-suite-comparison-v1", "suite": suite, "request_count": len(requests),
            "requests_sha256": request_hash,
            "prompt_token_range": [min(len(r["prompt"]) for r in requests), max(len(r["prompt"]) for r in requests)],
            "binary_sha256": binary_hash, "launcher_sha256": record["launcher_sha256"],
            "launcher_source_commit": record["repository_commit"], "target_config_sha256": target,
            "image_content_sha256": image, "draft_pack_manifest_sha256": pack_hash, "configuration": dict(SHARED),
            "repeats": repeats, "run_order": [f"{e['mode']}-r{e['repeat']}" for e in entries],
            "all_token_arrays_and_finish_reasons_equal": True, "output_tokens_per_run": rows[0]["output_tokens"],
            "modes": modes, "relative_to_serial": relative, "matched_depth_comparison": matched, "runs": rows,
            "timing_method": "Per run: post-first-token outputs divided by native total seconds minus summed prefill times, cross-checked against per-reply seconds and the native step clock. Per mode: median, minimum and maximum over repeats.",
            "scope": "Public-launcher one-stream eager GPU runs with profiling enabled. A median over a few repeats is not a confidence interval. Output equivalence is not a quality evaluation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, help="exported suite directory containing suite.json")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    result = verify(args.evidence)
    path = args.evidence / "comparison.json"
    if args.write_summary:
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        require(load(path) == result, "stored suite comparison differs from recomputed evidence")
    print(json.dumps({k: result[k] for k in ("suite", "repeats", "run_order", "relative_to_serial", "matched_depth_comparison")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

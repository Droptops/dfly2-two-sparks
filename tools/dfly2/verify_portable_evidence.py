#!/usr/bin/env python3
"""Verify recorded portable-launcher GPU runs using only the standard library.

This reads evidence; it does not launch a process on either GPU node.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from verify_evidence import load, relative_to_serial, require, sha256, summarize_run

ROOT = Path(__file__).resolve().parents[2]
MODES = ("serial", "dspark", "dfly2")
LAUNCHER_FILES = ("tools/dfly2/run_pair.py", "tools/dfly2/cluster_config.py")
HASH = re.compile(r"[0-9a-f]{64}\Z")


def current_launcher_digest(root=ROOT):
    return hashlib.sha256(b"".join((Path(root) / name).read_bytes() for name in LAUNCHER_FILES)).hexdigest()


def provenance(evidence):
    """The launcher bytes that produced a receipt set, pinned to their commit.

    Receipts describe the launcher that ran, not whatever the repository holds
    later. Editing the launcher must not silently re-label old measurements,
    so the digest is recorded beside them together with its source commit.
    """
    record = load(Path(evidence) / "provenance.json")
    require(record.get("schema") == "dfly2-evidence-provenance-v1", "unknown provenance schema")
    require(record.get("launcher_files") == list(LAUNCHER_FILES), "unexpected launcher file list")
    for key, pattern in (("launcher_sha256", HASH), ("repository_commit", re.compile(r"[0-9a-f]{40}\Z"))):
        require(isinstance(record.get(key), str) and pattern.fullmatch(record[key]), f"invalid provenance {key}")
    return record


def verify(evidence, *, historical=None, launcher_digest=None):
    evidence = Path(evidence)
    historical = Path(historical) if historical is not None else evidence.parent
    request_path = evidence / "chat-requests.json"
    request_hash = sha256(request_path)
    require(request_hash == sha256(historical / "chat-requests.json"), "portable request fixture differs from historical fixture")
    requests = load(request_path)["requests"]
    require(len(requests) == 6, "expected six chat requests")
    build = load(historical / "public-build-validation.json")
    binary_hash = build["built_executable_sha256"]["/opt/tensorfold/bin/tf-dsv41-lanes"]
    pack_hash = sha256(historical / "draft-pack-manifest.json")
    launcher_commit = None
    if launcher_digest is None:
        record = provenance(evidence)
        launcher_digest, launcher_commit = record["launcher_sha256"], record["repository_commit"]
    _, historical_reference = summarize_run(load(historical / "chat-serial.json"), requests, request_hash, None)
    reference, config, target_hash, image_hash, cache_bytes = None, None, None, None, None
    rows = []
    for mode in MODES:
        filename = f"chat-{mode}.json"
        run = load(evidence / filename)
        require(run["mode"] == mode and run.get("validation") == "portable-launcher", "wrong portable receipt")
        require(all(n["binary_sha256"] == binary_hash for n in run["nodes"]), "run differs from freshly built public binary")
        require(all(n["launcher_sha256"] == launcher_digest for n in run["nodes"]), "run differs from the recorded published launcher source")
        configs = {n["target_config_sha256"] for n in run["nodes"]}
        require(len(configs) == 1, "ranks used different target configurations")
        current_target = next(iter(configs))
        images = {n["image_content_sha256"] for n in run["nodes"]}
        require(len(images) == 1, "ranks used different image contents")
        current_image = next(iter(images))
        current_cache = sorted((n["rank"], n["target_rank_cache_bytes"]) for n in run["nodes"])
        if config is None:
            config, target_hash, image_hash, cache_bytes = run["configuration"], current_target, current_image, current_cache
        require(run["configuration"] == config, "inference settings differ across modes")
        require(current_target == target_hash and current_cache == cache_bytes, "target metadata differs across modes")
        require(current_image == image_hash, "image contents differ across modes")
        for key, value in (("world_size", 2), ("streams", 1), ("parallel", 1), ("cuda_graphs", False), ("profiling", True)):
            require(config[key] == value, "unexpected portable inference settings")
        require(type(config["pool_tokens"]) is int and config["pool_tokens"] == 8192, "portable validation requires the 8192-token pool")
        require(all(len(r["prompt"]) + r["max_tokens"] + 16 <= config["pool_tokens"] for r in requests), "request exceeds pool")
        expected_pack = pack_hash if mode == "dfly2" else None
        require(all(n["pack_manifest_sha256"] == expected_pack for n in run["nodes"]), "unexpected draft pack identity")
        row, actual = summarize_run(run, requests, request_hash, reference)
        reference = actual if reference is None else reference
        row["receipt"] = filename
        rows.append(row)
    return {"schema": "dfly2-portable-comparison-v2", "request_count": len(requests),
            "requests_sha256": request_hash, "binary_sha256": binary_hash,
            "launcher_sha256": launcher_digest, "launcher_source_commit": launcher_commit,
            "target_config_sha256": target_hash,
            "image_content_sha256": image_hash,
            "draft_pack_manifest_sha256": pack_hash, "configuration": config,
            "all_token_arrays_and_finish_reasons_equal": True,
            "matches_historical_serial_tokens_and_finish_reasons": reference == historical_reference,
            "output_tokens_per_run": rows[0]["output_tokens"], "runs": rows,
            "relative_to_serial": relative_to_serial(rows),
            "timing_method": "Post-first-token outputs divided by native total seconds minus summed prefill times; native times rounded to 0.01 seconds; cross-checked against per-reply seconds and the native step clock",
            "scope": "Fresh public-launcher one-stream eager GPU runs, profiling enabled, six short prompts, one run per mode. Output equivalence is not a quality evaluation or a statistical throughput study. The drafting modes do not share a proposal policy; see docs/RESULTS.md."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "evidence/portable-validation")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    result = verify(args.evidence)
    path = args.evidence / "comparison.json"
    if args.write_summary:
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        require(load(path) == result, "stored portable comparison differs from recomputed evidence")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

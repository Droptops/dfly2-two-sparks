#!/usr/bin/env python3
"""Summarize matched run_pair receipts; report estimated decode timing explicitly."""
import argparse
import json
from pathlib import Path


def summarize(paths):
    runs = [json.loads(p.read_text()) for p in paths]
    if not runs or runs[0]["mode"] != "serial":
        raise ValueError("First receipt must be the serial reference")
    reference = runs[0]
    expected = {r["name"]: (r["reply"], r["reason"]) for r in reference["native"]["replies"]}
    request_hash = reference["preflight"]["nodes"][0]["requests_sha256"]
    rows = []
    for path, run in zip(paths, runs):
        if not run["success"] or not run["native"]["complete"]:
            raise ValueError(f"Incomplete or failed run: {path}")
        if any(n["requests_sha256"] != request_hash for n in run["preflight"]["nodes"]):
            raise ValueError(f"Unmatched requests: {path}")
        replies = run["native"]["replies"]
        actual = {r["name"]: (r["reply"], r["reason"]) for r in replies}
        prefill_s = sum(r["prefill_s"] for r in run["native"]["records"] if "prefill_s" in r)
        summary = run["native"]["summary"]
        decode_s = summary["seconds"] - prefill_s
        decode_tokens = sum(max(0, len(r["reply"]) - 1) for r in replies)
        rows.append({"mode": run["mode"], "run_id": run["run_id"], "receipt": str(path),
                     "binary_sha256": run["preflight"]["nodes"][0]["binary_sha256"],
                     "exact_reference_tokens_and_reasons": actual == expected,
                     "output_tokens": summary["tokens"], "drafted": summary["drafted"],
                     "accepted": summary["accepted"], "rounds": summary["steps"],
                     "total_seconds_including_prefill": summary["seconds"],
                     "summed_reported_prefill_seconds": round(prefill_s, 4),
                     "estimated_decode_seconds": round(decode_s, 4), "decode_tokens": decode_tokens,
                     "estimated_decode_tokens_per_second": round(decode_tokens / decode_s, 2) if decode_s > 0 else None,
                     "containers_removed": all(v.get("removed") for v in run["cleanup"].values())})
    return {"requests_sha256": request_hash, "request_count": len(expected), "runs": rows,
            "method": "Matched fixed tokenized requests, one stream, eager execution, profiling enabled. Decode seconds estimated as total request time minus summed reported prefill times; timings rounded to 0.01s by native recorder. Decode token count excludes the first token produced during each prefill. These are one-run measurements, not a confidence interval or a production/CUDA-graph comparison.",
            "all_equal": all(r["exact_reference_tokens_and_reasons"] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.receipts)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["all_equal"] else 1)


if __name__ == "__main__":
    main()

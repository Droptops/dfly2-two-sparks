#!/usr/bin/env python3
"""Extract unchanged target embeddings and one EXL3 head into a compact cache.

CPU-only range copying. No tensor conversion, model loading, GPU work, remote
access, or source modification. Partial outputs and an incomplete sidecar are
retained on failure; choose a new output path to retry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

from proposal_replay import HIDDEN, VOCAB, RankCache
from reference_replay import require, sha256


def extract_head_cache(source, output, rank, *, hidden=HIDDEN, vocab=VOCAB, chunk_bytes=16 << 20):
    """Copy exactly four indexed tensors; dimensions are overridable for fixtures."""
    source, output = Path(source), Path(output)
    sidecar = output.with_name(output.name + ".json")
    require(type(rank) is int and rank in (0, 1), "rank must be 0 or 1")
    require(type(hidden) is int and hidden > 0 and type(vocab) is int and vocab > 0 and vocab % 2 == 0,
            "invalid target dimensions")
    require(type(chunk_bytes) is int and chunk_bytes > 0, "chunk size must be positive")
    require(not output.exists() and not output.is_symlink(), "output cache already exists")
    require(not sidecar.exists() and not sidecar.is_symlink(), "output sidecar already exists")
    cache = RankCache(source, hidden=hidden, vocab=vocab)
    lo, hi = rank * (vocab // 2), (rank + 1) * (vocab // 2)
    keys = ["embed.weight|None"] + [f"head|({lo}, {hi})|None|{part}" for part in ("tr", "suh", "svh")]
    require(all(key in cache.entries for key in keys), "source is missing required embedding/head entries")
    # The generic index parser already checks byte counts, offsets and bounds.
    # Validate these selected entries' roles before creating any output artifact.
    tr, suh, svh = [cache.entries[key] for key in keys[1:]]
    require(tr["dtype"] == "int16" and len(tr["shape"]) == 3, "head trellis must be rank-three int16")
    require(suh["dtype"] == "float16" and suh["shape"] == [hidden], "head input scales differ")
    require(svh["dtype"] == "float16" and svh["shape"] == [vocab // 2], "head output scales differ")
    source_stat = source.stat()
    report = {"schema": "dfly2-head-cache-extract-v1", "complete": False, "rank": rank,
              "hidden": hidden, "vocab": vocab, "output_file": output.name,
              "source": {"file": source.name, "file_bytes": cache.size, "data_end": cache.data_end,
                         "index_sha256": cache.provenance["index_sha256"],
                         "footer_sha256": cache.provenance["footer_sha256"]},
              "transformation": "byte-for-byte copies of exactly four entries; only index offsets change",
              "entries": []}
    index, at = [], 0
    stream_hash = hashlib.sha256()
    started = time.time()
    created = False
    try:
        # Exclusive creation also handles another process winning a preflight race.
        with output.open("xb") as dst:
            created = True
            with source.open("rb") as src:
                for key in keys:
                    entry = cache.entries[key]
                    original_offset, length = entry["offset"], entry["bytes"]
                    src.seek(original_offset)
                    h = hashlib.sha256()
                    remaining = length
                    while remaining:
                        data = src.read(min(remaining, chunk_bytes))
                        require(bool(data), f"short source read for {key}")
                        dst.write(data)
                        h.update(data)
                        stream_hash.update(data)
                        remaining -= len(data)
                    index.append([key, entry["dtype"], entry["shape"], at, length])
                    report["entries"].append({"key": key, "dtype": entry["dtype"], "shape": entry["shape"],
                                              "source_offset": original_offset, "source_end": original_offset + length,
                                              "output_offset": at, "output_end": at + length,
                                              "bytes": length, "sha256": h.hexdigest()})
                    at += length
            raw_index = json.dumps(index, separators=(",", ":")).encode("utf-8")
            footer = struct.pack("<Q", at) + b"TFDSRK01"
            dst.write(raw_index)
            dst.write(footer)
            stream_hash.update(raw_index)
            stream_hash.update(footer)
        current_stat = source.stat()
        require((current_stat.st_size, current_stat.st_mtime_ns) == (source_stat.st_size, source_stat.st_mtime_ns),
                "source cache changed during extraction")
        # Check both the output parser and the bytes read from disk after closing.
        compact = RankCache(output, hidden=hidden, vocab=vocab)
        require(set(compact.entries) == set(keys), "compact cache entries differ")
        actual_hash = sha256(output)
        require(actual_hash == stream_hash.hexdigest(), "written cache bytes differ from copied stream")
        report.update(output_bytes=output.stat().st_size, output_sha256=actual_hash,
                      output_index_sha256=compact.provenance["index_sha256"],
                      output_footer_sha256=compact.provenance["footer_sha256"],
                      data_bytes=at, complete=True)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if created and output.exists():
            report["partial_output_bytes"] = output.stat().st_size
            report["partial_output_sha256"] = sha256(output)
        raise
    finally:
        if created:
            report["wall_seconds"] = round(time.time() - started, 3)
            # Completion status is published last. Refuse to replace any sidecar.
            with sidecar.open("x", encoding="utf-8") as f:
                f.write(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    report = extract_head_cache(args.source, args.output, args.rank)
    print(json.dumps({key: report[key] for key in ("complete", "rank", "output_file", "output_bytes", "output_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Free-running original-Tencent proposals using the deployed EXL3 head.

The head probe is admitted only after scoring the captured native correction
vectors reproduces all five captured proposal IDs. This is a diagnostic, not a
serving backend or throughput benchmark. No target weights are dequantized here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import time

import reference_replay as reference


DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "int16": 2,
               "int32": 4, "int64": 8, "uint8": 1, "float8_e4m3fn": 1}
HIDDEN, VOCAB, STEPS = 5120, 129280, 5


def digest(data):
    return hashlib.sha256(data).hexdigest()


class RankCache:
    """Read the native rank-cache index and selected bytes, never the whole model."""

    def __init__(self, path, *, hidden=HIDDEN, vocab=VOCAB):
        self.path = Path(path)
        self.hidden, self.vocab = hidden, vocab
        reference.require(self.path.is_file() and not self.path.is_symlink(), "rank cache is missing or linked")
        self.size = self.path.stat().st_size
        reference.require(self.size >= 16, "short rank-cache footer")
        with self.path.open("rb") as f:
            f.seek(-16, 2)
            footer = f.read(16)
            reference.require(footer[8:] == b"TFDSRK01", "wrong rank-cache magic")
            self.data_end = struct.unpack("<Q", footer[:8])[0]
            length = self.size - 16 - self.data_end
            reference.require(0 <= length <= 1 << 30, "bad rank-cache index range")
            f.seek(self.data_end)
            raw = f.read(length)
        reference.require(len(raw) == length, "short rank-cache index")
        rows = json.loads(raw)
        reference.require(isinstance(rows, list), "rank-cache index must be a list")
        self.entries = {}
        at = 0
        for row in rows:
            reference.require(isinstance(row, list) and len(row) == 5, "invalid rank-cache entry")
            key, dtype, shape, offset, length = row
            reference.require(isinstance(key, str) and key not in self.entries, "duplicate or invalid rank-cache key")
            reference.require(dtype in DTYPE_BYTES and isinstance(shape, list) and len(shape) <= 4,
                              "invalid rank-cache dtype or shape")
            reference.require(all(type(x) is int and x >= 0 for x in shape), "invalid rank-cache dimension")
            reference.require(type(offset) is int and type(length) is int and offset == at and length >= 0,
                              "rank-cache entries overlap or leave gaps")
            reference.require(math.prod(shape) * DTYPE_BYTES[dtype] == length and offset + length <= self.data_end,
                              "rank-cache shape or bounds mismatch")
            self.entries[key] = {"dtype": dtype, "shape": shape, "offset": offset, "bytes": length}
            at = offset + length
        reference.require(at == self.data_end, "rank-cache data has unindexed trailing bytes")
        embed = self.entries.get("embed.weight|None")
        reference.require(embed is not None and embed["dtype"] == "bfloat16" and embed["shape"] == [vocab, hidden],
                          "rank-cache embedding shape/dtype differs")
        self.provenance = {"file": self.path.name, "file_bytes": self.size,
                           "index_sha256": digest(raw), "footer_sha256": digest(footer),
                           "data_end": self.data_end, "head_entries": {}, "embedding_rows": {}}
        sidecar = self.path.with_name(self.path.name + ".json")
        if sidecar.is_file():
            sidecar_raw = sidecar.read_bytes()
            extraction = json.loads(sidecar_raw)
            if isinstance(extraction, dict) and extraction.get("schema") == "dfly2-head-cache-extract-v1":
                reference.require(extraction.get("complete") is True, "compact cache extraction is incomplete")
                reference.require(extraction.get("output_bytes") == self.size,
                                  "compact cache extraction output bytes differ")
                reference.require(extraction.get("output_index_sha256") == self.provenance["index_sha256"],
                                  "compact cache extraction index hash differs")
                reference.require(extraction.get("output_footer_sha256") == self.provenance["footer_sha256"],
                                  "compact cache extraction footer hash differs")
                # Only a recognized compact-cache sidecar triggers a complete
                # file hash; original ~106 GB caches without it are never hashed.
                reference.require(extraction.get("output_sha256") == reference.sha256(self.path),
                                  "compact cache extraction output hash differs")
                self.provenance["extraction"] = {"sidecar_file": sidecar.name, "sidecar_sha256": digest(sidecar_raw),
                                                 "manifest": extraction}

    def read_range(self, offset, length):
        reference.require(0 <= offset <= offset + length <= self.data_end, "rank-cache read outside data")
        with self.path.open("rb") as f:
            f.seek(offset)
            data = f.read(length)
        reference.require(len(data) == length, "short rank-cache tensor read")
        return data

    def hash_head(self, rank):
        reference.require(rank in (0, 1) and self.vocab % 2 == 0, "head probe requires TP2")
        lo, hi = rank * (self.vocab // 2), (rank + 1) * (self.vocab // 2)
        for part in ("tr", "suh", "svh"):
            key = f"head|({lo}, {hi})|None|{part}"
            entry = self.entries.get(key)
            reference.require(entry is not None, f"rank-cache head entry missing: {key}")
            h = hashlib.sha256()
            with self.path.open("rb") as f:
                f.seek(entry["offset"])
                remaining = entry["bytes"]
                while remaining:
                    data = f.read(min(remaining, 16 << 20))
                    reference.require(bool(data), "short rank-cache head read")
                    h.update(data)
                    remaining -= len(data)
            self.provenance["head_entries"][key] = {**entry, "sha256": h.hexdigest()}
        return self.provenance

    def embedding_bytes(self, token):
        reference.require(type(token) is int and 0 <= token < self.vocab, "embedding token outside vocabulary")
        entry = self.entries["embed.weight|None"]
        data = self.read_range(entry["offset"] + token * self.hidden * 2, self.hidden * 2)
        self.provenance["embedding_rows"][str(token)] = digest(data)
        return data

    def embedding(self, token, torch, device):
        return torch.frombuffer(bytearray(self.embedding_bytes(token)), dtype=torch.bfloat16).clone().to(device)


@dataclass(frozen=True)
class ProbeConfig:
    executable: Path
    target_model: Path
    rank0_cache: Path
    rank1_cache: Path
    grouped_cubin: Path
    timeout: float = 300

    def __post_init__(self):
        reference.require(type(self.timeout) in (int, float) and math.isfinite(self.timeout) and self.timeout > 0,
                          "head timeout must be positive and finite")

    @classmethod
    def from_args(cls, args):
        return cls(*(Path(getattr(args, name)).resolve() for name in
                     ("head_probe", "head_target_model", "head_rank0_cache", "head_rank1_cache", "head_grouped_cubin")),
                   timeout=getattr(args, "head_timeout", 300))


def summarize_logits(logits, torch):
    """Native greedy tie rule: largest FP32 score, then smallest token ID."""
    reference.require(logits.ndim == 2 and logits.shape[1] >= 2, "head logits must be [rows,vocab]")
    reference.require(bool(torch.isfinite(logits).all()), "head logits contain nonfinite values")
    results = []
    for row in logits:
        token = int(torch.argmax(row))
        score = float(row[token])
        other = row.clone()
        other[token] = -float("inf")
        runnerup = int(torch.argmax(other))
        runnerup_score = float(row[runnerup])
        results.append({"token": token, "score": score, "runnerup_token": runnerup,
                        "runnerup_score": runnerup_score, "margin": score - runnerup_score})
    return results


def score(config, input_path, output_dir, rows, torch):
    """Run the native head and independently validate its output against raw logits."""
    input_path, output_dir = Path(input_path), Path(output_dir)
    reference.require(type(rows) is int and 1 <= rows <= 128, "invalid head row count")
    reference.require(input_path.stat().st_size == rows * HIDDEN * 2, "head input byte count differs")
    reference.require(not output_dir.exists(), "head output directory already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(p) for p in (config.executable, config.target_model, config.rank0_cache, config.rank1_cache,
                            config.grouped_cubin, input_path, output_dir)]
    log = output_dir.parent / (output_dir.name + ".subprocess.log")
    started = time.time()
    # Opening the log exclusively preserves evidence even if the subprocess fails.
    with log.open("xb") as f:
        process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=config.timeout, check=False)
    reference.require(process.returncode == 0, f"native head probe failed with exit {process.returncode}; see {log.name}")
    logits_path = output_dir / "logits.bin"
    logits = reference.read_tensor(logits_path, "F32", (rows, VOCAB), torch)
    actual = summarize_logits(logits, torch)
    recorded = json.loads((output_dir / "top1.json").read_text())
    reference.require(recorded.get("schema") == "dfly2-head-probe-v1" and recorded.get("rows") == rows and
                      recorded.get("vocab") == VOCAB, "native head summary schema differs")
    reference.require(recorded.get("m_per_launch") == 1, "native head must score each row with m=1")
    for field, path in (("input_sha256", input_path), ("cubin_sha256", config.grouped_cubin),
                        ("logits_sha256", logits_path)):
        reference.require(recorded.get(field) == reference.sha256(path), f"native head {field} differs from actual file")
    reference.require(isinstance(recorded.get("results"), list) and len(recorded["results"]) == rows,
                      "native head summary row count differs")
    for expected, got in zip(actual, recorded["results"]):
        reference.require(isinstance(got, dict), "invalid native head summary row")
        for key in ("token", "runnerup_token"):
            reference.require(type(got.get(key)) is int and got[key] == expected[key], "native head summary token differs from logits")
        for key in ("score", "runnerup_score", "margin"):
            value = got.get(key)
            reference.require(type(value) in (int, float) and math.isfinite(value) and
                              math.isclose(value, expected[key], rel_tol=1e-6, abs_tol=1e-6),
                              "native head summary score differs from logits")
    return logits, {"results": actual, "input_sha256": reference.sha256(input_path),
                    "logits_sha256": reference.sha256(logits_path), "top1_sha256": reference.sha256(output_dir / "top1.json"),
                    "subprocess_log_sha256": reference.sha256(log), "wall_seconds": round(time.time() - started, 3)}


def correction_context(captured, torch):
    pos = captured["positions"].unsqueeze(0)
    valid = captured["valid"][-1]
    indices = torch.arange(valid.shape[0], device=pos.device, dtype=torch.int32).masked_fill(valid == 0, -1)
    return {"position_ids": pos, "latent": captured["context"][-1].unsqueeze(0),
            "indices": indices.view(1, 1, -1).expand(1, 6, -1), "block_size": 6}


def run_free_running(model, captured, normalized, embedding, score_vector, torch):
    """Original correction, called causally with only reference-selected predecessors.

    Callbacks provide the actual target embedding row and deployed EXL3 head.
    Model math is never substituted. This helper is shape-generic for CPU wiring
    tests; the outer diagnostic enforces the released model's exact dimensions.
    """
    reference.require(normalized.ndim == 2 and normalized.shape[0] == 6, "normalized hidden must contain six slots")
    previous = torch.zeros_like(normalized)
    previous[0] = previous[1] = captured["anchor_mask_embeddings"][0]
    context = correction_context(captured, torch)
    vectors, results, tokens = [], [], []
    for slot in range(1, 6):
        corrected = model.hidden_correction(normalized.unsqueeze(0), previous.unsqueeze(0), context)
        vector = corrected[0, slot].detach().clone()
        reference.require(bool(torch.isfinite(vector).all()), "reference correction produced nonfinite values")
        result = score_vector(slot, vector)
        token = result["results"][0]["token"]
        vectors.append(vector)
        tokens.append(token)
        results.append(result)
        if slot < 5:
            previous[slot + 1] = embedding(token)
    return {"tokens": tokens, "head_calls": results, "corrected": torch.stack(vectors),
            "previous_embeddings": previous}


def run_proposal_parity(model, captured, reference_normalized, capture_dir, output_dir, args, torch):
    """Preserve diagnostic evidence and return a qualified free-running parity report."""
    capture_dir, output_dir = Path(capture_dir), Path(output_dir)
    reference.require(not output_dir.exists(), "proposal parity output directory already exists")
    output_dir.mkdir(parents=True)
    config = args if isinstance(args, ProbeConfig) else ProbeConfig.from_args(args)
    report = {"schema": "dfly2-reference-proposals-v1", "complete": False, "passed": False,
              "head_gate_passed": False, "scope": "five greedy draft proposals from original Tencent correction and backbone, using actual deployed TP2 EXL3 head and target embeddings",
              "does_not_establish": ["target capture/index-selection correctness", "independent EXL3 head mathematical correctness",
                                     "native original-pass logit equality or margin equality (original logits were not captured)",
                                     "unquantized-target proposal parity", "sampling distribution parity", "speed parity"],
              "steps_completed": [], "embedding_cache": []}
    started = time.time()
    try:
        reference.require(tuple(reference_normalized.shape) == (6, HIDDEN), "reference normalized shape differs")
        reference.require(tuple(captured["corrected"].shape) == (5, HIDDEN), "captured correction shape differs")
        reference.require(tuple(captured["proposals"].shape) == (5,), "captured proposal shape differs")
        meta, _, capture_hashes = reference.capture_metadata(capture_dir)
        report["capture_tensor_sha256"] = capture_hashes
        report["native_proposals"] = captured["proposals"].tolist()
        report["probe_binary_sha256"] = reference.sha256(config.executable)
        report["grouped_cubin_sha256"] = reference.sha256(config.grouped_cubin)
        report["target_config_sha256"] = reference.sha256(config.target_model / "config.json")
        caches = [RankCache(config.rank0_cache), RankCache(config.rank1_cache)]
        report["embedding_cache"] = [cache.provenance for cache in caches]
        for rank, cache in enumerate(caches):
            cache.hash_head(rank)

        def embedding(token):
            raw = [cache.embedding_bytes(token) for cache in caches]
            reference.require(raw[0] == raw[1], "target embedding rows differ between TP2 caches")
            return torch.frombuffer(bytearray(raw[0]), dtype=torch.bfloat16).clone().to(reference_normalized.device)

        reference.require(torch.equal(embedding(meta["anchor_token"]), captured["anchor_mask_embeddings"][0]),
                          "actual target anchor embedding differs from capture")
        reference.require(torch.equal(embedding(int(model.config.mask_token_id)), captured["anchor_mask_embeddings"][1]),
                          "actual target mask embedding differs from capture")
        for index, token in enumerate(report["native_proposals"][:4], start=2):
            reference.require(torch.equal(embedding(token), captured["previous_embeddings"][index]),
                              "actual target predecessor embedding differs from capture")
        _, gate = score(config, capture_dir / "corrected.bin", output_dir / "native-head-gate", STEPS, torch)
        report["native_head_gate"] = gate
        gate_tokens = [row["token"] for row in gate["results"]]
        report["head_gate_passed"] = gate_tokens == report["native_proposals"]
        reference.require(report["head_gate_passed"], "shared-head gate failed: captured native proposal IDs do not reproduce")

        def score_vector(slot, vector):
            path = output_dir / f"reference-slot-{slot}.bin"
            reference.write_bf16(path, vector)
            _, result = score(config, path, output_dir / f"reference-head-slot-{slot}", 1, torch)
            report["steps_completed"].append({"slot": slot, **result})
            return result

        free = run_free_running(model, captured, reference_normalized, embedding, score_vector, torch)
        reference.write_bf16(output_dir / "reference_free_corrected.bin", free["corrected"])
        reference.write_bf16(output_dir / "reference_free_previous_embeddings.bin", free["previous_embeddings"])
        proposals = output_dir / "reference_free_proposals.bin"
        proposals.write_bytes(struct.pack("<5q", *free["tokens"]))
        report["reference_proposals"] = free["tokens"]
        report["slot_matches"] = [a == b for a, b in zip(free["tokens"], report["native_proposals"])]
        report["matching_prefix_length"] = next((i for i, same in enumerate(report["slot_matches"]) if not same), STEPS)
        report["passed"] = all(report["slot_matches"])
        report["complete"] = True
        report["output_sha256"] = {p.name: reference.sha256(p) for p in output_dir.glob("reference_free_*.bin")}
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["wall_seconds"] = round(time.time() - started, 3)
        (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

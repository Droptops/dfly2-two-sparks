#!/usr/bin/env python3
"""Byte-preserving TP2 pack for AngelSlim/DS-V4.1-Flash-DFly2-B6.

No requantization: E2M1 nibbles, E8M0 exponents, BF16 and F32 bytes stay
unchanged. Every expert's intermediate dimension is divided between ranks;
attention heads/groups are divided the same way. The target's shared embedding
and output head are deliberately absent. Requires only the Python standard
library, and never imports or executes checkpoint code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import shutil
import struct
import tempfile

FORMAT = "tensorfold-dfly2-tp2-v1"
DTYPE_SIZE = {"BF16": 2, "F32": 4, "I8": 1, "U8": 1, "F8_E8M0": 1}


def expected_tensors():
    """The released B6 schema, independently checked against all four headers."""
    out = {}

    def add(name, dtype, *shape):
        out[name] = (dtype, tuple(shape))

    add("model.final_norm.weight", "BF16", 5120)
    for n in ("enorm", "hnorm"):
        add(f"model.mtp_fusion.{n}.weight", "BF16", 5120)
    add("model.mtp_fusion.eh_proj.weight", "BF16", 5120, 10240)
    add("model.mtp_fusion.gate", "BF16", 5120)
    for n in ("hidden_norm", "embed_norm"):
        add(f"model.hidden_correction.{n}.weight", "BF16", 5120)
    for n in ("gate_proj", "up_proj"):
        add(f"model.hidden_correction.{n}.weight", "BF16", 5120, 10240)
    add("model.hidden_correction.down_proj.weight", "BF16", 5120, 5120)
    p = "model.hidden_correction.attention"
    for n, shape in {
        "q_in": (1280, 10240), "q_norm": (1280,), "wq_b": (32768, 1280),
        "wo_a": (8192, 4096), "wo_b": (5120, 8192), "kv_in": (512, 10240),
        "kv_norm": (512,), "block_gate": (64, 10240),
    }.items():
        add(f"{p}.{n}.weight", "BF16", *shape)
    for n, shape in {
        "attn_sink": (64,), "block_sink": (64,), "block_gate_bias": (64, 16),
        "block_rel_bias": (64, 16),
    }.items():
        add(f"{p}.{n}", "F32", *shape)
    for layer in range(3):
        p = f"model.layers.{layer}"
        for n in ("attn", "ffn"):
            add(f"{p}.{n}_norm.weight", "BF16", 5120)
            for kind, shape in {"fn": (24, 20480), "base": (24,), "scale": (3,)}.items():
                add(f"{p}.hc_{n}_{kind}", "F32", *shape)
        add(f"{p}.attn.attn_sink", "F32", 64)
        for n, shape in {
            "wq_a": (1280, 5120), "q_norm": (1280,), "wq_b": (32768, 1280),
            "wkv": (512, 5120), "kv_norm": (512,), "wo_a": (8192, 4096),
            "wo_b": (5120, 8192),
        }.items():
            add(f"{p}.attn.{n}.weight", "BF16", *shape)
        add(f"{p}.ffn.gate.weight", "BF16", 128, 5120)
        add(f"{p}.ffn.gate.bias", "F32", 128)
        add(f"{p}.ffn.gate.bias_vl", "F32", 128)
        for w, (n, k) in {"w1": (2304, 5120), "w2": (5120, 2304), "w3": (2304, 5120)}.items():
            add(f"{p}.ffn.shared_experts.{w}.weight", "BF16", n, k)
            for expert in range(128):
                add(f"{p}.ffn.experts.{expert}.{w}.weight", "I8", n, k // 2)
                add(f"{p}.ffn.experts.{expert}.{w}.scale", "F8_E8M0", n, k // 32)
    return out


def validate_config(config):
    if config.get("architectures") != ["DFlyDeepseekV41DraftModel"]:
        raise ValueError("wrong drafter architecture")
    t, d = config["text_config"], config["dfly_config"]
    expected = {"hidden_size": 5120, "vocab_size": 129280, "num_hidden_layers": 40,
                "num_attention_heads": 64, "head_dim": 512, "qk_rope_head_dim": 64,
                "q_lora_rank": 1280, "o_lora_rank": 1024, "o_groups": 8,
                "moe_intermediate_size": 2304, "sliding_window": 128, "hc_mult": 4,
                "index_topk": 512, "hc_sinkhorn_iters": 20}
    for key, value in expected.items():
        if t.get(key) != value:
            raise ValueError(f"unsupported {key}: {t.get(key)!r}, expected {value}")
    wanted = {"block_size": 6, "causal_block": False, "target_kv_layers": [28, 32, 36],
              "target_index_layers": [28, 32, 36], "index_topk": 512,
              "mask_token_id": 128799, "mtp_fusion": "slot0", "dense_dtype": "bfloat16"}
    for key, value in wanted.items():
        if d.get(key) != value:
            raise ValueError(f"unsupported dfly_config.{key}: {d.get(key)!r}")
    if d.get("hidden_correction") != {"intermediate_size": 5120}:
        raise ValueError("unsupported hidden correction")
    if d.get("attentive_correction") != {"source_layer": -1, "block_keys": True,
            "block_mode": "gated", "block_kv_in": "hidden_embed", "max_block_slots": 16}:
        raise ValueError("unsupported attentive correction")
    if d.get("rope") != {"theta": 160000.0, "original_seq_len": 65536,
            "factor": 16.0, "beta_fast": 32.0, "beta_slow": 1.0}:
        raise ValueError("unsupported draft RoPE")


class Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.handles, self.maps, self.tensors = [], [], {}
        self.files = []
        try:
            index = json.loads((self.directory / "model.safetensors.index.json").read_text())
            self.index = index["weight_map"]
            for name in sorted(set(self.index.values())):
                if Path(name).name != name or not name.endswith(".safetensors"):
                    raise ValueError("unsafe shard name")
                path = self.directory / name
                handle = path.open("rb")
                self.handles.append(handle)
                mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
                self.maps.append(mm)
                if len(mm) < 8:
                    raise ValueError("truncated shard")
                size = struct.unpack_from("<Q", mm)[0]
                if size > 32 << 20 or size + 8 > len(mm):
                    raise ValueError("invalid header length")
                header = json.loads(mm[8:8 + size])
                base = 8 + size
                ranges = []
                for key, value in header.items():
                    if key == "__metadata__":
                        continue
                    dtype, shape, (lo, hi) = value["dtype"], tuple(value["shape"]), value["data_offsets"]
                    if dtype not in DTYPE_SIZE or any(type(x) is not int or x <= 0 for x in shape):
                        raise ValueError(f"invalid tensor {key}")
                    if not (0 <= lo <= hi <= len(mm) - base) or hi - lo != math.prod(shape) * DTYPE_SIZE[dtype]:
                        raise ValueError(f"invalid tensor range: {key}")
                    if key in self.tensors or self.index.get(key) != name:
                        raise ValueError(f"duplicate tensor or index mismatch: {key}")
                    self.tensors[key] = (dtype, shape, mm, base + lo, base + hi)
                    ranges.append((lo, hi))
                ranges.sort()
                if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
                    raise ValueError(f"overlapping tensors in {name}")
                self.files.append(path)
            expected = expected_tensors()
            if self.tensors.keys() != expected.keys() or self.index.keys() != expected.keys():
                raise ValueError(f"schema mismatch; missing={set(expected)-set(self.tensors)}, extra={set(self.tensors)-set(expected)}")
            for key, shape in expected.items():
                if self.tensors[key][:2] != shape:
                    raise ValueError(f"{key}: {self.tensors[key][:2]} != {shape}")
        except BaseException:
            self.close()
            raise

    def close(self):
        for mm in self.maps:
            mm.close()
        for handle in self.handles:
            handle.close()
        self.maps.clear()
        self.handles.clear()

    def read_slice(self, name, axis, rank):
        dtype, shape, mm, lo, hi = self.tensors[name]
        return split_bytes(memoryview(mm)[lo:hi], shape, DTYPE_SIZE[dtype], axis, rank)


def split_shape(shape, axis):
    shape = list(shape)
    if axis is not None:
        if shape[axis] % 2:
            raise ValueError("TP2 split must be exact")
        shape[axis] //= 2
    return tuple(shape)


def split_bytes(data, shape, itemsize, axis, rank):
    """Keep raw element bytes; axis 0 is contiguous, axis 1 gathers row slices."""
    if rank not in (0, 1):
        raise ValueError("rank must be 0 or 1")
    if len(data) != math.prod(shape) * itemsize:
        raise ValueError("tensor byte count mismatch")
    if axis is None:
        return bytes(data)
    split_shape(shape, axis)
    if axis == 0:
        half = len(data) // 2
        return bytes(data[rank * half:(rank + 1) * half])
    if axis != 1 or len(shape) != 2:
        raise ValueError("only matrix column splitting is supported")
    row_bytes = shape[1] * itemsize
    half = row_bytes // 2
    out = bytearray(shape[0] * half)
    for row in range(shape[0]):
        lo = row * row_bytes + rank * half
        out[row * half:(row + 1) * half] = data[lo:lo + half]
    return out


def split_axis(name):
    if ".experts." in name or ".shared_experts." in name:
        return 1 if ".w2." in name else 0
    if any(name.endswith(s) for s in (".wq_b.weight", ".wo_a.weight", ".attn_sink",
            ".block_gate.weight", ".block_gate_bias", ".block_rel_bias", ".block_sink")):
        return 0
    if name.endswith(".wo_b.weight"):
        return 1
    # Hidden correction's dense MLP is also TP2 intermediate-sharded.
    if name in ("model.hidden_correction.gate_proj.weight", "model.hidden_correction.up_proj.weight"):
        return 0
    if name == "model.hidden_correction.down_proj.weight":
        return 1
    return None


def pack_plan():
    schema = expected_tensors()
    out = []
    for name, (dtype, shape) in schema.items():
        if ".ffn.experts." in name:
            continue
        out.append((name, dtype, split_shape(shape, split_axis(name)), [name]))
    for layer in range(3):
        for w in ("w1", "w2", "w3"):
            for kind in ("weight", "scale"):
                names = [f"model.layers.{layer}.ffn.experts.{e}.{w}.{kind}" for e in range(128)]
                shape = split_shape(schema[names[0]][1], split_axis(names[0]))
                out.append((f"model.layers.{layer}.ffn.experts.{w}.{kind}", "U8", (128,) + shape, names))
    return sorted(out)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_rank(checkpoint, destination, rank):
    plan = pack_plan()
    header = {"__metadata__": {"format": FORMAT, "rank": str(rank), "world_size": "2",
                              "fp4_nibble_order": "low_first", "fp4_scale_block": "32"}}
    offset = 0
    for name, dtype, shape, _ in plan:
        size = math.prod(shape) * DTYPE_SIZE[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with destination.open("xb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for _, _, _, source_names in plan:
            for name in source_names:
                f.write(checkpoint.read_slice(name, split_axis(name), rank))
        f.flush()
        os.fsync(f.fileno())
    return {"file": destination.name, "bytes": destination.stat().st_size,
            "tensor_bytes": offset, "sha256": sha256(destination)}


def pack(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    config = json.loads((source / "config.json").read_text())
    validate_config(config)
    checkpoint = Checkpoint(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".packing-", dir=destination.parent))
    try:
        manifest = {"format": FORMAT, "world_size": 2, "source_model": "AngelSlim/DS-V4.1-Flash-DFly2-B6",
                    "source_config_sha256": sha256(source / "config.json"),
                    "source_files": [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in checkpoint.files],
                    "ranks": [], "conversion": "byte-preserving slices; no requantization"}
        for rank in range(2):
            print(f"packing rank {rank}", flush=True)
            manifest["ranks"].append(write_rank(checkpoint, staging / f"rank{rank}.safetensors", rank))
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        (staging / "dfly2-pack.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # Public weights are read by the peer's NFS root-squashed container user.
        # mkdtemp defaults to 0700, which prevents that user traversing the pack.
        staging.chmod(0o755)
        staging.rename(destination)
        return manifest
    except BaseException:
        # Only this freshly created staging directory, never the source/destination.
        shutil.rmtree(staging)
        raise
    finally:
        checkpoint.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(pack(args.source, args.destination), indent=2))

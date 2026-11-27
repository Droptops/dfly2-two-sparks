#!/usr/bin/env python3
"""Export/verify numerical replay summaries without tensors or private paths.

Offline only. Public verification checks recorded consistency and artifact
hashes, not the original numerical calculation or provenance authenticity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct

REFERENCE_COMMIT = "e02ea36e65d44e6e165c4ba73c139d500e12f2d9"
THRESHOLDS = {"rtol": 0.02, "atol": 0.02, "max_relative_rms": 0.01}
VOCAB = 129280
CAPTURE_KEYS = {"target_hidden", "anchor_mask_embeddings", "positions", "context", "valid",
                "initial_block", "normalized", "previous_embeddings", "corrected", "proposals"}
SHAPES = {"initial_block": [6, 5120], "normalized": [6, 5120], "layers": [3, 6, 5120],
          "layers_residual": [3, 6, 4, 5120], "corrected": [5, 5120],
          "correction_given_native_hidden": [5, 5120]}
CORE_COMPARISONS = {"initial_block", "normalized", "corrected", "correction_given_native_hidden"}
SOURCE_ROLES = {"reference_replay", "proposal_replay", "pack_weights", "head_probe_source"}
LIMITATIONS = ["Recorded consistency only; hashes do not authenticate the reports or attest GPU execution.",
              "Independent numerical replay requires regeneration with the pinned code, weights, target head and capture state.",
              "No target capture/index-selection, independent EXL3 mathematics, unquantized-target or sampling-distribution parity claim.",
              "Head-gate margins are probe margins; original native pass logits and margins were not captured.",
              "Diagnostic wall time includes loading, subprocesses, synchronization and I/O; no throughput claim."]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def keys(value, allowed, required=()):
    require(isinstance(value, dict) and set(required) <= set(value) <= set(allowed), "unknown or missing fields")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA256")
    return value


def number(value, low=None):
    require(type(value) in (int, float) and math.isfinite(value) and (low is None or value >= low),
            "invalid finite numeric value")
    return value


def integer(value, low=0, high=None):
    require(type(value) is int and value >= low and (high is None or value <= high), "invalid integer")
    return value


def boolean(value):
    require(type(value) is bool, "invalid boolean")
    return value


def tokens(values, count=5):
    require(isinstance(values, list) and len(values) == count, "proposal count differs")
    return [integer(v, 0, VOCAB-1) for v in values]


def proposal_hash(values):
    return hashlib.sha256(struct.pack("<5q", *tokens(values))).hexdigest()


def load(path):
    def no_duplicates(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def hashes(value, required, allowed=None):
    keys(value, required if allowed is None else allowed, required)
    return {k: digest(v) for k, v in value.items()}


def comparison(value, name):
    core = {"passed", "finite", "shape"}
    extra = {"max_abs", "rms_abs", "reference_rms", "relative_rms", "cosine_similarity",
             "different_elements", "outside_tolerance", "elements"}
    require(name in SHAPES, "unknown comparison")
    keys(value, core | extra, core)
    finite, passed = boolean(value["finite"]), boolean(value["passed"])
    require(value["shape"] == SHAPES[name], "comparison shape differs")
    if not finite:
        require(set(value) == core and not passed, "nonfinite comparison cannot pass")
        return dict(value)
    require(set(value) == core | extra, "finite comparison lacks metrics")
    total = integer(value["elements"], 1)
    require(total == math.prod(value["shape"]), "element count differs from shape")
    different = integer(value["different_elements"], 0, total)
    outside = integer(value["outside_tolerance"], 0, different)
    maximum, rms, scale, relative = [number(value[k], 0) for k in ("max_abs", "rms_abs", "reference_rms", "relative_rms")]
    require(-1.000001 <= number(value["cosine_similarity"]) <= 1.000001, "invalid cosine similarity")
    require(rms <= maximum+1e-6 and math.isclose(relative, rms/max(scale, 1e-30), rel_tol=1e-6, abs_tol=1e-12),
            "inconsistent RMS metrics")
    require(passed == (outside == 0 and relative <= THRESHOLDS["max_relative_rms"]), "numeric pass flag differs")
    require((different == 0) == (maximum == 0), "mismatch count contradicts maximum error")
    return dict(value)


def head(value, rows):
    allowed = {"results", "input_sha256", "logits_sha256", "top1_sha256", "subprocess_log_sha256", "wall_seconds"}
    keys(value, allowed, allowed)
    out = {k: digest(value[k]) for k in allowed if k.endswith("sha256")}
    out["wall_seconds"] = number(value["wall_seconds"], 0)
    require(isinstance(value["results"], list) and len(value["results"]) == rows, "head row count differs")
    out["results"] = []
    for row in value["results"]:
        fields = {"token", "score", "runnerup_token", "runnerup_score", "margin"}
        keys(row, fields, fields)
        token, second = tokens([row["token"], row["runnerup_token"]], 2)
        score, runnerup, margin = (number(row[k]) for k in ("score", "runnerup_score", "margin"))
        require(token != second and score >= runnerup and margin >= 0, "invalid head top-two result")
        require(math.isclose(margin, score-runnerup, rel_tol=1e-6, abs_tol=1e-6), "head margin differs from scores")
        require(score != runnerup or token < second, "head tie violates lowest-ID rule")
        out["results"].append(dict(row))
    return out


def cache(value, rank):
    """Keep hashes/ranges, replacing all raw cache names with numeric rank/roles."""
    allowed = {"file", "file_bytes", "data_end", "index_sha256", "footer_sha256", "head_entries", "embedding_rows", "extraction"}
    keys(value, allowed, allowed-{"extraction"})
    size, end = integer(value["file_bytes"], 16), integer(value["data_end"])
    require(end <= size-16, "cache data extent invalid")
    out = {"rank": rank, "file_bytes": size, "data_end": end,
           "index_sha256": digest(value["index_sha256"]), "footer_sha256": digest(value["footer_sha256"]),
           "head_entries": {}, "embedding_rows": {}}
    lo, hi = rank*(VOCAB//2), (rank+1)*(VOCAB//2)
    role_keys = {f"head|({lo}, {hi})|None|{part}": part for part in ("tr", "suh", "svh")}
    require(isinstance(value["head_entries"], dict) and set(value["head_entries"]) <= set(role_keys), "unexpected head entry")
    for key, item in value["head_entries"].items():
        fields = {"dtype", "shape", "offset", "bytes", "sha256"}
        keys(item, fields, fields)
        part = role_keys[key]
        require(item["dtype"] == ("int16" if part == "tr" else "float16"), "head dtype differs")
        shape = item["shape"]
        require(isinstance(shape, list) and len(shape) == (3 if part == "tr" else 1), "head rank differs")
        require(all(integer(d, 1) for d in shape), "invalid head shape")
        if part != "tr":
            require(shape == [5120 if part == "suh" else VOCAB//2], "head scale shape differs")
        length, offset = integer(item["bytes"], 1), integer(item["offset"])
        require(length == math.prod(shape)*2 and offset+length <= end, "head entry bounds differ")
        out["head_entries"][part] = {**item, "sha256": digest(item["sha256"])}
    for token, value_hash in value["embedding_rows"].items():
        require(isinstance(token, str) and re.fullmatch(r"0|[1-9][0-9]*", token), "invalid embedding row key")
        integer(int(token), 0, VOCAB-1)
        out["embedding_rows"][token] = digest(value_hash)
    if "extraction" in value:
        extraction = value["extraction"]
        keys(extraction, {"sidecar_file", "sidecar_sha256", "manifest"}, {"sidecar_file", "sidecar_sha256", "manifest"})
        record = extraction["manifest"]
        require(record.get("schema") == "dfly2-head-cache-extract-v1" and record.get("complete") is True,
                "cache extraction incomplete")
        source = record["source"]
        require(record["rank"] == rank and record["hidden"] == 5120 and record["vocab"] == VOCAB,
                "cache extraction dimensions/rank differ")
        require(record["output_bytes"] == size and record["data_bytes"] == end and
                record["output_index_sha256"] == out["index_sha256"] and record["output_footer_sha256"] == out["footer_sha256"],
                "cache extraction identity differs")
        out["extraction"] = {"sidecar_sha256": digest(extraction["sidecar_sha256"]),
                             "output_sha256": digest(record["output_sha256"]),
                             "source_file_bytes": integer(source["file_bytes"], 16),
                             "source_data_end": integer(source["data_end"]),
                             "source_index_sha256": digest(source["index_sha256"]),
                             "source_footer_sha256": digest(source["footer_sha256"]), "entries": []}
        roles = {"embed.weight|None": "embedding", **role_keys}
        require(len(record["entries"]) == 4 and {e["key"] for e in record["entries"]} == set(roles),
                "cache extraction entries differ")
        for item in record["entries"]:
            role = roles[item["key"]]
            rec = {"role": role, "sha256": digest(item["sha256"]), **{k: integer(item[k]) for k in
                   ("source_offset", "source_end", "output_offset", "output_end", "bytes")}}
            require(rec["source_end"]-rec["source_offset"] == rec["bytes"] == rec["output_end"]-rec["output_offset"] and
                    rec["source_end"] <= source["data_end"] and rec["output_end"] <= end, "extraction range mismatch")
            if role in out["head_entries"]:
                entry = out["head_entries"][role]
                require(rec["sha256"] == entry["sha256"] and rec["bytes"] == entry["bytes"] and
                        rec["output_offset"] == entry["offset"], "extracted head identity differs")
            out["extraction"]["entries"].append(rec)
    return out


def proposal(raw):
    allowed = {"schema", "complete", "passed", "head_gate_passed", "scope", "does_not_establish", "steps_completed",
               "embedding_cache", "capture_tensor_sha256", "native_proposals", "probe_binary_sha256", "grouped_cubin_sha256",
               "target_config_sha256", "native_head_gate", "reference_proposals", "slot_matches", "matching_prefix_length",
               "output_sha256", "error", "wall_seconds"}
    keys(raw, allowed, {"schema", "complete", "passed", "head_gate_passed", "steps_completed", "embedding_cache"})
    require(raw["schema"] == "dfly2-reference-proposals-v1", "unknown proposal schema")
    out = {k: boolean(raw[k]) for k in ("complete", "passed", "head_gate_passed")}
    out.update(error_present=bool(raw.get("error")), steps_completed=[], embedding_cache=[])
    if "wall_seconds" in raw: out["wall_seconds"] = number(raw["wall_seconds"], 0)
    for key in ("probe_binary_sha256", "grouped_cubin_sha256", "target_config_sha256"):
        if key in raw: out[key] = digest(raw[key])
    if "capture_tensor_sha256" in raw:
        out["capture_tensor_sha256"] = hashes(raw["capture_tensor_sha256"], CAPTURE_KEYS, CAPTURE_KEYS | {"layers", "layers_residual"})
    if "native_proposals" in raw: out["native_proposals"] = tokens(raw["native_proposals"])
    if "native_proposals" in out and "capture_tensor_sha256" in out:
        require(proposal_hash(out["native_proposals"]) == out["capture_tensor_sha256"]["proposals"], "native IDs/file hash differs")
    require(isinstance(raw["embedding_cache"], list) and len(raw["embedding_cache"]) in (0, 2), "cache rank count differs")
    out["embedding_cache"] = [cache(c, i) for i, c in enumerate(raw["embedding_cache"])]
    if "native_head_gate" in raw:
        out["native_head_gate"] = head(raw["native_head_gate"], 5)
        require("native_proposals" in out, "head gate lacks native IDs")
        gate_ids = [r["token"] for r in out["native_head_gate"]["results"]]
        require(out["head_gate_passed"] == (gate_ids == out["native_proposals"]), "head gate flag differs from IDs")
        require(out["native_head_gate"]["input_sha256"] == out["capture_tensor_sha256"]["corrected"], "head gate input hash differs from capture")
    else:
        require(not out["head_gate_passed"], "head gate passed without scores")
    require(isinstance(raw["steps_completed"], list) and len(raw["steps_completed"]) <= 5, "invalid proposal steps")
    for slot, item in enumerate(raw["steps_completed"], 1):
        require(item.get("slot") == slot and out["head_gate_passed"], "proposal steps lack ordered passed gate")
        out["steps_completed"].append({"slot": slot, **head({k:v for k,v in item.items() if k != "slot"}, 1)})
    optional = {"reference_proposals", "slot_matches", "matching_prefix_length"}
    if optional & set(raw):
        require(optional <= set(raw), "incomplete proposal comparison fields")
        ids = tokens(raw["reference_proposals"])
        require(len(out["steps_completed"]) == 5 and ids == [s["results"][0]["token"] for s in out["steps_completed"]],
                "reference IDs differ from head steps")
        matches = [a == b for a,b in zip(ids, out["native_proposals"])]
        prefix = next((i for i,v in enumerate(matches) if not v), 5)
        require(raw["slot_matches"] == matches and all(type(v) is bool for v in raw["slot_matches"]) and
                type(raw["matching_prefix_length"]) is int and raw["matching_prefix_length"] == prefix,
                "proposal matches/prefix differs from IDs")
        out.update(reference_proposals=ids, slot_matches=matches, matching_prefix_length=prefix)
    if "output_sha256" in raw:
        names = {"reference_free_corrected.bin", "reference_free_previous_embeddings.bin", "reference_free_proposals.bin"}
        out["output_sha256"] = hashes(raw["output_sha256"], names)
        require("reference_proposals" in out and out["output_sha256"]["reference_free_proposals.bin"] == proposal_hash(out["reference_proposals"]),
                "reference IDs/file hash differs")
    if out["complete"]:
        require(optional <= set(out) and out["head_gate_passed"] and not out["error_present"] and
                {"output_sha256", "probe_binary_sha256", "grouped_cubin_sha256", "target_config_sha256", "capture_tensor_sha256"} <= set(out),
                "complete proposal report lacks evidence")
        require(len(out["embedding_cache"]) == 2 and all(set(c["head_entries"]) == {"tr", "suh", "svh"} for c in out["embedding_cache"]),
                "complete proposal report lacks heads")
        require(out["embedding_cache"][0]["embedding_rows"] == out["embedding_cache"][1]["embedding_rows"],
                "completed proposal has unequal embedding row hashes")
    require(out["passed"] == (out["complete"] and out["head_gate_passed"] and all(out.get("slot_matches", [False]))),
            "proposal pass flag differs")
    return out


def weights(raw):
    fields = {"source_config_sha256", "source_files", "parameters_loaded", "source_tensors_accounted_for",
              "unused_vision_biases", "expert_loading", "config"}
    keys(raw, fields, fields)
    out = {"source_config_sha256": digest(raw["source_config_sha256"]),
           "parameters_loaded": integer(raw["parameters_loaded"], 1),
           "source_tensors_accounted_for": integer(raw["source_tensors_accounted_for"], 1),
           "configuration_sha256": hashlib.sha256(json.dumps(raw["config"], sort_keys=True, allow_nan=False).encode()).hexdigest(),
           "source_files": [], "unused_vision_bias_equals_text": []}
    require(isinstance(raw["source_files"], list) and raw["source_files"], "empty checkpoint provenance")
    for index, item in enumerate(raw["source_files"], 1):
        keys(item, {"file", "bytes", "sha256"}, {"file", "bytes", "sha256"})
        out["source_files"].append({"index": index, "bytes": integer(item["bytes"], 1), "sha256": digest(item["sha256"])})
    for item in raw["unused_vision_biases"]:
        keys(item, {"tensor", "reason", "equals_text_bias"}, {"tensor", "reason", "equals_text_bias"})
        out["unused_vision_bias_equals_text"].append(boolean(item["equals_text_bias"]))
    return out


def case(raw, proposal_raw=None):
    fields = {"schema", "complete", "passed", "reference_commit", "capture", "capture_tensor_sha256", "torch", "device",
              "thresholds", "scope", "import_mode", "does_not_establish", "weights", "comparisons", "numeric_passed",
              "proposal_parity", "native_proposals", "output_sha256", "error", "wall_seconds"}
    required = {"schema", "complete", "passed", "reference_commit", "capture", "capture_tensor_sha256", "torch", "device", "thresholds"}
    keys(raw, fields, required)
    require(raw["schema"] == "dfly2-reference-replay-v1" and raw["reference_commit"] == REFERENCE_COMMIT, "reference source pin differs")
    require(raw["thresholds"] == THRESHOLDS, "fixed numerical tolerances differ")
    meta = raw["capture"]
    require(meta.get("schema") == "dfly2-reference-capture-v1" and meta.get("steps") == 5 and meta.get("world_size") == 2,
            "capture schema/geometry differs")
    native = meta["native"]
    require(native.get("norm_rounding") in ("fused", "double") and math.isclose(number(native["rms_norm_eps"], 0), 1e-20, rel_tol=1e-6),
            "capture normalization differs")
    capture = {"q0": integer(meta["q0"], 1), "anchor_token": integer(meta["anchor_token"], 0, VOCAB-1),
               "steps": 5, "world_size": 2, "rms_norm_eps": native["rms_norm_eps"], "norm_rounding": native["norm_rounding"],
               **{k: digest(native[k]) for k in ("binary_sha256", "checkpoint_manifest_sha256", "requests_sha256")}}
    require(isinstance(raw["torch"], str) and re.fullmatch(r"[0-9][0-9a-zA-Z.+_-]{0,95}", raw["torch"]), "invalid runtime version")
    require(isinstance(raw["device"], str) and re.fullmatch(r"cpu|cuda(?::[0-9]+)?", raw["device"]), "invalid runtime device")
    out = {"complete": boolean(raw["complete"]), "passed": boolean(raw["passed"]), "reference_commit": REFERENCE_COMMIT,
           "capture": capture, "capture_tensor_sha256": hashes(raw["capture_tensor_sha256"], CAPTURE_KEYS, CAPTURE_KEYS | {"layers", "layers_residual"}),
           "runtime": {"torch": raw["torch"], "device": raw["device"]}, "thresholds": dict(THRESHOLDS),
           "error_present": bool(raw.get("error")), "comparisons": {}, "numeric_passed": None}
    if "wall_seconds" in raw: out["wall_seconds"] = number(raw["wall_seconds"], 0)
    if "weights" in raw: out["weights"] = weights(raw["weights"])
    if "comparisons" in raw:
        require(CORE_COMPARISONS <= set(raw["comparisons"]) <= set(SHAPES), "comparison inventory differs")
        out["comparisons"] = {k:comparison(v,k) for k,v in raw["comparisons"].items()}
        out["numeric_passed"] = all(v["passed"] for v in out["comparisons"].values())
        require(raw.get("numeric_passed") is out["numeric_passed"], "aggregate numerical pass differs")
    else:
        require("numeric_passed" not in raw, "numeric pass lacks comparisons")
    if "native_proposals" in raw: out["native_proposals"] = tokens(raw["native_proposals"])
    if "native_proposals" in out:
        require(proposal_hash(out["native_proposals"]) == out["capture_tensor_sha256"]["proposals"], "case native IDs/file hash differs")
    if "output_sha256" in raw:
        names = {"reference_"+k+".bin" for k in out["comparisons"]}
        out["output_sha256"] = hashes(raw["output_sha256"], names)
    if "proposal_parity" in raw:
        if proposal_raw is None:
            require(not out["complete"] and not out["passed"] and out["error_present"] and
                    raw["proposal_parity"].get("complete") is False and raw["proposal_parity"].get("passed") is False,
                    "successful proposal report is missing")
            out["proposal_report_missing"] = True
        else:
            out["proposal"] = proposal(proposal_raw)
            for key in ("complete", "passed", "head_gate_passed", "native_proposals", "reference_proposals", "slot_matches", "matching_prefix_length"):
                if key in raw["proposal_parity"]:
                    require(raw["proposal_parity"][key] == out["proposal"].get(key), "parent/child proposal summary differs")
            if "capture_tensor_sha256" in out["proposal"]:
                require(out["proposal"]["capture_tensor_sha256"] == out["capture_tensor_sha256"], "proposal/case capture hashes differ")
            if "native_proposals" in out:
                require(out["native_proposals"] == out["proposal"].get("native_proposals"), "proposal/case native IDs differ")
    else:
        require(proposal_raw is None, "unlinked proposal report")
    if out["complete"]:
        require(not out["error_present"] and {"weights", "native_proposals", "output_sha256"} <= set(out) and out["comparisons"],
                "complete replay lacks evidence")
        require("proposal" not in out or out["proposal"]["complete"], "complete replay has incomplete proposal")
    expected = out["complete"] and out["numeric_passed"] is True and ("proposal" not in out or out["proposal"]["passed"])
    require(out["passed"] == expected, "replay pass flag differs")
    return out


def child_path(parent, name):
    require(isinstance(name, str) and not Path(name).is_absolute(), "report path must be relative")
    path = (parent/name).resolve()
    require(path.is_relative_to(parent.resolve()) and not (parent/name).is_symlink(), "report path escapes batch")
    return path


def export_case(path):
    raw = load(path)
    child = None
    child_hash = None
    if "proposal_parity" in raw:
        child_file = child_path(path.parent, raw["proposal_parity"]["report"])
        if child_file.exists(): child, child_hash = load(child_file), sha(child_file)
    else:
        require(not (path.parent/"proposal-parity/report.json").exists(), "unlinked proposal report would be omitted")
    return {"raw_report_sha256": sha(path), "raw_proposal_report_sha256": child_hash, "result": case(raw, child)}


def execution(value):
    fixed = {"world_size": 2, "streams": 1, "parallel": 1, "pool_tokens": 8192,
             "cuda_graphs": False, "profiling": True, "drafting": True}
    keys(value, set(fixed)|{"fixed_draft_depth", "experimental"}, set(fixed)|{"fixed_draft_depth", "experimental"})
    for key, expected in fixed.items():
        require(type(value[key]) is type(expected) and value[key] == expected, "capture execution setting differs")
    require(value["fixed_draft_depth"] is None or type(value["fixed_draft_depth"]) is int and value["fixed_draft_depth"] == 5,
            "capture must have five draft steps")
    controls = value["experimental"]
    flags = {"device_feedback", "rdma_best", "detailed_profile", "adaptive", "copies"}
    keys(controls, flags|{"capture", "reference_norms"}, flags|{"capture"})
    if "reference_norms" in controls: flags=flags|{"reference_norms"}
    for key in flags: boolean(controls[key])
    require(not controls["adaptive"] and not controls["copies"], "capture policy differs")
    cap = controls["capture"]
    keys(cap, {"enabled", "limit", "every"}, {"enabled", "limit", "every"})
    require(cap["enabled"] is True, "capture not enabled")
    integer(cap["limit"], 1, 64); integer(cap["every"], 1, 4096)
    return value


def run_provenance(path):
    raw = load(path)
    require(raw.get("success") is True and raw.get("mode") == "dfly2" and raw.get("native", {}).get("complete") is True,
            "capture serving run did not complete successfully")
    settings = execution(raw["execution"])
    nodes = raw["preflight"]["nodes"]
    require(len(nodes) == 2 and {n["rank"] for n in nodes} == {0, 1}, "run lacks two rank identities")
    hash_keys = {"binary_sha256", "requests_sha256", "pack_manifest_sha256", "launcher_sha256", "image_content_sha256", "target_config_sha256"}
    identities = [{k: digest(n[k]) for k in hash_keys} for n in nodes]
    require(identities[0] == identities[1] and all(n.get("gpu") == "NVIDIA GB10" for n in nodes), "run rank identities/hardware differ")
    for rank in ("0", "1"):
        state, cleanup = raw["states"][rank], raw["cleanup"][rank]
        require(state.get("Status") == "exited" and type(state.get("ExitCode")) is int and state["ExitCode"] == 0 and
                state.get("OOMKilled") is False and not state.get("Error") and cleanup.get("removed") is True,
                "capture run rank failure or missing cleanup")
    profiles = [r["profile"] for r in raw["native"]["records"] if "profile" in r and r.get("rank") == 0]
    require(len(profiles) == 1, "capture run lacks native pass counter")
    return {"raw_run_sha256": sha(path), "execution": settings, "identities": identities[0],
            "gpu": "NVIDIA GB10", "successful_ranks": [0, 1], "cleaned_ranks": [0, 1],
            "recorded_draft_passes": integer(profiles[0]["passes"], 1)}


def check_run_provenance(g):
    boolean(g["run_provenance_provided"])
    if not g["run_provenance_provided"]:
        require("run_provenance" not in g, "undeclared run provenance")
        return
    proof = g["run_provenance"]
    fields = {"raw_run_sha256", "execution", "identities", "gpu", "successful_ranks", "cleaned_ranks", "recorded_draft_passes"}
    keys(proof, fields, fields)
    digest(proof["raw_run_sha256"])
    execution(proof["execution"])
    require(proof["gpu"] == "NVIDIA GB10" and proof["successful_ranks"] == proof["cleaned_ranks"] == [0, 1], "invalid rank lifecycle")
    ids = hashes(proof["identities"], {"binary_sha256", "requests_sha256", "pack_manifest_sha256", "launcher_sha256", "image_content_sha256", "target_config_sha256"})
    count = integer(proof["recorded_draft_passes"], 1)
    controls = proof["execution"]["experimental"]
    cap = controls["capture"]
    expected = [i*cap["every"] for i in range(min(cap["limit"], (count+cap["every"]-1)//cap["every"]))]
    require(g["capture_pass_indices"] == expected, "capture names/count differ from run controls/pass count")
    if g["variant"] in ("legacy", "optimized"):
        enabled = g["variant"] == "optimized"
        require(controls["device_feedback"] is enabled and controls["rdma_best"] is enabled, "variant label contradicts recorded controls")
    for entry in g["cases"]:
        if entry["status"] != "reported": continue
        result = entry["result"]
        capture = result["capture"]
        require(capture["norm_rounding"] == ("double" if controls.get("reference_norms", False) else "fused"),
                "capture normalization differs from recorded controls")
        for a,b in (("binary_sha256", "binary_sha256"), ("requests_sha256", "requests_sha256"),
                    ("checkpoint_manifest_sha256", "pack_manifest_sha256")):
            require(capture[a] == ids[b], "run/capture source identity differs")
        prop = result.get("proposal", {})
        if "target_config_sha256" in prop:
            require(prop["target_config_sha256"] == ids["target_config_sha256"], "probe/run target config differs")


def group(spec, plan_parent):
    keys(spec, {"fixture", "variant", "report", "expected_cases", "case_prefix", "run_result"}, {"fixture", "variant", "report", "expected_cases"})
    require(spec["fixture"] in ("chat", "long", "heldout") and spec["variant"] in ("legacy", "optimized", "candidate"), "unreviewed group label")
    expected = integer(spec["expected_cases"], 1, 4096)
    path = (plan_parent/spec["report"]).resolve()
    raw = load(path)
    out = {"fixture": spec["fixture"], "variant": spec["variant"], "expected_cases": expected,
           "raw_batch_report_sha256": sha(path), "cases": [], "batch_error_present": bool(raw.get("error")),
           "batch_complete": boolean(raw["complete"]), "batch_passed": boolean(raw["passed"]),
           "run_provenance_provided": "run_result" in spec, "capture_pass_indices": []}
    if "run_result" in spec:
        out["run_provenance"] = run_provenance((plan_parent/spec["run_result"]).resolve())
    if raw.get("schema") == "dfly2-reference-replay-v1":
        require(expected == 1 and "case_prefix" not in spec, "single replay expected count/prefix differs")
        out.update(complete=boolean(raw["complete"]), passed=boolean(raw["passed"]), error_present=bool(raw.get("error")),
                   batch_expected_cases=1, batch_attempted_cases=1)
        out["cases"].append({"index": 1, "batch_index": 1, "status": "reported", **export_case(path)})
        require("run_result" not in spec, "run provenance needs a batch with captured pass names")
        return out
    require(raw.get("schema") == "dfly2-reference-batch-v1" and raw.get("reference_commit") == REFERENCE_COMMIT,
            "unknown batch schema/reference")
    requested, attempts = raw["captures_requested"], raw["cases"]
    require(isinstance(requested, list) and requested and all(isinstance(n,str) for n in requested) and
            len(set(requested)) == len(requested), "requested case inventory differs")
    require(isinstance(attempts, list) and len(attempts) <= len(requested), "attempt count exceeds inventory")
    if "case_prefix" in spec:
        prefix = spec["case_prefix"]
        require(isinstance(prefix, str) and prefix and PurePosixPath(prefix).parts == (prefix,) and "\\" not in prefix and prefix not in (".", ".."),
                "case prefix must be one relative path component")
        selected = {i for i, name in enumerate(requested, 1) if PurePosixPath(name).parts and PurePosixPath(name).parts[0] == prefix}
    else:
        selected = set(range(1, len(requested)+1))
    require(len(selected) == expected, "requested prefix case inventory differs")
    for i in sorted(selected):
        leaf = PurePosixPath(requested[i-1]).name
        match = re.fullmatch(r"capture-([0-9]{6,})", leaf)
        require(match is not None, "unexpected captured pass name")
        out["capture_pass_indices"].append(int(match[1]))
    out.update(batch_expected_cases=len(requested), batch_attempted_cases=len(attempts),
               weights_load_count=integer(raw["weights_load_count"], 0, 1))
    for name in ("weights_load_seconds", "wall_seconds"):
        if name in raw: out[name] = number(raw[name], 0)
    failed_runtime = False
    all_cases = []
    for index, name in enumerate(requested, 1):
        entry = {"index": index, "batch_index": index, "status": "not_attempted"}
        if index <= len(attempts):
            attempt = attempts[index-1]
            require(not failed_runtime and attempt["capture"] == name, "batch continued after failure or changed order")
            file = child_path(path.parent, attempt["report"])
            if file.exists():
                entry.update(status="reported", **export_case(file))
                result = entry["result"]
                require(result["complete"] == boolean(attempt["complete"]) and result["passed"] == boolean(attempt["passed"]),
                        "batch/case completion flags differ")
                if "capture_tensor_sha256" in attempt:
                    require(attempt["capture_tensor_sha256"] == result["capture_tensor_sha256"], "batch/case hashes differ")
                failed_runtime = not result["complete"]
            else:
                require(attempt.get("complete") is False and attempt.get("passed") is False and attempt.get("error"),
                        "missing successful case report")
                entry.update(status="missing_report", error_present=True)
                failed_runtime = True
        all_cases.append(entry)
    reported = [c["result"] for c in all_cases if c["status"] == "reported"]
    if out["batch_complete"]:
        require(len(reported) == len(requested) and all(r["complete"] for r in reported) and not out["batch_error_present"], "batch falsely complete")
    require(out["batch_passed"] == (out["batch_complete"] and all(r["passed"] for r in reported)), "batch pass flag differs")
    out["cases"] = [{**c, "index": i} for i,c in enumerate((c for c in all_cases if c["batch_index"] in selected), 1)]
    out["complete"] = all(c["status"] == "reported" and c["result"]["complete"] for c in out["cases"])
    out["passed"] = out["complete"] and all(c["result"]["passed"] for c in out["cases"])
    out["error_present"] = any(c["status"] == "missing_report" or c.get("result", {}).get("error_present", False) for c in out["cases"])
    check_run_provenance(out)
    return out


def summarize(report):
    reported = [c["result"] for g in report["groups"] for c in g["cases"] if c["status"] == "reported"]
    proposals = [r["proposal"] for r in reported if "proposal" in r]
    return {"groups": len(report["groups"]), "expected_cases": sum(g["expected_cases"] for g in report["groups"]),
            "reported_cases": len(reported), "complete_cases": sum(r["complete"] for r in reported),
            "unique_captured_tensor_sets": len({json.dumps(r["capture_tensor_sha256"], sort_keys=True) for r in reported}),
            "numerical_passes": sum(r["numeric_passed"] is True for r in reported),
            "numerical_failures": sum(r["numeric_passed"] is False for r in reported),
            "proposal_reports": len(proposals), "head_gate_passes": sum(r["head_gate_passed"] for r in proposals),
            "missing_proposal_reports": sum(r.get("proposal_report_missing", False) for r in reported),
            "complete_proposal_comparisons": sum(r["complete"] for r in proposals),
            "exact_proposal_passes": sum(r["passed"] for r in proposals),
            "matched_slots": sum(sum(r.get("slot_matches", [])) for r in proposals),
            "compared_slots": sum(len(r.get("slot_matches", [])) for r in proposals),
            "all_declared_cases_complete": all(g["complete"] for g in report["groups"]),
            "all_declared_cases_passed": all(g["passed"] for g in report["groups"]),
            "all_source_batches_complete": all(g["batch_complete"] for g in report["groups"]),
            "all_source_batches_passed": all(g["batch_passed"] for g in report["groups"])}


def export(plan_path, output):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = load(plan_path)
    keys(plan, {"schema", "groups", "source_files"}, {"schema", "groups"})
    require(plan["schema"] == "dfly2-reference-export-plan-v1" and isinstance(plan["groups"], list) and plan["groups"], "invalid export plan")
    require(not output.exists(), "export destination already exists")
    report = {"schema": "dfly2-reference-public-v1", "reference_commit": REFERENCE_COMMIT,
              "thresholds": dict(THRESHOLDS), "purpose": "numerical_diagnostic", "limitations": LIMITATIONS,
              "export_time_source_sha256": {}, "groups": [group(g, plan_path.parent) for g in plan["groups"]]}
    for role, item in plan.get("source_files", {}).items():
        require(role in SOURCE_ROLES, "unknown source role")
        keys(item, {"path", "sha256"}, {"path", "sha256"})
        actual = sha((plan_path.parent/item["path"]).resolve())
        require(actual == digest(item["sha256"]), "declared source file hash differs")
        report["export_time_source_sha256"][role] = actual
    report["summary"] = summarize(report)
    validate_public(report)
    output.mkdir(parents=True)
    write(output/"reference.json", report)
    write(output/"manifest.json", {"schema": "dfly2-reference-public-manifest-v1", "reference_sha256": sha(output/"reference.json"),
                                    "exporter_sha256": sha(Path(__file__)), "raw_plan_sha256": sha(plan_path)})
    return verify(output)


def validate_public(report):
    """Rebuild raw-shaped whitelisted adapters, then reapply every semantic gate."""
    fields = {"schema", "reference_commit", "thresholds", "purpose", "limitations", "export_time_source_sha256", "groups", "summary"}
    keys(report, fields, fields)
    require(report["schema"] == "dfly2-reference-public-v1" and report["reference_commit"] == REFERENCE_COMMIT and
            report["thresholds"] == THRESHOLDS and report["purpose"] == "numerical_diagnostic" and report["limitations"] == LIMITATIONS,
            "public report scope/pins differ")
    hashes(report["export_time_source_sha256"], set(), SOURCE_ROLES)
    require(isinstance(report["groups"], list) and report["groups"], "empty public report")
    batches = {}
    for g in report["groups"]:
        validate_public_group(g)
        identity = g["raw_batch_report_sha256"]
        batch = batches.setdefault(identity, {"fields": {k:v for k,v in g.items() if k.startswith("batch_")}, "cases": {}})
        require(batch["fields"] == {k:v for k,v in g.items() if k.startswith("batch_")}, "same-batch declarations differ")
        for c in g["cases"]:
            require(c["batch_index"] not in batch["cases"], "batch case exported more than once")
            batch["cases"][c["batch_index"]] = c
    for batch in batches.values():
        meta, cases = batch["fields"], batch["cases"]
        require(set(cases) == set(range(1, meta["batch_expected_cases"]+1)), "export dropped requested cases from a source batch")
        attempted = [c for c in cases.values() if c["status"] != "not_attempted"]
        require(len(attempted) == meta["batch_attempted_cases"], "batch attempt count differs")
        require({c["batch_index"] for c in attempted} == set(range(1, len(attempted)+1)), "batch attempts are not a prefix")
        stopped = False
        for index in sorted(cases):
            c = cases[index]
            require(not stopped or c["status"] == "not_attempted", "batch continued after runtime failure")
            stopped = c["status"] != "reported" or not c["result"]["complete"]
        results = [c["result"] for c in attempted if c["status"] == "reported"]
        if meta["batch_complete"]:
            require(len(results) == len(cases) and all(r["complete"] for r in results) and not meta["batch_error_present"], "source batch falsely complete")
        require(meta["batch_passed"] == (meta["batch_complete"] and all(r["passed"] for r in results)), "source batch pass differs")
    require(report["summary"] == summarize(report), "summary differs from case evidence")


def validate_public_group(g):
    fields = {"fixture", "variant", "expected_cases", "raw_batch_report_sha256", "cases", "error_present", "complete", "passed",
              "weights_load_count", "weights_load_seconds", "wall_seconds", "batch_complete", "batch_passed", "batch_error_present",
              "batch_expected_cases", "batch_attempted_cases", "run_provenance_provided", "run_provenance", "capture_pass_indices"}
    keys(g, fields, fields-{"weights_load_count", "weights_load_seconds", "wall_seconds", "run_provenance"})
    require(g["fixture"] in ("chat", "long", "heldout") and g["variant"] in ("legacy", "optimized", "candidate"), "unreviewed group label")
    digest(g["raw_batch_report_sha256"])
    expected = integer(g["expected_cases"], 1, 4096)
    for name in ("complete", "passed", "error_present", "batch_complete", "batch_passed", "batch_error_present"): boolean(g[name])
    integer(g["batch_expected_cases"], expected)
    integer(g["batch_attempted_cases"], 0, g["batch_expected_cases"])
    if "weights_load_count" in g: integer(g["weights_load_count"], 0, 1)
    for name in ("weights_load_seconds", "wall_seconds"):
        if name in g: number(g[name], 0)
    require(isinstance(g["cases"], list) and len(g["cases"]) == expected, "public case inventory differs")
    require(isinstance(g["capture_pass_indices"], list) and len(g["capture_pass_indices"]) in (0, expected), "capture pass inventory differs")
    for value in g["capture_pass_indices"]: integer(value)
    stopped = False
    results = []
    for index, entry in enumerate(g["cases"], 1):
        require(type(entry.get("index")) is int and entry["index"] == index, "public case order differs")
        integer(entry.get("batch_index"), 1, g["batch_expected_cases"])
        status = entry.get("status")
        require(status in ("reported", "missing_report", "not_attempted"), "invalid case status")
        if status == "reported":
            keys(entry, {"index", "batch_index", "status", "raw_report_sha256", "raw_proposal_report_sha256", "result"},
                 {"index", "batch_index", "status", "raw_report_sha256", "raw_proposal_report_sha256", "result"})
            require(not stopped, "reported case after runtime failure/unattempted case")
            digest(entry["raw_report_sha256"])
            if entry["raw_proposal_report_sha256"] is not None: digest(entry["raw_proposal_report_sha256"])
            require((entry["raw_proposal_report_sha256"] is not None) == ("proposal" in entry["result"]), "proposal hash/report mismatch")
            validate_public_case(entry["result"])
            results.append(entry["result"])
            stopped = not entry["result"]["complete"]
        elif status == "missing_report":
            keys(entry, {"index", "batch_index", "status", "error_present"}, {"index", "batch_index", "status", "error_present"})
            require(not stopped and entry["error_present"] is True, "missing-report failure lacks evidence")
            stopped = True
        else:
            keys(entry, {"index", "batch_index", "status"}, {"index", "batch_index", "status"})
            stopped = True
    if g["complete"]:
        require(not g["error_present"] and len(results) == expected and all(r["complete"] for r in results), "public batch falsely complete")
    require(g["complete"] == (len(results) == expected and all(r["complete"] for r in results)), "group completeness differs")
    require(g["passed"] == (g["complete"] and all(r["passed"] for r in results)), "public batch pass differs")
    check_run_provenance(g)


def raw_cache(clean):
    fields = {"rank", "file_bytes", "data_end", "index_sha256", "footer_sha256", "head_entries", "embedding_rows", "extraction"}
    keys(clean, fields, fields-{"extraction"})
    rank = integer(clean["rank"], 0, 1)
    lo, hi = rank*(VOCAB//2), (rank+1)*(VOCAB//2)
    out = {k:v for k,v in clean.items() if k not in ("rank", "head_entries", "extraction")}
    out["file"] = "redacted"
    out["head_entries"] = {f"head|({lo}, {hi})|None|{p}": v for p,v in clean["head_entries"].items()}
    if "extraction" in clean:
        ext = clean["extraction"]
        fields = {"sidecar_sha256", "output_sha256", "source_file_bytes", "source_data_end", "source_index_sha256", "source_footer_sha256", "entries"}
        keys(ext, fields, fields)
        entries = []
        for entry in ext["entries"]:
            keys(entry, {"role", "sha256", "source_offset", "source_end", "output_offset", "output_end", "bytes"},
                 {"role", "sha256", "source_offset", "source_end", "output_offset", "output_end", "bytes"})
            require(entry["role"] in ("embedding", "tr", "suh", "svh"), "unknown extraction role")
            key = "embed.weight|None" if entry["role"] == "embedding" else f"head|({lo}, {hi})|None|{entry['role']}"
            entries.append({"key": key, **{k:v for k,v in entry.items() if k != "role"}})
        out["extraction"] = {"sidecar_file": "redacted", "sidecar_sha256": ext["sidecar_sha256"], "manifest": {
            "schema": "dfly2-head-cache-extract-v1", "complete": True, "rank": rank, "hidden": 5120, "vocab": VOCAB,
            "output_bytes": clean["file_bytes"], "data_bytes": clean["data_end"], "output_sha256": ext["output_sha256"],
            "output_index_sha256": clean["index_sha256"], "output_footer_sha256": clean["footer_sha256"], "entries": entries,
            "source": {"file_bytes": ext["source_file_bytes"], "data_end": ext["source_data_end"],
                       "index_sha256": ext["source_index_sha256"], "footer_sha256": ext["source_footer_sha256"]}}}
    return out


def raw_proposal(clean):
    raw = {k:v for k,v in clean.items() if k not in ("error_present", "embedding_cache")}
    raw.update(schema="dfly2-reference-proposals-v1", embedding_cache=[raw_cache(c) for c in clean["embedding_cache"]])
    if clean["error_present"]: raw["error"] = "redacted"
    return raw


def validate_public_case(clean):
    fields = {"complete", "passed", "reference_commit", "capture", "capture_tensor_sha256", "runtime", "thresholds", "error_present",
              "comparisons", "numeric_passed", "wall_seconds", "weights", "native_proposals", "output_sha256", "proposal", "proposal_report_missing"}
    required = fields-{"wall_seconds", "weights", "native_proposals", "output_sha256", "proposal", "proposal_report_missing"}
    keys(clean, fields, required)
    boolean(clean["error_present"])
    capture_fields = {"q0", "anchor_token", "steps", "world_size", "rms_norm_eps", "norm_rounding",
                      "binary_sha256", "checkpoint_manifest_sha256", "requests_sha256"}
    keys(clean["capture"], capture_fields, capture_fields)
    keys(clean["runtime"], {"torch", "device"}, {"torch", "device"})
    cap = clean["capture"]
    raw = {k:v for k,v in clean.items() if k not in ("runtime", "error_present", "weights", "proposal", "proposal_report_missing")}
    raw.update(schema="dfly2-reference-replay-v1", torch=clean["runtime"]["torch"], device=clean["runtime"]["device"],
               capture={"schema": "dfly2-reference-capture-v1", **{k:cap[k] for k in ("q0", "anchor_token", "steps", "world_size")},
                        "native": {k:v for k,v in cap.items() if k not in ("q0", "anchor_token", "steps", "world_size")}})
    if not clean["comparisons"]:
        require(clean["numeric_passed"] is None, "empty comparisons claim numerical result")
        raw.pop("comparisons"); raw.pop("numeric_passed")
    if clean["error_present"]: raw["error"] = "redacted"
    if "weights" in clean:
        w = clean["weights"]
        fields = {"source_config_sha256", "parameters_loaded", "source_tensors_accounted_for", "configuration_sha256", "source_files", "unused_vision_bias_equals_text"}
        keys(w, fields, fields)
        digest(w["configuration_sha256"])
        for i, entry in enumerate(w["source_files"], 1):
            keys(entry, {"index", "bytes", "sha256"}, {"index", "bytes", "sha256"})
            require(type(entry["index"]) is int and entry["index"] == i, "checkpoint file order differs")
        raw["weights"] = {k:w[k] for k in ("source_config_sha256", "parameters_loaded", "source_tensors_accounted_for")}
        raw["weights"].update(source_files=[{"file": "redacted", "bytes": x["bytes"], "sha256": x["sha256"]} for x in w["source_files"]],
                              config={}, expert_loading="redacted", unused_vision_biases=[{"tensor": "redacted", "reason": "redacted", "equals_text_bias": x}
                                                                                       for x in w["unused_vision_bias_equals_text"]])
    prop = None
    if "proposal" in clean:
        boolean(clean["proposal"]["error_present"])
        prop = raw_proposal(clean["proposal"])
        require(proposal(prop) == clean["proposal"], "public proposal changed under validation")
        raw["proposal_parity"] = {"report": "redacted", "complete": clean["proposal"]["complete"], "passed": clean["proposal"]["passed"]}
    if "proposal_report_missing" in clean:
        require(clean["proposal_report_missing"] is True and "proposal" not in clean, "invalid missing-proposal flag")
        raw["proposal_parity"] = {"report": "redacted", "complete": False, "passed": False}
    rebuilt = case(raw, prop)
    if "weights" in clean:
        rebuilt["weights"]["configuration_sha256"] = clean["weights"]["configuration_sha256"]
    require(rebuilt == clean, "public case changed under validation")


def verify(folder):
    folder = Path(folder).resolve()
    require({p.name for p in folder.iterdir()} == {"manifest.json", "reference.json"} and
            all(not p.is_symlink() and p.is_file() for p in folder.iterdir()), "unexpected public bundle contents")
    manifest = load(folder/"manifest.json")
    fields = {"schema", "reference_sha256", "exporter_sha256", "raw_plan_sha256"}
    keys(manifest, fields, fields)
    require(manifest["schema"] == "dfly2-reference-public-manifest-v1", "unknown public manifest")
    for key in fields-{"schema"}: digest(manifest[key])
    require(manifest["reference_sha256"] == sha(folder/"reference.json"), "public report checksum differs")
    report = load(folder/"reference.json")
    validate_public(report)
    return report["summary"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--plan", type=Path)
    selection.add_argument("--verify", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.plan is not None:
        p.error("--output is required with --plan") if a.output is None else None
        result = export(a.plan, a.output)
    else:
        require(a.output is None, "--output cannot accompany --verify")
        result = verify(a.verify)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

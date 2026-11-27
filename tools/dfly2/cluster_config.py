"""Validated cluster configuration and read-only RoCE discovery. No shell snippets."""
import hashlib
import ipaddress
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from urllib.parse import urlsplit


class ConfigError(ValueError):
    pass


def keys(value, allowed, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ConfigError(f"Expected keys {sorted(allowed)}; required {sorted(required)}")


def text(value, pattern, label):
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ConfigError(f"Invalid {label}")
    return value


def integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ConfigError(f"{label} must be an integer in {low}..{high}")
    return value


def absolute(value):
    if not isinstance(value, str) or any(ord(c) < 32 or c == "," for c in value):
        raise ConfigError("Paths must be absolute Linux paths without control characters or commas")
    p = PurePosixPath(value)
    if not p.is_absolute() or ".." in p.parts or value != str(p) or str(p) == "/":
        raise ConfigError("Paths must be normalized absolute Linux paths")
    return str(p)


def validate(value):
    keys(value, {"version", "image", "nodes", "paths", "execution", "stopped", "experimental"}, {"version", "nodes", "paths"})
    if value["version"] != 1 or type(value["version"]) is not int:
        raise ConfigError("Unsupported cluster configuration version")
    out = {"version": 1, "image": text(value.get("image", "dfly2-two-sparks:experimental"), r"[A-Za-z0-9][A-Za-z0-9._/:@-]{0,254}", "Docker image")}
    if not isinstance(value["nodes"], list) or len(value["nodes"]) != 2:
        raise ConfigError("Exactly two nodes are required, head first")
    out["nodes"] = []
    for rank, raw in enumerate(value["nodes"]):
        keys(raw, {"hostname", "address", "ssh_target", "ssh_port", "repo_dir", "config_path", "fabric"}, {"hostname", "address", "repo_dir", "config_path", "fabric"})
        node = {"hostname": text(raw["hostname"], r"[A-Za-z0-9][A-Za-z0-9-]{0,62}", "short hostname"),
                "repo_dir": absolute(raw["repo_dir"]), "config_path": absolute(raw["config_path"])}
        if not isinstance(raw["address"], str):
            raise ConfigError("Each fabric address must be an IPv4 string")
        try:
            address = ipaddress.IPv4Address(raw["address"])
        except (ValueError, TypeError) as exc:
            raise ConfigError("Each fabric address must be IPv4") from exc
        if address.is_loopback or address.is_multicast or address.is_unspecified:
            raise ConfigError("Fabric addresses must be unicast interface addresses")
        node["address"] = str(address)
        if rank == 1 and "ssh_target" not in raw:
            raise ConfigError("Worker ssh_target is required")
        if "ssh_target" in raw:
            node["ssh_target"] = text(raw["ssh_target"], r"[A-Za-z0-9_][A-Za-z0-9_.-]*@[A-Za-z0-9][A-Za-z0-9_.-]*", "user@SSH-host")
        node["ssh_port"] = integer(raw.get("ssh_port", 22), 1, 65535, "SSH port")
        f = raw["fabric"]
        keys(f, {"interface", "hcas", "gid_index"}, {"interface", "hcas", "gid_index"})
        if not isinstance(f["hcas"], list) or not 1 <= len(f["hcas"]) <= 8:
            raise ConfigError("Expected 1..8 explicit HCA device names")
        hcas = [text(x, r"[A-Za-z0-9_][A-Za-z0-9_.-]*", "HCA device") for x in f["hcas"]]
        if len(set(hcas)) != len(hcas):
            raise ConfigError("Duplicate HCA device")
        node["fabric"] = {"interface": text(f["interface"], r"[A-Za-z0-9_][A-Za-z0-9_.-]*", "network interface"), "hcas": hcas,
                          "gid_index": integer(f["gid_index"], 0, 255, "GID index")}
        out["nodes"].append(node)
    if any(out["nodes"][0][k] == out["nodes"][1][k] for k in ("hostname", "address")):
        raise ConfigError("Nodes must have distinct hostnames and fabric addresses")
    names = {"model_root", "experiment_root", "target_model", "engram", "rank_cache", "kit", "token_map", "binary", "weights_probe", "requests", "draft_pack", "results"}
    keys(value["paths"], names, names)
    paths = {k: absolute(v) for k, v in value["paths"].items()}
    root, experiment = PurePosixPath(paths["model_root"]), PurePosixPath(paths["experiment_root"])
    if experiment == root or not experiment.is_relative_to(root):
        raise ConfigError("experiment_root must be a strict subdirectory of model_root")
    for key, path in paths.items():
        if key != "model_root" and not PurePosixPath(path).is_relative_to(root):
            raise ConfigError(f"{key} must be inside model_root")
    for key in ("binary", "weights_probe", "requests", "results"):
        if not PurePosixPath(paths[key]).is_relative_to(experiment) or paths[key] == str(experiment):
            raise ConfigError(f"{key} must be inside experiment_root")
    out["paths"] = paths
    if "experimental" in value:
        exp = value["experimental"]
        keys(exp, {"device_feedback", "rdma_best", "detailed_profile", "adaptive", "copies", "capture", "dfly_depth", "reference_norms"})
        flags = {}
        for key in ("device_feedback", "rdma_best", "detailed_profile", "adaptive", "reference_norms"):
            flag = exp.get(key, False)
            if type(flag) is not bool:
                raise ConfigError(f"experimental.{key} must be boolean")
            flags[key] = flag
        if "copies" in exp:
            if type(exp["copies"]) is not bool:
                raise ConfigError("experimental.copies must be boolean")
            flags["copies"] = exp["copies"]
        if "dfly_depth" in exp:
            flags["dfly_depth"] = integer(exp["dfly_depth"], 0, 5, "DFly fixed depth")
            if flags["adaptive"]:
                raise ConfigError("Fixed DFly depth and adaptive policy are mutually exclusive")
        if "capture" in exp:
            if flags["adaptive"]:
                raise ConfigError("Reference capture requires fixed five-draft execution")
            if flags.get("dfly_depth", 5) != 5:
                raise ConfigError("Reference capture requires all five drafts")
            cap = exp["capture"]
            keys(cap, {"directory", "limit", "every"}, {"directory"})
            capture_path = PurePosixPath(absolute(cap["directory"]))
            if capture_path == experiment or not capture_path.is_relative_to(experiment):
                raise ConfigError("Capture directory must be strictly inside experiment_root")
            flags["capture"] = {"directory": str(capture_path),
                                "limit": integer(cap.get("limit", 4), 1, 64, "capture limit"),
                                "every": integer(cap.get("every", 1), 1, 4096, "capture interval")}
        out["experimental"] = flags
    execution = value.get("execution", {})
    keys(execution, {"port", "pool", "min_available_gib", "timeout_seconds"})
    out["execution"] = {"port": integer(execution.get("port", 4460), 1024, 65535, "rendezvous port"),
                        "pool": integer(execution.get("pool", 8192), 8192, 8192, "validated pool"),
                        "min_available_gib": integer(execution.get("min_available_gib", 112), 112, 1024, "available GiB"),
                        "timeout_seconds": integer(execution.get("timeout_seconds", 1800), 30, 7200, "timeout")}
    stopped = value.get("stopped", {})
    keys(stopped, {"dashboard_url", "systemd_user_units", "systemd_units", "container_labels", "container_names"})
    stop = {"dashboard_url": stopped.get("dashboard_url")}
    if stop["dashboard_url"] is not None:
        if not isinstance(stop["dashboard_url"], str):
            raise ConfigError("Dashboard URL must be a string or null")
        u = urlsplit(stop["dashboard_url"])
        if (u.scheme not in {"http", "https"} or u.hostname not in {"localhost", "127.0.0.1", "::1", out["nodes"][0]["address"], out["nodes"][0]["hostname"]}
                or u.username or u.password or u.query or u.fragment):
            raise ConfigError("Dashboard URL must address the head without credentials/query/fragment")
    for key in ("systemd_user_units", "systemd_units", "container_labels", "container_names"):
        values = stopped.get(key, [])
        if not isinstance(values, list) or len(values) > 32:
            raise ConfigError(f"{key} must be a list with at most 32 entries")
        pattern = r"[A-Za-z0-9_][A-Za-z0-9_.@:-]*\.service" if key.endswith("units") else (r"[A-Za-z0-9_.-]+=[A-Za-z0-9_.-]+" if key == "container_labels" else r"[A-Za-z0-9][A-Za-z0-9_.-]*")
        stop[key] = [text(x, pattern, key) for x in values]
    out["stopped"] = stop
    return out


def load(path):
    path = Path(path)
    if path.stat().st_size > 65536:
        raise ConfigError("Cluster configuration exceeds 64 KiB")
    return validate(json.loads(path.read_text(encoding="utf-8")))


def identity(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fabric(config, rank):
    """Verify route, active HCA ports and mapped RoCEv2 GIDs against live sysfs."""
    node, peer = config["nodes"][rank], config["nodes"][1-rank]
    f = node["fabric"]
    def ip_json(*args):
        return json.loads(subprocess.check_output(["ip", *args], text=True, timeout=10))
    route = ip_json("-j", "route", "get", peer["address"])[0]
    if route.get("prefsrc") != node["address"] or route.get("dev") != f["interface"]:
        raise ConfigError("Peer route does not match configured source address/interface")
    primary_seen = False
    for name in f["hcas"]:
        hca = Path("/sys/class/infiniband") / name
        port = hca / "ports/1"
        if not (port / "state").read_text().startswith("4:"):
            raise ConfigError(f"HCA {name} port1 is not active")
        gid = str(f["gid_index"])
        mapped = ipaddress.IPv6Address((port / "gids" / gid).read_text().strip()).ipv4_mapped
        if (port / "gid_attrs/types" / gid).read_text().strip() != "RoCE v2" or mapped is None:
            raise ConfigError(f"HCA {name} GID is not mapped RoCE v2")
        matches = []
        for netdev in (hca / "device/net").iterdir():
            addrs = ip_json("-4", "-j", "addr", "show", "dev", netdev.name)
            if any(str(mapped) == a["local"] for dev in addrs for a in dev.get("addr_info", [])):
                matches.append(netdev.name)
        if not matches:
            raise ConfigError(f"HCA {name} GID has no matching local interface address")
        primary_seen |= str(mapped) == node["address"] and f["interface"] in matches
    if not primary_seen:
        raise ConfigError("Configured HCAs do not include the primary fabric address")
    return {"address": node["address"], "interface": f["interface"], "hcas": ",".join(f["hcas"]), "gid": str(f["gid_index"])}


def gpu_environment(f):
    return {"NCCL_SOCKET_IFNAME": "=" + f["interface"], "NCCL_IB_HCA": "=" + f["hcas"],
            "NCCL_IB_GID_INDEX": f["gid"], "NCCL_NET": "IB", "NCCL_IB_DISABLE": "0",
            "NCCL_IB_MERGE_NICS": "1", "NCCL_NET_PLUGIN": "none", "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,NET,GRAPH", "TP_SOCKET_IFNAME": f["interface"],
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

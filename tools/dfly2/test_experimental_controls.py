"""Control/provenance tests for opt-in GPU experiments; no remote or GPU actions."""
import copy
import json
from pathlib import Path
import unittest

import cluster_config as cluster
import run_pair as pair

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/cluster.example.json"


class ExperimentalControls(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(EXAMPLE.read_text())
        self.raw["experimental"] = {"device_feedback": True, "rdma_best": True,
                                    "detailed_profile": True, "adaptive": True, "copies": False, "reference_norms": True}
        pair.configure(cluster.validate(self.raw))
        paths = pair.CLUSTER["paths"]
        self.job = {"rank": 0, "mode": "dfly2", "kind": "lanes", "run_id": "a" * 32,
                    "binary": paths["binary"], "requests": paths["requests"], "pack": paths["draft_pack"]}
        self.pre = {"fabric": {"gid": "3", "hcas": "mlx5_0", "interface": "enp1s0f0"},
                    "binary_sha256": "b" * 64, "requests_sha256": "c" * 64}

    def tearDown(self):
        pair.configure(cluster.load(EXAMPLE))

    def test_effective_controls_and_recorded_settings_agree(self):
        for mode in pair.MODES:
            args = pair.create_argv({**self.job, "mode": mode}, self.pre, Path("/tmp/cid"))
            settings = pair.execution_settings(mode)["experimental"]
            for env, key in (("TF_DFLY_DEVICE_FEEDBACK", "device_feedback"), ("TF_DFLY_RDMA_BEST", "rdma_best"), ("TF_DFLY_PROFILE", "detailed_profile"), ("TF_DFLY_REFERENCE_NORMS", "reference_norms")):
                self.assertEqual(env + "=1" in args, mode == "dfly2")
                self.assertEqual(settings[key], mode == "dfly2")
            self.assertEqual("--dfly-adaptive" in args, mode == "dfly2")
            self.assertEqual(args[args.index("--copies") + 1], "0")
            self.assertFalse(settings["copies"])

    def test_capture_is_bounded_and_head_only_with_provenance(self):
        self.raw["experimental"]["adaptive"] = False
        path = self.raw["paths"]["experiment_root"] + "/captures/new"
        self.raw["experimental"]["capture"] = {"directory": path, "limit": 3, "every": 7}
        pair.configure(cluster.validate(self.raw))
        for rank in (0, 1):
            args = pair.create_argv({**self.job, "rank": rank}, self.pre, Path("/tmp/cid"))
            self.assertEqual("TF_DFLY_CAPTURE_DIR=" + path in args, rank == 0)
            self.assertEqual("TF_DFLY_CAPTURE_BINARY_SHA256=" + "b" * 64 in args, rank == 0)
            self.assertEqual(f"type=bind,src={path},dst={path}" in args, rank == 0)
        recorded = pair.execution_settings("dfly2")["experimental"]
        self.assertEqual(recorded["capture"], {"enabled": True, "limit": 3, "every": 7})
        self.assertNotIn(path, json.dumps(recorded))

    def test_invalid_controls_rejected_and_identity_changes(self):
        first = cluster.identity(cluster.validate(self.raw))
        for key, val in (("device_feedback", 1), ("copies", "0"), ("reference_norms", "1"), ("unknown", True)):
            raw = copy.deepcopy(self.raw)
            raw["experimental"][key] = val
            with self.assertRaises(cluster.ConfigError):
                cluster.validate(raw)
        self.raw["experimental"]["device_feedback"] = False
        self.assertNotEqual(first, cluster.identity(cluster.validate(self.raw)))
        self.assertEqual(cluster.validate(self.raw), cluster.validate(cluster.validate(self.raw)))

    def test_capture_cannot_write_outside_experiment_or_mix_adaptive(self):
        for path in ("/tmp/capture", self.raw["paths"]["experiment_root"]):
            self.raw["experimental"].update(adaptive=False, capture={"directory": path})
            with self.assertRaises(cluster.ConfigError):
                cluster.validate(self.raw)
        self.raw["experimental"].update(adaptive=True, capture={"directory": self.raw["paths"]["experiment_root"] + "/captures"})
        with self.assertRaises(cluster.ConfigError):
            cluster.validate(self.raw)

    def test_fixed_depth_is_recorded_and_cannot_mix_with_adaptive(self):
        self.raw["experimental"]["dfly_depth"] = 2
        with self.assertRaises(cluster.ConfigError):
            cluster.validate(self.raw)
        self.raw["experimental"]["adaptive"] = False
        pair.configure(cluster.validate(self.raw))
        self.assertEqual(pair.execution_settings("dfly2")["fixed_draft_depth"], 2)
        args = pair.create_argv(self.job, self.pre, Path("/tmp/cid"))
        self.assertEqual(args[args.index("--fix-k") + 1], "2")
        self.raw["experimental"]["copies"] = True
        pair.configure(cluster.validate(self.raw))
        with self.assertRaises(pair.Refusal):
            pair.execution_settings("dfly2")
        self.assertTrue(pair.execution_settings("dspark")["experimental"]["copies"])


if __name__ == "__main__":
    unittest.main()

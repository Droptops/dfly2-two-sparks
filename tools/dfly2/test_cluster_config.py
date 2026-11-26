import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import cluster_config as cluster

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/cluster.example.json"


class ClusterTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(EXAMPLE.read_text())

    def test_example_normalizes_idempotently(self):
        config = cluster.validate(self.raw)
        self.assertEqual(config, cluster.validate(config))
        self.assertEqual(config["image"], "dfly2-two-sparks:experimental")
        self.assertEqual(cluster.identity(config), cluster.identity(cluster.validate(config)))

    def test_unknown_docker_options_and_wrong_types_rejected(self):
        for key, val in (("docker_args", ["--privileged"]), ("env", {"LD_PRELOAD": "/tmp/x"}), ("version", True), ("nodes", [])):
            bad = copy.deepcopy(self.raw)
            bad[key] = val
            with self.subTest(key=key), self.assertRaises(cluster.ConfigError): cluster.validate(bad)
        for val in (True, 0, "8192", 16384):
            bad = copy.deepcopy(self.raw)
            bad["execution"]["pool"] = val
            with self.subTest(pool=val), self.assertRaises(cluster.ConfigError): cluster.validate(bad)

    def test_injected_or_duplicate_node_identity_rejected(self):
        for key, value in (("ssh_target", "user@host -oProxyCommand=bad"), ("ssh_target", "-oBad@host"), ("hostname", "host;id"), ("address", "127.0.0.1"), ("address", 123), ("ssh_port", "22")):
            bad = copy.deepcopy(self.raw)
            bad["nodes"][1][key] = value
            with self.subTest(key=key), self.assertRaises(cluster.ConfigError): cluster.validate(bad)
        for key in ("hostname", "address"):
            bad = copy.deepcopy(self.raw)
            bad["nodes"][1][key] = bad["nodes"][0][key]
            with self.assertRaises(cluster.ConfigError): cluster.validate(bad)
        for hca in ("..", "-bad", "../mlx5_0"):
            bad = copy.deepcopy(self.raw)
            bad["nodes"][0]["fabric"]["hcas"] = [hca]
            with self.assertRaises(cluster.ConfigError): cluster.validate(bad)

    def test_paths_and_output_scope(self):
        for value in ("relative", "/", "/srv/models/../etc/passwd", "/srv/models/a,b", "/srv/models/a\tb"):
            with self.subTest(path=value), self.assertRaises(cluster.ConfigError): cluster.absolute(value)
        for key, value in (("experiment_root", "/srv/models"), ("results", "/srv/models/results"), ("binary", "/usr/bin/docker")):
            bad = copy.deepcopy(self.raw)
            bad["paths"][key] = value
            with self.assertRaises(cluster.ConfigError): cluster.validate(bad)

    def test_dashboard_and_service_filters_cannot_add_options(self):
        for value in ([], "http://other-host/status", "http://user:secret@localhost/status", "http://localhost/status?key=x"):
            bad = copy.deepcopy(self.raw)
            bad["stopped"]["dashboard_url"] = value
            with self.assertRaises(cluster.ConfigError): cluster.validate(bad)
        for key, value in (("systemd_units", ["model.service;id"]), ("systemd_units", ["--bad.service"]), ("container_labels", ["x=1 --all"]), ("container_names", [".*"])):
            bad = copy.deepcopy(self.raw)
            bad["stopped"][key] = value
            with self.assertRaises(cluster.ConfigError): cluster.validate(bad)

    def test_live_route_must_match_configured_identity_before_sysfs(self):
        config = cluster.validate(self.raw)
        for route in ({"dev": "wrong", "prefsrc": config["nodes"][0]["address"]}, {"dev": config["nodes"][0]["fabric"]["interface"], "prefsrc": "192.0.2.99"}):
            with patch.object(cluster.subprocess, "check_output", return_value=json.dumps([route])) as call:
                with self.assertRaisesRegex(cluster.ConfigError, "route"):
                    cluster.fabric(config, 0)
                self.assertEqual(call.call_args[0][0], ["ip", "-j", "route", "get", config["nodes"][1]["address"]])


if __name__ == "__main__":
    unittest.main()

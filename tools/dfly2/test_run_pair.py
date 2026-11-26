import json
import copy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import shlex

import cluster_config as cluster
import run_pair as pair
import run_weights_probe as probe

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/cluster.example.json"


def image_fixture():
    return {"Id": "sha256:" + "a" * 64, "Architecture": "arm64", "Os": "linux",
            "Config": {"Env": ["PATH=/usr/bin"], "Entrypoint": ["/bin/sh"], "WorkingDir": "/workspace"},
            "RootFS": {"Type": "layers", "Layers": ["sha256:" + "b" * 64, "sha256:" + "c" * 64]}}


class PairTests(unittest.TestCase):
    def setUp(self):
        pair.configure(cluster.load(EXAMPLE))
        paths = pair.CLUSTER["paths"]
        self.config = {"rank": 0, "mode": "dfly2", "kind": "lanes", "run_id": "a" * 32,
                       "binary": paths["binary"], "requests": paths["requests"], "pack": paths["draft_pack"],
                       "config_identity": cluster.identity(pair.CLUSTER)}
        self.pre = {"fabric": {"gid": "3", "hcas": "mlx5_0", "interface": "enp1s0f0"}}

    def test_modes_are_isolated_and_baseline_does_not_set_empty_dfly_directory(self):
        for mode in pair.MODES:
            cfg = {**self.config, "mode": mode}
            args = pair.create_argv(cfg, self.pre, Path("/tmp/cid"))
            self.assertIn("ai.spark-dfly2.experimental=1", args)
            self.assertNotIn("--rm", args)
            self.assertEqual(args[args.index("--entrypoint") + 1], cfg["binary"])
            self.assertEqual(args[args.index("--drafts") + 1], "0" if mode == "serial" else "1")
            self.assertEqual(args[args.index("--serial") + 1], "1" if mode == "serial" else "0")
            self.assertEqual(args[args.index("--graphs") + 1], "0")
            self.assertIn("TF_DS_REPLAY_FLOOR=1", args)
            self.assertEqual(any(x.startswith("TF_DS_DFLY2_DIR=") for x in args), mode == "dfly2")

    def test_matched_depth_mode_fixes_five_drafts_and_settings_are_recorded(self):
        for mode in pair.MODES:
            args = pair.create_argv({**self.config, "mode": mode}, self.pre, Path("/tmp/cid"))
            settings = pair.execution_settings(mode)
            self.assertEqual("--fix-k" in args, mode == "dspark_fixed5")
            self.assertEqual(settings["fixed_draft_depth"], 5 if mode == "dspark_fixed5" else None)
            self.assertEqual(settings["drafting"], mode != "serial")
            # The record and the container arguments come from one source.
            self.assertEqual(args[args.index("--pool") + 1], str(settings["pool_tokens"]))
            self.assertEqual(args[args.index("--streams") + 1], str(settings["streams"]))
            self.assertEqual(args[args.index("--parallel") + 1], str(settings["parallel"]))
            self.assertEqual(args[args.index("--profile") + 1], "1")
        fixed = pair.create_argv({**self.config, "mode": "dspark_fixed5"}, self.pre, Path("/tmp/cid"))
        self.assertEqual(fixed[fixed.index("--fix-k") + 1], "5")
        self.assertFalse(any(x.startswith("TF_DS_DFLY2_DIR=") for x in fixed))
        with self.assertRaises(pair.Refusal):
            pair.execution_settings("dspark_fixed9")

    def test_cleanup_refuses_foreign_label_and_uses_recorded_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rank0.cid"
            cid = "f" * 64
            path.write_text(cid)
            info = {"Id": cid, "Name": "/dfly2-validation-r0", "State": {"Status": "created"},
                    "Labels": {pair.LABEL: "1", pair.RUN_LABEL: "b" * 32, pair.RANK_LABEL: "0"}}
            def command(args, **kwargs):
                if args[1] == "inspect":
                    return subprocess.CompletedProcess(args, 0, json.dumps(info), "")
                return subprocess.CompletedProcess(args, 0, "", "")
            with patch.object(pair, "hostname", return_value=pair.HEAD), patch.object(pair, "journal", return_value=path), patch.object(pair, "command", side_effect=command) as call:
                with self.assertRaisesRegex(pair.Refusal, "labels disagree"):
                    pair.node_action("cleanup", self.config)
                self.assertEqual(len(call.call_args_list), 1)
                info["Labels"][pair.RUN_LABEL] = self.config["run_id"]
                self.assertEqual(pair.node_action("cleanup", self.config)["id"], cid)
                self.assertEqual(call.call_args[0][0], ["docker", "rm", "-f", cid])

    def test_missing_creation_journal_never_adopts_existing_container(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pair, "hostname", return_value=pair.HEAD), patch.object(pair, "command") as command:
            with patch.object(pair, "journal", return_value=Path(tmp) / "absent.cid"):
                self.assertEqual(pair.node_action("cleanup", self.config), {"removed": False})
            command.assert_not_called()

    def test_preexisting_name_collision_is_never_adopted(self):
        cid = "f" * 64
        with patch.object(pair, "command", return_value=subprocess.CompletedProcess([], 0, cid, "")) as command:
            with self.assertRaisesRegex(pair.Refusal, "already in use"):
                pair.check_container_names(self.config)
            self.assertEqual(command.call_args[0][0][:4], ["docker", "ps", "-aq", "--no-trunc"])
        with patch.object(pair, "command", side_effect=[subprocess.CompletedProcess([], 0, cid, ""), subprocess.CompletedProcess([], 0, "", "")]):
            pair.check_container_names(self.config, allow_id=cid)

    def test_pack_manifest_checks_format_rank_files_and_sizes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pair, "ROOT", Path(tmp)):
            pack = Path(tmp) / "pack"
            pack.mkdir()
            (pack / "config.json").write_text("{}")
            (pack / "rank0.safetensors").write_bytes(b"fixture")
            manifest = {"format": "tensorfold-dfly2-tp2-v1", "world_size": 2,
                        "ranks": [{"file": "rank0.safetensors", "bytes": 7, "sha256": "a" * 64},
                                  {"file": "rank1.safetensors", "bytes": 7, "sha256": "b" * 64}]}
            path = pack / "dfly2-pack.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(pair.check_pack(pack, 0), pair.digest(path))
            manifest["ranks"][0]["bytes"] = 6
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(pair.Refusal, "size"):
                pair.check_pack(pack, 0)
            manifest["ranks"][0]["file"] = "../outside"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(pair.Refusal, "two rank files"):
                pair.check_pack(pack, 0)

    def test_stop_requested_idle_is_allowed_but_starting_and_inflight_refused(self):
        from io import BytesIO
        pair.CLUSTER["stopped"]["dashboard_url"] = "http://127.0.0.1:4000/status"
        state = {"phase": "idle", "active_model": None, "desired_model": None,
                 "loaded": {pair.HEAD: [], pair.WORKER: []}, "inflight": 0, "stop_requested": True}
        with patch.object(pair.urllib.request, "urlopen", side_effect=lambda *a, **kw: BytesIO(json.dumps(state).encode())):
            self.assertEqual(pair.stopped_dashboard()["phase"], "idle")
            state["phase"] = "starting"
            with self.assertRaises(pair.Refusal): pair.stopped_dashboard()
            state["phase"], state["inflight"] = "idle", 1
            with self.assertRaises(pair.Refusal): pair.stopped_dashboard()

    def test_truncated_failed_or_missing_native_replies_are_not_success(self):
        reply = {"rank": 0, "request": 0, "name": "short", "tokens": 2, "token_sha": "1" * 12, "equal": True, "reply": [1, 2], "reason": "length"}
        summary = {"rank": 0, "requests": 1, "all_equal": True}
        text = json.dumps(reply) + "\n" + json.dumps(summary)
        self.assertTrue(pair.parse_native(text, 1)["complete"])
        self.assertFalse(pair.parse_native(json.dumps(summary), 1)["complete"])
        self.assertFalse(pair.parse_native(text, 2)["complete"])
        self.assertFalse(pair.parse_native(text + '\n{"error":"BadCapture"}', 1)["complete"])
        self.assertFalse(pair.parse_native(text + "\n" + json.dumps(reply), 1)["complete"])

    def test_reference_compares_full_tokens_and_reason_even_with_identical_hash(self):
        ref = {"mode": "serial", "success": True, "run_id": "reference",
               "preflight": {"nodes": [{"requests_sha256": "same"}]},
               "native": {"replies": [{"name": "short", "tokens": 2, "token_sha": "a", "reply": [1, 2], "reason": "length"}]}}
        result = json.loads(json.dumps(ref))
        result["mode"] = "dfly2"
        self.assertTrue(pair.compare_reference(result, ref)["equal"])
        result["native"]["replies"][0]["reply"] = [1, 3]
        self.assertFalse(pair.compare_reference(result, ref)["equal"])
        result["native"]["replies"][0]["reply"] = [1, 2]
        result["native"]["replies"][0]["reason"] = "eos"
        self.assertFalse(pair.compare_reference(result, ref)["equal"])
        result["preflight"]["nodes"][0]["requests_sha256"] = "different"
        with self.assertRaises(pair.Refusal): pair.compare_reference(result, ref)

    def test_default_path_is_read_only_preflight(self):
        with patch.object(pair, "preflight_pair", return_value={}) as preflight, patch.object(pair, "run_pair") as run, patch("builtins.print"):
            self.assertEqual(pair.main(["--config", str(EXAMPLE)]), 0)
        preflight.assert_called_once()
        run.assert_not_called()

    def test_mismatched_host_and_config_refused_before_docker(self):
        with patch.object(pair, "hostname", return_value="wrong-host"), patch.object(pair, "command") as command:
            with self.assertRaisesRegex(pair.Refusal, "identity"):
                pair.node_action("create", self.config)
            command.assert_not_called()
        with patch.object(pair, "hostname", return_value=pair.HEAD), patch.object(pair, "command") as command:
            with self.assertRaisesRegex(pair.Refusal, "configurations disagree"):
                pair.node_action("cleanup", {**self.config, "config_identity": "bad"})
            command.assert_not_called()

    def test_ssh_quotes_paths_and_uses_strict_host_keys(self):
        node = pair.CLUSTER["nodes"][1]
        node["repo_dir"] = "/opt/repo with spaces;echo unsafe"
        node["config_path"] = "/etc/config's file.json"
        with patch.object(pair, "command", return_value=subprocess.CompletedProcess([], 0, '{"ok":true,"result":{}}', "")) as command:
            pair.rpc(1, "preflight", self.config)
        argv = command.call_args[0][0]
        self.assertIn("StrictHostKeyChecking=yes", argv)
        self.assertEqual(shlex.split(argv[-1]), ["python3", node["repo_dir"] + "/tools/dfly2/run_pair.py", "node", "--config", node["config_path"]])
        self.assertEqual(json.loads(command.call_args.kwargs["input_text"])["config"]["rank"], 1)

    def test_inherited_dfly_environment_refused_for_both_baselines(self):
        info = image_fixture()
        info["Config"]["Env"].append("TF_DS_DFLY2_DIR=")
        with patch.object(pair, "command", return_value=subprocess.CompletedProcess([], 0, json.dumps(info), "")):
            for mode in ("serial", "dspark"):
                with self.assertRaisesRegex(pair.Refusal, "inherits"):
                    pair.check_image(mode)
            self.assertEqual(pair.check_image("dfly2"), (info["Id"], pair.image_content_fingerprint(info)))

    def test_pair_refuses_different_image_or_script_bytes(self):
        node = dict.fromkeys(("binary_sha256", "requests_sha256", "pack_manifest_sha256", "launcher_sha256", "config_identity", "image_content_sha256", "target_config_sha256"), "same")
        for key in ("image_content_sha256", "launcher_sha256", "config_identity"):
            other = {**node, key: "different"}
            with patch.object(pair, "hostname", return_value=pair.HEAD), patch.object(pair, "rpc", side_effect=[node, other]):
                with self.assertRaisesRegex(pair.Refusal, key):
                    pair.preflight_pair(self.config)

    def test_identical_runtime_with_different_store_ids_is_accepted(self):
        original = image_fixture()
        transferred = copy.deepcopy(original)
        transferred.update(Id="sha256:" + "d" * 64, RepoTags=["copied:tag"], Variant="")
        transferred["Config"] = dict(reversed(list(transferred["Config"].items())))
        fingerprint = pair.image_content_fingerprint(original)
        self.assertEqual(pair.image_content_fingerprint(transferred), fingerprint)
        node = dict.fromkeys(("binary_sha256", "requests_sha256", "pack_manifest_sha256", "launcher_sha256", "config_identity", "target_config_sha256"), "same")
        nodes = [{**node, "image_id": info["Id"], "image_content_sha256": pair.image_content_fingerprint(info)} for info in (original, transferred)]
        with patch.object(pair, "hostname", return_value=pair.HEAD), patch.object(pair, "rpc", side_effect=nodes):
            self.assertEqual(pair.preflight_pair(self.config)["nodes"], nodes)

    def test_changed_environment_layers_or_variant_fails_pair_comparison(self):
        original = image_fixture()
        changed_env, changed_layers, reordered_layers, changed_variant, changed_command = [copy.deepcopy(original) for _ in range(5)]
        changed_env["Config"]["Env"].append("CUDA_MODULE_LOADING=EAGER")
        changed_layers["RootFS"]["Layers"][0] = "sha256:" + "f" * 64
        reordered_layers["RootFS"]["Layers"].reverse()
        changed_variant["Variant"] = "v8"
        changed_command["Config"]["Entrypoint"] = ["/another/entrypoint"]
        node = dict.fromkeys(("binary_sha256", "requests_sha256", "pack_manifest_sha256", "launcher_sha256", "config_identity", "target_config_sha256"), "same")
        node["image_content_sha256"] = pair.image_content_fingerprint(original)
        for changed in (changed_env, changed_layers, reordered_layers, changed_variant, changed_command):
            other = {**node, "image_content_sha256": pair.image_content_fingerprint(changed)}
            self.assertNotEqual(node["image_content_sha256"], other["image_content_sha256"])
            with patch.object(pair, "hostname", return_value=pair.HEAD), patch.object(pair, "rpc", side_effect=[node, other]):
                with self.assertRaisesRegex(pair.Refusal, "image_content_sha256"):
                    pair.preflight_pair(self.config)

    def test_image_content_refuses_missing_or_invalid_platform_config_layers(self):
        for key, value in (("Architecture", "amd64"), ("Os", "windows"), ("Variant", None), ("Config", None), ("Config", []), ("RootFS", None),
                           ("RootFS", {"Type": "layers", "Layers": []}), ("RootFS", {"Type": "layers", "Layers": ["not-a-digest"]}),
                           ("Config", {"Env": [123]}), ("Config", {"Env": {}}), ("Config", {"Env": ""}),
                           ("Config", {"bad": float("nan")})):
            info = image_fixture()
            info[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(pair.Refusal):
                pair.image_content_fingerprint(info)
        for key in ("Architecture", "Os", "Config", "RootFS"):
            info = image_fixture()
            del info[key]
            with self.subTest(missing=key), self.assertRaises(pair.Refusal):
                pair.image_content_fingerprint(info)

    def test_output_and_request_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            root.mkdir()
            with patch.object(pair, "EXPERIMENT", root):
                with self.assertRaises(pair.Refusal): pair.new_output(Path(tmp) / "outside")
                with self.assertRaises(pair.Refusal): pair.new_output(root)
                out = pair.new_output(root / "results/run1")
                with self.assertRaises(FileExistsError): pair.new_output(out)
                request = root / "requests.json"
                valid = {"name": "short", "prompt": [1, 2], "max_tokens": 128}
                request.write_text(json.dumps({"requests": [valid]}))
                self.assertEqual(len(pair.read_requests(request)[1]), 1)
                for changed in ({"prompt": [True]}, {"name": 'bad"name'}, {"max_tokens": 8192}, {"images": ["image"]}):
                    request.write_text(json.dumps({"requests": [{**valid, **changed}]}))
                    with self.assertRaises(pair.Refusal): pair.read_requests(request)

    def test_probe_defaults_to_preflight_and_requires_release_receipt(self):
        with patch.object(pair, "node_action", return_value={}) as action, patch.object(probe, "run") as run, patch("builtins.print"):
            self.assertEqual(probe.main(["--config", str(EXAMPLE), "--rank", "1"]), 0)
        self.assertEqual(action.call_args[0][0], "preflight")
        run.assert_not_called()
        logs = {"stdout": "", "stderr": json.dumps({"rank": 1, "ok": True, "all_uploaded_bytes_equal": True, "allocation_released": True})}
        self.assertTrue(probe.verified(logs, 1)[0])
        self.assertFalse(probe.verified(logs, 0)[0])
        logs["stderr"] = logs["stderr"].replace('"allocation_released": true', '"allocation_released": false')
        self.assertFalse(probe.verified(logs, 1)[0])

    def test_weight_probe_container_is_bounded_and_offline(self):
        cfg = {**self.config, "kind": "weights", "binary": pair.CLUSTER["paths"]["weights_probe"]}
        args = pair.create_argv(cfg, self.pre, Path("/tmp/cid"))
        self.assertIn("--network=none", args)
        self.assertEqual(args[args.index("--memory") + 1], "12g")
        self.assertEqual(args[args.index("--name") + 1], "dfly2-weights-r0")
        self.assertNotIn("--requests", args)


if __name__ == "__main__":
    unittest.main()

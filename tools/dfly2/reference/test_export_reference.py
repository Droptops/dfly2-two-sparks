"""Invented report mutation tests; never evidence of GPU/reference parity."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

if __package__:
    from . import export_reference as e
else:
    import export_reference as e


class ReferenceExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="dfly-reference-export-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.batch_path = self.root/"PRIVATE-HOST"/"report.json"
        self.batch_path.parent.mkdir()
        self.native = [10, 11, 12, 13, 14]
        self.cases = []
        requested = []
        attempts = []
        for label in ("PRIVATE-legacy", "PRIVATE-optimized"):
            name = label+"/capture-000000"
            requested.append(name)
            path = self.batch_path.parent/name/"report.json"
            path.parent.mkdir(parents=True)
            raw, prop = self.make_case()
            e.write(path, raw)
            (path.parent/"proposal-parity").mkdir()
            e.write(path.parent/"proposal-parity/report.json", prop)
            self.cases.append(path)
            attempts.append({"capture": name, "report": name+"/report.json", "complete": True, "passed": True,
                             "capture_tensor_sha256": raw["capture_tensor_sha256"]})
        self.batch = {"schema": "dfly2-reference-batch-v1", "complete": True, "passed": True,
                      "reference_commit": e.REFERENCE_COMMIT, "captures_requested": requested,
                      "weights_load_count": 1, "weights_load_seconds": 7.1, "wall_seconds": 10.0, "cases": attempts}
        self.plan = {"schema": "dfly2-reference-export-plan-v1", "groups": [
            {"fixture": "chat", "variant": "legacy", "report": str(self.batch_path), "expected_cases": 1, "case_prefix": "PRIVATE-legacy"},
            {"fixture": "chat", "variant": "optimized", "report": str(self.batch_path), "expected_cases": 1, "case_prefix": "PRIVATE-optimized"}]}
        self.plan_path = self.root/"plan.json"
        self.save()

    def save(self):
        e.write(self.batch_path, self.batch)
        e.write(self.plan_path, self.plan)

    def metric(self, name):
        shape = e.SHAPES[name]
        return dict(passed=True, finite=True, shape=shape, max_abs=0, rms_abs=0, reference_rms=1, relative_rms=0,
                    cosine_similarity=1, different_elements=0, outside_tolerance=0, elements=e.math.prod(shape))

    def head(self, ids, input_hash="b"*64):
        return {"results": [dict(token=i, score=10.0, runnerup_token=1000, runnerup_score=9.0, margin=1.0) for i in ids],
                "input_sha256": input_hash, "logits_sha256": "d"*64, "top1_sha256": "e"*64,
                "subprocess_log_sha256": "f"*64, "wall_seconds": .1}

    def cache(self, rank):
        lo, hi = rank*(e.VOCAB//2), (rank+1)*(e.VOCAB//2)
        entries = {}
        offset = 0
        for part, shape in (("tr", [1, 1, 1]), ("suh", [5120]), ("svh", [e.VOCAB//2])):
            length = e.math.prod(shape)*2
            entries[f"head|({lo}, {hi})|None|{part}"] = dict(dtype="int16" if part == "tr" else "float16",
                    shape=shape, offset=offset, bytes=length, sha256="a"*64)
            offset += length
        return dict(file="PRIVATE-CACHE.bin", file_bytes=offset+100, data_end=offset,
                    index_sha256="c"*64, footer_sha256="d"*64, head_entries=entries,
                    embedding_rows={str(i): "e"*64 for i in self.native+[7]})

    def make_case(self):
        capture_hashes = {k: "b"*64 for k in e.CAPTURE_KEYS}
        capture_hashes["proposals"] = e.proposal_hash(self.native)
        prop = {"schema": "dfly2-reference-proposals-v1", "complete": True, "passed": True, "head_gate_passed": True,
                "steps_completed": [{"slot": i, **self.head([token])} for i, token in enumerate(self.native, 1)],
                "embedding_cache": [self.cache(0), self.cache(1)], "capture_tensor_sha256": capture_hashes,
                "native_proposals": self.native, "probe_binary_sha256": "1"*64, "grouped_cubin_sha256": "2"*64,
                "target_config_sha256": "3"*64, "native_head_gate": self.head(self.native),
                "reference_proposals": self.native, "slot_matches": [True]*5, "matching_prefix_length": 5,
                "output_sha256": {"reference_free_corrected.bin": "4"*64, "reference_free_previous_embeddings.bin": "5"*64,
                                  "reference_free_proposals.bin": e.proposal_hash(self.native)}, "wall_seconds": .7}
        comps = {k: self.metric(k) for k in sorted(e.CORE_COMPARISONS)}
        raw = {"schema": "dfly2-reference-replay-v1", "complete": True, "passed": True, "reference_commit": e.REFERENCE_COMMIT,
               "capture": {"schema": "dfly2-reference-capture-v1", "q0": 42, "anchor_token": 7, "steps": 5, "world_size": 2,
                           "native": dict(binary_sha256="6"*64, checkpoint_manifest_sha256="7"*64, requests_sha256="8"*64,
                                          rms_norm_eps=1e-20, norm_rounding="fused")},
               "capture_tensor_sha256": capture_hashes, "torch": "2.11.0a0+nv26.7", "device": "cuda",
               "thresholds": e.THRESHOLDS, "weights": {"source_config_sha256": "9"*64,
                   "source_files": [{"file": "PRIVATE-SOURCE.bin", "bytes": 256, "sha256": "a"*64}],
                   "parameters_loaded": 123, "source_tensors_accounted_for": 125,
                   "unused_vision_biases": [{"tensor": "PRIVATE-TENSOR", "reason": "PRIVATE-REASON", "equals_text_bias": False}],
                   "expert_loading": "PRIVATE-DESCRIPTION", "config": {"_name_or_path": "/PRIVATE/PATH", "hidden_size": 5120}},
               "comparisons": comps, "numeric_passed": True,
               "proposal_parity": {"report": "proposal-parity/report.json", **{k:prop[k] for k in
                    ("complete", "passed", "head_gate_passed", "native_proposals", "reference_proposals", "slot_matches", "matching_prefix_length")}},
               "native_proposals": self.native, "output_sha256": {"reference_"+k+".bin": "a"*64 for k in comps}, "wall_seconds": 1.0}
        return raw, prop

    def export(self):
        folder = self.root/"public"
        summary = e.export(self.plan_path, folder)
        return folder, summary

    def mutate_case(self, fn, index=0):
        path = self.cases[index]
        raw = e.load(path)
        prop_path = path.parent/"proposal-parity/report.json"
        prop = e.load(prop_path)
        fn(raw, prop)
        e.write(path, raw); e.write(prop_path, prop)

    def mutate_public(self, fn):
        folder, _ = self.export()
        path = folder/"reference.json"
        report = e.load(path); fn(report); e.write(path, report)
        manifest = e.load(folder/"manifest.json")
        manifest["reference_sha256"] = e.sha(path); e.write(folder/"manifest.json", manifest)
        return folder

    def test_shared_batch_export_verifies_and_removes_private_data(self):
        folder, summary = self.export()
        self.assertEqual(summary["exact_proposal_passes"], 2)
        self.assertEqual(summary["compared_slots"], 10)
        self.assertEqual(e.verify(folder), summary)
        text = (folder/"reference.json").read_text()
        self.assertNotIn("PRIVATE", text)
        self.assertNotIn("_name_or_path", text)
        self.assertNotIn("config_name_or_path", text)
        self.assertNotIn("embedding_bytes", text)
        report = e.load(folder/"reference.json")
        self.assertFalse(report["groups"][0]["run_provenance_provided"])
        self.assertEqual(report["groups"][1]["cases"][0]["batch_index"], 2)

    def test_missing_prefix_cases_rejected(self):
        self.plan["groups"][0]["expected_cases"] = 2; self.save()
        with self.assertRaisesRegex(ValueError, "prefix case inventory"): self.export()

    def test_dropped_group_rejected(self):
        self.plan["groups"].pop(); self.save()
        with self.assertRaisesRegex(ValueError, "dropped requested"): self.export()

    def test_duplicate_group_rejected(self):
        self.plan["groups"].append(copy.deepcopy(self.plan["groups"][0])); self.save()
        with self.assertRaisesRegex(ValueError, "more than once"): self.export()

    def test_numerical_negative_is_preserved(self):
        def change(raw, prop):
            metric = raw["comparisons"]["normalized"]
            metric.update(passed=False, max_abs=.1, rms_abs=.001, relative_rms=.001, different_elements=100, outside_tolerance=1)
            raw.update(numeric_passed=False, passed=False)
        self.mutate_case(change)
        self.batch["passed"] = False; self.batch["cases"][0]["passed"] = False; self.save()
        folder, result = self.export()
        self.assertEqual(result["numerical_failures"], 1)
        self.assertEqual(result["exact_proposal_passes"], 2)
        self.assertFalse(result["all_declared_cases_passed"])
        self.assertTrue(result["all_declared_cases_complete"])

    def test_nonfinite_negative_preserved(self):
        def change(raw, prop):
            raw["comparisons"]["normalized"] = {"finite": False, "passed": False, "shape": [6,5120]}
            raw.update(numeric_passed=False, passed=False)
        self.mutate_case(change)
        self.batch["passed"] = False; self.batch["cases"][0]["passed"] = False; self.save()
        self.assertEqual(self.export()[1]["numerical_failures"], 1)

    def test_proposal_mismatch_keeps_exact_ids_prefix_and_margins(self):
        def change(raw, prop):
            ids = self.native.copy(); ids[1] = 99
            prop.update(reference_proposals=ids, slot_matches=[True,False,True,True,True], matching_prefix_length=1, passed=False)
            prop["steps_completed"][1]["results"][0]["token"] = 99
            prop["output_sha256"]["reference_free_proposals.bin"] = e.proposal_hash(ids)
            for key in ("reference_proposals", "slot_matches", "matching_prefix_length", "passed"):
                raw["proposal_parity"][key] = prop[key]
            raw["passed"] = False
        self.mutate_case(change)
        self.batch["passed"] = False; self.batch["cases"][0]["passed"] = False; self.save()
        folder, result = self.export()
        self.assertEqual(result["matched_slots"], 9)
        proposal = e.load(folder/"reference.json")["groups"][0]["cases"][0]["result"]["proposal"]
        self.assertEqual(proposal["matching_prefix_length"], 1)
        self.assertEqual(proposal["reference_proposals"][1], 99)

    def test_runtime_failure_retains_unattempted_group(self):
        def change(raw, prop):
            raw.update(complete=False, passed=False, error="RuntimeError: PRIVATE-PATH failure")
            raw.pop("proposal_parity")
        self.mutate_case(change)
        (self.cases[0].parent/"proposal-parity/report.json").unlink()
        self.batch.update(complete=False, passed=False, error="RuntimeError: PRIVATE-HOST")
        self.batch["cases"] = [dict(self.batch["cases"][0], complete=False, passed=False, error="PRIVATE")]
        self.save()
        folder, result = self.export()
        report = e.load(folder/"reference.json")
        self.assertEqual(result["reported_cases"], 1)
        self.assertEqual(report["groups"][1]["cases"][0]["status"], "not_attempted")
        self.assertNotIn("PRIVATE", (folder/"reference.json").read_text())

    def test_completed_group_distinct_from_failed_batch(self):
        def change(raw, prop):
            raw.update(complete=False, passed=False, error="RuntimeError: PRIVATE")
            raw.pop("proposal_parity")
        self.mutate_case(change, index=1)
        (self.cases[1].parent/"proposal-parity/report.json").unlink()
        self.batch.update(complete=False, passed=False, error="RuntimeError: PRIVATE")
        self.batch["cases"][1].update(complete=False, passed=False, error="PRIVATE"); self.save()
        folder, _ = self.export()
        group = e.load(folder/"reference.json")["groups"][0]
        self.assertTrue(group["complete"])
        self.assertTrue(group["passed"])
        self.assertFalse(group["batch_complete"])

    def test_numeric_pass_flag_mutation_rejected(self):
        self.mutate_case(lambda raw,p: raw["comparisons"]["normalized"].update(passed=False))
        with self.assertRaisesRegex(ValueError, "numeric pass flag"): self.export()

    def test_relaxed_tolerance_rejected(self):
        self.mutate_case(lambda raw,p: raw["thresholds"].update(rtol=.5))
        with self.assertRaisesRegex(ValueError, "tolerances differ"): self.export()

    def test_rms_inconsistency_rejected(self):
        self.mutate_case(lambda raw,p: raw["comparisons"]["normalized"].update(relative_rms=.001))
        with self.assertRaisesRegex(ValueError, "RMS metrics"): self.export()

    def test_head_margin_mutation_rejected(self):
        self.mutate_case(lambda raw,p: p["native_head_gate"]["results"][0].update(margin=9))
        with self.assertRaisesRegex(ValueError, "margin differs"): self.export()

    def test_native_id_hash_mutation_rejected(self):
        self.mutate_case(lambda raw,p: p["native_proposals"].__setitem__(0, 99))
        with self.assertRaisesRegex(ValueError, "IDs/file hash"): self.export()

    def test_prefix_mutation_rejected(self):
        self.mutate_case(lambda raw,p: p.update(matching_prefix_length=4))
        with self.assertRaisesRegex(ValueError, "prefix differs"): self.export()

    def test_report_path_escape_rejected(self):
        self.batch["cases"][0]["report"] = "../outside/report.json"; self.save()
        with self.assertRaisesRegex(ValueError, "escapes batch"): self.export()

    def test_checksum_mutation_rejected(self):
        folder, _ = self.export()
        with (folder/"reference.json").open("a") as out: out.write(" ")
        with self.assertRaisesRegex(ValueError, "checksum differs"): e.verify(folder)

    def test_public_injected_private_field_rejected_even_with_rehashed_manifest(self):
        folder = self.mutate_public(lambda r:r["groups"][0]["cases"][0]["result"]["runtime"].update(host="PRIVATE"))
        with self.assertRaisesRegex(ValueError, "fields"): e.verify(folder)

    def test_summary_mutation_rejected_even_with_rehashed_manifest(self):
        folder = self.mutate_public(lambda r:r["summary"].update(matched_slots=500))
        with self.assertRaisesRegex(ValueError, "summary differs"): e.verify(folder)

    def test_public_prefix_mutation_rejected_even_with_rehashed_manifest(self):
        folder = self.mutate_public(lambda r:r["groups"][0]["cases"][0]["result"]["proposal"].update(matching_prefix_length=1))
        with self.assertRaisesRegex(ValueError, "prefix differs"): e.verify(folder)

    def test_private_raw_tensor_artifact_rejected_in_public_bundle(self):
        folder, _ = self.export(); (folder/"corrected.bin").write_bytes(b"private")
        with self.assertRaisesRegex(ValueError, "bundle contents"): e.verify(folder)

    def test_existing_export_refused(self):
        self.export()
        with self.assertRaisesRegex(ValueError, "already exists"): self.export()

    def add_run(self, group_index):
        enabled = group_index == 1
        settings = {"world_size":2, "streams":1, "parallel":1, "pool_tokens":8192, "cuda_graphs":False,
                    "profiling":True, "drafting":True, "fixed_draft_depth":5,
                    "experimental":dict(device_feedback=enabled, rdma_best=enabled, detailed_profile=False, adaptive=False,
                                        copies=False, capture=dict(enabled=True, limit=1, every=50))}
        ids = dict(binary_sha256="6"*64, requests_sha256="8"*64, pack_manifest_sha256="7"*64,
                   launcher_sha256="a"*64, image_content_sha256="b"*64, target_config_sha256="3"*64)
        run = dict(success=True, mode="dfly2", execution=settings, native=dict(complete=True,records=[dict(rank=0,profile=dict(passes=4))]),
                   preflight=dict(nodes=[dict(rank=i,gpu="NVIDIA GB10",host="PRIVATE",**ids) for i in (0,1)]),
                   states={str(i):dict(Status="exited",ExitCode=0,OOMKilled=False,Error="") for i in (0,1)},
                   cleanup={str(i):dict(removed=True,id="PRIVATE") for i in (0,1)})
        path = self.root/f"run{group_index}.json"; e.write(path,run)
        self.plan["groups"][group_index]["run_result"] = str(path); self.save()
        return path

    def test_run_receipt_attests_controls_and_capture_counts(self):
        self.add_run(0); self.add_run(1)
        folder, _ = self.export()
        groups = e.load(folder/"reference.json")["groups"]
        self.assertTrue(all(g["run_provenance_provided"] for g in groups))
        self.assertTrue(groups[1]["run_provenance"]["execution"]["experimental"]["device_feedback"])

    def test_run_wrong_binary_rejected(self):
        path = self.add_run(0); run = e.load(path)
        for node in run["preflight"]["nodes"]: node["binary_sha256"] = "f"*64
        e.write(path,run)
        with self.assertRaisesRegex(ValueError, "run/capture source identity"): self.export()

    def test_run_missing_cleanup_rejected(self):
        path = self.add_run(0); run = e.load(path); run["cleanup"]["1"]["removed"] = False; e.write(path,run)
        with self.assertRaisesRegex(ValueError, "missing cleanup"): self.export()

    def test_run_wrong_control_label_rejected(self):
        path = self.add_run(0); run = e.load(path); run["execution"]["experimental"]["rdma_best"] = True; e.write(path,run)
        with self.assertRaisesRegex(ValueError, "label contradicts"): self.export()

    def test_run_dropped_capture_rejected(self):
        path = self.add_run(0); run = e.load(path)
        run["execution"]["experimental"]["capture"].update(every=1,limit=4); e.write(path,run)
        with self.assertRaisesRegex(ValueError, "names/count differ"): self.export()

    def test_double_norms_need_recorded_opt_in(self):
        path=self.add_run(0)
        self.mutate_case(lambda raw,p:raw['capture']['native'].update(norm_rounding='double'))
        with self.assertRaisesRegex(ValueError,'normalization differs from recorded'):self.export()
        run=e.load(path);run['execution']['experimental']['reference_norms']=True;e.write(path,run)
        self.assertEqual(self.export()[1]['reported_cases'],2)

    def test_reference_norm_control_rejects_integer(self):
        path=self.add_run(0);run=e.load(path);run['execution']['experimental']['reference_norms']=1;e.write(path,run)
        with self.assertRaisesRegex(ValueError,'invalid boolean'):self.export()

    def test_missing_failed_proposal_report_remains_visible(self):
        def change(raw,prop):
            raw.update(complete=False,passed=False,error='RuntimeError: PRIVATE')
            raw['proposal_parity']={'report':'proposal-parity/report.json','complete':False,'passed':False}
        self.mutate_case(change,index=1)
        (self.cases[1].parent/'proposal-parity/report.json').unlink()
        self.batch.update(complete=False,passed=False,error='PRIVATE')
        self.batch['cases'][1].update(complete=False,passed=False,error='PRIVATE');self.save()
        self.assertEqual(self.export()[1]['missing_proposal_reports'],1)

    def test_gate_failure_retains_negative_ids_and_stops(self):
        def change(raw,prop):
            prop.update(complete=False,passed=False,head_gate_passed=False,error='ValueError: PRIVATE',steps_completed=[])
            prop['native_head_gate']['results'][0]['token']=99
            for key in ('reference_proposals','slot_matches','matching_prefix_length','output_sha256'):prop.pop(key)
            raw.update(complete=False,passed=False,error='ValueError: PRIVATE')
            raw['proposal_parity']={'report':'proposal-parity/report.json','complete':False,'passed':False}
        self.mutate_case(change,index=1)
        self.batch.update(complete=False,passed=False,error='PRIVATE')
        self.batch['cases'][1].update(complete=False,passed=False,error='PRIVATE');self.save()
        folder,summary=self.export()
        self.assertEqual(summary['head_gate_passes'],1)
        prop=e.load(folder/'reference.json')['groups'][1]['cases'][0]['result']['proposal']
        self.assertEqual(prop['native_head_gate']['results'][0]['token'],99)
        self.assertFalse(prop['head_gate_passed'])


if __name__ == "__main__":
    unittest.main()

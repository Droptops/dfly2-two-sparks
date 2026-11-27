"""Invented small receipts test freeze policy; these are not GPU evidence."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import unittest
from unittest import mock

from . import analyze_matrix as m
from . import selection as s
from . import test_analyze_matrix as matrix_tests


class SelectionTests(unittest.TestCase):
    def setUp(self):
        helper = matrix_tests.MatrixTests('runTest')
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        self.root = helper.root/'selection-repo'
        (self.root/'evidence/experimental').mkdir(parents=True)
        (self.root/'evidence/fixtures').mkdir()
        (self.root/'experimental').mkdir()
        self.h = helper.h
        self.prototype = helper.analyze()[1][0]
        self.fixture = copy.deepcopy(helper.fixture)
        self.fixture['requests'][0]['max_tokens'] = 256
        self.tasks = {'suite': 'heldout-test', 'first_seed': 1, 'tasks': [
            {'name': 'chat_greedy_test', 'temperature': 0, 'category': 'test'}]}
        self.tasks_path = self.root/'evidence/fixtures/heldout-tasks.json'
        m.dump(self.tasks_path, self.tasks)
        self.heldout = copy.deepcopy(self.fixture)
        self.heldout['metadata'] = {'suite': 'heldout-test', 'frozen_tasks_sha256': m.sha(self.tasks_path),
                                    'categories': {'chat_greedy_test': 'test'}}
        m.dump(self.root/'fixture.json', self.fixture)
        m.dump(self.root/'heldout.json', self.heldout)
        self.fixture_hash = m.sha(self.root/'fixture.json')
        self.patch = mock.patch.object(m, 'helpers', return_value=self.h)
        self.patch.start(); self.addCleanup(self.patch.stop)
        patch = mock.patch.dict(s.FIXTURE_HASHES, {'chat': self.fixture_hash, 'long': self.fixture_hash})
        patch.start(); self.addCleanup(patch.stop)
        m.dump(self.root/'experimental/native.lock.json', {'invented': True})
        lock_hash = hashlib.sha256(json.dumps({'invented': True}, sort_keys=True).encode()).hexdigest()
        self.build = {'version': 'v5', 'status': 'complete', 'source': {'experimental_lock_sha256': lock_hash},
            'public_image': {'build_exit_code': 0, 'binaries_sha256': {'tf-dsv41-lanes': 'b'*64},
                             'image_content_sha256': 'c'*64},
            'private_record_sha256': {'public_native_build_receipt': '9'*64}}
        m.dump(self.root/'evidence/experimental-build-v5.json', self.build)
        self.make_bundle(s.SWEEP, s.ORDER, 1, self.fixture, 0)
        self.private = {'schema': 'dfly2-candidate-freeze-v1', 'utc': '2026-11-26T01:00:00Z',
            'rule': {'schema': 'dfly2-private-candidate-selection-v1', 'source_matrix': s.SWEEP,
                     'declared_before_sweep_completion': True, 'criterion': s.PRIVATE_CRITERION, 'tie_breaker': s.PRIVATE_TIE},
            'rule_sha256': '8'*64,
            'selection_analysis_sha256': m.sha(self.folder(s.SWEEP)/'analysis.json'),
            'selected_sweep_variant': 'depth1', 'selected_exploratory_decode_tokens_per_second': 2.5,
            'candidate': {'name': 'candidate', 'mode': 'dfly2', 'args': s.expected_args('depth1')},
            'effective_execution': s.sweep_controls('depth1'), 'cluster_config_sha256': 'PRIVATE-CONFIG-DIGEST',
            'build_receipt_sha256': '9'*64, 'heldout_tasks_sha256': m.sha(self.tasks_path),
            'heldout_requests_sha256': m.sha(self.root/'heldout.json'), 'heldout_used_for_selection': False}
        self.private_path = self.root/'private.json'
        self.public_path = self.root/'evidence/experimental-selection-v5.json'
        m.dump(self.private_path, self.private)
        self.record = s.export(self.private_path, self.public_path, self.root)

    def folder(self, name): return self.root/'evidence/experimental'/name

    def make_bundle(self, name, names, repeats, fixture, minute):
        folder = self.folder(name); folder.mkdir()
        m.dump(folder/'requests.json', fixture)
        variants = []
        expected = {'serial': ('serial', s.controls('serial')),
            'legacy': ('dfly2', s.controls('dfly2')),
            'candidate': ('dfly2', s.sweep_controls('depth1')),
            'dspark_fixed5_nocopies': ('dspark_fixed5', s.controls('dspark_fixed5', depth=5, copies=False)),
            'dspark': ('dspark', s.controls('dspark'))}
        for variant in names:
            mode, execution = (('serial' if variant == 'serial' else 'dfly2'), s.sweep_controls(variant)) if name == s.SWEEP else expected[variant]
            variants.append({'name': variant, 'mode': mode, 'execution': execution})
        manifest = {'schema': 'dfly2-lab-matrix-public-v1', 'purpose': 'benchmark',
                    'complete': True, 'repeats': repeats, 'variants': variants, 'runs': []}
        runs = []
        for i, (variant, repeat) in enumerate(m.schedule(names, repeats), 1):
            v = next(v for v in variants if v['name'] == variant)
            run = copy.deepcopy(self.prototype)
            run.update(mode=v['mode'], execution=copy.deepcopy(v['execution']))
            for node in run['nodes']:
                node['requests_sha256'] = m.sha(folder/'requests.json')
                node['pack_manifest_sha256'] = 'e'*64 if v['mode'] == 'dfly2' else None
            start = datetime(2026, 11, 26, tzinfo=timezone.utc) + timedelta(minutes=minute, seconds=2*i)
            run['started_utc'] = start.isoformat().replace('+00:00', 'Z')
            run['finished_utc'] = (start+timedelta(seconds=1)).isoformat().replace('+00:00', 'Z')
            if v['execution']['experimental']['adaptive']:
                run['records'].append({'rank': 0, 'request': 0, 'dfly_policy': dict(
                    schema='observed-depth-v1', decisions=0, calibrations=0, probes=0,
                    chosen_depth_counts=[0]*6, conditional_survival=[.5]*5, tried=[0]*5, accepted=[0]*5,
                    verify_costs=[dict(n=0, ewma_ms=0)]*6, draft_costs=[dict(n=0, ewma_ms=0)]*6)})
            filename = f'run-{i:03d}-{variant}-r{repeat}.json'
            manifest['runs'].append({'order': i, 'variant': variant, 'repeat': repeat, 'file': filename})
            m.dump(folder/filename, run); runs.append(run)
        m.dump(folder/'matrix.json', manifest)
        analysis = m.compare_exported(manifest, runs, fixture['requests'], m.sha(folder/'requests.json'), self.h)
        m.dump(folder/'analysis.json', analysis)

    def complete(self):
        self.make_bundle('benchmark-chat', s.VARIANTS, 3, self.fixture, 61)
        self.make_bundle('benchmark-long', s.VARIANTS, 3, self.fixture, 62)
        self.make_bundle('heldout', ['serial', 'candidate', 'dspark'], 3, self.heldout, 63)

    def rewrite(self, name, mutate):
        folder = self.folder(name)
        manifest = m.load(folder/'matrix.json')
        runs = [m.load(folder/e['file']) for e in manifest['runs']]
        fixture = m.load(folder/'requests.json')
        mutate(manifest, runs)
        m.dump(folder/'matrix.json', manifest)
        for e, run in zip(manifest['runs'], runs): m.dump(folder/e['file'], run)
        m.dump(folder/'analysis.json', m.compare_exported(manifest, runs, fixture['requests'], m.sha(folder/'requests.json'), self.h))

    def test_staged_export_is_allowlisted_and_not_complete(self):
        self.assertNotIn('PRIVATE-', json.dumps(self.record))
        self.assertEqual(s.export(self.private_path, self.public_path, self.root), self.record)
        self.assertFalse(s.verify(self.public_path, self.root)['complete'])
        with self.assertRaisesRegex(ValueError, 'all 39'): s.verify(self.public_path, self.root, True)

    def test_all_39_runs_and_three_rotations_pass(self):
        self.complete()
        result = s.verify(self.public_path, self.root, True)
        self.assertTrue(result['complete'])
        self.assertEqual(result['successful_evaluation_runs'], 39)

    def test_equal_rate_tie_requires_earliest_candidate(self):
        bad = copy.deepcopy(self.record); bad['selected_sweep_variant'] = 'depth2'
        with self.assertRaisesRegex(ValueError, 'winner/tie'): s.verify_record(bad, self.root)
        self.private['selected_sweep_variant'] = 'depth2'; m.dump(self.private_path, self.private)
        with self.assertRaisesRegex(ValueError, 'winner/effective'): s.export(self.private_path, self.public_path, self.root)

    def test_frozen_sweep_digests_and_disqualification_rejected(self):
        path = self.folder(s.SWEEP)/'matrix.json'
        manifest = m.load(path)
        manifest['known_issues'] = [{'schema': 'dfly2-known-issue-v1', 'code': 'initial_draft_depth_bypass', 'affected_variants': ['depth0']}]
        m.dump(path, manifest)
        with self.assertRaises(ValueError): s.verify(self.public_path, self.root)

    def test_even_semantically_unchanged_frozen_sweep_bytes_are_bound(self):
        path = self.folder(s.SWEEP)/'analysis.json'
        path.write_text(json.dumps(m.load(path)), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'file digests'): s.verify(self.public_path, self.root)

    def test_individually_valid_retuned_candidate_is_rejected(self):
        self.complete()
        def change(manifest, runs):
            for v in manifest['variants']:
                if v['name'] == 'candidate': v['execution'] = s.sweep_controls('depth2')
            for e, run in zip(manifest['runs'], runs):
                if e['variant'] == 'candidate': run['execution'] = s.sweep_controls('depth2')
        self.rewrite('benchmark-long', change)
        with self.assertRaisesRegex(ValueError, 'controls retuned'): s.verify(self.public_path, self.root, True)

    def test_individually_valid_different_binary_is_rejected(self):
        self.complete()
        def change(manifest, runs):
            for run in runs:
                for node in run['nodes']: node['binary_sha256'] = 'f'*64
        self.rewrite('benchmark-long', change)
        with self.assertRaisesRegex(ValueError, 'cross-matrix'): s.verify(self.public_path, self.root, True)

    def test_wrong_final_build_and_private_provenance_are_rejected(self):
        self.build['public_image']['binaries_sha256']['tf-dsv41-lanes'] = 'a'*64
        m.dump(self.root/'evidence/experimental-build-v5.json', self.build)
        with self.assertRaisesRegex(ValueError, 'final public build'): s.verify(self.public_path, self.root)

    def test_freeze_chronology_and_heldout_hash_are_rejected(self):
        self.complete()
        for key, value, pattern in [('frozen_utc', '2026-11-26T00:00:01Z', 'predates'),
                                    ('frozen_utc', '2026-11-26T02:00:00Z', 'precede')]:
            bad = copy.deepcopy(self.record); bad[key] = value
            with self.assertRaisesRegex(ValueError, pattern): s.verify_record(bad, self.root, True)
        bad = copy.deepcopy(self.record); bad['heldout']['requests_sha256'] = '7'*64
        with self.assertRaisesRegex(ValueError, 'request SHA'): s.verify_record(bad, self.root, True)
        bad = copy.deepcopy(self.record); bad['heldout']['tasks_sha256'] = '7'*64
        with self.assertRaisesRegex(ValueError, 'task SHA'): s.verify_record(bad, self.root, True)

    def test_public_extra_fields_and_timestamp_offsets_rejected(self):
        bad = copy.deepcopy(self.record); bad['private_path'] = 'PRIVATE-PATH'
        with self.assertRaisesRegex(ValueError, 'fields'): s.verify_record(bad, self.root)
        bad = copy.deepcopy(self.record); bad['frozen_utc'] = '2026-11-26T01:00:00+01:00'
        with self.assertRaisesRegex(ValueError, 'explicit UTC'): s.verify_record(bad, self.root)

    def test_rotation_and_exact_output_gates_remain_active(self):
        self.complete()
        folder = self.folder('benchmark-chat')
        manifest = m.load(folder/'matrix.json')
        manifest['runs'][5]['repeat'] = 1; m.dump(folder/'matrix.json', manifest)
        with self.assertRaisesRegex(ValueError, 'rotated schedule'): s.verify(self.public_path, self.root, True)
        manifest['runs'][5]['repeat'] = 2; m.dump(folder/'matrix.json', manifest)
        path = folder/manifest['runs'][2]['file']; run = m.load(path)
        reply = next(r for r in run['records'] if 'reply' in r)
        reply['reply'][0] = 99
        reply['token_sha'] = hashlib.sha256(','.join(map(str, reply['reply'])).encode()).hexdigest()[:12]
        m.dump(path, run)
        with self.assertRaisesRegex(ValueError, 'token arrays'): s.verify(self.public_path, self.root, True)

    def test_consistently_changed_chat_outputs_between_matrices_are_rejected(self):
        self.complete()
        def change(manifest, runs):
            for run in runs:
                reply = next(r for r in run['records'] if 'reply' in r)
                reply['reply'][0] = 99
                reply['token_sha'] = hashlib.sha256(','.join(map(str, reply['reply'])).encode()).hexdigest()[:12]
        self.rewrite('benchmark-chat', change)
        with self.assertRaisesRegex(ValueError, 'between frozen sweep'): s.verify(self.public_path, self.root, True)


if __name__ == '__main__': unittest.main()

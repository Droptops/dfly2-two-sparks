#!/usr/bin/env python3
"""Export a frozen v5 candidate and verify its subsequent public evaluation.

This checks offline consistency, not authenticity of GPU execution or of the
recorded wall clock. A staged freeze is valid before evaluation bundles exist;
--require-complete additionally requires all 39 final runs.
"""
import argparse
import copy
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

if __package__:
    from . import analyze_matrix as m
else:
    import analyze_matrix as m

ROOT = Path(__file__).resolve().parents[3]
SWEEP = 'sweep-v5-chat'
ORDER = ['serial', 'depth0', 'depth1', 'depth2', 'depth3', 'depth4', 'depth5', 'adaptive', 'norms']
ELIGIBLE = ORDER[2:]
FINAL = {'benchmark-chat': 'chat', 'benchmark-long': 'long', 'heldout': 'heldout'}
VARIANTS = ['serial', 'legacy', 'candidate', 'dspark_fixed5_nocopies', 'dspark']
FIXTURE_HASHES = {
    'chat': '5243088b517b75309cb57a5d72a379a468339ed2aad98b2e58b533f0b6cd6f23',
    'long': '856411dc8409168f65ad1e2bb6640ae2a5d3b526b36b8101f0ffb22715cae539',
}
RULE = {'source_matrix': SWEEP, 'eligible_variants_in_order': ELIGIBLE,
        'criterion': 'maximum_aggregate_estimated_decode_tokens_per_second_median',
        'tie_breaker': 'earliest_in_frozen_sweep_order', 'selection_repeats': 1,
        'heldout_used_for_selection': False}
EVALUATION = {'matrices': list(FINAL), 'repeats': 3, 'rotation': 'left_by_repeat_index',
              'expected_successful_runs': 39, 'exact_token_arrays_and_finish_reasons_required': True}
LIMITATION = ('Offline consistency checks do not authenticate GPU execution or recorded timestamps. '
              'Selection on one exploratory chat repeat can overfit and does not establish Tencent drafter parity.')
PRIVATE_CRITERION = ('Highest aggregate estimated decode tokens per second among completed exact-output '
    'depth1..5, adaptive, and normalization candidates. Use complete analyzer output only. '
    'Serial and depth0 are controls. No held-out tasks are used for selection.')
PRIVATE_TIE = 'If rates are numerically equal, retain earlier variant in the frozen sweep order.'


def utc(value):
    m.require(isinstance(value, str) and re.fullmatch(
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)', value),
        'timestamp must be an explicit UTC instant')
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def controls(mode, *, depth=None, feedback=False, adaptive=False, norms=False, copies=None):
    return {**m.SHARED, 'drafting': mode != 'serial', 'fixed_draft_depth': depth,
            'experimental': {'device_feedback': feedback, 'rdma_best': feedback,
                'detailed_profile': False, 'adaptive': adaptive, 'reference_norms': norms,
                'copies': mode in ('dspark', 'dspark_fixed5') if copies is None else copies}}


def sweep_controls(name):
    if name == 'serial': return controls('serial')
    return controls('dfly2', feedback=True,
                    depth=int(name[-1]) if name.startswith('depth') else None,
                    adaptive=name == 'adaptive', norms=name == 'norms')


def expected_args(name):
    return ['--feedback', '--rdma'] + (
        ['--depth', name[-1]] if name.startswith('depth') else
        ['--adaptive'] if name == 'adaptive' else ['--reference-norms'])


def bundle(folder, repo):
    result = m.verify_export(folder, repo)
    manifest = m.load(folder/'matrix.json')
    m.require(result['complete'] and result['benchmark_eligible']
              and result['all_successful_token_arrays_and_finish_reasons_equal']
              and not result['failure_retained'] and not manifest.get('known_issues'),
              'selection/evaluation requires a complete eligible exact-output matrix')
    runs = [m.load(folder/e['file']) for e in manifest['runs']]
    previous = None
    for run in runs:
        start, finish = utc(run['started_utc']), utc(run['finished_utc'])
        m.require(start <= finish and (previous is None or previous <= start), 'matrix timestamp order')
        previous = finish
    return manifest, runs, result


def bundle_hashes(folder, manifest):
    names = ['matrix.json', 'analysis.json', 'requests.json'] + [e['file'] for e in manifest['runs']]
    return {name: m.sha(folder/name) for name in names}


def winner(manifest, analysis):
    m.require([v['name'] for v in manifest['variants']] == ORDER and manifest['repeats'] == 1,
              'selection sweep must use the frozen nine-variant order once')
    for v in manifest['variants']:
        m.require(v['mode'] == ('serial' if v['name'] == 'serial' else 'dfly2')
                  and v['execution'] == sweep_controls(v['name']), 'selection sweep controls changed')
    m.require(analysis['successful_runs'] == 9, 'selection requires all nine sweep runs')
    choices = [v for v in analysis['variants'] if v['variant'] in ELIGIBLE]
    m.require([v['variant'] for v in choices] == ELIGIBLE, 'selection candidates missing or reordered')
    return max(choices, key=lambda v: Decimal(str(v['estimated_decode_tokens_per_second']['median'])))


def identity(runs):
    for run in runs:
        for node in run['nodes']:
            m.require(type(node['rank']) is int and node['rank'] in (0, 1)
                      and node['gpu'] == 'NVIDIA GB10', 'unexpected cross-matrix rank/GPU identity')
            m.integer(node['target_rank_cache_bytes'], 1)
    first = runs[0]['nodes'][0]
    result = {key: m.digest(first[key]) for key in m.HASH_FIELDS}
    result['target_rank_cache_bytes'] = [
        n['target_rank_cache_bytes'] for n in sorted(runs[0]['nodes'], key=lambda n: n['rank'])]
    packs = {m.digest(n['pack_manifest_sha256']) for run in runs if run['mode'] == 'dfly2'
             for n in run['nodes']}
    m.require(len(packs) == 1, 'one frozen drafter pack identity required')
    result['pack_manifest_sha256'] = next(iter(packs))
    return result


def outputs(run):
    return [(r['name'], r['reply'], r['reason']) for r in run['records'] if 'reply' in r]


def build_identity(repo, runs):
    path = repo/'evidence/experimental-build-v5.json'
    build = m.load(path)
    m.require(build['version'] == 'v5' and build['status'] == 'complete'
              and build['public_image']['build_exit_code'] == 0, 'completed v5 public build required')
    value = identity(runs)
    m.require(value['binary_sha256'] == build['public_image']['binaries_sha256']['tf-dsv41-lanes']
              and value['image_content_sha256'] == build['public_image']['image_content_sha256'],
              'matrix does not use the final public build lane/image identity')
    lock_hash = hashlib.sha256(json.dumps(m.load(repo/'experimental/native.lock.json'), sort_keys=True).encode()).hexdigest()
    m.require(lock_hash == build['source']['experimental_lock_sha256'], 'current experimental lock differs from build')
    return {**value, 'build_receipt_sha256': m.sha(path), 'experimental_lock_sha256': lock_hash}


def validate_heldout(document, tasks, tasks_hash):
    metadata = document['metadata']
    m.require(metadata.get('frozen_tasks_sha256') == tasks_hash
              and metadata.get('suite') == tasks['suite'], 'heldout task metadata differs from freeze')
    requests = document['requests']
    m.require(len(requests) == len(tasks['tasks']), 'heldout task/request count mismatch')
    m.require(metadata.get('categories') == {t['name']: t['category'] for t in tasks['tasks']},
              'heldout category metadata differs from tasks')
    for i, (request, task) in enumerate(zip(requests, tasks['tasks'])):
        m.require(all(request[k] == value for k, value in {
            'name': task['name'], 'temperature': task['temperature'], 'max_tokens': 256,
            'top_p': 1, 'top_k': 20, 'seed': tasks['first_seed'] + i}.items()),
            'heldout request sampling controls differ from frozen tasks')


def frozen_parts(repo):
    folder = repo/'evidence/experimental'/SWEEP
    manifest, runs, analysis = bundle(folder, repo)
    m.require(analysis['requests_sha256'] == FIXTURE_HASHES['chat'], 'selection requires the reviewed chat fixture')
    return folder, manifest, runs, analysis, winner(manifest, analysis)


def export(freeze_path, output, repo=ROOT):
    """Write only reviewed public fields; never copy paths/configuration digests."""
    repo, freeze_path, output = Path(repo), Path(freeze_path), Path(output)
    private = m.load(freeze_path)
    m.require(private['schema'] == 'dfly2-candidate-freeze-v1', 'unknown private freeze schema')
    rule = private['rule']
    m.require(rule['schema'] == 'dfly2-private-candidate-selection-v1'
              and rule['source_matrix'] == SWEEP and rule['declared_before_sweep_completion'] is True
              and rule['criterion'] == PRIVATE_CRITERION and rule['tie_breaker'] == PRIVATE_TIE,
              'private selection rule differs from reviewed rule')
    folder, manifest, runs, analysis, selected = frozen_parts(repo)
    name = selected['variant']
    rate = selected['estimated_decode_tokens_per_second']['median']
    m.require(private['selected_sweep_variant'] == name
              and private['selected_exploratory_decode_tokens_per_second'] == rate
              and private['effective_execution'] == selected['execution']
              and private['candidate'] == {'name': 'candidate', 'mode': 'dfly2', 'args': expected_args(name)},
              'private freeze winner/effective controls differ from verified sweep')
    m.require(private['selection_analysis_sha256'] == m.sha(folder/'analysis.json'),
              'private freeze analysis digest differs from public sweep')
    build = m.load(repo/'evidence/experimental-build-v5.json')
    m.require(private['build_receipt_sha256'] == build['private_record_sha256']['public_native_build_receipt'],
              'private freeze build receipt is not the public v5 build')
    record = {'schema': 'dfly2-public-candidate-selection-v1', 'frozen_utc': private['utc'],
        'rule': copy.deepcopy(RULE), 'selected_sweep_variant': name,
        'selected_exploratory_decode_tokens_per_second': rate,
        'effective_execution': copy.deepcopy(selected['execution']),
        'source_sweep': {'folder': SWEEP, 'files_sha256': bundle_hashes(folder, manifest)},
        'build_identity': build_identity(repo, runs),
        'heldout': {'tasks_sha256': m.digest(private['heldout_tasks_sha256']),
                    'requests_sha256': m.digest(private['heldout_requests_sha256']),
                    'used_for_selection': private['heldout_used_for_selection']},
        'evaluation': copy.deepcopy(EVALUATION),
        'provenance': {'private_freeze_sha256': m.sha(freeze_path),
                       'private_rule_sha256': m.digest(private['rule_sha256']),
                       'private_selection_analysis_sha256': m.digest(private['selection_analysis_sha256'])},
        'limitation': LIMITATION}
    verify_record(record, repo)
    # A candidate freeze is write-once; identical re-export is harmless.
    if output.exists():
        m.require(m.load(output) == record, 'refusing to replace a different published candidate freeze')
    else:
        with output.open('x', encoding='utf-8', newline='\n') as f:
            json.dump(record, f, indent=2, allow_nan=False)
            f.write('\n')
    return record


def verify_record(record, repo=ROOT, require_complete=False):
    repo = Path(repo)
    keys = {'schema', 'frozen_utc', 'rule', 'selected_sweep_variant',
            'selected_exploratory_decode_tokens_per_second', 'effective_execution',
            'source_sweep', 'build_identity', 'heldout', 'evaluation', 'provenance', 'limitation'}
    m.require(set(record) == keys and record['schema'] == 'dfly2-public-candidate-selection-v1',
              'unknown public selection schema/fields')
    m.require(record['rule'] == RULE and record['evaluation'] == EVALUATION
              and record['limitation'] == LIMITATION, 'selection rule/evaluation scope changed')
    frozen = utc(record['frozen_utc'])
    folder, manifest, runs, analysis, selected = frozen_parts(repo)
    m.require(record['source_sweep'] == {'folder': SWEEP, 'files_sha256': bundle_hashes(folder, manifest)},
              'frozen sweep file digests changed')
    m.require(record['selected_sweep_variant'] == selected['variant']
              and record['selected_exploratory_decode_tokens_per_second'] == selected['estimated_decode_tokens_per_second']['median']
              and record['effective_execution'] == selected['execution'], 'frozen winner/tie or controls changed')
    m.execution(record['effective_execution'], 'dfly2')
    expected_identity = build_identity(repo, runs)
    m.require(record['build_identity'] == expected_identity, 'frozen build identity changed')
    m.require(utc(runs[-1]['finished_utc']) <= frozen, 'freeze predates completed sweep')
    heldout = record['heldout']
    m.require(set(heldout) == {'tasks_sha256', 'requests_sha256', 'used_for_selection'}
              and heldout['used_for_selection'] is False, 'heldout must be excluded from selection')
    tasks_path = repo/'evidence/fixtures/heldout-tasks.json'
    tasks = m.load(tasks_path)
    m.require(m.digest(heldout['tasks_sha256']) == m.sha(tasks_path), 'frozen heldout task SHA differs')
    m.digest(heldout['requests_sha256'])
    provenance = record['provenance']
    m.require(set(provenance) == {'private_freeze_sha256', 'private_rule_sha256', 'private_selection_analysis_sha256'},
              'unknown selection provenance fields')
    for value in provenance.values(): m.digest(value)
    m.require(provenance['private_selection_analysis_sha256'] == m.sha(folder/'analysis.json'),
              'frozen selection analysis SHA differs')
    completed = []
    total = 0
    previous_finish = frozen
    missing = False
    for name, fixture in FINAL.items():
        target = repo/'evidence/experimental'/name
        if not target.exists():
            missing = True
            continue
        m.require(not missing, 'final evaluation matrices published out of order')
        final, final_runs, final_analysis = bundle(target, repo)
        names = VARIANTS if fixture != 'heldout' else ['serial', 'candidate', 'dspark']
        expected = {'serial': ('serial', controls('serial')),
                    'legacy': ('dfly2', controls('dfly2')),
                    'candidate': ('dfly2', record['effective_execution']),
                    'dspark_fixed5_nocopies': ('dspark_fixed5', controls('dspark_fixed5', depth=5, copies=False)),
                    'dspark': ('dspark', controls('dspark'))}
        m.require(final['repeats'] == 3 and [v['name'] for v in final['variants']] == names,
                  'final evaluation must have three rotated repeats of frozen variants')
        for v in final['variants']:
            m.require((v['mode'], v['execution']) == expected[v['name']], 'final candidate/baseline controls retuned')
        m.require(final_analysis['successful_runs'] == len(names)*3, 'final evaluation run count differs')
        m.require(identity(final_runs) == identity(runs), 'cross-matrix binary/image/launcher/target/cache/pack identity changed')
        m.require(utc(final_runs[0]['started_utc']) >= previous_finish, 'final runs precede freeze or previous evaluation')
        previous_finish = utc(final_runs[-1]['finished_utc'])
        expected_hash = heldout['requests_sha256'] if fixture == 'heldout' else FIXTURE_HASHES[fixture]
        m.require(final_analysis['requests_sha256'] == expected_hash, 'final request SHA differs from frozen fixture')
        if fixture == 'chat':
            m.require(outputs(final_runs[0]) == outputs(runs[0]), 'chat outputs differ between frozen sweep and evaluation')
        if fixture == 'heldout': validate_heldout(m.load(target/'requests.json'), tasks, heldout['tasks_sha256'])
        total += final_analysis['successful_runs']
        completed.append(name)
    if require_complete:
        m.require(completed == list(FINAL) and total == 39, 'complete evaluation requires all 39 final runs')
    return {'schema': 'dfly2-selection-verification-v1', 'selected_sweep_variant': selected['variant'],
            'completed_evaluation_matrices': completed, 'successful_evaluation_runs': total,
            'complete': total == 39, 'scope': 'offline_consistency_only'}


def verify(path, repo=ROOT, require_complete=False):
    return verify_record(m.load(path), repo, require_complete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--freeze', type=Path, help='private frozen candidate receipt to sanitize')
    choice.add_argument('--verify', type=Path, help='public frozen candidate receipt')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        if args.output is None: parser.error('--freeze requires --output')
        export(args.freeze, args.output, args.repo)
        path = args.output
    else:
        if args.output is not None: parser.error('--output is only valid with --freeze')
        path = args.verify
    print(json.dumps(verify(path, args.repo, args.require_complete), indent=2))


if __name__ == '__main__':
    main()

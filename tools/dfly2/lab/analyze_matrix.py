#!/usr/bin/env python3
"""Offline validation/export of DFly2 experimental matrices. Starts no processes.

This is deliberately separate from the historical evidence schemas. It retains
experimental controls and nested policy/profile records, and never presents
capture or detailed-profile jobs as throughput benchmarks.
"""
import argparse
import copy
from decimal import Decimal
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
APPROVED_FIXTURES = {
    '5243088b517b75309cb57a5d72a379a468339ed2aad98b2e58b533f0b6cd6f23',
    '856411dc8409168f65ad1e2bb6640ae2a5d3b526b36b8101f0ffb22715cae539',
}
SHARED = {'world_size': 2, 'streams': 1, 'parallel': 1, 'pool_tokens': 8192,
          'cuda_graphs': False, 'profiling': True}
HASH_FIELDS = ('binary_sha256', 'launcher_sha256', 'image_content_sha256', 'target_config_sha256')
POLICY_FIELDS = {'schema', 'decisions', 'calibrations', 'probes', 'chosen_depth_counts',
                 'conditional_survival', 'tried', 'accepted', 'verify_costs', 'draft_costs'}
PROFILE_FIELDS = {'schema', 'passes', 'total_pass_ms', 'host_readbacks', 'depths', 'categories', 'scope'}
PROFILE_CATEGORIES = {'context', 'fusion', 'mixing', 'attention', 'route', 'routed_moe', 'shared_moe',
                      'correction_input', 'correction_mlp', 'correction_attention', 'correction_merge',
                      'head', 'argmax', 'collective', 'feedback_gap'}
PROFILE_SCOPE = 'elapsed stream intervals, including enqueue gaps; instrumentation enabled'
TOKEN = re.compile(r'[a-z][a-z0-9_-]{0,63}\Z')
HEX64 = re.compile(r'[0-9a-f]{64}\Z')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def helpers(repo):
    root = (Path(repo)/'tools/dfly2').resolve()
    names = ('run_pair', 'cluster_config', 'sanitize_evidence', 'verify_evidence')
    for name in names:
        if name in sys.modules:
            require(Path(sys.modules[name].__file__).resolve() == root/(name+'.py'),
                    'helper imported from a different checkout; use a fresh Python process')
    sys.path.insert(0, str(root))
    modules = {name: importlib.import_module(name) for name in names}
    require(all(Path(module.__file__).resolve() == root/(name+'.py')
                for name,module in modules.items()), 'helper checkout mismatch')
    return SimpleNamespace(**modules)


def numeric(value):
    if type(value) in (int, bool) or value is None:
        return value
    if type(value) is float:
        require(math.isfinite(value), 'non-finite numerical evidence')
        return value
    if isinstance(value, list):
        return [numeric(x) for x in value]
    if isinstance(value, dict):
        require(all(TOKEN.fullmatch(k) for k in value), 'unexpected numerical key')
        return {k: numeric(v) for k, v in value.items()}
    raise ValueError('non-numerical value in metric evidence')


def digest(value):
    require(isinstance(value, str) and HEX64.fullmatch(value), 'invalid SHA-256')
    return value


def integer(value, low=0):
    require(type(value) is int and value >= low, 'invalid integer counter')
    return value


def policy(value):
    require(set(value) == POLICY_FIELDS and value['schema'] == 'observed-depth-v1', 'unknown policy schema/fields')
    out = {'schema': value['schema'], **numeric({k:v for k,v in value.items() if k != 'schema'})}
    for key in ('decisions', 'calibrations', 'probes'):
        integer(out[key])
    require(len(out['chosen_depth_counts']) == 6, 'policy depth histogram shape')
    require(sum(integer(x) for x in out['chosen_depth_counts']) == out['decisions'], 'policy decisions differ from histogram')
    require(out['calibrations'] <= out['decisions'] and out['probes'] <= out['decisions'], 'policy subset counts invalid')
    require(len(out['conditional_survival']) == len(out['tried']) == len(out['accepted']) == 5, 'policy prefix array shape')
    require(all(0 <= x <= 1 for x in out['conditional_survival']), 'policy survival range')
    require(all(0 <= integer(a) <= integer(t) for a,t in zip(out['accepted'],out['tried'])), 'policy acceptance counts invalid')
    for key in ('verify_costs', 'draft_costs'):
        require(len(out[key]) == 6, 'policy cost array shape')
        for cost in out[key]:
            require(set(cost) == {'n', 'ewma_ms'} and integer(cost['n']) >= 0 and cost['ewma_ms'] >= 0, 'policy cost invalid')
    return out


def profile(value):
    require(set(value) == PROFILE_FIELDS and value['schema'] == 'dfly2-stream-profile-v1', 'unknown detailed profile schema/fields')
    require(value['scope'] == PROFILE_SCOPE, 'unreviewed profile text')
    out = {'schema': value['schema'], 'scope': value['scope'],
           **numeric({k:v for k,v in value.items() if k not in ('schema','scope')})}
    integer(out['passes']); integer(out['host_readbacks'])
    require(out['total_pass_ms'] >= 0 and len(out['depths']) == 6, 'profile timing/depth shape')
    require(sum(integer(x) for x in out['depths']) == out['passes'] and out['depths'][0] == 0, 'profile pass counts')
    require(set(out['categories']) == PROFILE_CATEGORIES, 'unknown profile categories')
    for value in out['categories'].values():
        require(set(value) == {'calls','total_ms'} and integer(value['calls']) >= 0 and value['total_ms'] >= 0,
                'invalid category metric')
    return out


def fixture(data):
    require(isinstance(data, dict) and set(data) == {'requests','metadata'}, 'invalid fixture envelope')
    requests = data['requests']
    require(isinstance(requests,list) and requests, 'empty fixture')
    names = []
    for r in requests:
        require(set(r) == {'name','prompt','max_tokens','temperature','top_p','top_k','seed'}, 'unexpected request fields')
        require(isinstance(r['name'],str) and TOKEN.fullmatch(r['name']), 'unsafe request name')
        require(r['prompt'] and all(type(x) is int and 0 <= x < 129280 for x in r['prompt']), 'invalid prompt tokens')
        require(type(r['max_tokens']) is int and 0 < r['max_tokens'] <= 8192, 'invalid output budget')
        require(len(r['prompt'])+r['max_tokens']+16 <= 8192, 'request exceeds validated pool')
        names.append(r['name'])
    require(len(set(names)) == len(names), 'duplicate request names')
    return requests


def execution(value, mode):
    require(mode in ('serial','dspark','dspark_fixed5','dfly2'), 'unknown native mode')
    require(set(value) in (set(SHARED)|{'drafting','fixed_draft_depth'},
                           set(SHARED)|{'drafting','fixed_draft_depth','experimental'}), 'unknown execution keys')
    for k,v in SHARED.items():
        require(type(value[k]) is type(v) and value[k] == v, 'unsupported shared inference setting '+k)
    require(value['drafting'] is (mode != 'serial'), 'drafting flag contradicts mode')
    depth = value['fixed_draft_depth']
    require(depth is None or (type(depth) is int and 0 <= depth <= 5), 'invalid fixed depth')
    require(mode != 'serial' or depth is None, 'serial depth override')
    require(mode != 'dspark_fixed5' or depth == 5, 'fixed-five DSpark depth changed')
    if 'experimental' in value:
        exp = value['experimental']
        flags = {'device_feedback','rdma_best','detailed_profile','adaptive','copies'}
        require(set(exp) in (flags,flags|{'capture'},flags|{'reference_norms'},flags|{'capture','reference_norms'}), 'unknown effective controls')
        if 'reference_norms' in exp: flags=flags|{'reference_norms'}
        require(all(type(exp[x]) is bool for x in flags), 'nonboolean effective control')
        require(mode == 'dfly2' or not any(exp[x] for x in flags-{'copies'}), 'DFly controls active on another mode')
        require(mode != 'dfly2' or exp['copies'] is False, 'DFly suffix copies unsupported')
        require(mode != 'serial' or exp['copies'] is False, 'serial copies enabled')
        require(not exp['adaptive'] or depth is None, 'fixed/adaptive conflict')
        if 'capture' in exp:
            cap = exp['capture']
            require(mode == 'dfly2' and set(cap) == {'enabled','limit','every'} and cap['enabled'] is True, 'invalid capture controls')
            require(1 <= integer(cap['limit']) <= 64 and 1 <= integer(cap['every']) <= 4096, 'capture bounds')
    return copy.deepcopy(value)


def sanitize_records(records, h, requests):
    out = []
    names = {r['name'] for r in requests}
    for r in records:
        rank = r.get('rank')
        if 'dfly_policy' in r:
            require(set(r) == {'rank','request','dfly_policy'} and rank == 0, 'invalid policy record envelope')
            out.append({'rank':0,'request':integer(r['request']),'dfly_policy':policy(r['dfly_policy'])})
            continue
        if 'dfly2_profile' in r:
            require(set(r) == {'rank','dfly2_profile'} and rank in (0,1), 'invalid profile record envelope')
            out.append({'rank':rank,'dfly2_profile':profile(r['dfly2_profile'])})
            continue
        keys = h.sanitize_evidence.REPLY_KEYS if 'reply' in r else h.sanitize_evidence.PREFILL_KEYS if 'prefill_s' in r else None
        if keys:
            require(rank == 0 and r.get('name') in names, 'unknown reply/prefill request')
            clean = {k:r[k] for k in keys if k in r}
            for k,v in clean.items():
                if k not in ('name','reason','token_sha'): numeric(v)
            if 'reply' in r:
                require(r['reason'] in ('stop','length') and re.fullmatch(r'[0-9a-f]{12}',r['token_sha']), 'invalid reply metadata')
            out.append(clean)
        else:
            for name, keys in h.sanitize_evidence.METRIC_KEYS.items():
                if name in r and rank == 0:
                    out.append({'rank':0,name:{k:numeric(v) for k,v in r[name].items() if k in keys}})
    return out


def read_extra_records(result_path):
    path = Path(result_path).parent/'rank1.stdout.jsonl'
    if not path.exists(): return []
    records=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        try: value=json.loads(line)
        except ValueError: continue
        if isinstance(value,dict) and ('dfly2_profile' in value or 'dfly_policy' in value):
            require(value.get('rank') == 1, 'worker log has wrong rank')
            records.append(value)
    return records


def sanitize_raw(raw, path, requests, h):
    mode=raw['mode']
    clean={'schema':'dfly2-lab-run-v1','mode':mode,'success':raw.get('success') is True,
           'complete':raw.get('native',{}).get('complete') is True,
           'execution':execution(raw['execution'],mode),'nodes':[], 'lifecycle':[]}
    for k in ('started_utc','finished_utc'):
        require(isinstance(raw.get(k),str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z',raw[k]), 'invalid run timestamp')
        clean[k]=raw[k]
    for node in raw.get('preflight',{}).get('nodes',[]):
        require(node['rank'] in (0,1) and node['gpu']=='NVIDIA GB10', 'unexpected hardware')
        clean['nodes'].append({'rank':node['rank'],'gpu':node['gpu'],
            **{k:digest(node[k]) for k in (*HASH_FIELDS,'requests_sha256')},
            'pack_manifest_sha256':digest(node['pack_manifest_sha256']) if node['pack_manifest_sha256'] else None,
            'target_rank_cache_bytes':integer(node['rank_cache']['bytes'],1)})
    for rank in ('0','1'):
        state=raw.get('states',{}).get(rank,{})
        cleanup=raw.get('cleanup',{}).get(rank,{})
        status=state.get('Status')
        require(status is None or status in ('created','running','paused','restarting','removing','exited','dead'), 'unreviewed state text')
        clean['lifecycle'].append({'rank':int(rank),'status':status,'exit_code':numeric(state.get('ExitCode')),
                                 'oom_killed':state.get('OOMKilled') is True, 'error_present':bool(state.get('Error')),
                                 'removed':cleanup.get('removed') is True})
    native=raw.get('native',{})
    clean['records']=sanitize_records(native.get('records',[])+read_extra_records(path),h,requests)
    clean['summary']={k:numeric(v) for k,v in (native.get('summary') or {}).items() if k in h.sanitize_evidence.SUMMARY_KEYS}
    if not clean['success']:
        clean['failure']={'category':'run_failed','error_present':bool(raw.get('error')),
                          'equivalence_reported_false':raw.get('equivalence',{}).get('equal') is False if raw.get('equivalence') else False}
    return clean


def summarize_clean(run, requests, request_hash, reference, h):
    adapter=copy.deepcopy({**run,'schema':'dfly2-public-evidence-v1','configuration':run['execution']})
    # The historical helper assumes a draft pass every round. Adaptive depth
    # may skip passes, so its optimistic bound must subtract actual pass count,
    # not mean pass cost times every serial/verification round. Feed a derived
    # per-round cost only to that calculation; preserve the measured mean below.
    profiles=[r['profile'] for r in adapter['records'] if 'profile' in r]
    require(len(profiles)==1,'expected one native profile')
    p=profiles[0]
    rounds=integer(p['rounds'],1); passes=integer(p['passes'])
    require(passes<=rounds,'more draft passes than single-stream rounds')
    actual_pass_ms=Decimal(str(p['pass_ms']))
    require(actual_pass_ms.is_finite() and actual_pass_ms>=0,'invalid draft pass timing')
    effective=actual_pass_ms*passes/rounds
    p['pass_ms']=float(effective)
    result,actual=h.verify_evidence.summarize_run(adapter,requests,request_hash,reference)
    result['native_profile'].update(pass_ms=float(actual_pass_ms),passes=passes,
                                   draft_ms_per_round=float(effective))
    return result,actual


def schedule(variants,repeats):
    return [(name,repeat+1) for repeat in range(repeats)
            for name in variants[repeat%len(variants):]+variants[:repeat%len(variants)]]


def spread(values):
    values=[Decimal(str(x)) for x in values if x is not None]
    if not values: return None
    return {k:float(v.quantize(Decimal('.01'))) for k,v in
            [('median',statistics.median(values)),('min',min(values)),('max',max(values))]}


def known_issues(value, variant_names):
    """Allow only reviewed, path-free disqualifications; never waive run checks."""
    require(isinstance(value,list) and value, 'known issues must be a nonempty list')
    names=set(variant_names); seen=set(); result=[]
    for issue in value:
        require(isinstance(issue,dict) and set(issue)=={'schema','code','affected_variants'},
                'unknown known issue schema/fields')
        require(issue['schema']=='dfly2-known-issue-v1'
                and issue['code']=='initial_draft_depth_bypass', 'unknown known issue schema/code')
        require(issue['code'] not in seen, 'duplicate known issue code')
        seen.add(issue['code'])
        affected=issue['affected_variants']
        require(isinstance(affected,list) and affected
                and all(isinstance(name,str) and name in names for name in affected),
                'known issue must reference declared variants')
        require(len(set(affected))==len(affected), 'duplicate known issue affected variant')
        result.append(copy.deepcopy(issue))
    return result


def compare_exported(manifest, runs, requests, request_hash, h):
    require(manifest['schema']=='dfly2-lab-matrix-public-v1', 'unknown public matrix schema')
    purpose=manifest['purpose']
    require(purpose in ('benchmark','profile','capture'), 'unknown measurement purpose')
    repeats=integer(manifest['repeats'],1)
    variants=manifest['variants']; names=[v['name'] for v in variants]
    require(names and len(set(names))==len(names) and all(TOKEN.fullmatch(x) for x in names), 'invalid variant names')
    issues=known_issues(manifest['known_issues'],names) if 'known_issues' in manifest else []
    depth_bypass={name for issue in issues if issue['code']=='initial_draft_depth_bypass'
                  for name in issue['affected_variants']}
    require(variants[0]['mode']=='serial', 'first variant must be serial reference')
    declared={v['name']:v for v in variants}
    for v in variants:
        require(set(v)=={'name','mode','execution'}, 'unknown public variant fields')
        execution(v['execution'],v['mode'])
    expected=schedule(names,repeats)
    entries=manifest['runs']
    require(entries and len(entries)==len(runs)<=len(expected), 'missing/extra attempted runs')
    require(len({e['file'] for e in entries})==len(entries),'duplicate receipt filename')
    require(type(manifest['complete']) is bool, 'complete must be boolean')
    if manifest['complete']: require(len(entries)==len(expected), 'completed matrix missing repeats')
    identity=None; pack=None; reference=None; previous_finish=None; rows=[]; failed=False
    for index,(entry,run) in enumerate(zip(entries,runs)):
        require(set(entry)=={'order','variant','repeat','file'} and entry['order']==index+1, 'malformed run entry')
        require((entry['variant'],entry['repeat'])==expected[index], 'order differs from rotated schedule')
        require(Path(entry['file']).name==entry['file'], 'unsafe receipt name')
        variant=declared[entry['variant']]
        run_keys={'schema','mode','success','complete','execution','nodes','lifecycle','started_utc','finished_utc','records','summary'}
        require(set(run) in (run_keys,run_keys|{'failure'}),'unknown public receipt fields')
        require(run['schema']=='dfly2-lab-run-v1' and run['mode']==variant['mode'], 'wrong run variant')
        require(type(run['success']) is type(run['complete']) is bool,'nonboolean completion flag')
        require(sanitize_records(run['records'],h,requests)==run['records'],'unreviewed public record fields')
        require(run['execution']==variant['execution'], 'effective controls differ from declared variant')
        exp=run['execution'].get('experimental',{})
        if purpose=='benchmark':
            require(not exp.get('detailed_profile') and not exp.get('capture'), 'instrumented/capture run cannot be a benchmark')
        require(not failed, 'attempt continued after failure')
        if not run['success']:
            require(set(run.get('failure',{}))=={'category','error_present','equivalence_reported_false'}
                    and run['failure']['category']=='run_failed','unknown failure metadata')
            failed=True
            require(index==len(runs)-1 and manifest['complete'] is False, 'failed attempt must stop incomplete matrix')
            continue
        require(run['complete'], 'successful native run incomplete')
        require(run['started_utc']<=run['finished_utc'], 'run clock reversed')
        require(previous_finish is None or previous_finish<=run['started_utc'], 'runs overlap or order changed')
        previous_finish=run['finished_utc']
        require(sorted(n['rank'] for n in run['nodes'])==[0,1], 'missing rank evidence')
        for n in run['nodes']:
            require(set(n)=={'rank','gpu','requests_sha256','pack_manifest_sha256','target_rank_cache_bytes'}|set(HASH_FIELDS),
                    'unknown public node metadata')
        require(len(run['lifecycle'])==2 and all(x['status']=='exited' and x['exit_code']==0 and not x['oom_killed']
                    and type(x['exit_code']) is int and not x['error_present'] and x['removed'] for x in run['lifecycle']), 'rank failure/OOM/cleanup failure')
        require(all(set(x)=={'rank','status','exit_code','oom_killed','error_present','removed'} for x in run['lifecycle']),
                'unknown lifecycle metadata')
        require(all(all(type(x[k]) is bool for k in ('oom_killed','error_present','removed')) for x in run['lifecycle']),
                'nonboolean lifecycle flags')
        require(sorted(x['rank'] for x in run['lifecycle'])==[0,1],'duplicate lifecycle rank')
        for field in HASH_FIELDS:
            require(len({digest(n[field]) for n in run['nodes']})==1, 'ranks disagree on '+field)
        current=tuple(run['nodes'][0][k] for k in HASH_FIELDS)+tuple(sorted((n['rank'],n['target_rank_cache_bytes']) for n in run['nodes']))
        if identity is None: identity=current
        require(current==identity, 'native/image/launcher/target/cache changed within matrix')
        if run['mode']=='dfly2':
            packs={digest(n['pack_manifest_sha256']) for n in run['nodes']}
            require(len(packs)==1, 'rank drafter pack mismatch')
            if pack is None: pack=next(iter(packs))
            require(packs=={pack}, 'drafter weights changed within matrix')
        else:
            require(all(n['pack_manifest_sha256'] is None for n in run['nodes']), 'unexpected draft pack in another mode')
        profiles=[r for r in run['records'] if 'dfly2_profile' in r]
        policies=[r for r in run['records'] if 'dfly_policy' in r]
        if run['execution']['fixed_draft_depth']==0 and entry['variant'] not in depth_bypass:
            native_profiles=[r['profile'] for r in run['records'] if 'profile' in r]
            require(len(native_profiles)==1 and integer(native_profiles[0]['passes'])==0,
                    'fixed zero depth performed draft passes')
            for reply in (r for r in run['records'] if 'reply' in r):
                require(all(type(reply.get(k)) is int and reply[k]==0 for k in ('drafted','accepted')),
                        'fixed zero depth drafted or accepted tokens')
                require('rounds' not in reply or (type(reply['rounds']) is int
                        and reply['rounds']==len(reply['reply'])-1), 'fixed zero depth round count differs from serial decoding')
        require(bool(profiles)==bool(exp.get('detailed_profile')), 'detailed profile evidence/control mismatch')
        require(bool(policies)==bool(exp.get('adaptive')), 'adaptive policy evidence/control mismatch')
        if policies:
            require([r['request'] for r in policies]==list(range(len(requests))), 'missing/duplicate policy request')
            for r in policies: policy(r['dfly_policy'])
        if profiles:
            require(sorted(r['rank'] for r in profiles)==[0,1], 'need detailed profile from both ranks')
            native_profiles=[r['profile'] for r in run['records'] if 'profile' in r]
            require(len(native_profiles)==1,'need native profile alongside detailed profiles')
            for r in profiles:
                p=profile(r['dfly2_profile'])
                require(p['passes']==native_profiles[0]['passes'],'detailed/native pass counts differ')
                reads=p['passes'] if exp.get('device_feedback') else sum(i*n for i,n in enumerate(p['depths']))
                require(p['host_readbacks']==reads, 'feedback readback count contradicts execution path')
            require(profiles[0]['dfly2_profile']['depths']==profiles[1]['dfly2_profile']['depths'], 'profile ranks disagree on depths')
        row,actual=summarize_clean(run,requests,request_hash,reference,h)
        if reference is None: reference=actual
        row.update(variant=entry['variant'],repeat=entry['repeat'],order=entry['order'],receipt=entry['file'])
        rows.append(row)
    eligible=manifest['complete'] and not failed and purpose=='benchmark' and not issues
    result={'schema':'dfly2-lab-analysis-v1','purpose':purpose,'complete':manifest['complete'],
            'benchmark_eligible':eligible,'attempted_runs':len(runs),'successful_runs':len(rows),'requests_sha256':request_hash,
            'all_successful_token_arrays_and_finish_reasons_equal':bool(rows),'failure_retained':failed,
            'measurement_scope':'One-stream eager, 8192-token pool. Exact target outputs do not establish Tencent drafter parity.'}
    if issues: result['known_issues']=issues
    if not eligible:
        result['reason']=('Declared known issues disqualify this matrix from throughput comparisons; raw run receipts remain available.'
                          if issues else 'Capture/profile runs or incomplete matrices are excluded from throughput comparisons.')
        return result
    grouped=[]
    for name in names:
        subset=[r for r in rows if r['variant']==name]
        require(len(subset)==repeats,'repeat count mismatch')
        per_request=[]
        for q,name_q in enumerate(r['name'] for r in requests):
            items=[r['requests'][q] for r in subset]
            per_request.append({'name':name_q,'prompt_tokens':items[0]['prompt_tokens'],'output_tokens':items[0]['output_tokens'],
                **{k:spread([r[k] for r in items]) for k in
                   ('estimated_decode_tokens_per_second','estimated_decode_seconds','prefill_seconds','total_seconds','tokens_per_round')},
                'end_to_end_tokens_per_second':spread([Decimal(r['output_tokens'])/Decimal(str(r['total_seconds'])) for r in items])})
        grouped.append({'variant':name,'mode':declared[name]['mode'],'execution':declared[name]['execution'],'runs':repeats,
                        **{k:spread([r[k] for r in subset]) for k in ('estimated_decode_tokens_per_second','end_to_end_tokens_per_second',
                           'estimated_decode_seconds','total_seconds_including_prefill','tokens_per_round')},
                        'draft_pass_ms':spread([r['native_profile']['pass_ms'] for r in subset]),
                        'draft_ms_per_round':spread([r['native_profile']['draft_ms_per_round'] for r in subset]),
                        'requests':per_request})
    result.update(variants=grouped,runs=rows, output_tokens_per_run=rows[0]['output_tokens'],
                  timing_method='Per run: output tokens after the first per request divided by total minus prefill; cross-checked against reply and native step clocks. Median/min/max across repeats; no confidence-interval claim.')
    return result


def analyze_private(manifest_path, repo):
    path=Path(manifest_path).resolve(); data=load(path); h=helpers(repo)
    require(data.get('schema')=='dfly2-lab-matrix-v1','unknown private matrix schema')
    resolve=lambda p: (path.parent/str(p)).resolve()
    fixture_path=resolve(data['fixture']); payload=fixture_path.read_bytes(); requests=fixture(json.loads(payload))
    variants=[]; configs={}; invariant=None
    for v in data['variants']:
        require(set(v)=={'name','mode','config','execution'},'unknown private variant fields')
        cfg=h.cluster_config.load(resolve(v['config']))
        h.run_pair.configure(cfg)
        require(h.run_pair.execution_settings(v['mode'])==v['execution'],'declared controls do not match private config')
        shared=config_invariants(cfg)
        if invariant is None: invariant=shared
        require(shared==invariant,'private configuration changed beyond declared experimental controls/config paths')
        configs[v['name']]=cfg
        variants.append({k:v[k] for k in ('name','mode','execution')})
    public={'schema':'dfly2-lab-matrix-public-v1','purpose':data['purpose'],'repeats':data['repeats'],
            'complete':data['complete'],'variants':variants,'runs':[]}
    if 'known_issues' in data:
        public['known_issues']=known_issues(data['known_issues'],[v['name'] for v in variants])
    runs=[]; run_ids=set(); container_ids=set(); rank_cache_names=None
    for entry in data['runs']:
        fields={'order','variant','repeat','result'}
        require(set(entry) in (fields,fields|{'config'}),'unknown private attempt fields')
        result_path=resolve(entry['result']); raw=load(result_path)
        require(raw['run_id'] not in run_ids,'duplicate attempted run UUID'); run_ids.add(raw['run_id'])
        cfg=h.cluster_config.load(resolve(entry['config'])) if 'config' in entry else configs[entry['variant']]
        require(config_invariants(cfg)==invariant,'attempt configuration changed shared invariants')
        h.run_pair.configure(cfg)
        declared=next(v for v in variants if v['name']==entry['variant'])
        require(h.run_pair.execution_settings(declared['mode'])==declared['execution'],
                'attempt config effective controls differ from declared variant')
        nodes=raw.get('preflight',{}).get('nodes',[])
        require(all(n['config_identity']==h.cluster_config.identity(cfg) for n in nodes),'receipt config digest differs from variant config')
        if raw.get('success'):
            require(set(raw.get('states',{}))==set(raw.get('cleanup',{}))=={'0','1'},
                    'successful attempt lacks both rank lifecycle receipts')
            require(all({'Status','ExitCode','OOMKilled','Error'}<=set(state)
                        and type(state['OOMKilled']) is bool and type(state['ExitCode']) is int
                        for state in raw['states'].values()),'missing or malformed rank lifecycle evidence')
            require(all(n['launcher_sha256']==h.run_pair.launcher_identity() for n in nodes),
                    'analyze using the exact launcher checkout that produced these runs')
            current=sorted((n['rank'],n['rank_cache']['name']) for n in nodes)
            if rank_cache_names is None: rank_cache_names=current
            require(current==rank_cache_names,'rank cache files changed across variants')
            for cleanup in raw['cleanup'].values():
                require(cleanup['id'] not in container_ids,'container ownership ID reused')
                container_ids.add(cleanup['id'])
        clean=sanitize_raw(raw,result_path,requests,h)
        filename=f"run-{entry['order']:03d}-{entry['variant']}-r{entry['repeat']}.json"
        public['runs'].append({k:entry[k] for k in ('order','variant','repeat')}|{'file':filename})
        runs.append(clean)
    analysis=compare_exported(public,runs,requests,hashlib.sha256(payload).hexdigest(),h)
    return public,runs,payload,analysis


def verify_export(folder,repo):
    folder=Path(folder); manifest=load(folder/'matrix.json')
    payload=(folder/'requests.json').read_bytes(); requests=fixture(json.loads(payload))
    runs=[load(folder/e['file']) for e in manifest['runs']]
    result=compare_exported(manifest,runs,requests,hashlib.sha256(payload).hexdigest(),helpers(repo))
    require(result==load(folder/'analysis.json'),'stored analysis differs from recomputed evidence')
    return result


def make_manifest(variant_specs, fixture_path, repo, *, purpose='benchmark', repeats=3):
    """Build a planned private manifest; coordinator appends runs after attempts."""
    h=helpers(repo)
    variants=[]
    for name,mode,config_path in variant_specs:
        cfg=h.cluster_config.load(config_path)
        h.run_pair.configure(cfg)
        variants.append({'name':name,'mode':mode,'config':str(Path(config_path).resolve()),
                         'execution':h.run_pair.execution_settings(mode)})
    return {'schema':'dfly2-lab-matrix-v1','purpose':purpose,'fixture':str(Path(fixture_path).resolve()),
            'repeats':repeats,'variants':variants,'complete':False,'runs':[]}


def config_invariants(config):
    shared=copy.deepcopy(config)
    shared.pop('experimental',None)
    for node in shared['nodes']:node.pop('config_path')
    return shared


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=ROOT)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--manifest',type=Path)
    group.add_argument('--verify-export',type=Path)
    p.add_argument('--output',type=Path,help='private analysis JSON')
    p.add_argument('--export',type=Path,help='new sanitized evidence directory; never overwrites')
    p.add_argument('--approved-fixture-sha256',action='append',default=[])
    args=p.parse_args()
    if args.verify_export:
        result=verify_export(args.verify_export,args.repo)
    else:
        manifest,runs,payload,result=analyze_private(args.manifest,args.repo)
        if args.output: dump(args.output,result)
        if args.export:
            require(hashlib.sha256(payload).hexdigest() in APPROVED_FIXTURES|set(args.approved_fixture_sha256),
                    'fixture must be explicitly reviewed before public export')
            args.export.mkdir(parents=True,exist_ok=False)
            (args.export/'requests.json').write_bytes(payload)
            dump(args.export/'matrix.json',manifest)
            for entry,run in zip(manifest['runs'],runs): dump(args.export/entry['file'],run)
            dump(args.export/'analysis.json',result)
            verify_export(args.export,args.repo)
    print(json.dumps({k:result[k] for k in ('purpose','complete','benchmark_eligible','attempted_runs','successful_runs','failure_retained')},indent=2))


if __name__=='__main__': main()

"""Offline mutation tests use invented receipts, never GPU measurement claims."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

if __package__:
    from . import analyze_matrix as m
else:
    import analyze_matrix as m

REPO=Path(__file__).resolve().parents[3]


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='dfly-matrix-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.h=m.helpers(REPO)
        self.fixture={'requests':[dict(name='chat_greedy_test',prompt=[1,2,3],max_tokens=3,
                                      temperature=0,top_p=1,top_k=20,seed=1)],
                      'metadata':{'suite':'synthetic-test'}}
        self.fixture_path=self.root/'requests.json';m.dump(self.fixture_path,self.fixture)
        self.fixture_hash=m.sha(self.fixture_path)
        self.configs={}
        specs=[]
        for name,mode,controls in [('serial','serial',{}),('dfly','dfly2',{})]:
            cfg=m.load(REPO/'examples/cluster.example.json')
            cfg['experimental']=controls
            for node in cfg['nodes']: node['config_path']=f'/etc/dfly2-two-sparks/{name}.json'
            self.configs[name]=self.h.cluster_config.validate(cfg)
            path=self.root/(name+'-config.json');m.dump(path,cfg)
            specs.append((name,mode,path))
        self.manifest=m.make_manifest(specs,self.fixture_path,REPO,repeats=2)
        self.manifest['complete']=True
        self.raws=[]
        for order,(variant,repeat) in enumerate(m.schedule(['serial','dfly'],2),1):
            raw=self.raw(order,variant)
            path=self.root/f'run{order}'/'result.json';path.parent.mkdir()
            m.dump(path,raw);self.raws.append(path)
            self.manifest['runs'].append(dict(order=order,variant=variant,repeat=repeat,result=str(path)))
        self.path=self.root/'manifest.json';self.save()

    def save(self): m.dump(self.path,self.manifest)

    def raw(self,order,variant):
        v=next(v for v in self.manifest['variants'] if v['name']==variant)
        dfly=v['mode']=='dfly2'
        ids=[10,11,12]
        reply=dict(rank=0,request=0,name='chat_greedy_test',tokens=3,reason='length',
                   token_sha=hashlib.sha256(','.join(map(str,ids)).encode()).hexdigest()[:12],
                   rounds=2,drafted=10 if dfly else 0,accepted=0,min_rows=1,seconds=1.0,reply=ids)
        prefill=dict(rank=0,request=0,name='chat_greedy_test',prompt_tokens=3,cached=0,restored=0,prefill_s=.2)
        summary=dict(rank=0,requests=1,tokens=3,steps=2,shared_rounds=0,drafted=reply['drafted'],accepted=0,seconds=1.0)
        profile=dict(rounds=2,rows_a_round=6 if dfly else 1,step_ms=400,send_ms=0,enqueue_ms=1,
                     forward_ms=200 if dfly else 400,absorb_ms=0,logits_ms=0,sample_ms=0,
                     passes=2 if dfly else 0,streams_a_pass=1,pass_ms=200 if dfly else 0,prefills=1,prefill_ms=200)
        nodes=[]
        for rank in (0,1):
            nodes.append(dict(rank=rank,host='PRIVATE-HOST',gpu='NVIDIA GB10',binary_sha256='b'*64,
                              launcher_sha256=self.h.run_pair.launcher_identity(),image_content_sha256='c'*64,
                              target_config_sha256='d'*64,requests_sha256=self.fixture_hash,
                              pack_manifest_sha256='e'*64 if dfly else None,
                              config_identity=self.h.cluster_config.identity(self.configs[variant]),
                              rank_cache=dict(name=f'private-rank-{rank}',bytes=1000+rank)))
        return dict(run_id=f'{order:032x}',mode=v['mode'],success=True,execution=v['execution'],
                    started_utc=f'2026-11-26T00:00:{order*2:02d}Z',finished_utc=f'2026-11-26T00:00:{order*2+1:02d}Z',
                    preflight=dict(nodes=nodes),
                    states={str(rank):dict(Status='exited',ExitCode=0,OOMKilled=False,Error='') for rank in (0,1)},
                    cleanup={str(rank):dict(removed=True,id=f'{order*2+rank:064x}') for rank in (0,1)},
                    native=dict(complete=True,records=[prefill,reply,dict(rank=0,profile=profile)],replies=[reply],summary=summary))

    def mutate(self,index,fn):
        raw=m.load(self.raws[index]);fn(raw);m.dump(self.raws[index],raw)

    def analyze(self): return m.analyze_private(self.path,REPO)

    def test_complete_matrix_and_export_recompute(self):
        manifest,runs,payload,result=self.analyze()
        self.assertTrue(result['benchmark_eligible'])
        self.assertEqual(result['variants'][1]['estimated_decode_tokens_per_second'],dict(median=2.5,min=2.5,max=2.5))
        self.assertEqual(result['variants'][1]['requests'][0]['end_to_end_tokens_per_second']['median'],3)
        export=self.root/'export';export.mkdir();(export/'requests.json').write_bytes(payload)
        m.dump(export/'matrix.json',manifest);m.dump(export/'analysis.json',result)
        for entry,run in zip(manifest['runs'],runs):m.dump(export/entry['file'],run)
        self.assertEqual(m.verify_export(export,REPO),result)
        self.assertNotIn('PRIVATE-HOST',json.dumps(runs))
        self.assertNotIn('private-rank',json.dumps(runs))
        self.assertNotIn('config_identity',json.dumps(runs))
        self.assertNotIn('run_id',json.dumps(runs))

    def test_missing_repeat_rejected(self):
        self.manifest['runs'].pop();self.save()
        with self.assertRaisesRegex(ValueError,'missing repeats'):self.analyze()

    def test_known_issue_retains_receipts_but_disqualifies_complete_matrix(self):
        _,original_runs,_,_=self.analyze()
        issue=dict(schema='dfly2-known-issue-v1',code='initial_draft_depth_bypass',affected_variants=['dfly'])
        self.manifest['known_issues']=[issue];self.save()
        manifest,runs,payload,result=self.analyze()
        self.assertEqual(runs,original_runs)
        self.assertEqual(manifest['known_issues'],[issue])
        self.assertEqual(result['known_issues'],[issue])
        self.assertTrue(result['complete'])
        self.assertTrue(result['all_successful_token_arrays_and_finish_reasons_equal'])
        self.assertEqual(result['successful_runs'],4)
        self.assertFalse(result['benchmark_eligible'])
        self.assertNotIn('variants',result)
        self.assertNotIn('runs',result)
        export=self.root/'disqualified';export.mkdir();(export/'requests.json').write_bytes(payload)
        m.dump(export/'matrix.json',manifest);m.dump(export/'analysis.json',result)
        for entry,run in zip(manifest['runs'],runs):m.dump(export/entry['file'],run)
        self.assertEqual(m.verify_export(export,REPO),result)

    def test_known_issue_private_and_public_schema_mutations_refused(self):
        manifest,runs,_,_=self.analyze()
        issue=dict(schema='dfly2-known-issue-v1',code='initial_draft_depth_bypass',affected_variants=['dfly'])
        mutations=[None,[],{},[{}],[dict(issue,schema='future')],[dict(issue,code='unknown')],
                   [dict(issue,private_path='PRIVATE-PATH')],[dict(issue,affected_variants=[])],
                   [dict(issue,affected_variants='dfly')],[dict(issue,affected_variants=['missing'])],
                   [dict(issue,affected_variants=['dfly','dfly'])],
                   [dict(issue,affected_variants=[{}])],[issue,issue]]
        for value in mutations:
            with self.subTest(value=value):
                self.manifest['known_issues']=value;self.save()
                with self.assertRaisesRegex(ValueError,'known issue'):self.analyze()
                public=copy.deepcopy(manifest);public['known_issues']=value
                with self.assertRaisesRegex(ValueError,'known issue'):
                    m.compare_exported(public,runs,self.fixture['requests'],self.fixture_hash,self.h)

    def test_known_issue_does_not_waive_output_equivalence_checks(self):
        self.manifest['known_issues']=[dict(schema='dfly2-known-issue-v1',
            code='initial_draft_depth_bypass',affected_variants=['dfly'])]
        self.save()
        def change(r):
            reply=r['native']['records'][1];reply['reply'][0]=99
            reply['token_sha']=hashlib.sha256(','.join(map(str,reply['reply'])).encode()).hexdigest()[:12]
        self.mutate(1,change)
        with self.assertRaisesRegex(ValueError,'token arrays'):self.analyze()

    def test_fixed_zero_depth_gate_and_only_applicable_issue_exemption(self):
        manifest,runs,_,_=self.analyze()
        manifest['variants'][1]['execution']['fixed_draft_depth']=0
        for run in runs:
            if run['mode']=='dfly2':run['execution']['fixed_draft_depth']=0
        def compare(values):
            return m.compare_exported(manifest,values,self.fixture['requests'],self.fixture_hash,self.h)
        with self.assertRaisesRegex(ValueError,'zero depth performed draft passes'):compare(runs)
        issue=dict(schema='dfly2-known-issue-v1',code='initial_draft_depth_bypass',affected_variants=['serial'])
        manifest['known_issues']=[issue]
        with self.assertRaisesRegex(ValueError,'zero depth performed draft passes'):compare(runs)
        issue['affected_variants']=['dfly']
        self.assertFalse(compare(runs)['benchmark_eligible'])
        del manifest['known_issues']
        clean=copy.deepcopy(runs)
        for run in clean:
            if run['mode']=='dfly2':
                run['summary'].update(drafted=0,accepted=0)
                for record in run['records']:
                    if 'reply' in record:record.update(drafted=0,accepted=0)
                    if 'profile' in record:record['profile'].update(passes=0,pass_ms=0,rows_a_round=1)
        self.assertTrue(compare(clean)['benchmark_eligible'])
        for key,value in [('drafted',1),('accepted',1),('rounds',1)]:
            with self.subTest(key=key):
                bad=copy.deepcopy(clean)
                next(r for r in bad[1]['records'] if 'reply' in r)[key]=value
                with self.assertRaisesRegex(ValueError,'fixed zero depth'):compare(bad)

    def test_helpers_from_another_checkout_rejected(self):
        other=type('OtherHelper',(),{'__file__':str(self.root/'run_pair.py')})()
        with mock.patch.dict(m.sys.modules,{'run_pair':other}):
            with self.assertRaisesRegex(ValueError,'different checkout'):m.helpers(REPO)

    def test_optional_reference_norms_strict_bool_and_dfly_only(self):
        for name in ('serial','dfly'):
            v=next(v for v in self.manifest['variants'] if v['name']==name)
            value=copy.deepcopy(v['execution'])
            value['experimental'].pop('reference_norms',None)
            m.execution(value,v['mode'])
            value['experimental']['reference_norms']=False
            m.execution(value,v['mode'])
            value['experimental']['reference_norms']=1
            with self.assertRaisesRegex(ValueError,'nonboolean'):m.execution(value,v['mode'])
            value['experimental']['reference_norms']=True
            if name=='serial':
                with self.assertRaisesRegex(ValueError,'another mode'):m.execution(value,v['mode'])
            else:m.execution(value,v['mode'])

    def test_wrong_rotation_rejected(self):
        self.manifest['runs'][2]['repeat']=1;self.save()
        with self.assertRaisesRegex(ValueError,'rotated schedule'):self.analyze()

    def test_changed_binary_rejected(self):
        self.mutate(1,lambda r:r['preflight']['nodes'][0].update(binary_sha256='f'*64))
        with self.assertRaisesRegex(ValueError,'ranks disagree'):self.analyze()

    def test_config_change_outside_controls_rejected(self):
        path=self.root/'dfly-config.json';cfg=m.load(path);cfg['execution']['port']+=1;m.dump(path,cfg)
        with self.assertRaisesRegex(ValueError,'configuration changed'):self.analyze()

    def test_execution_override_rejected(self):
        self.mutate(1,lambda r:r['execution']['experimental'].update(device_feedback=True))
        with self.assertRaisesRegex(ValueError,'effective controls'):self.analyze()

    def test_changed_tokens_rejected_even_with_rehashed_token_list(self):
        def change(r):
            reply=r['native']['records'][1];reply['reply'][0]=99
            reply['token_sha']=hashlib.sha256(','.join(map(str,reply['reply'])).encode()).hexdigest()[:12]
        self.mutate(1,change)
        with self.assertRaisesRegex(ValueError,'token arrays'):self.analyze()

    def test_rank_oom_and_cleanup_rejected(self):
        self.mutate(1,lambda r:r['states']['1'].update(OOMKilled=True))
        with self.assertRaisesRegex(ValueError,'rank failure'):self.analyze()

    def test_failed_attempt_retained_without_aggregate_rates(self):
        self.mutate(1,lambda r:r.update(success=False,error='Refusal: private-host runtime failure'))
        self.manifest['runs']=self.manifest['runs'][:2];self.manifest['complete']=False;self.save()
        manifest,runs,_,result=self.analyze()
        self.assertEqual(len(runs),2)
        self.assertTrue(result['failure_retained'])
        self.assertFalse(result['benchmark_eligible'])
        self.assertNotIn('variants',result)
        self.assertNotIn('private-host',json.dumps(runs))

    def test_continuation_after_failure_rejected(self):
        self.mutate(1,lambda r:r.update(success=False,error='test failure'))
        self.manifest['complete']=False;self.save()
        with self.assertRaisesRegex(ValueError,'failed attempt'):self.analyze()

    def test_capture_purpose_never_emits_rates(self):
        self.manifest['purpose']='capture';self.save()
        result=self.analyze()[-1]
        self.assertFalse(result['benchmark_eligible'])
        self.assertNotIn('variants',result)

    def test_benchmark_cannot_hide_instrumentation(self):
        cfg=m.load(self.root/'dfly-config.json');cfg['experimental']['detailed_profile']=True
        m.dump(self.root/'dfly-config.json',cfg)
        self.configs['dfly']=self.h.cluster_config.validate(cfg)
        self.h.run_pair.configure(cfg)
        self.manifest['variants'][1]['execution']=self.h.run_pair.execution_settings('dfly2')
        for index in (1,2):m.dump(self.raws[index],self.raw(index+1,'dfly'))
        self.save()
        with self.assertRaisesRegex(ValueError,'cannot be a benchmark'):self.analyze()

    def test_unknown_nested_metric_text_refused(self):
        value=dict(schema='observed-depth-v1',decisions=0,calibrations=0,probes=0,
                   chosen_depth_counts=[0]*6,conditional_survival=[.5]*5,tried=[0]*5,accepted=[0]*5,
                   verify_costs=[dict(n=0,ewma_ms=0)]*6,draft_costs=[dict(n=0,ewma_ms=0)]*6)
        self.assertEqual(m.policy(value),value)
        value['verify_costs'][0]['ewma_ms']='private-path'
        with self.assertRaisesRegex(ValueError,'non-numerical'):m.policy(value)

    def test_nested_profile_round_trip_and_counts(self):
        value=dict(schema='dfly2-stream-profile-v1',passes=2,total_pass_ms=42.,host_readbacks=10,
                   depths=[0,0,0,0,0,2],scope=m.PROFILE_SCOPE,
                   categories={k:dict(calls=1,total_ms=.125) for k in m.PROFILE_CATEGORIES})
        self.assertEqual(m.profile(value),value)
        value['depths'][5]=1
        with self.assertRaisesRegex(ValueError,'pass counts'):m.profile(value)

    def test_free_draft_bound_uses_actual_pass_count_when_rounds_skip_drafting(self):
        raw=m.load(self.raws[1])
        p=raw['native']['records'][2]['profile']
        p.update(passes=1,pass_ms=600,rows_a_round=3.5)
        raw['native']['records'][1]['drafted']=5
        raw['native']['summary']['drafted']=5
        clean=m.sanitize_raw(raw,self.raws[1],self.fixture['requests'],self.h)
        row,_=m.summarize_clean(clean,self.fixture['requests'],self.fixture_hash,None,self.h)
        self.assertEqual(row['estimated_decode_tokens_per_second'],2.5)
        self.assertEqual(row['native_profile']['pass_ms'],600)
        self.assertEqual(row['native_profile']['draft_ms_per_round'],300)
        self.assertEqual(row['decode_tokens_per_second_if_draft_pass_were_free'],10)

    def test_public_verifier_rejects_unknown_private_field(self):
        manifest,runs,payload,_=self.analyze()
        runs[0]['nodes'][0]['hostname']='PRIVATE-HOST'
        with self.assertRaisesRegex(ValueError,'unknown public node'):
            m.compare_exported(manifest,runs,self.fixture['requests'],self.fixture_hash,self.h)

    def test_two_rank_detailed_profiles_retained_without_benchmark_claim(self):
        cfg=m.load(self.root/'dfly-config.json');cfg['experimental']['detailed_profile']=True
        m.dump(self.root/'dfly-config.json',cfg)
        self.configs['dfly']=self.h.cluster_config.validate(cfg)
        self.h.run_pair.configure(cfg)
        self.manifest['variants'][1]['execution']=self.h.run_pair.execution_settings('dfly2')
        self.manifest['purpose']='profile'
        details=dict(schema='dfly2-stream-profile-v1',passes=2,total_pass_ms=42.,host_readbacks=10,
                     depths=[0,0,0,0,0,2],scope=m.PROFILE_SCOPE,
                     categories={k:dict(calls=1,total_ms=.125) for k in m.PROFILE_CATEGORIES})
        for index in (1,2):
            raw=self.raw(index+1,'dfly')
            raw['native']['records'].append(dict(rank=0,dfly2_profile=details))
            m.dump(self.raws[index],raw)
            (self.raws[index].parent/'rank1.stdout.jsonl').write_text(json.dumps(dict(rank=1,dfly2_profile=details))+'\n')
        self.save()
        _,runs,_,result=self.analyze()
        self.assertFalse(result['benchmark_eligible'])
        self.assertNotIn('variants',result)
        self.assertEqual([r['rank'] for r in runs[1]['records'] if 'dfly2_profile' in r],[0,1])
        self.assertEqual(next(r['dfly2_profile'] for r in runs[1]['records'] if 'dfly2_profile' in r),details)
        (self.raws[1].parent/'rank1.stdout.jsonl').unlink()
        with self.assertRaisesRegex(ValueError,'both ranks'):self.analyze()

    def test_unique_attempt_config_paths_preserve_identity_checks(self):
        cfg=m.load(self.root/'dfly-config.json')
        for node in cfg['nodes']:node['config_path']='/etc/dfly2-two-sparks/dfly-r2.json'
        normalized=self.h.cluster_config.validate(cfg)
        path=self.root/'dfly-r2-config.json';m.dump(path,cfg)
        self.manifest['runs'][2]['config']=str(path)
        raw=m.load(self.raws[2])
        for n in raw['preflight']['nodes']:n['config_identity']=self.h.cluster_config.identity(normalized)
        m.dump(self.raws[2],raw);self.save()
        self.assertTrue(self.analyze()[-1]['benchmark_eligible'])
        cfg['experimental']['rdma_best']=True;m.dump(path,cfg)
        with self.assertRaisesRegex(ValueError,'attempt config effective controls'):self.analyze()


if __name__=='__main__':unittest.main()

#!/usr/bin/env python3
"""Offline consistency checks for the reviewed v5 build summary.

These checks do not authenticate execution or recreate private logs/binaries.
The complete source tree still requires prepare_experimental.py regeneration.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import prepare_experimental as source

FINAL_LANES = '9469e5b23e08237caad1477b6fb3250bc2a65b8025b4c52fcca06a9cf45c0f0e'
MOUNTED_LANES = '078f89d13e638584d6ba473b1ef98acf78a2121eaa3719fd917c26c296ebf442'
BINARIES = {'tf-dsv41-lanes','tf-dsv41-dfly-weights','tf-dsv41-dfly-head'}
RUNTIME_BUNDLES = ('sweep-v5-chat','benchmark-chat','benchmark-long','heldout')
PRIVATE_RECORDS = {'public_native_build_receipt','earlier_mounted_build_receipt','validation_receipt',
                   'final_image_build_receipt','host_tests_receipt','host_tests_log','cuda_info_log',
                   'diagnostic_receipt','feedback_report','normalization_report'}
HEX = re.compile(r'[0-9a-f]{64}\Z')
HUNK = re.compile(rb'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')


def require(ok, message):
    if not ok: raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_map(value, names=None):
    require(isinstance(value,dict) and value and
            all(isinstance(v,str) and HEX.fullmatch(v) for v in value.values()), 'invalid reported hash map')
    if names is not None: require(set(value)==set(names), 'reported hash inventory differs')


def integer(value, expected, label):
    require(type(value) is int and value==expected, label+' count/status differs')


def published_patch_files(root, records):
    """Replay only files fully introduced by public patches, without upstream IO.

    Existing upstream files cannot be reconstructed from partial context hunks.
    Their declared digests are not presented as independently recomputed here.
    """
    files={}; touched=set()
    for record in records:
        payload=(root/record['path']).read_bytes()
        chunks=re.split(rb'(?=^diff --git )',payload,flags=re.M)
        for chunk in chunks:
            if not chunk: continue
            lines=chunk.splitlines(keepends=True)
            match=re.fullmatch(rb'diff --git a/(\S+) b/(\S+)\r?\n?',lines[0])
            require(match and match[1]==match[2], 'unsupported patch path/rename')
            name=match[2].decode('utf-8'); touched.add(name)
            fresh=any(line.startswith(b'new file mode ') for line in lines)
            if not fresh and name not in files: continue
            old=[] if fresh else files[name].splitlines(keepends=True)
            result=[]; cursor=0; index=0
            while index<len(lines):
                header=HUNK.match(lines[index])
                if not header: index+=1; continue
                start=max(int(header[1])-1,0)
                require(cursor<=start<=len(old), 'patch hunk position differs')
                result.extend(old[cursor:start]); cursor=start
                before=cursor; added=0; index+=1
                while index<len(lines) and not lines[index].startswith(b'@@ '):
                    line=lines[index]; index+=1
                    require(line[:1] in (b' ',b'+',b'-'), 'unsupported patch line')
                    if line[:1] in (b' ',b'-'):
                        require(cursor<len(old) and old[cursor]==line[1:], 'patch context differs')
                        cursor+=1
                    if line[:1] in (b' ',b'+'):
                        result.append(line[1:]); added+=1
                require(cursor-before==int(header[2] or b'1') and added==int(header[4] or b'1'),
                        'patch hunk counts differ')
            result.extend(old[cursor:]); files[name]=b''.join(result)
    return files,touched


def verify(root):
    root=Path(root).resolve()
    path=root/'evidence/experimental-build-v5.json'
    if not path.exists(): return None
    require(path.is_file() and not path.is_symlink(), 'build summary must be a real file')
    report=source.load(path)
    require(set(report)=={'schema','version','status','platform','historical_validation_applies','source','public_image',
            'separate_source_host_tests','synthetic_gpu_checks','earlier_mounted_build','retained_v4_source',
            'private_record_sha256','limitations'}, 'unknown experimental build summary fields')
    require(report['schema']=='dfly2-experimental-build-validation-v1' and report['version']=='v5'
            and report['status']=='complete' and report['platform']=='linux/arm64'
            and report['historical_validation_applies'] is False, 'invalid v5 build summary scope')
    exp,base=source.locks(root/'experimental/native.lock.json',root)
    stated=report['source']
    require(stated['experimental_lock_path']=='experimental/native.lock.json'
            and stated['lock_hash_method']=='SHA256 of UTF-8 json.dumps(value, sort_keys=True)', 'source lock/hash method differs')
    require(stated['experimental_lock_sha256']==source.canonical_hash(exp)
            and stated['baseline_lock_sha256']==source.canonical_hash(base), 'source lock provenance mismatch')
    require(stated['patches']==exp['patches'] and stated['prepared_trees']==exp['prepared_trees'],
            'source patch/tree provenance mismatch')
    require(stated['upstream_revisions']=={name:base[name]['revision'] for name in ('recipe','tensorfold','inputs')},
            'upstream revision provenance mismatch')
    require(stated['build_inputs_sha256']==source.build_inputs(root), 'build input bytes changed')
    hash_map({k:stated[k] for k in ('source_files_sha256','baseline_source_files_sha256')})
    require(stated['source_files_sha256']!=stated['baseline_source_files_sha256'], 'experimental source manifest equals baseline')
    files,_=published_patch_files(root,base['integration_patches']+exp['patches'])
    _,changed=published_patch_files(root,exp['patches'])
    hash_map(stated['native_files_sha256'],changed)
    integer(stated['native_files_checked'],len(changed),'native source files')
    require(len(changed)==15, 'v5 native file inventory changed')
    reconstructed=0
    for name in changed & files.keys():
        require(hashlib.sha256(files[name]).hexdigest()==stated['native_files_sha256'][name],
                'reconstructed native source hash differs: '+name)
        reconstructed+=1
    require(reconstructed==7, 'reconstructable native file inventory changed')
    for name in ('preparation_exit_code','source_check_exit_code'): integer(stated[name],0,name)

    image=report['public_image']
    for field,key in [('base_image','base_image'),('zig_version','zig_version'),('zig_archive_sha256','zig_sha256')]:
        require(image[field]==base['toolchain'][key], 'public image toolchain differs')
    integer(image['build_exit_code'],0,'public image build')
    start=datetime.strptime(image['build_started_utc'],'%Y-%m-%dT%H:%M:%SZ')
    finish=datetime.strptime(image['build_finished_utc'],'%Y-%m-%dT%H:%M:%SZ')
    require(start<=finish and image['cuda_architecture']=='sm_121', 'image timing/architecture differs')
    hash_map({'image_content_sha256':image['image_content_sha256']})
    require(image['transferred_content_verified'] is True and image['staged_binaries_matched_on_ranks']==[0,1],
            'missing two-rank transfer/binary consistency')
    stores=image['store_specific_image_ids_by_rank']
    require(len(stores)==2 and [s['rank'] for s in stores]==[0,1]
            and all(set(s)=={'rank','image_id'} and re.fullmatch(r'sha256:[0-9a-f]{64}',s['image_id']) for s in stores),
            'invalid rank image IDs')
    hash_map(image['binaries_sha256'],BINARIES)
    earlier=report['earlier_mounted_build']['binaries_sha256']; hash_map(earlier,BINARIES)
    # Reviewed v5 identities distinguish two actual builds; they are not a
    # reproducible-build assertion or a cryptographic attestation of execution.
    require(image['binaries_sha256']['tf-dsv41-lanes']==FINAL_LANES
            and earlier['tf-dsv41-lanes']==MOUNTED_LANES, 'final/mounted lane binary provenance differs')
    cuda=image['cuda_info']
    require(cuda==dict(exit_code=0,device='NVIDIA GB10',compute_capability=121,
                      driver_cuda_api_version=13000,embedded_kernels=True), 'CUDA inspection differs')
    host=report['separate_source_host_tests']
    for key,value in [('exit_code',0),('tests_passed',594),('tests_total',594),('build_steps_succeeded',23),('build_steps_total',23)]:
        integer(host[key],value,'host '+key)
    require(host['gpu_access_requested'] is False, 'host test scope changed')
    gpu=report['synthetic_gpu_checks']
    require(gpu['source_sha256']==stated['native_files_sha256']['zig/kernels/cuda/dsv41_dfly2.cu'],
            'GPU diagnostic source differs from native build')
    require(gpu['device']=='NVIDIA GB10' and gpu['artifact_kind']=='cubin' and gpu['compiled_for_checks'] is True
            and gpu['compile_flags']==['-cubin','-O3','--std=c++17','--fmad=false','--ftz=false','-arch=sm_121'],
            'GPU diagnostic compilation scope differs')
    hash_map({'artifact_sha256':gpu['artifact_sha256']})
    for group,count in [('feedback',13),('reference_norms',8)]:
        for field,value in [('exit_code',0),('groups_passed',count),('groups_total',count)]:
            integer(gpu[group][field],value,group+' '+field)
    require(gpu['reference_norms']['primitive_helper_sha256']==digest(root/'tools/dfly2/test_primitives.py'),
            'primitive helper bytes changed')
    old=report['retained_v4_source']
    old_lock,_=source.locks(root/'experimental/v4.native.lock.json',root)
    require(old['lock_path']=='experimental/v4.native.lock.json' and old['lock_sha256']==source.canonical_hash(old_lock)
            and old['prepared_trees']==old_lock['prepared_trees'] and old['prepared_trees']!=exp['prepared_trees']
            and old['depth_sweep_qualified'] is False and old['disqualification']=='initial_draft_depth_bypass',
            'retained v4 source/disqualification provenance differs')
    hash_map(report['private_record_sha256'],PRIVATE_RECORDS)
    require(report['private_record_sha256']['public_native_build_receipt']!=
            report['private_record_sha256']['earlier_mounted_build_receipt'], 'distinct build receipt identities collapsed')
    require(isinstance(report['limitations'],list) and len(report['limitations'])==4
            and all(isinstance(x,str) and x for x in report['limitations']), 'missing build evidence limitations')
    checked=0
    for name in RUNTIME_BUNDLES:
        folder=root/'evidence/experimental'/name
        if not folder.exists(): continue
        manifest=source.load(folder/'matrix.json')
        for entry in manifest['runs']:
            require(Path(entry['file']).name==entry['file'], 'unsafe runtime receipt path')
            run=source.load(folder/entry['file'])
            for node in run['nodes']:
                require(node['binary_sha256']==image['binaries_sha256']['tf-dsv41-lanes']
                        and node['image_content_sha256']==image['image_content_sha256'],
                        'v5 runtime identity differs from final public build')
            checked+=1
    return {'native_files_reported':len(changed),'native_files_reconstructed':reconstructed,
            'runtime_receipts_bound':checked,'execution_authenticated':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    print(json.dumps(verify(parser.parse_args().repo),sort_keys=True))

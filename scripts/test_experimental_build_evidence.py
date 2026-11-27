"""Mutate a copy of the reviewed public build evidence; no GPU execution."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import verify_experimental_build as v

REPO=Path(__file__).resolve().parents[1]


class ExperimentalBuildEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='dfly-build-evidence-')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        paths=['evidence/experimental-build-v5.json','upstream.lock.json',
               'experimental/native.lock.json','experimental/v4.native.lock.json',
               'tools/dfly2/test_primitives.py',*v.source.BUILD_INPUTS]
        for lock_name in ('experimental/native.lock.json','experimental/v4.native.lock.json'):
            paths.extend(p['path'] for p in v.source.load(REPO/lock_name)['patches'])
        paths.extend(p['path'] for p in v.source.load(REPO/'upstream.lock.json')['integration_patches'])
        for name in set(paths):
            dest=self.root/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(REPO/name,dest)
        self.path=self.root/'evidence/experimental-build-v5.json'
        self.original=v.source.load(self.path)

    def write(self,value):
        self.path.write_text(json.dumps(value)+'\n',encoding='utf-8')

    def test_actual_public_summary_reconstructs_source_without_execution_claim(self):
        self.assertEqual(v.verify(self.root),dict(native_files_reported=15,native_files_reconstructed=7,
                                                runtime_receipts_bound=0,execution_authenticated=False))

    def test_patch_bytes_changed_without_lock_update_refused(self):
        patch=self.root/'experimental/0002-initial-draft-depth.patch'
        patch.write_bytes(patch.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'SHA256'):v.verify(self.root)

    def test_rehashed_kernel_claim_cannot_override_reconstructed_source(self):
        value=copy.deepcopy(self.original)
        value['source']['native_files_sha256']['zig/kernels/cuda/dsv41_dfly2.cu']='a'*64
        value['synthetic_gpu_checks']['source_sha256']='a'*64
        self.write(value)
        with self.assertRaisesRegex(ValueError,'reconstructed native source'):v.verify(self.root)

    def test_build_input_and_primitive_helper_bytes_are_independently_checked(self):
        for name,message in [('build/Dockerfile','build input bytes'),
                             ('tools/dfly2/test_primitives.py','primitive helper bytes')]:
            with self.subTest(name=name):
                path=self.root/name;original=path.read_bytes()
                path.write_bytes(original+b'\n')
                with self.assertRaisesRegex(ValueError,message):v.verify(self.root)
                path.write_bytes(original)

    def test_mounted_build_cannot_impersonate_final_image(self):
        value=copy.deepcopy(self.original)
        value['public_image']['binaries_sha256']=copy.deepcopy(value['earlier_mounted_build']['binaries_sha256'])
        self.write(value)
        with self.assertRaisesRegex(ValueError,'final/mounted lane binary provenance'):v.verify(self.root)

    def test_source_provenance_counts_and_v4_disqualification_mutations_refused(self):
        mutations=[('source','experimental_lock_sha256','a'*64),
                   ('source','prepared_trees',self.original['retained_v4_source']['prepared_trees']),
                   ('source','native_files_checked',14),
                   ('source','native_files_checked',True),
                   ('separate_source_host_tests','tests_passed',593),
                   ('retained_v4_source','depth_sweep_qualified',True),
                   ('retained_v4_source','lock_sha256','a'*64)]
        for section,key,bad in mutations:
            with self.subTest(section=section,key=key):
                value=copy.deepcopy(self.original);value[section][key]=bad;self.write(value)
                with self.assertRaises(ValueError):v.verify(self.root)

    def test_final_runtime_receipts_bound_to_binary_and_image_content(self):
        folder=self.root/'evidence/experimental/sweep-v5-chat';folder.mkdir(parents=True)
        (folder/'matrix.json').write_text(json.dumps(dict(runs=[dict(file='run.json')])))
        node=dict(binary_sha256=v.FINAL_LANES,image_content_sha256=self.original['public_image']['image_content_sha256'])
        receipt=folder/'run.json'
        receipt.write_text(json.dumps(dict(nodes=[node])))
        self.assertEqual(v.verify(self.root)['runtime_receipts_bound'],1)
        for key,bad in [('binary_sha256',v.MOUNTED_LANES),('image_content_sha256','a'*64)]:
            with self.subTest(key=key):
                changed=dict(node);changed[key]=bad
                receipt.write_text(json.dumps(dict(nodes=[changed])))
                with self.assertRaisesRegex(ValueError,'runtime identity'):v.verify(self.root)


if __name__=='__main__':unittest.main()

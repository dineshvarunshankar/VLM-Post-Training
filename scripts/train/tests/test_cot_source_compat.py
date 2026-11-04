"""Known historical generator profiles must preserve every semantic input."""
from copy import deepcopy
import importlib
import unittest

HISTORICAL = {
    'source_cache.py':'44554c141b47c56ccdb4a4a0c87fb722d6b772e12b98191f66821b11727be1ba',
    'medical.py':'79a73834b7112ffbf069f136ebcbb6f706f7d784ac74bdfabe50501bb23be0ad',
    'preflight.py':'5e0d0b8156a94e94c605122567f62fd9fc04e51b2bea9601259058bfb0f40c90',
    'evidence_media.py':'3515ae5790379f73031aa6a3e1bf5458e281098349d32348f8b800099d0398e5',
    'gemini_media.py':'85cd8afffff481fade48120f0c35174da02116e6a9acbfc358ab420155e85ef0',
    'quality.py':'8458fbb401bebdfbd5d256fa780d73ad532b798f8a16b5e3b49c017e8049bd8d',
    'generate.py':'95bdbfb6970b8f2e21ec7b2208a97b4f206d52bf41f1af945e459c975dcf6b62',
    'preview.py':'f6c6f0b6cc68d4679a8f0f7ca3c91cddfc4fbeaff02671d2759fd11b0487e5fe',
    'transport_media.py':'c491890308ff0bfae94661996b4f4e9d02e8ceb99d7a8c5b07ef235f3595f0f4',
    'request_runtime.py':'7205b8b4a202a2f3d035bcc589d404ce95cb3cfc9781ecbedc195cad16ce5b25',
}


class SourceCompatibilityTests(unittest.TestCase):
    def module(self):
        try: result=importlib.import_module('scripts.train.shared.cot_source_compat')
        except ModuleNotFoundError: result=None
        self.assertIsNotNone(result,'Audited generator compatibility is missing')
        return result

    def test_exact_known_profile_allows_only_scheduler_change(self):
        module=self.module()
        previous={'implementation':HISTORICAL,'assembled_input_sha256':'a'*64,'prompt_sha256':'b'*64}
        current=deepcopy(previous); current['implementation']=module.current_implementation()
        self.assertTrue(module.compatible_inputs(previous,current))
        self.assertTrue(module.known_implementation(HISTORICAL))
        for name in ('prompt_sha256','assembled_input_sha256'):
            altered=deepcopy(previous); altered[name]='c'*64
            self.assertFalse(module.compatible_inputs(altered,current))
        unknown=deepcopy(HISTORICAL); unknown['medical.py']='0'*64
        self.assertFalse(module.known_implementation(unknown))
        altered=deepcopy(previous); altered['implementation']=unknown
        self.assertFalse(module.compatible_inputs(altered,current))

if __name__=='__main__': unittest.main()

class RestoredSchedulerProfilesTests(unittest.TestCase):
    def test_registered_scheduler_profiles_keep_semantic_checks(self):
        from scripts.train.shared import cot_source_compat as compat
        from scripts.train.shared.data import read_json
        from scripts.train.shared.cot_data import fingerprint
        profiles=read_json(compat.PROFILE_PATH)['profiles']
        for key in ('4dd57aee9e22ce5c4b22fe22a2d288beea4922d8f9886570fa917612fc279fc3','d00af033eabb93fbcdc8f6a1146432f499beb5b9db8d5f93ab164ab93bdf1629','a0bb630eb60eada6a60b71018253a4846518cf4c92e8e3a0f9da426e61ce09f7'):
            with self.subTest(profile=key):
                self.assertIn(key,profiles)
                implementation=profiles[key]['implementation']
                self.assertEqual(fingerprint(implementation),key)
                old={'implementation':implementation,'video_sha256':'v','annotation_sha256':{'labels.json':'a'},'assembled_input_sha256':'i','prompt_sha256':'p'}
                current=dict(old,implementation=compat.current_implementation())
                self.assertTrue(compat.known_implementation(implementation))
                self.assertTrue(compat.compatible_inputs(old,current))
                for field in ('video_sha256','annotation_sha256','assembled_input_sha256','prompt_sha256'):
                    bad=dict(old);bad[field]='changed'
                    self.assertFalse(compat.compatible_inputs(bad,current))
                bad_impl=dict(implementation);bad_impl['medical.py']='0'*64
                self.assertFalse(compat.known_implementation(bad_impl))
                self.assertFalse(compat.compatible_inputs(dict(old,implementation=bad_impl),current))

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

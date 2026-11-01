from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

from scripts.train.shared.config import evaluation_settings
from scripts.train.shared.run_config import load_run, compile_run
from scripts.train.shared.runtime import save_run_metadata

ROOT = Path(__file__).resolve().parents[1]
GREEDY = dict(do_sample=False, temperature=0.0, top_p=1.0, top_k=0,
              min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0)


class EvaluationDecodingTest(unittest.TestCase):
    def test_saved_policy_records_effective_evaluation_decoding(self):
        sampling = dict(GREEDY, do_sample=True, temperature=0.7,
                        presence_penalty=1.5)
        for override in (None, GREEDY):
            with self.subTest(override=override), TemporaryDirectory() as directory:
                root = Path(directory)
                for filename in ('catalog.yaml', 'prompts.yaml', 'train.jsonl'):
                    (root / filename).write_text('')
                args = SimpleNamespace(run_location='abci', resume=None,
                                       catalog=root / 'catalog.yaml',
                                       prompts=root / 'prompts.yaml', split_dir=root)
                config = {'evaluation': {'max_new_tokens': 1024}}
                if override is not None:
                    config['evaluation']['generation'] = deepcopy(override)
                output = root / 'output'
                with patch.dict('os.environ', {'TRIAGE_RUN_CONFIG': ''}):
                    save_run_metadata(args, output, {'generation': sampling},
                                      config, {}, {}, {'train': []})
                self.assertEqual(json.loads((output / 'generation_policy.json').read_text()),
                                 sampling if override is None else GREEDY)
                self.assertEqual(json.loads((output / 'run.json').read_text())['model']['generation'],
                                 sampling)

    def test_override_does_not_change_model_or_sampling_defaults(self):
        profile = {'generation': dict(GREEDY, do_sample=True, temperature=0.7,
                                      presence_penalty=1.5)}
        before = deepcopy(profile)
        config = {'evaluation': {'max_new_tokens': 1024, 'generation': GREEDY}}
        result = evaluation_settings(config, profile)
        self.assertEqual(result['generation'], GREEDY)
        result['generation']['temperature'] = 0.3
        self.assertEqual(profile, before)
        self.assertEqual(config['evaluation']['generation'], GREEDY)
        self.assertEqual(evaluation_settings({'evaluation': {'max_new_tokens': 1024}}, profile)['generation'], before['generation'])

    def test_run_contract_accepts_explicit_evaluation_generation(self):
        run = load_run(ROOT / 'configs/runs/gemma12b_sft_stabilized_medical.yaml')
        run['shared']['evaluation']['generation'] = GREEDY
        plan = compile_run(run)
        self.assertEqual(evaluation_settings(plan['experiment'], plan['run']['model_settings'])['generation'], GREEDY)


if __name__ == '__main__':
    unittest.main()

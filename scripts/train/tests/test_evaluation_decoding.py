from copy import deepcopy
from pathlib import Path
import unittest

from scripts.train.shared.config import evaluation_settings
from scripts.train.shared.run_config import load_run, compile_run

ROOT = Path(__file__).resolve().parents[1]
GREEDY = dict(do_sample=False, temperature=0.0, top_p=1.0, top_k=0,
              min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0)


class EvaluationDecodingTest(unittest.TestCase):
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

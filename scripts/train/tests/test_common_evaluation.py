import unittest
from scripts.train.shared.config import select_medical_examples
from scripts.train.shared import metrics

POLICY = {'sources': ['native', 'stabilized'], 'aggregation': 'equal_source_macro_f1'}
DEFINITION = {'subcategories': {'observables': {'cues': [{'name': 'cue'}], 'selection_rule': 'multi_select_or_none'}}}
CATALOG = {'categories': {'medical': DEFINITION}}

def row(source, prediction=1):
    return {'reference': {'variant': source, 'category': 'medical'},
            'answer': '{"cue":1}', 'prediction': '{"cue":%d}' % prediction}

class CommonEvaluationTests(unittest.TestCase):
    def test_training_sources_stay_independent(self):
        config = {'dataset': {'sources': {'native': False, 'stabilized': True}},
                  'mixture': {'medical_vqa': {'excluded_categories': [], 'context_sources': ['native']}},
                  'evaluation': POLICY}
        splits = {s: [row('native'), row('stabilized')] for s in ('train', 'validation', 'test')}
        selected = select_medical_examples(splits, config)
        self.assertEqual(len(selected['train']), 1)
        self.assertEqual(len(selected['validation']), 2)
        self.assertEqual(len(selected['test']), 2)
        config['evaluation'] = {}
        self.assertEqual(len(select_medical_examples(splits, config)['validation']), 1)

    def test_equal_view_weight_not_pooled_example_counts(self):
        scorer = getattr(metrics, 'score_medical_evaluation', None)
        self.assertIsNotNone(scorer)
        rows = [row('native')] * 3 + [row('stabilized', 0)]
        report = scorer(rows, CATALOG, POLICY)
        self.assertEqual(report['metrics']['macro_f1'], .5)
        self.assertAlmostEqual(report['metrics']['pooled_macro_f1'], 6/7)
        self.assertEqual(report['medical_sources']['native']['metrics']['macro_f1'], 1)
        self.assertEqual(report['medical_sources']['stabilized']['metrics']['macro_f1'], 0)
        self.assertEqual(scorer(rows, CATALOG, {}), metrics.score_answers(rows, CATALOG))

    def test_missing_view_cannot_silently_change_score(self):
        scorer = getattr(metrics, 'score_medical_evaluation', None)
        self.assertIsNotNone(scorer)
        with self.assertRaisesRegex(ValueError, 'sources'):
            scorer([row('native')], CATALOG, POLICY)

    def test_loader_reads_both_held_out_views_only(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from scripts.train.shared.data import load_splits
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ('train', 'validation', 'test'):
                (root / (split + '.jsonl')).write_text('\n'.join(json.dumps(
                    {'variant': source, 'clip': split + source, 'category': 'medical'})
                    for source in ('native', 'stabilized')))
            loaded = []
            def load_clip(path, *args, **kwargs):
                loaded.append(path.name)
                return {}, {'categories': {'medical': {'cue': 1}}}, path.name
            with patch('scripts.train.shared.data.read_yaml', return_value=CATALOG), \
                 patch('scripts.train.shared.data.frozen_split_membership', return_value=None), \
                 patch('scripts.train.shared.data.reject_pairs', return_value=set()), \
                 patch('scripts.train.shared.data.clip_path', side_effect=lambda root, variant, clip: root / clip), \
                 patch('scripts.train.shared.data.load_clip', side_effect=load_clip), \
                 patch('scripts.train.shared.data.read_json', return_value={'view_type': 'aerial'}), \
                 patch('scripts.train.shared.data.render_prompt', return_value=('system', 'question')):
                splits = load_splits(root, root, 'catalog', 'prompts',
                                     sources={'native': False, 'stabilized': True},
                                     evaluation_sources=['native', 'stabilized'])
            self.assertEqual([len(splits[s]) for s in ('train', 'validation', 'test')], [1, 2, 2])
            self.assertNotIn('trainnative', loaded)
            self.assertIn('validationnative', loaded)

    def test_config_rejects_incomplete_common_policy(self):
        from copy import deepcopy
        from pathlib import Path
        from scripts.train.shared.run_config import load_run, compile_run
        run = load_run(Path('scripts/train/configs/runs/gemma12b_sft_stabilized_medical.yaml'))
        compile_run(run)
        for change in ({'sources': ['stabilized'], 'aggregation': 'equal_source_macro_f1'},
                       {'sources': ['native', 'stabilized']}, {'aggregation': 'pooled'}):
            changed = deepcopy(run)
            changed['shared']['evaluation'] = {'max_new_tokens': 1024, **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                compile_run(changed)

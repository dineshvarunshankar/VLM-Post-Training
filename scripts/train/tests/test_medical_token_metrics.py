"""Label diagnostics must exclude prompts, reasoning, and auxiliary answers."""
import importlib
import math
from types import SimpleNamespace
import unittest

import torch


class MedicalTokenMetricsTests(unittest.TestCase):
    def module(self):
        try:
            module = importlib.import_module('scripts.train.shared.medical_token_metrics')
        except ModuleNotFoundError:
            module = None
        self.assertIsNotNone(module, 'Medical value diagnostics are not implemented')
        return module

    def test_value_spans_exclude_digits_in_keys_and_reasoning(self):
        module = self.module()
        answer = '{"label1":0,"injury":1}'
        suffix = 'Reasoning says {"injury":0}. </think>\n' + answer + '<end>'
        # Character tokens make the expected mask independent of a tokenizer.
        offsets = [(i, i + 1) for i in range(len(suffix))]
        groups = module.target_groups(suffix, answer, offsets, medical=True)
        self.assertEqual(''.join(c for c, g in zip(suffix, groups) if g == 1), '01')
        self.assertEqual(''.join(c for c, g in zip(suffix, groups) if g == 2), '{"label1":,"injury":}')
        self.assertEqual(module.target_groups(suffix, answer, offsets, medical=False), [0] * len(suffix))

    def test_shifted_logits_and_fused_outputs_produce_same_counts(self):
        module = self.module()
        labels = torch.tensor([[-100, -100, 1, 0], [-100, 1, 0, -100]])
        groups = torch.tensor([[0, 0, 1, 2], [0, 1, 2, 0]])
        logits = torch.zeros(2, 4, 2)
        # Row 0: value wrong with p=.25, syntax right with p=.75.
        logits[0, 1] = torch.tensor([math.log(.75), math.log(.25)])
        logits[0, 2] = torch.tensor([math.log(.75), math.log(.25)])
        expected = torch.tensor([[math.log(4), 0, 1, -math.log(.75), 1, 1],
                                 [math.log(2), 0, 1, math.log(2), 1, 1]], dtype=torch.float64)
        actual = module.batch_statistics(SimpleNamespace(logits=logits), labels, groups)
        torch.testing.assert_close(actual, expected)
        fused = SimpleNamespace(log_probs=torch.tensor([[0., math.log(.25), math.log(.75)],
                                                        [-math.log(2), -math.log(2), 0.]]),
                                is_top1=torch.tensor([[False, False, True], [False, True, False]]),
                                label_mask=labels[:, 1:] != -100)
        torch.testing.assert_close(module.batch_statistics(fused, labels, groups), expected)
        metrics = module.summarize(actual.sum(0), 'eval')
        self.assertAlmostEqual(metrics['eval_medical_value_loss'], math.log(8) / 2)
        self.assertEqual(metrics['eval_medical_value_accuracy'], 0.)
        self.assertEqual(metrics['eval_medical_format_accuracy'], 1.)
        self.assertEqual(metrics['eval_medical_value_tokens'], 2)

    def test_empty_group_reports_coverage_not_fake_perfect_accuracy(self):
        metrics = self.module().summarize(torch.zeros(6), 'test')
        self.assertEqual(metrics['test_medical_value_tokens'], 0)
        self.assertNotIn('test_medical_value_loss', metrics)
        self.assertNotIn('test_medical_value_accuracy', metrics)


if __name__ == '__main__':
    unittest.main()

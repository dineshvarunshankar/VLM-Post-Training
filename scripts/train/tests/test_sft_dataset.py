"""Exercise mixed CoT provenance through the actual Arrow/TRL boundary."""
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from datasets import Dataset
from scripts.train.tests.test_cot_data import fixture, CATALOG, REPO
from scripts.train.shared.data import read_yaml
from scripts.train.shared.run_config import compile_run, load_run
from scripts.train.shared.cot_data import build_cot_splits
from scripts.train.shared.subject_check import _row
from scripts.train.shared.sft_dataset import build_sft_dataset as convert


class SFTDatasetTests(unittest.TestCase):
    def rows(self, root):
        _, medical = fixture(root, variant='stabilized')
        config = compile_run(load_run(REPO/'scripts/train/configs/cot/qwen3.5-9b.yaml'))['experiment']
        splits, _ = build_cot_splits({'train': medical, 'validation': [], 'test': []},
            media_root=root, catalog=read_yaml(CATALOG), config=config)
        rows = splits['train']
        rows.append(_row({'subject_is_human_or_human_like_manikin': 0, 'subject_is_not_human_or_manikin': 1},
                         {'id': 'negative', 'category': 'subject_check'}, image='/negative.png'))
        return rows

    def test_real_mixed_rows_preserve_targets_media_and_original_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.rows(Path(directory))
            original = deepcopy(rows)
            dataset = convert(rows)
            self.assertIsInstance(dataset, Dataset)
            self.assertEqual(rows, original)
            self.assertEqual(len(dataset), 4)
            for index, row in enumerate(original):
                for key in ('task', 'answer', 'reasoning', 'video', 'image'):
                    self.assertEqual(dataset[index].get(key), row.get(key))
                identity = dataset[index]['target_identity']
                self.assertEqual(json.loads(identity) if isinstance(row['target_identity'], dict) else identity,
                                 row['target_identity'])
            self.assertEqual(dataset[-1]['image'], '/negative.png')
            subset = dataset.select([2, 0])
            self.assertEqual([x['task'] for x in subset], [original[2]['task'], original[0]['task']])
            self.assertNotEqual(dataset._fingerprint, subset._fingerprint)

    def test_real_trl_accepts_and_consumes_all_mixed_rows(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast, GPT2Config, GPT2LMHeadModel
        from trl import SFTConfig, SFTTrainer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # The tiny CPU model tests TRL's dataset contract, not video encoding.
            # Media preservation is checked separately against the full dataset.
            dataset = convert(self.rows(root)).remove_columns('image')
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
                {'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, 'a': 3}, unk_token='[UNK]')),
                unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
            seen = []
            def collate(rows):
                seen.extend(rows)
                tokens = torch.tensor([[3, 3, 2]] * len(rows))
                return {'input_ids': tokens, 'attention_mask': torch.ones_like(tokens), 'labels': tokens.clone()}
            model = GPT2LMHeadModel(GPT2Config(vocab_size=4, n_layer=1, n_head=1,
                n_embd=8, n_positions=8, bos_token_id=2, eos_token_id=2, pad_token_id=1))
            args = SFTConfig(output_dir=str(root/'output'), use_cpu=True, bf16=False, fp16=False,
                max_steps=1, per_device_train_batch_size=4, gradient_checkpointing=False, report_to='none',
                save_strategy='no', disable_tqdm=True, remove_unused_columns=False,
                dataset_kwargs={'skip_prepare_dataset': True})
            trainer = SFTTrainer(model=model, args=args, processing_class=tokenizer,
                                 data_collator=collate, train_dataset=dataset)
            trainer.train()
            self.assertEqual(trainer.state.global_step, 1)
            self.assertEqual(len(seen), 4)
            self.assertEqual(sorted(x['task'] for x in seen), sorted(x['task'] for x in dataset))


if __name__ == '__main__':
    unittest.main()

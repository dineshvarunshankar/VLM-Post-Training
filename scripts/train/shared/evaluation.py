"""Shared vLLM medical evaluation for SFT and RL."""

import time
from pathlib import Path
from accelerate.utils import gather_object

from scripts.train.shared.data import atomic_json
from scripts.train.shared.decoding import answer_schema
from scripts.train.shared.metrics import score_answers
from scripts.train.rl.video import prepare_video_prompt
from scripts.train.rl.rollout.features import encode_video_features


class MedicalVLLMEvaluator:
    def __init__(self, client, feature_config, processor, catalog, generation, run_id, max_tokens=1024):
        self.client, self.feature_config, self.processor = client, feature_config, processor
        self.catalog, self.generation, self.run_id = catalog, dict(generation), run_id
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError('evaluation.max_new_tokens must be a positive integer')
        self.max_tokens = max_tokens

    def __call__(self, trainer, dataset, prefix):
        if dataset is None or not len(dataset):
            raise ValueError('Medical validation cannot be empty')
        started = time.monotonic()
        trainer.vllm_generation.sync_weights()
        trainer._last_loaded_step = trainer.state.global_step
        trainer.accelerator.wait_for_everyone()
        version = f'{self.run_id}:{prefix}:step-{trainer.state.global_step}'
        records = []
        rank, world = trainer.accelerator.process_index, trainer.accelerator.num_processes
        for index in range(rank, len(dataset), world):
            row = dataset[index]
            packet = prepare_video_prompt(trainer.video_collator, [row])
            if int(packet['attention_mask'].sum()) + self.max_tokens > trainer.video_collator.max_length:
                raise ValueError('Validation prompt and answer exceed the context budget')
            encoded = encode_video_features(packet, policy_version=version, **self.feature_config)
            category = row['reference']['category']
            settings = self.generation
            output = self.client.generate(
                [encoded], policy_version=version,
                temperature=settings['temperature'] if settings.get('do_sample', True) else 0,
                top_p=settings['top_p'], top_k=settings['top_k'],
                repetition_penalty=settings['repetition_penalty'],
                presence_penalty=settings.get('presence_penalty', 0),
                max_tokens=self.max_tokens, seed=trainer.args.seed + index,
                schemas=[answer_schema(self.catalog['categories'][category])],
            )
            prediction = self.processor.tokenizer.decode(output['completion_ids'][0], skip_special_tokens=True)
            records.append(dict(index=index, reference=row['reference'], answer=row['answer'], prediction=prediction))
        records = sorted(gather_object(records), key=lambda item: item['index'])
        if [row['index'] for row in records] != list(range(len(dataset))):
            raise ValueError('Every validation example must be evaluated exactly once')
        report = score_answers(records, self.catalog)
        report['backend'] = 'vllm'
        report['split'] = 'test' if prefix == 'test' else 'validation'
        report['decoding'] = dict(self.generation, max_new_tokens=self.max_tokens, constraint='category_json_schema')
        metrics = {f'{prefix}_{key}': value for key, value in report['metrics'].items()}
        metrics[f'{prefix}_runtime'] = time.monotonic() - started
        if trainer.is_world_process_zero():
            atomic_json(Path(trainer.args.output_dir) / f'{prefix}-step-{trainer.state.global_step}.json', report)
        trainer.log(metrics)
        if prefix != 'test':
            trainer.control = trainer.callback_handler.on_evaluate(trainer.args, trainer.state, trainer.control, metrics)
        return metrics

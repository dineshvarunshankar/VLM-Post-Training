"""Shared vLLM evaluation with medical-only checkpoint selection."""

import time
from pathlib import Path
from accelerate.utils import gather_object

from scripts.train.shared.data import atomic_json
from scripts.train.shared.decoding import answer_schema
from scripts.train.shared.metrics import score_answers
from scripts.train.shared.auxiliary_evaluation import AuxiliaryEvaluation, score_auxiliary
from scripts.train.rl.video import prepare_video_prompt
from scripts.train.rl.rollout.features import encode_video_features


class MedicalVLLMEvaluator:
    def __init__(self, client, feature_config, processor, catalog, generation, run_id, max_tokens=1024, *, experiment=None, medical_templates=None):
        self.client, self.feature_config, self.processor = client, feature_config, processor
        self.catalog, self.generation, self.run_id = catalog, dict(generation), run_id
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError('evaluation.max_new_tokens must be a positive integer')
        self.max_tokens = max_tokens
        self.auxiliary = AuxiliaryEvaluation(experiment, medical_templates, catalog)

    def predict(self, trainer, row, version, seed, schema):
        packet = prepare_video_prompt(trainer.video_collator, [row])
        if int(packet['attention_mask'].sum()) + self.max_tokens > trainer.video_collator.max_length:
            raise ValueError('Validation prompt and answer exceed the context budget')
        encoded = encode_video_features(packet, policy_version=version, **self.feature_config)
        settings = self.generation
        output = self.client.generate(
            [encoded], policy_version=version,
            temperature=settings['temperature'] if settings.get('do_sample', True) else 0,
            top_p=settings['top_p'], top_k=settings['top_k'],
            repetition_penalty=settings['repetition_penalty'],
            presence_penalty=settings.get('presence_penalty', 0),
            min_p=settings['min_p'],
            max_tokens=self.max_tokens, seed=seed,
            schemas=[schema],
        )
        prediction = self.processor.tokenizer.decode(output['completion_ids'][0], skip_special_tokens=True)
        return prediction

    def __call__(self, trainer, dataset, prefix):
        if dataset is None or not len(dataset):
            raise ValueError('Medical validation cannot be empty')
        started = time.monotonic()
        auxiliary_rows, auxiliary_preparation = self.auxiliary.prepare(dataset, prefix)
        trainer.vllm_generation.sync_weights()
        trainer._last_loaded_step = trainer.state.global_step
        trainer.accelerator.wait_for_everyone()
        version = f'{self.run_id}:{prefix}:step-{trainer.state.global_step}'
        records = []
        rank, world = trainer.accelerator.process_index, trainer.accelerator.num_processes
        for index in range(rank, len(dataset), world):
            row = dataset[index]
            prediction = self.predict(trainer, row, version, trainer.args.seed + index,
                                      answer_schema(self.catalog['categories'][row['reference']['category']]))
            records.append(dict(index=index, reference=row['reference'], answer=row['answer'], prediction=prediction))
        records = sorted(gather_object(records), key=lambda item: item['index'])
        if [row['index'] for row in records] != list(range(len(dataset))):
            raise ValueError('Every validation example must be evaluated exactly once')
        report = score_answers(records, self.catalog)
        auxiliary_records = []
        for index in range(rank, len(auxiliary_rows), world):
            row = auxiliary_rows[index]
            prediction = self.predict(trainer, row, version, trainer.args.seed + len(dataset) + index,
                                      row['reward_context']['schema'])
            auxiliary_records.append(dict(index=index, reference=row['reference'], prediction=prediction,
                                          reward_context=row['reward_context']))
        auxiliary_records = sorted(gather_object(auxiliary_records), key=lambda item: item['index'])
        if [row['index'] for row in auxiliary_records] != list(range(len(auxiliary_rows))):
            raise ValueError('Every auxiliary evaluation example must be evaluated exactly once')
        report['auxiliary'] = score_auxiliary(auxiliary_records, self.auxiliary.scoring)
        report['auxiliary']['preparation'] = auxiliary_preparation
        report['backend'] = 'vllm'
        report['split'] = 'test' if prefix == 'test' else 'validation'
        report['decoding'] = dict(self.generation, max_new_tokens=self.max_tokens, constraint='task_json_schema')
        metrics = {f'{prefix}_{key}': value for key, value in report['metrics'].items()}
        for task, values in report['auxiliary']['tasks'].items():
            for key in ('score', 'invalid_rate', 'examples'):
                metrics[f'{prefix}_aux/{task}/{key}'] = values[key]
            for key, value in values['components'].items():
                metrics[f'{prefix}_aux/{task}/{key}'] = value
        for source, tasks in report['auxiliary']['sources'].items():
            for task, values in tasks.items():
                metrics[f'{prefix}_aux/{source}/{task}/score'] = values['score']
        metrics[f'{prefix}_runtime'] = time.monotonic() - started
        if trainer.is_world_process_zero():
            atomic_json(Path(trainer.args.output_dir) / f'{prefix}-step-{trainer.state.global_step}.json', report)
        trainer.log(metrics)
        if prefix != 'test':
            trainer.control = trainer.callback_handler.on_evaluate(trainer.args, trainer.state, trainer.control, metrics)
        return metrics

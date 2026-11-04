"""Shared vLLM evaluation with medical-only checkpoint selection."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from accelerate.utils import gather_object

from scripts.train.shared.data import atomic_json
from scripts.train.shared.decoding import answer_schema
from scripts.train.shared.metrics import score_medical_evaluation
from scripts.train.shared.auxiliary_evaluation import AuxiliaryEvaluation, score_auxiliary
from scripts.train.rl.video import prepare_video_prompt
from scripts.train.rl.rollout.features import encode_video_features


class MedicalVLLMEvaluator:
    def __init__(self, client, feature_config, processor, catalog, generation, run_id, max_tokens=1024, *, experiment=None, medical_templates=None, lifecycle=None, subject_check_rows=None, subject_reasoning_parser=None):
        self.subject_check_rows = subject_check_rows or {}
        self.subject_reasoning_parser = subject_reasoning_parser
        self.lifecycle = lifecycle
        self.client, self.feature_config, self.processor = client, feature_config, processor
        self.catalog, self.generation, self.run_id = catalog, dict(generation), run_id
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError('evaluation.max_new_tokens must be a positive integer')
        self.max_tokens = max_tokens
        self.evaluation_policy = (experiment or {}).get("evaluation", {})
        self.request_batch_size = self.evaluation_policy.get('request_batch_size', 1)
        if type(self.request_batch_size) is not int or self.request_batch_size < 1:
            raise ValueError('evaluation.request_batch_size must be a positive integer')
        self.auxiliary = AuxiliaryEvaluation(experiment, medical_templates, catalog)

    def encode(self, trainer, row, version):
        packet = prepare_video_prompt(trainer.video_collator, [row])
        if int(packet['attention_mask'].sum()) + self.max_tokens > trainer.video_collator.max_length:
            raise ValueError('Validation prompt and answer exceed the context budget')
        encoded = encode_video_features(packet, policy_version=version, **self.feature_config)
        return {"task": "subject_check", "packet": encoded} if row.get("task") == "subject_check" else encoded

    def predict(self, trainer, row, version, seed, schema):
        return self.predict_encoded(self.encode(trainer, row, version), version, seed, schema)

    def predict_many(self, trainer, rows, version, seeds, schemas):
        if not len(rows) == len(seeds) == len(schemas):
            raise ValueError('Each evaluation row needs a seed and schema')
        results = []
        for start in range(0, len(rows), self.request_batch_size):
            stop = min(start + self.request_batch_size, len(rows))
            # Processor/collator access stays serial; only HTTP generation overlaps.
            packets = [self.encode(trainer, row, version) for row in rows[start:stop]]
            if len(packets) == 1:
                results.append(self.predict_encoded(packets[0], version, seeds[start], schemas[start]))
            else:
                with ThreadPoolExecutor(max_workers=len(packets)) as pool:
                    futures = [pool.submit(self.predict_encoded, packet, version, seeds[index], schemas[index])
                               for index, packet in zip(range(start, stop), packets)]
                    results.extend(future.result() for future in futures)
        return results

    def predict_encoded(self, encoded, version, seed, schema):
        direct_options = {}
        if encoded.get('task') == 'subject_check':
            if self.subject_reasoning_parser not in ('qwen3', 'gemma4'):
                raise ValueError('Subject-check generation requires a native reasoning parser')
            direct_options = dict(enable_thinking=False, reasoning_parser=self.subject_reasoning_parser)
            encoded = encoded['packet']
        settings = self.generation
        output = self.client.generate(
            [encoded], policy_version=version,
            temperature=settings['temperature'] if settings.get('do_sample', True) else 0,
            top_p=settings['top_p'], top_k=settings['top_k'],
            repetition_penalty=settings['repetition_penalty'],
            presence_penalty=settings.get('presence_penalty', 0),
            min_p=settings['min_p'],
            max_tokens=self.max_tokens, seed=seed,
            schemas=[schema], **direct_options,
        )
        prediction = self.processor.tokenizer.decode(output['completion_ids'][0], skip_special_tokens=True)
        return prediction

    def __call__(self, trainer, dataset, prefix, *, additional_metrics=None):
        if dataset is None or not len(dataset):
            raise ValueError('Medical validation cannot be empty')
        started = time.monotonic()
        auxiliary_rows, auxiliary_preparation = self._phase(
            'prepare auxiliary rows', lambda: self.auxiliary.prepare(dataset, prefix))
        if self.lifecycle is not None:
            with self.lifecycle.generation(trainer.vllm_generation.sync_weights):
                trainer._last_loaded_step = trainer.state.global_step
                return self._evaluate(trainer, dataset, prefix, started, additional_metrics, auxiliary_rows, auxiliary_preparation)
        trainer.vllm_generation.sync_weights()
        trainer._last_loaded_step = trainer.state.global_step
        trainer.accelerator.wait_for_everyone()
        return self._evaluate(trainer, dataset, prefix, started, additional_metrics, auxiliary_rows, auxiliary_preparation)

    def _phase(self, label, operation):
        return self.lifecycle.phase(label, operation) if self.lifecycle is not None else operation()

    def _prediction_phase(self, label, operation):
        from scripts.train.shared.distributed_local_work import run_local_work
        return self._phase(label, lambda: run_local_work(label, operation))

    def evaluate_subject_check(self, trainer, prefix, version, seed_offset):
        from scripts.train.shared.subject_check import subject_check_schema
        from scripts.train.shared.subject_check_evaluation import score_subject_check
        split = 'test' if prefix == 'test' else 'validation'
        subjects = self.subject_check_rows.get(split, [])
        rank, world = trainer.accelerator.process_index, trainer.accelerator.num_processes
        indices = list(range(rank, len(subjects), world))
        rows = [subjects[i] for i in indices]
        predictions = self._prediction_phase('subject check prediction', lambda: self.predict_many(
            trainer, rows, version, [trainer.args.seed + seed_offset + i for i in indices],
            [subject_check_schema() for row in rows]))
        records = []
        for index, row, prediction in zip(indices, rows, predictions):
            record = dict(index=index, reference=row['reference'], answer=row['answer'])
            if isinstance(prediction, dict):
                record.update(prediction=prediction['answer'], raw_completion=prediction['raw'], diagnostics=prediction['diagnostics'])
            else:
                record['prediction'] = prediction
            records.append(record)
        records = sorted(gather_object(records), key=lambda row: row['index'])
        if [r['index'] for r in records] != list(range(len(subjects))):
            raise ValueError('Every subject-check heldout row must be evaluated exactly once')
        return score_subject_check(records)

    def _evaluate(self, trainer, dataset, prefix, started, additional_metrics, auxiliary_rows, auxiliary_preparation):
        version = f'{self.run_id}:{prefix}:step-{trainer.state.global_step}'
        records = []
        rank, world = trainer.accelerator.process_index, trainer.accelerator.num_processes
        indices = list(range(rank, len(dataset), world))
        rows = [dataset[index] for index in indices]
        predictions = self._prediction_phase('medical prediction', lambda: self.predict_many(trainer, rows, version,
            [trainer.args.seed + index for index in indices],
            [answer_schema(self.catalog['categories'][row['reference']['category']]) for row in rows]))
        for index, row, prediction in zip(indices, rows, predictions):
            records.append(dict(index=index, reference=row['reference'], answer=row['answer'], prediction=prediction))
        records = sorted(gather_object(records), key=lambda item: item['index'])
        if [row['index'] for row in records] != list(range(len(dataset))):
            raise ValueError('Every validation example must be evaluated exactly once')
        report = score_medical_evaluation(records, self.catalog, self.evaluation_policy)
        auxiliary_records = []
        indices = list(range(rank, len(auxiliary_rows), world))
        rows = [auxiliary_rows[index] for index in indices]
        predictions = self._prediction_phase('auxiliary prediction', lambda: self.predict_many(trainer, rows, version,
            [trainer.args.seed + len(dataset) + index for index in indices],
            [row['reward_context']['schema'] for row in rows]))
        for index, row, prediction in zip(indices, rows, predictions):
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
        for source, values in report.get('medical_sources', {}).items():
            for key, value in values['metrics'].items():
                metrics[f'{prefix}_{source}_{key}'] = value
        for task, values in report['auxiliary']['tasks'].items():
            for key in ('score', 'invalid_rate', 'examples'):
                metrics[f'{prefix}_aux/{task}/{key}'] = values[key]
            for key, value in values['components'].items():
                metrics[f'{prefix}_aux/{task}/{key}'] = value
        for source, tasks in report['auxiliary']['sources'].items():
            for task, values in tasks.items():
                metrics[f'{prefix}_aux/{source}/{task}/score'] = values['score']
        if self.subject_check_rows:
            from scripts.train.shared.subject_check_evaluation import subject_metric_values
            report['subject_check'] = self.evaluate_subject_check(trainer, prefix, version, len(dataset) + len(auxiliary_rows))
            metrics.update(subject_metric_values(report['subject_check'], prefix))
        if additional_metrics:
            metrics.update(additional_metrics)
            report['metrics']['loss'] = additional_metrics[f'{prefix}_loss']
            report['loss_scope'] = 'medical answer-only targets; mean per-example loss'
            report['loss_runtime'] = additional_metrics[f'{prefix}_loss_runtime']
        metrics[f'{prefix}_runtime'] = time.monotonic() - started + report.get('loss_runtime', 0)
        if trainer.is_world_process_zero():
            atomic_json(Path(trainer.args.output_dir) / f'{prefix}-step-{trainer.state.global_step}.json', report)
        trainer.log(metrics)
        if prefix != 'test':
            trainer.control = trainer.callback_handler.on_evaluate(trainer.args, trainer.state, trainer.control, metrics)
        return metrics

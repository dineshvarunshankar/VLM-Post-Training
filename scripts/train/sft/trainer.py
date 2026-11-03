"""TRL SFT with shared vLLM medical checkpoint evaluation."""

import time

from trl import SFTTrainer
from scripts.train.shared.vllm import SFTWeightSync


class MedicalSFTTrainer(SFTTrainer):
    def __init__(self, *args, medical_evaluator, vllm_server_url,
                 train_sampling_plan=None, mixture_config=None, **kwargs):
        if not callable(medical_evaluator) or not vllm_server_url:
            raise ValueError("SFT requires the shared medical evaluator and a vLLM server URL")
        self.train_sampling_plan = train_sampling_plan
        self.mixture_config = mixture_config
        self.medical_evaluator = medical_evaluator
        super().__init__(*args, **kwargs)
        if self.is_deepspeed_enabled:
            raise ValueError("Evaluation weight transfer supports FSDP2 or unsharded models")
        self.video_collator = self.data_collator
        from scripts.train.shared.collator import AnswerOnlyCollator
        if isinstance(self.video_collator, AnswerOnlyCollator):
            self.video_collator.medical_token_metrics = True
        self.vllm_generation = SFTWeightSync(self, vllm_server_url)

    def _teacher_loss_loop(self, *args, **kwargs):
        """Collect diagnostics during the existing forward pass, without logits gathering."""
        self._medical_metric_totals = None
        self._collect_medical_metrics = True
        try:
            output = self.evaluation_loop(*args, **kwargs)
            if self._medical_metric_totals is not None:
                from scripts.train.shared.medical_token_metrics import summarize
                output.metrics.update(summarize(self._medical_metric_totals, kwargs['metric_key_prefix']))
            return output
        finally:
            self._collect_medical_metrics = False
            self._medical_metric_totals = None

    def _get_train_sampler(self, train_dataset=None):
        if self.train_sampling_plan is None:
            return super()._get_train_sampler(train_dataset)
        from scripts.train.shared.mixture import EpochMixtureSampler

        return EpochMixtureSampler(
            self.train_sampling_plan,
            seed=self.args.data_seed
            if self.args.data_seed is not None
            else self.args.seed,
            shuffle=self.mixture_config["shuffle"],
            resample=self.mixture_config["resample_each_epoch"],
        )

    def get_eval_dataloader(self, eval_dataset=None):
        if (getattr(getattr(self, 'video_collator', None), 'answer_format', None) == 'cot'
                and eval_dataset is not None and not isinstance(eval_dataset, str)):
            # Trainer uses one "eval" key for every explicit dataset. Fingerprints
            # separate task/split/subset contents and reuse unchanged worker pools.
            fingerprint = getattr(eval_dataset, '_fingerprint', None)
            key = 'cot:' + fingerprint if isinstance(fingerprint, str) else None
            cached = getattr(self, '_eval_dataloaders', {})
            if self.args.dataloader_persistent_workers and key in cached:
                return cached[key]
            return self._get_dataloader(
                dataset=eval_dataset, description="CoT teacher evaluation",
                batch_size=self.args.eval_batch_size,
                sampler_fn=self._get_eval_sampler, dataloader_key=key,
            )
        return super().get_eval_dataloader(eval_dataset)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        if dataset is None or not len(dataset):
            raise ValueError("Medical validation cannot be empty")
        if getattr(getattr(self, 'video_collator', None), 'answer_format', None) == 'cot':
            return self._evaluate_cot(dataset, ignore_keys, metric_key_prefix)
        # Use Trainer's distributed loss-only loop: retain answer masking,
        # discard logits, and let Accelerate handle gathering and tail padding.
        # Calling super().evaluate() would emit an early on_evaluate event
        # without the medical score needed by checkpoint-selection callbacks.
        started = time.monotonic()
        was_training = self.model.training
        try:
            output = self._teacher_loss_loop(
                self.get_eval_dataloader(dataset),
                description="Answer-only validation loss",
                prediction_loss_only=True,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            loss_key = f"{metric_key_prefix}_loss"
            if loss_key not in output.metrics:
                raise RuntimeError("Validation loss pass returned no loss")
            loss_metrics = {
                **{k: v for k, v in output.metrics.items() if k.startswith(f'{metric_key_prefix}_medical_')},
                loss_key: output.metrics[loss_key],
                f"{metric_key_prefix}_loss_runtime": time.monotonic() - started,
            }
            return self.medical_evaluator(
                self, dataset, metric_key_prefix, additional_metrics=loss_metrics
            )
        finally:
            self.model.train(was_training)


    def _evaluate_cot(self, dataset, ignore_keys, prefix):
        """Score every human row, but compute teacher loss only where validated."""
        started = time.monotonic()
        covered = [i for i in range(len(dataset)) if dataset[i].get("teacher_available") and dataset[i].get("reasoning")]
        metrics = {prefix + "_teacher_loss_expected": len(dataset),
                   prefix + "_teacher_loss_covered": len(covered)}
        was_training = self.model.training
        try:
            if covered:
                teachers = dataset.select(covered) if hasattr(dataset, "select") else [dataset[i] for i in covered]
                output = self._teacher_loss_loop(self.get_eval_dataloader(teachers),
                    description="Covered medical CoT teacher loss", prediction_loss_only=True,
                    ignore_keys=ignore_keys, metric_key_prefix=prefix)
                if prefix + "_loss" not in output.metrics:
                    raise RuntimeError("Covered CoT validation loss pass returned no loss")
                metrics[prefix + "_loss"] = output.metrics[prefix + "_loss"]
                metrics.update({k: v for k, v in output.metrics.items() if k.startswith(prefix + '_medical_')})
            auxiliary = getattr(self.medical_evaluator, "teacher_caption_rows", lambda prefix: [])(prefix)
            metrics[prefix + "_caption_teacher_loss_covered"] = len(auxiliary)
            if auxiliary:
                from datasets import Dataset
                teachers = Dataset.from_list(auxiliary)
                output = self.evaluation_loop(self.get_eval_dataloader(teachers),
                    description="Covered caption teacher loss", prediction_loss_only=True,
                    ignore_keys=ignore_keys, metric_key_prefix=prefix + "_caption")
                metrics[prefix + "_caption_loss"] = output.metrics[prefix + "_caption_loss"]
            metrics[prefix + "_loss_runtime"] = time.monotonic() - started
            return self.medical_evaluator(self, dataset, prefix, additional_metrics=metrics)
        finally:
            self.model.train(was_training)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        groups = inputs.pop('medical_token_groups', None)
        task_ids = inputs.pop("cot_task_ids", None)
        if task_ids is not None and model.training:
            import torch
            from scripts.train.shared.cot_format import TASK_IDS
            counts = torch.stack([(inputs["labels"] != -100).sum(dim=1)[task_ids == task_id].sum()
                                  for task_id in TASK_IDS.values()]).detach().to(self.accelerator.device)
            if not hasattr(self, "_cot_supervised_tokens"):
                self._cot_supervised_tokens = counts.new_zeros(len(TASK_IDS))
            self._cot_supervised_tokens += counts
        if groups is None or not getattr(self, '_collect_medical_metrics', False):
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        from scripts.train.shared.medical_token_metrics import batch_statistics
        labels = inputs['labels']
        captured = []
        def capture(module, args, outputs):
            if not captured:
                captured.append(batch_statistics(outputs, labels, groups))
        handle = model.register_forward_hook(capture)
        try:
            result = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError('Medical token diagnostics did not observe model outputs')
        # Gather per-example rows so Accelerate drops duplicated final-batch
        # samples before aggregation. Never average per-rank averages.
        totals = self.accelerator.gather_for_metrics(captured[0]).sum(0)
        if self._medical_metric_totals is None:
            self._medical_metric_totals = totals
        else:
            self._medical_metric_totals += totals
        return result

    def log(self, logs, *args, **kwargs):
        if hasattr(self, "_cot_supervised_tokens"):
            from scripts.train.shared.cot_format import TASK_IDS
            totals = self.accelerator.reduce(self._cot_supervised_tokens.clone(), reduction="sum").tolist()
            logs = dict(logs, **{f"train/supervised_tokens/{task}":total for task,total in zip(TASK_IDS,totals)})
        return super().log(logs, *args, **kwargs)

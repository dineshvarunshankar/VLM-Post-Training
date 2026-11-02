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
        self.vllm_generation = SFTWeightSync(self, vllm_server_url)

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

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        if dataset is None or not len(dataset):
            raise ValueError("Medical validation cannot be empty")
        # Use Trainer's distributed loss-only loop: retain answer masking,
        # discard logits, and let Accelerate handle gathering and tail padding.
        # Calling super().evaluate() would emit an early on_evaluate event
        # without the medical score needed by checkpoint-selection callbacks.
        started = time.monotonic()
        was_training = self.model.training
        try:
            output = self.evaluation_loop(
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
                loss_key: output.metrics[loss_key],
                f"{metric_key_prefix}_loss_runtime": time.monotonic() - started,
            }
            return self.medical_evaluator(
                self, dataset, metric_key_prefix, additional_metrics=loss_metrics
            )
        finally:
            self.model.train(was_training)

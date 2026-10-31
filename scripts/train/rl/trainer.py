"""TRL GSPO extension that retains native-video inputs through rollout and loss."""

import json
from contextlib import contextmanager

import torch
from trl import GRPOTrainer

from scripts.train.shared.collator import AnswerOnlyCollator
from scripts.train.rl.video import completion_inputs, prepare_video_prompt, stack_video_prompts, video_log_probs
from scripts.train.rl.rollout.weights import install_policy_weight_export


VIDEO_FIELDS = ('video','system','question','required_frame_indices','video_metadata')


def video_example(row):
    """Whitelist model inputs; ground truth and reward metadata stay outside messages."""
    return {key:row[key] for key in VIDEO_FIELDS if key in row and row[key] is not None}


class VideoGSPOTrainer(GRPOTrainer):
    def __init__(self, *args, video_collator, video_rollout, train_sampling_plan=None, mixture_config=None, medical_evaluator=None, **kwargs):
        config = kwargs.get('args')
        if config is None:
            raise ValueError('Pass GRPOConfig through the args keyword')
        if config.importance_sampling_level != 'sequence' or config.loss_type != 'grpo':
            raise ValueError('GSPO requires sequence ratios and response-mean (grpo) loss normalization')
        if not callable(video_rollout):
            raise ValueError('A native-video rollout transport is required')
        if kwargs.get('rollout_func') is not None:
            raise ValueError('Use video_rollout, not an additional rollout_func')
        if kwargs.get('tools') or kwargs.get('environment_factory'):
            raise ValueError('This trainer implements single-turn answer-only video tasks')
        self.train_sampling_plan = train_sampling_plan
        self.mixture_config = mixture_config
        self.medical_evaluator = medical_evaluator
        self.video_collator = video_collator
        self.video_rollout = video_rollout
        self._video_prompts = None
        super().__init__(*args, rollout_func=self._rollout_video, **kwargs)
        if self.use_vllm:
            install_policy_weight_export(self.vllm_generation)

    def _get_train_sampler(self, dataset=None):
        if self.train_sampling_plan is None:
            return super()._get_train_sampler(dataset)
        from scripts.train.rl.sampling import RLQuestionSampler
        return RLQuestionSampler(
            self.train_sampling_plan,
            seed=self.args.data_seed if self.args.data_seed is not None else self.args.seed,
            shuffle=self.mixture_config['shuffle'], resample=self.mixture_config['resample_each_epoch'],
            questions_per_rollout=self.args.generation_batch_size // self.num_generations,
            num_generations=self.num_generations,
            steps_per_generation=self.num_iterations * self.args.steps_per_generation,
        )

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        if self.medical_evaluator is None:
            raise ValueError('Medical macro-F1 evaluator is required for RL checkpoint selection')
        dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        return self.medical_evaluator(self, dataset, metric_key_prefix)

    @contextmanager
    def _using_video_prompts(self, prompts):
        previous = self._video_prompts
        self._video_prompts = prompts
        try:
            yield
        finally:
            self._video_prompts = previous

    def _rollout_video(self, prompts, trainer):
        if self._video_prompts is None or len(prompts) != len(self._video_prompts):
            raise ValueError('Video rollout prompts are not aligned with the training batch')
        # Transport gets model inputs only; each entry requests one independent response.
        output = self.video_rollout(self._video_prompts, trainer)
        required = ('prompt_ids','completion_ids','logprobs')
        if any(key not in output or len(output[key]) != len(prompts) for key in required):
            raise ValueError('Video transport must return one completion and its log probabilities per input')
        for packet, ids, completion, logps in zip(self._video_prompts,output['prompt_ids'],output['completion_ids'],output['logprobs']):
            expected = packet['input_ids'][0][packet['attention_mask'][0].bool()].tolist()
            if ids != expected:
                raise ValueError('vLLM rollout tokens differ from the training video processor')
            if not completion or len(completion) != len(logps):
                raise ValueError('Completion tokens and sampling log probabilities do not align')
        return output

    def _generate_and_score_completions(self, inputs):
        cache, packets = {}, []
        for row in inputs:
            example = video_example(row)
            if row['prompt'] != AnswerOnlyCollator.messages(example):
                raise ValueError('Dataset prompt differs from the approved video message construction')
            key = json.dumps(example,sort_keys=True)
            if key not in cache:
                cache[key] = prepare_video_prompt(self.video_collator,[example])
            packets.append(cache[key])
        with self._using_video_prompts(packets):
            output = super()._generate_and_score_completions(inputs)
        # A list is shuffled and sliced alongside responses by TRL's minibatch utilities.
        output['video_prompts'] = packets
        return output

    def _compute_loss(self, model, inputs):
        if 'video_prompts' not in inputs:
            raise ValueError('Native-video inputs were lost during rollout minibatch preparation')
        with self._using_video_prompts(inputs['video_prompts']):
            return super()._compute_loss(model,inputs)

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask,
            logits_to_keep, batch_size=None, compute_entropy=False, compute_aux_loss=False, **kwargs):
        if self._video_prompts is None or len(self._video_prompts) != input_ids.shape[0]:
            raise ValueError('Policy inputs lack aligned native-video processor outputs')
        if compute_aux_loss:
            raise ValueError('MoE auxiliary loss is not configured for this video recipe')
        chunk_size = batch_size or input_ids.shape[0]
        prompt_width = input_ids.shape[1] - logits_to_keep
        all_logps, all_entropy = [], []
        for start in range(0,input_ids.shape[0],chunk_size):
            end = min(start+chunk_size,input_ids.shape[0])
            prompt = stack_video_prompts(self._video_prompts[start:end],
                pad_token_id=self._tokenizer.pad_token_id,prompt_width=prompt_width,device=input_ids.device)
            if not torch.equal(prompt['attention_mask'],attention_mask[start:end,:prompt_width]):
                raise ValueError('Rollout prompt mask differs from cached video input')
            model_inputs = completion_inputs(prompt,input_ids[start:end,prompt_width:],
                attention_mask[start:end,prompt_width:],expected_prompt_ids=input_ids[start:end,:prompt_width],
                max_length=self.video_collator.max_length)
            with self.accelerator.autocast():
                logps, entropy = video_log_probs(model,model_inputs,logits_to_keep)
            all_logps.append(logps)
            if compute_entropy:
                all_entropy.append(entropy)
        return torch.cat(all_logps), torch.cat(all_entropy) if compute_entropy else None, None

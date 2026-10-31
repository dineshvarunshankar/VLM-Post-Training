"""Connect cached TRL video prompts to processed-feature HTTP generation."""

from scripts.train.rl.rollout.features import encode_video_features


class VideoServerRollout:
    """Callable for VideoGSPOTrainer.video_rollout.

    All training ranks must use the same run_id. TRL must synchronize weights
    before calling this adapter; its step marker is checked here. Verifying that
    synchronization includes every vision adapter remains a launch prerequisite.
    """

    def __init__(self, client, feature_config, *, run_id):
        if not isinstance(run_id, str) or not run_id.strip() or len(run_id) > 900:
            raise ValueError('A shared nonempty run ID of at most 900 characters is required')
        self.client = client
        self.feature_config = dict(feature_config)
        self.run_id = run_id

    def __call__(self, packets, trainer):
        args = trainer.args
        if not trainer.use_vllm or args.vllm_mode != 'server':
            raise ValueError('Native-video rollout requires TRL vLLM server mode')
        step = trainer.state.global_step
        if getattr(trainer, '_last_loaded_step', None) != step:
            raise ValueError('TRL has not synchronized the current policy step to vLLM')
        if not packets:
            raise ValueError('A video rollout batch cannot be empty')
        policy_version = f'{self.run_id}:step-{step}'
        encoded, prepared = {}, []
        for packet in packets:
            key = id(packet)
            if key not in encoded:
                prompt_length = int(packet['attention_mask'].sum())
                if prompt_length + args.max_completion_length > trainer.video_collator.max_length:
                    raise ValueError('Video prompt and response budget exceed the context limit')
                encoded[key] = encode_video_features(
                    packet, policy_version=policy_version, **self.feature_config,
                )
            prepared.append(encoded[key])
        return self.client.generate(
            prepared, policy_version=policy_version,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_completion_length,
        )

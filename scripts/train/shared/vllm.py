"""vLLM feature configuration and SFT evaluation weight synchronization."""

import os


def weight_transfer_port():
    port = int(os.environ.get('TRIAGE_VLLM_GROUP_PORT', '51216'))
    if not 1024 <= port <= 65535:
        raise ValueError('Weight-transfer port must be between 1024 and 65535')
    return port


def feature_configuration(config, processor):
    kind = config.model_type
    if kind == 'qwen3_5':
        start, end = config.vision_start_token_id, config.vision_end_token_id
    elif kind in ('gemma4', 'gemma4_unified'):
        start = processor.tokenizer.convert_tokens_to_ids(processor.boi_token)
        end = processor.tokenizer.convert_tokens_to_ids(processor.eoi_token)
    else:
        raise ValueError(f'Unsupported rollout architecture: {kind}')
    return dict(model_type=kind, video_token_id=processor.video_token_id,
                image_token_id=getattr(processor, "image_token_id", None),
                vision_start_token_id=start, vision_end_token_id=end,
                spatial_merge_size=getattr(config.vision_config, 'spatial_merge_size', None),
                pooling_kernel_size=getattr(config.vision_config, 'pooling_kernel_size', None))


class SenderAccelerator:
    """Override only TRL's server sender predicate; preserve real device/collectives."""
    def __init__(self, accelerator, sender_rank):
        self._accelerator = accelerator
        self._sender_rank = sender_rank

    @property
    def is_main_process(self):
        return self._accelerator.process_index == self._sender_rank

    def __getattr__(self, name):
        return getattr(self._accelerator, name)


def install_bounded_prefix_reset(generation):
    """Bound the sender-only post-transfer HTTP control without retry delays."""
    if not generation.accelerator.is_main_process:
        return
    client = generation.vllm_client

    def reset_prefix_cache():
        import requests
        response = requests.post(client.base_url + '/reset_prefix_cache',
                                 headers=client.session.headers, timeout=(5, 60))
        if response.status_code != 200:
            raise RuntimeError('vLLM prefix cache reset failed')
        # TRL 1.16/vLLM>=0.26 reads JSON but does not inspect success.
        return response.json()

    client.reset_prefix_cache = reset_prefix_cache


class SFTWeightSync:
    """Initialize weight transfer after the trainer has prepared distributed weights."""

    def __init__(self, trainer, server_url):
        self.trainer = trainer
        self.server_url = server_url
        self.generation = None
        self.generations = {}
        from scripts.train.shared.eval_replicas import replica_plan_from_environment
        self.replica_plan = replica_plan_from_environment(trainer.accelerator.num_processes)

    def sync_weights(self):
        from trl.generation.vllm_generation import VLLMGeneration
        from scripts.train.rl.rollout.weights import install_policy_weight_export

        trainer = self.trainer
        model = trainer.accelerator.unwrap_model(trainer.model_wrapped)
        if self.replica_plan is not None:
            # Every rank traverses the full tensors in the same replica order.
            # Never move this loop or TRL communicator calls to a worker thread.
            for replica in self.replica_plan['replicas']:
                key = replica['replica_id']
                if key not in self.generations:
                    generation = VLLMGeneration(
                        model=model, accelerator=SenderAccelerator(trainer.accelerator, replica['sender_rank']),
                        processing_class=trainer.processing_class, mode='server',
                        server_base_url=replica['url'], group_port=replica['group_port'])
                    install_policy_weight_export(generation)
                    install_bounded_prefix_reset(generation)
                    self.generations[key] = generation
                generation = self.generations[key]
                generation.model = model
                error = None
                try:
                    generation.sync_weights()
                except Exception as caught:
                    error = caught
                # TRL resets its prefix cache only on the sender after transport.
                # Complete this boundary before any rank enters the next replica's
                # constructor barrier or full-tensor collectives.
                import torch.distributed as distributed
                status = {'rank': trainer.accelerator.process_index,
                          'error': type(error).__name__ if error else None}
                if distributed.is_initialized():
                    statuses = [None] * distributed.get_world_size()
                    distributed.all_gather_object(statuses, status)
                else:
                    statuses = [status]
                failures = [str(item['rank']) for item in statuses if item['error']]
                if failures:
                    raise RuntimeError(
                        f'Evaluation replica {key} weight synchronization failed on rank(s) '
                        + ', '.join(failures)) from error
            return
        if self.generation is None:
            self.generation = VLLMGeneration(
                model=model, accelerator=trainer.accelerator,
                processing_class=trainer.processing_class,
                mode="server", server_base_url=self.server_url, group_port=weight_transfer_port(),
            )
            install_policy_weight_export(self.generation)
        self.generation.model = model
        self.generation.sync_weights()

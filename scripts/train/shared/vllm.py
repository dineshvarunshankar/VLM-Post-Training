"""vLLM feature configuration and SFT evaluation weight synchronization."""

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
                vision_start_token_id=start, vision_end_token_id=end,
                spatial_merge_size=getattr(config.vision_config, 'spatial_merge_size', None),
                pooling_kernel_size=getattr(config.vision_config, 'pooling_kernel_size', None))


class SFTWeightSync:
    """Initialize weight transfer after the trainer has prepared distributed weights."""

    def __init__(self, trainer, server_url):
        self.trainer = trainer
        self.server_url = server_url
        self.generation = None

    def sync_weights(self):
        from trl.generation.vllm_generation import VLLMGeneration
        from scripts.train.rl.rollout.weights import install_policy_weight_export

        trainer = self.trainer
        model = trainer.accelerator.unwrap_model(trainer.model_wrapped)
        if self.generation is None:
            self.generation = VLLMGeneration(
                model=model, accelerator=trainer.accelerator,
                processing_class=trainer.processing_class,
                mode="server", server_base_url=self.server_url,
            )
            install_policy_weight_export(self.generation)
        self.generation.model = model
        self.generation.sync_weights()

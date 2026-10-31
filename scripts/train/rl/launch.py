"""Shared launcher for RL-only and the RL stage of an SFT-to-RL run."""


import argparse
import copy
import json
import os
from scripts.train.shared.storage_settings import local_weights_directory, local_dataset_directory
from pathlib import Path

from scripts.train.shared.paths import REPO_ROOT, CONFIG_DIR, ANSWER_ONLY_CONFIG_DIR
from scripts.train.shared.data import atomic_json, load_splits, read_yaml
from scripts.train.rl.recipe import inspect_dataset, inspect_recipe, load_recipe, resolve_model_recipe
from scripts.train.rl.runtime import training_arguments, validate_sft_start
from scripts.train.shared.config import load_training_config, select_medical_examples


def decode_question_rows(batch):
    rows = [json.loads(value) for value in batch['serialized_question']]
    return {key: [row[key] for row in rows] for key in rows[0]}


def QuestionDataset(rows):
    """Use a TRL-compatible Dataset without coercing heterogeneous reward targets."""
    from datasets import Dataset
    keys = sorted({key for row in rows for key in row})
    encoded = [json.dumps({key: row.get(key) for key in keys}) for row in rows]
    return Dataset.from_dict({'serialized_question': encoded}).with_transform(decode_question_rows)


def parse_args(recipe):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ANSWER_ONLY_CONFIG_DIR / 'recipes' / f'{recipe}.yaml')
    parser.add_argument('--models', type=Path, default=CONFIG_DIR / 'models.yaml')
    parser.add_argument('--model')
    for key in ('media-root', 'split-dir', 'catalog', 'prompts', 'output-dir', 'sft-run', 'resume'):
        parser.add_argument('--' + key, type=Path, default=local_dataset_directory() if key == 'media-root' else None)
    parser.add_argument('--output-root', type=Path, default=local_weights_directory('finetuned'))
    parser.add_argument('--weights-root', type=Path, default=local_weights_directory('base_weights'))
    parser.add_argument('--checkpoint')
    parser.add_argument('--revision')
    parser.add_argument('--run-name')
    parser.add_argument('--run-date')
    parser.add_argument('--vllm-server-url', default=os.environ.get('VLLM_SERVER_URL'))
    parser.add_argument('--vllm-model-name', default=os.environ.get('VLLM_MODEL_NAME'))
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--inspect', action='store_true', help='Inspect configuration, optionally the dataset; no model loading')
    return parser.parse_args()


def load_starting_adapter(model, directory):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    weights = load_peft_weights(str(directory), device='cpu', local_files_only=True)
    expected = get_peft_model_state_dict(model)
    if set(weights) != set(expected) or any(weights[k].shape != expected[k].shape for k in weights):
        raise ValueError('Selected SFT adapter tensors differ from the configured policy adapter')
    result = set_peft_model_state_dict(model, weights, adapter_name='default')
    if result.unexpected_keys:
        raise ValueError('Unexpected keys while loading the selected SFT adapter')


def verify_reference_adapter(model):
    import torch
    if 'ref' not in model.peft_config:
        raise ValueError('TRL must preserve the starting policy as a reference adapter')
    model.set_adapter('default')
    count = 0
    for name, parameter in model.named_parameters():
        if '.lora_' in name and '.default.' in name:
            reference = model.get_parameter(name.replace('.default.', '.ref.'))
            reference.requires_grad_(False)
            if not torch.equal(parameter.detach(), reference.detach()):
                raise ValueError('Reference adapter does not match the starting policy')
            count += 1
    if not count:
        raise ValueError('No reference adapter parameters were verified')


def main(expected_recipe):
    from dotenv import load_dotenv
    if not os.environ.get("TRIAGE_RUN_CONFIG"):
        load_dotenv(REPO_ROOT / '.env')
    args = parse_args(expected_recipe)
    args.config = args.config.resolve()
    recipe = load_recipe(args.config)
    if recipe['recipe'] != expected_recipe:
        raise ValueError(f'This command requires {expected_recipe}')
    selected_profile = read_yaml(args.models)['models'][args.model] if args.model else None
    if selected_profile is not None:
        recipe = resolve_model_recipe(recipe, selected_profile)
    if args.inspect:
        inspect_recipe(args.config, expected_recipe, model_profile=selected_profile)
        paths = (args.media_root, args.split_dir, args.catalog, args.prompts)
        if any(path is not None for path in paths):
            if not all(path is not None for path in paths):
                raise ValueError('Dataset inspection requires all four dataset/catalog/prompt paths')
            inspect_dataset(args)
        return 0
    for name in ('model', 'media_root', 'split_dir', 'catalog', 'prompts', 'vllm_server_url', 'vllm_model_name'):
        if not getattr(args, name):
            raise ValueError(f'--{name.replace("_", "-")} is required for training')
    if expected_recipe == 'sft_then_rl' and args.sft_run is None:
        raise ValueError('--sft-run must name the completed, verified SFT run')
    if expected_recipe == 'rl_only' and args.sft_run is not None:
        raise ValueError('RL-only must start from cloud base weights, not an SFT adapter')
    experiment = load_training_config(args.config.parent / recipe['experiment_config'])
    from scripts.train.shared.config import validate_grounding_readiness
    validate_grounding_readiness(experiment)
    if int(os.environ.get('WORLD_SIZE', '1')) != recipe['execution']['training_gpus']:
        raise ValueError('Launch with torchrun using execution.training_gpus processes')
    # Resolve hardware-dependent fields before touching any dataset or loading weights.
    training_arguments(recipe, experiment, 1, '/unused', args.vllm_server_url)
    if (experiment['dataset']['sources']['stabilized']
            and experiment['mixture']['label_questions']['grounding']['spatial']['enabled']
            and recipe['rewards']['spatial']['stabilized_full_credit_pixels'] is None):
        raise ValueError('Resolve stabilized spatial alignment/tolerance, or explicitly disable spatial label grounding')

    from scripts.train.shared.modeling import load_model_config, build_processor, load_lora_model
    from scripts.train.shared.runtime import validate_runtime, save_run_metadata
    from scripts.train.shared.media_geometry import collect_geometry
    from scripts.train.rl.data import build_rl_mixture
    from scripts.train.shared.collator import AnswerOnlyCollator
    from scripts.train.rl.trainer import VideoGSPOTrainer, video_example
    from scripts.train.rl.rollout.client import VideoRolloutClient
    from scripts.train.rl.rollout.server import VideoServerRollout
    from scripts.train.rl.evaluation import BaselineCheckpoint
    from scripts.train.shared.evaluation import MedicalVLLMEvaluator
    from scripts.train.shared.vllm import feature_configuration
    from scripts.train.rl.reward_data import make_reward_function
    from trl import GRPOConfig
    from transformers import set_seed

    profile = load_model_config(args)
    output = validate_runtime(args, profile, expected_recipe)
    start_adapter = None
    if args.sft_run:
        expected_sft = load_training_config(args.config.parent / recipe['sft_config'])
        start_adapter = validate_sft_start(
            args.sft_run, profile, experiment, args.catalog, args.prompts, args.split_dir,
            expected_sft_config=expected_sft,
        )
    splits = select_medical_examples(load_splits(args.media_root, args.split_dir, args.catalog, args.prompts,
                                               sources=experiment['dataset']['sources']), experiment)
    geometry = collect_geometry(splits['train'])
    pool, plan, report = build_rl_mixture(splits['train'], read_yaml(args.catalog), read_yaml(args.prompts),
                                        experiment, recipe, geometry)
    for row in pool:
        row['prompt'] = AnswerOnlyCollator.messages(video_example(row))
    train_args = training_arguments(recipe, experiment, report['examples_per_epoch'], output, args.vllm_server_url)
    from scripts.train.shared.tracking import configure_tracking, add_tracking_metadata
    train_args.update(configure_tracking(args, expected_recipe, output, repository=profile["repository"]))
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError('--max-steps must be positive')
        train_args['max_steps'] = args.max_steps
    set_seed(train_args['seed'])
    config, processor, collator = build_processor(profile, experiment)
    policy_config = copy.deepcopy(experiment)
    policy_config['lora']['lora_dropout'] = 0.0
    model, adapter_summary = load_lora_model(profile, config, policy_config)
    if start_adapter:
        load_starting_adapter(model, start_adapter)
    model.enable_input_require_grads()
    features = feature_configuration(config, processor)
    client = VideoRolloutClient(args.vllm_server_url, args.vllm_model_name, api_key=os.environ.get('VLLM_API_KEY'))
    client.check_compatibility()
    run_id = f'{args.run_date}:{args.model}:{args.run_name}'
    from scripts.train.shared.config import evaluation_settings

    evaluation = evaluation_settings(experiment, profile)
    evaluator = MedicalVLLMEvaluator(
        client, features, processor, read_yaml(args.catalog), evaluation['generation'],
        run_id, max_tokens=evaluation['max_new_tokens'],
    )
    trainer = VideoGSPOTrainer(
        model=model, args=GRPOConfig(**train_args), processing_class=processor,
        reward_funcs=make_reward_function(recipe['rewards']),
        train_dataset=QuestionDataset(pool), eval_dataset=QuestionDataset(splits['validation']),
        video_collator=collator, video_rollout=VideoServerRollout(client, features, run_id=run_id),
        train_sampling_plan=plan, mixture_config=experiment['mixture'], medical_evaluator=evaluator,
    )
    verify_reference_adapter(trainer.model)
    trainer.add_callback(BaselineCheckpoint(trainer))
    report['final_batch_padding_questions'] = (-report['examples_per_epoch']) % recipe['batch']['questions_per_rollout']
    metadata_config = dict(experiment, recipe=expected_recipe, rl=recipe,
                           starting_sft_run=str(args.sft_run.resolve()) if args.sft_run else None)
    if trainer.is_world_process_zero():
        save_run_metadata(args, output, profile, metadata_config, train_args, adapter_summary, splits, report)
    trainer.accelerator.wait_for_everyone()
    add_tracking_metadata(trainer, output)
    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    trainer.save_model(str(output / 'adapter'))
    if trainer.is_world_process_zero():
        processor.save_pretrained(output / 'adapter')
        atomic_json(output / 'train_metrics.json', result.metrics)
        atomic_json(output / 'checkpoint_selection.json', dict(
            metric='medical_category_macro_f1', best_score=trainer.state.best_metric,
            best_checkpoint=trainer.state.best_model_checkpoint, final_adapter=str(output / 'adapter'),
        ))
    from scripts.train.shared.runtime import evaluate_heldout, finish_training_and_export

    evaluate_heldout(
        trainer, QuestionDataset(splits["test"]) if splits["test"] else None, output,
    )
    finish_training_and_export(trainer, output)
    return 0

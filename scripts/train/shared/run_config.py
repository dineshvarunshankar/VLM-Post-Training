"""Resolve a self-contained run YAML into the existing training recipe interfaces."""

from copy import deepcopy
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from scripts.train.shared.config import fields, validate_mixture, validate_augmentation
from scripts.train.shared.paths import TRAIN_ROOT, CONFIG_DIR
from scripts.train.rl.recipe import validate_recipe, resolve_model_recipe, resolve_batch
from scripts.train.storage.cloud_models import run_destination, safe_relative

TOP_LEVEL = ('schema_version run model model_settings answer_format recipe paths cloud wandb '
             'execution dataset mixture sft rl env_file checkpoint sft_run resume')
PATH_FIELDS = 'media_root split_dir weights_root output_root catalog prompts'
SECRET_KEYS = {'api_key', 'password', 'secret', 'secret_key', 'access_token', 'auth_token',
               'application_credential_secret', 'aws_secret_access_key', 'wandb_api_key', 'token', 'access_key_id'}


def reject_secrets(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in SECRET_KEYS or str(key).lower().endswith(('_api_key', '_password', '_secret', '_token', '_access_key', '_access_key_id')):
                raise ValueError('Secret fields belong in the environment, not run YAML')
            reject_secrets(item)
    elif isinstance(value, list):
        for item in value:
            reject_secrets(item)


def load_run(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    reject_secrets(config)
    fields(config, TOP_LEVEL, 'run YAML')
    fields(config['paths'], PATH_FIELDS, 'paths')
    def resolve(value):
        if not isinstance(value, str) or not value.strip() or '${' in value:
            raise ValueError('Paths must be explicit strings; secrets alone belong in .env')
        p = Path(value).expanduser()
        return str((p if p.is_absolute() else path.parent / p).resolve())
    config['paths'] = {key: resolve(value) for key, value in config['paths'].items()}
    for key in ('env_file', 'checkpoint', 'sft_run', 'resume'):
        if config[key] is not None:
            config[key] = resolve(config[key])
    return config


def compile_run(config):
    """Pure configuration validation. Does not open datasets or load credentials/models."""
    reject_secrets(config)
    fields(config, TOP_LEVEL, 'run YAML')
    c = deepcopy(config)
    if type(c['schema_version']) is not int or c['schema_version'] != 1:
        raise ValueError('schema_version must be 1')
    if c['answer_format'] != 'answer_only':
        raise ValueError('Only answer_only is implemented')
    if c['recipe'] not in ('sft', 'rl_only', 'sft_then_rl'):
        raise ValueError('recipe must be sft, rl_only, or sft_then_rl')
    fields(c['run'], 'name date', 'run')
    date.fromisoformat(c['run']['date'])
    fields(c['paths'], PATH_FIELDS, 'paths')
    fields(c['cloud'], 'source_container source_prefix output_container output_prefix', 'cloud')
    for value in c['cloud'].values():
        safe_relative(value)
    fields(c['wandb'], 'entity project mode log_model watch', 'wandb')
    if c['wandb']['mode'] not in ('online', 'offline', 'disabled'):
        raise ValueError('wandb.mode must be online, offline, or disabled')
    fields(c['execution'], 'training_gpus training_devices vllm_server_url vllm_model_name', 'execution')
    gpus = c['execution']['training_gpus']
    if type(gpus) is not int or gpus < 1:
        raise ValueError('execution.training_gpus must be a positive integer')
    devices = c['execution']['training_devices']
    if devices is not None and (not isinstance(devices, list) or len(devices) != gpus
                               or len(set(map(str, devices))) != gpus):
        raise ValueError('training_devices must list one unique device per training GPU, or be null')
    url = urlsplit(c['execution']['vllm_server_url'])
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('Use an HTTP(S) vLLM URL without credentials, query, or fragment')
    fields(c['dataset'], 'modality sources', 'dataset')
    if c['dataset']['modality'] != 'RGB' or c['dataset']['sources'] not in (
            ['stabilized'], ['native', 'stabilized'], ['stabilized', 'native']):
        raise ValueError('Choose RGB stabilized or native + stabilized sources')
    if c['sft_run'] and c['recipe'] != 'sft_then_rl':
        raise ValueError('sft_run is only valid for sft_then_rl')
    experiment = deepcopy(c['sft'])
    if 'extends' in experiment or experiment.get('recipe') != 'sft_only':
        raise ValueError('sft must contain a complete sft_only recipe without extends')
    experiment['dataset'] = dict(modality='RGB', sources={k: k in c['dataset']['sources'] for k in ('native', 'stabilized')})
    experiment['mixture'] = deepcopy(c['mixture'])
    # Validate the declared task policy before applying dataset-source eligibility.
    all_sources = deepcopy(experiment)
    all_sources['dataset']['sources'] = {'native': True, 'stabilized': True}
    validate_mixture(all_sources)
    inactive_tasks = []
    for name in ('camera_view', 'counting', 'subject_localization'):
        task = experiment['mixture'][name]
        if task['enabled'] and not any(experiment['dataset']['sources'][v] for v in task['sources']):
            task['enabled'] = False
            inactive_tasks.append(name)
    validate_mixture(experiment)
    validate_augmentation(experiment)
    profile = c['model_settings']
    if not isinstance(c['model'], str) or not c['model']:
        raise ValueError('model must be a nonempty model identifier')
    if not isinstance(profile, dict) or not all(profile.get(k) for k in ('repository', 'revision', 'generation')):
        raise ValueError('model_settings requires repository, pinned revision, and generation settings')
    known_models = yaml.safe_load((CONFIG_DIR / 'models.yaml').read_text())['models']
    if c['model'] not in known_models or profile['repository'] != known_models[c['model']]['repository']:
        raise ValueError('model and model_settings.repository must identify the same supported model')
    rl = deepcopy(c['rl'])
    if 'extends' in rl:
        raise ValueError('rl must contain complete settings without extends')
    recipe = c['recipe']
    rl.update(recipe='sft_then_rl' if recipe == 'sft_then_rl' else 'rl_only', experiment_config='experiment.yaml')
    rl['initialization'] = dict(source='best_sft_checkpoint' if recipe == 'sft_then_rl' else 'cloud_base',
                                continue_adapter=recipe == 'sft_then_rl', fresh_optimizer=True)
    if recipe == 'sft_then_rl':
        rl['sft_config'] = 'experiment.yaml'
    rl['execution']['training_gpus'] = gpus
    rl = resolve_model_recipe(rl, profile)
    validate_recipe(rl)
    batch = resolve_batch(rl)
    if experiment['effective_batch_size'] % (gpus * experiment['training']['per_device_train_batch_size']):
        raise ValueError('SFT effective batch must be divisible by GPUs times per-device batch size')
    def output(stage):
        relative = run_destination('runs', profile['repository'], c['run']['date'], c['run']['name'],
                                   answer_format=c['answer_format'], recipe=stage)
        return str(Path(c['paths']['output_root']) / relative.removeprefix('runs/'))
    stages = []
    if recipe == 'sft' or (recipe == 'sft_then_rl' and not c['sft_run']):
        stages.append(dict(recipe='sft_only', script=str(TRAIN_ROOT / 'train_sft.py'),
                           config='experiment.yaml', output=output('sft_only')))
    if recipe != 'sft':
        stage = dict(recipe=recipe, script=str(TRAIN_ROOT / ('train_rl.py' if recipe == 'rl_only' else 'train_sft_then_rl.py')),
                     config='rl.yaml', output=output(recipe))
        if recipe == 'sft_then_rl':
            stage['sft_run'] = c['sft_run'] or stages[0]['output']
        stages.append(stage)
    return dict(run=c, experiment=experiment, rl=rl, models={'models': {c['model']: profile}},
                stages=stages, rl_batch=batch, inactive_tasks=inactive_tasks)

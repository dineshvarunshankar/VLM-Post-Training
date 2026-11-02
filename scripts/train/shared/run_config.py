"""Resolve recipe-specific runs and shared catalogs into pinned stage configurations."""

from copy import deepcopy
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from scripts.train.shared.config import fields, validate_mixture, validate_augmentation
from scripts.train.shared.paths import TRAIN_ROOT, CONFIG_DIR
from scripts.train.shared.contracts import validate_experiment, validate_rl_contract, validate_generation, keys, number
from scripts.train.rl.recipe import resolve_model_recipe, resolve_batch
from scripts.train.storage.cloud_models import run_destination, safe_relative

TOP_LEVEL = ('schema_version run model model_settings answer_format recipe paths cloud wandb '
             'execution dataset mixture sft rl env_file checkpoint sft_run resume')
RUN_FIELDS = ('schema_version run model answer_format recipe paths execution dataset mixture '
              'shared env_file checkpoint resume')
SHARED_FIELDS = 'method attention max_length video augmentation lora data_checks evaluation seed data_loading'
DATA_LOADING_FIELDS = ('dataloader_num_workers dataloader_persistent_workers dataloader_prefetch_factor '
                       'dataloader_pin_memory dataloader_multiprocessing_context')
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


def validate_run_fields(config):
    if not isinstance(config, dict):
        raise ValueError('Run YAML must be a mapping')
    version = config.get('schema_version')
    if type(version) is not int or version not in (1, 2):
        raise ValueError('schema_version must be 1 (legacy) or 2')
    if version == 1:
        fields(config, TOP_LEVEL, 'run YAML')
        return
    recipe = config.get('recipe')
    sections = {'sft': 'sft', 'rl_only': 'rl', 'sft_then_rl': 'sft rl sft_run'}
    if recipe not in sections:
        raise ValueError('recipe must be sft, rl_only, or sft_then_rl')
    required = RUN_FIELDS + ' ' + sections[recipe]
    # Resolved snapshots pin catalog values; editable templates contain none of these.
    if any(key in config for key in ('model_settings', 'cloud', 'wandb')):
        required += ' model_settings cloud wandb'
    fields(config, required, 'run YAML')


def resolve_catalogs(config):
    c = deepcopy(config)
    if c['schema_version'] == 2 and 'model_settings' not in c:
        profiles = yaml.safe_load((CONFIG_DIR / 'models.yaml').read_text())['models']
        if not isinstance(c['model'], str) or c['model'] not in profiles:
            raise ValueError('model must be a key from configs/models.yaml: ' + ', '.join(profiles))
        storage = yaml.safe_load((CONFIG_DIR / 'storage.yaml').read_text())
        c['model_settings'] = deepcopy(profiles[c['model']])
        c['cloud'] = deepcopy(storage['cloud'])
        c['wandb'] = deepcopy(storage['wandb'])
    reject_secrets(c)
    return c


def experiment_config(c):
    if c['schema_version'] == 1:
        experiment = deepcopy(c['sft'])
        if 'extends' in experiment or experiment.get('recipe') != 'sft_only':
            raise ValueError('sft must contain a complete sft_only recipe without extends')
        return experiment
    shared = deepcopy(c['shared'])
    fields(shared, SHARED_FIELDS, 'shared')
    data_loading = shared.pop('data_loading')
    fields(data_loading, DATA_LOADING_FIELDS, 'shared.data_loading')
    seed = shared.pop('seed')
    if type(seed) is not int or seed < 0:
        raise ValueError('shared.seed must be a nonnegative integer')
    if type(data_loading['dataloader_num_workers']) is not int or data_loading['dataloader_num_workers'] < 0:
        raise ValueError('shared.data_loading.dataloader_num_workers must be a nonnegative integer')
    shared['training'] = dict(seed=seed, **data_loading)
    if c['recipe'] != 'rl_only':
        fields(c['sft'], 'effective_batch_size training', 'sft')
        training = c['sft']['training']
        if not isinstance(training, dict) or set(training) & set(shared['training']):
            raise ValueError('sft.training must be a mapping without duplicated shared seed/data-loading fields')
        shared['training'].update(deepcopy(training))
        shared['effective_batch_size'] = c['sft']['effective_batch_size']
        shared['recipe'] = 'sft_only'
    return shared


def load_run(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    reject_secrets(config)
    validate_run_fields(config)
    fields(config['paths'], PATH_FIELDS, 'paths')
    def resolve(value):
        if not isinstance(value, str) or not value.strip() or '${' in value:
            raise ValueError('Paths must be explicit strings; secrets alone belong in .env')
        p = Path(value).expanduser()
        return str((p if p.is_absolute() else path.parent / p).resolve())
    config['paths'] = {key: resolve(value) for key, value in config['paths'].items()}
    for key in ('env_file', 'checkpoint', 'sft_run', 'resume'):
        if config.get(key) is not None:
            config[key] = resolve(config[key])
    return config


def compile_run(config):
    """Pure configuration validation. Does not open datasets or load credentials/models."""
    reject_secrets(config)
    validate_run_fields(config)
    c = resolve_catalogs(config)
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
    if c['wandb']['log_model'] not in ('false', 'end', 'checkpoint'):
        raise ValueError('wandb.log_model must be false, end, or checkpoint as a quoted string')
    if c['wandb']['watch'] not in ('false', 'gradients', 'parameters', 'all'):
        raise ValueError('wandb.watch must be false, gradients, parameters, or all as a quoted string')
    for key in ('entity', 'project'):
        if not isinstance(c['wandb'][key], str) or not c['wandb'][key].strip():
            raise ValueError(f'wandb.{key} must be a nonempty string')
    execution_fields = 'training_gpus training_devices vllm_server_url vllm_model_name'
    if 'vllm_gpu_sharing' in c['execution']:
        execution_fields += ' vllm_gpu_sharing'
        if type(c['execution']['vllm_gpu_sharing']) is not bool:
            raise ValueError('execution.vllm_gpu_sharing must be true or false')
        if c['execution']['vllm_gpu_sharing'] and c['recipe'] != 'sft':
            raise ValueError('execution.vllm_gpu_sharing currently supports SFT only')
    fields(c['execution'], execution_fields, 'execution')
    gpus = c['execution']['training_gpus']
    if type(gpus) is not int or gpus < 1:
        raise ValueError('execution.training_gpus must be a positive integer')
    devices = c['execution']['training_devices']
    if devices is not None and (not isinstance(devices, list) or len(devices) != gpus
                               or len(set(map(str, devices))) != gpus):
        raise ValueError('training_devices must list one unique device per training GPU, or be null')
    for key in ('vllm_server_url', 'vllm_model_name'):
        if not isinstance(c['execution'][key], str) or not c['execution'][key].strip():
            raise ValueError(f'execution.{key} must be a nonempty string')
    url = urlsplit(c['execution']['vllm_server_url'])
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('Use an HTTP(S) vLLM URL without credentials, query, or fragment')
    fields(c['dataset'], 'modality sources', 'dataset')
    if c['dataset']['modality'] != 'RGB' or c['dataset']['sources'] not in (
            ['stabilized'], ['native', 'stabilized'], ['stabilized', 'native']):
        raise ValueError('Choose RGB stabilized or native + stabilized sources')
    if c.get('sft_run') and c['recipe'] != 'sft_then_rl':
        raise ValueError('sft_run is only valid for sft_then_rl')
    experiment = experiment_config(c)
    validate_experiment(experiment, sft=c['recipe'] != 'rl_only' or c['schema_version'] == 1)
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
    keys(profile, 'repository revision generation vllm_tensor_parallel_size', 'model_settings', 'generation_source')
    number(profile['vllm_tensor_parallel_size'], 'model.vllm_tensor_parallel_size', integer=True, strict=True)
    validate_generation(profile['generation'])
    known_models = yaml.safe_load((CONFIG_DIR / 'models.yaml').read_text())['models']
    if c['model'] not in known_models or profile['repository'] != known_models[c['model']]['repository']:
        raise ValueError('model and model_settings.repository must identify the same supported model')
    recipe = c['recipe']
    rl, batch = None, None
    if recipe != 'sft' or c['schema_version'] == 1:
        rl = deepcopy(c['rl'])
        if not isinstance(rl, dict) or 'extends' in rl:
            raise ValueError('rl must contain complete settings without extends')
        validate_rl_contract(rl)
        rl.update(recipe='sft_then_rl' if recipe == 'sft_then_rl' else 'rl_only', experiment_config='experiment.yaml')
        rl['initialization'] = dict(source='best_sft_checkpoint' if recipe == 'sft_then_rl' else 'cloud_base',
                                    continue_adapter=recipe == 'sft_then_rl', fresh_optimizer=True)
        if recipe == 'sft_then_rl':
            rl['sft_config'] = 'experiment.yaml'
        rl['execution']['training_gpus'] = gpus
        rl = resolve_model_recipe(rl, profile)
        batch = resolve_batch(rl)
    if recipe != 'rl_only' or c['schema_version'] == 1:
        effective = experiment['effective_batch_size']
        per_device = experiment['training']['per_device_train_batch_size']
        if any(type(v) is not int or v < 1 for v in (effective, per_device)):
            raise ValueError('SFT batch sizes must be positive integers')
        if effective % (gpus * per_device):
            raise ValueError('SFT effective batch must be divisible by GPUs times per-device batch size')
    def output(stage):
        relative = run_destination('runs', profile['repository'], c['run']['date'], c['run']['name'],
                                   answer_format=c['answer_format'], recipe=stage)
        return str(Path(c['paths']['output_root']) / relative.removeprefix('runs/'))
    stages = []
    if recipe == 'sft' or (recipe == 'sft_then_rl' and not c.get('sft_run')):
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

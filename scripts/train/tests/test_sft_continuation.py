"""Continuation contracts: immutable inputs, optimizer continuity and stable LR tails."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


def api():
    name = 'scripts.train.sft.continuation'
    assert importlib.util.find_spec(name), 'Explicit continuation implementation is missing'
    return importlib.import_module(name)


POLICY = {'scheduler': 'cosine_tail', 'start_learning_rate': None}


def test_launch_extension_allows_only_epoch_and_policy():
    old = {'recipe': 'sft', 'sft': {'training': {'num_train_epochs': 5, 'learning_rate': 2e-5, 'lr_scheduler_type': 'cosine'}}, 'shared': {'seed': 7}}
    new = copy.deepcopy(old)
    new['sft']['training'].update(num_train_epochs=8, continuation=POLICY)
    api().validate_launch_extension(old, new)
    for mutate in [lambda c: c['shared'].update(seed=8),
                   lambda c: c['sft']['training'].update(learning_rate=1e-4),
                   lambda c: c['sft']['training'].update(num_train_epochs=4),
                   lambda c: c.update(recipe='rl_only')]:
        bad = copy.deepcopy(new)
        mutate(bad)
        with pytest.raises(ValueError):
            api().validate_launch_extension(old, bad)


def test_checkpoint_requires_complete_optimizer_scheduler_and_health(tmp_path):
    cp = tmp_path / 'checkpoints' / 'checkpoint-5'
    cp.mkdir(parents=True)
    (cp / 'trainer_state.json').write_text(json.dumps({'global_step': 5}))
    with pytest.raises(ValueError, match='optimizer|scheduler'):
        api().validate_checkpoint(cp, fsdp=False)
    torch.save({}, cp / 'optimizer.pt')
    torch.save({'last_epoch': 5}, cp / 'scheduler.pt')
    assert api().validate_checkpoint(cp, fsdp=False) == 5
    with pytest.raises(ValueError, match='optimizer'):
        api().validate_checkpoint(cp, fsdp=True)
    torch.save({}, cp / 'optimizer.bin')
    assert api().validate_checkpoint(cp, fsdp=True) == 5
    (cp / 'health_state.json').write_text(json.dumps({'stopped': True}))
    with pytest.raises(ValueError, match='stopped'):
        api().validate_checkpoint(cp, fsdp=True)


def old_optimizer(saved_step=5, total=10):
    from transformers import get_cosine_schedule_with_warmup
    p = torch.nn.Parameter(torch.ones(()))
    opt = torch.optim.AdamW([p], lr=2e-5)
    sched = get_cosine_schedule_with_warmup(opt, 0, total)
    for _ in range(saved_step):
        p.grad = torch.ones_like(p)
        opt.step()
        sched.step()
    return p, opt, sched


def restored_tail(saved_step=5, total=10, end=15, start=None):
    p, opt, old = old_optimizer(saved_step, total)
    optim_state = copy.deepcopy(opt.state_dict())
    old_state = copy.deepcopy(old.state_dict())
    tail = api().make_scheduler(opt, end, dict(POLICY, start_learning_rate=start))
    opt.load_state_dict(optim_state)
    tail.load_state_dict(old_state)
    return p, opt, tail


def test_tail_uses_restored_lr_first_and_reaches_zero_without_rebound():
    p, opt, tail = restored_tail()
    assert opt.param_groups[0]['lr'] == pytest.approx(1e-5)
    assert opt.state[p]['step'].item() == 5
    for _ in range(5):
        opt.step(); tail.step()
    assert opt.param_groups[0]['lr'] == pytest.approx(5e-6)
    for _ in range(7):
        opt.step(); tail.step()
    assert opt.param_groups[0]['lr'] == 0


def test_completed_schedule_requires_explicit_lr_and_preserves_adam():
    with pytest.raises(ValueError, match='positive|zero|exhausted'):
        restored_tail(saved_step=10)
    p, opt, tail = restored_tail(saved_step=10, start=2e-6)
    assert opt.param_groups[0]['lr'] == 2e-6
    assert opt.state[p]['step'].item() == 10
    with pytest.raises(ValueError, match='zero|exhausted'):
        restored_tail(start=2e-6)
    with pytest.raises(ValueError, match='horizon|step'):
        restored_tail(saved_step=10, end=10, start=2e-6)


def test_second_resume_keeps_original_anchor_and_rejects_policy_drift():
    p, opt, tail = restored_tail()
    for _ in range(5):
        opt.step(); tail.step()
    saved_opt, saved_sched = copy.deepcopy(opt.state_dict()), copy.deepcopy(tail.state_dict())
    tail2 = api().make_scheduler(opt, 15, POLICY)
    opt.load_state_dict(saved_opt); tail2.load_state_dict(saved_sched)
    assert opt.param_groups[0]['lr'] == pytest.approx(5e-6)
    for _ in range(5):
        opt.step(); tail2.step()
    assert opt.param_groups[0]['lr'] == 0
    wrong = api().make_scheduler(opt, 16, POLICY)
    opt.load_state_dict(saved_opt)
    with pytest.raises(ValueError, match='horizon|policy'):
        wrong.load_state_dict(saved_sched)


def test_metadata_provenance_keeps_original_bytes_and_rejects_data_changes(tmp_path):
    original = {'config': {'training': {'num_train_epochs': 5, 'learning_rate': 2e-5, 'lr_scheduler_type': 'cosine'}},
                'training_arguments': {'num_train_epochs': 5, 'learning_rate': 2e-5, 'lr_scheduler_type': 'cosine'},
                'data_sha256': {'train': 'original'}}
    dest = tmp_path / 'run.json'
    dest.write_text(json.dumps(original, indent=2) + '\n')
    raw = dest.read_bytes()
    continued = copy.deepcopy(original)
    for key in ('config',):
        continued[key]['training'].update(num_train_epochs=8, continuation=POLICY)
    continued['training_arguments'].update(num_train_epochs=8, continuation=POLICY)
    checkpoint = tmp_path / 'recovery' / 'checkpoint-5'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'trainer_state.json').write_text('{"global_step": 5}')
    api().save_continuation_metadata(dest, continued, checkpoint)
    assert dest.read_bytes() == raw
    records = list((tmp_path / 'continuations').glob('*.json'))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record['checkpoint'] == str(checkpoint.resolve())
    assert record['original_run_sha256']
    continued['data_sha256']['train'] = 'changed'
    with pytest.raises(ValueError):
        api().save_continuation_metadata(dest, continued, checkpoint)


def test_continuation_refuses_unarchived_completion_outputs(tmp_path):
    (tmp_path / 'merged').mkdir()
    with pytest.raises(ValueError, match='archive|completion'):
        api().validate_completion_outputs(tmp_path)
    (tmp_path / 'merged').rmdir()
    api().validate_completion_outputs(tmp_path)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, input_ids, labels=None):
        logits = self.weight * input_ids.float()
        return {'loss': ((logits - labels.float()) ** 2).mean(), 'logits': logits}


@pytest.mark.parametrize("saved_step, explicit_lr, first_lr, second_lr", [(10, 2e-6, 2e-6, 1.5e-6), (6, None, 6.909830056250527e-6, 6.25e-6)])
def test_real_trainer_resume_preserves_adam_and_first_update_lr_twice(tmp_path, saved_step, explicit_lr, first_lr, second_lr):
    from transformers import Trainer, TrainingArguments, TrainerCallback
    class ContinuedTrainer(api().ContinuationMixin, Trainer):
        pass
    class StopAt(TrainerCallback):
        def __init__(self, step):
            self.step = step
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == self.step:
                control.should_training_stop = True
                control.should_save = True
    rows = [{'input_ids': torch.tensor([1.]), 'labels': torch.tensor([0.])}] * 2
    def trainer(epochs, policy=None, stop=None):
        args = TrainingArguments(output_dir=str(tmp_path / 'checkpoints'), use_cpu=True,
                                 num_train_epochs=epochs, per_device_train_batch_size=1,
                                 learning_rate=2e-5, lr_scheduler_type='cosine',
                                 warmup_steps=0, save_strategy='epoch', report_to='none',
                                 disable_tqdm=True)
        obj = ContinuedTrainer(model=TinyModel(), args=args, train_dataset=rows,
                               callbacks=[StopAt(stop)] if stop else [])
        obj.continuation_policy = policy
        updates = []
        original = obj.create_optimizer
        def create_optimizer(*a, **kw):
            result = original(*a, **kw)
            def observe(opt, a, kw):
                param = opt.param_groups[0]['params'][0]
                updates.append((opt.param_groups[0]['lr'], float(opt.state.get(param, {}).get('step', 0))))
            obj.optimizer.register_step_pre_hook(observe)
            return result
        obj.create_optimizer = create_optimizer
        return obj, updates
    initial, _ = trainer(5, stop=saved_step)
    initial.train()
    checkpoint = tmp_path / 'checkpoints' / f'checkpoint-{saved_step}'
    # Preserve a best-checkpoint selection in the genuine saved Trainer state.
    state_path = checkpoint / 'trainer_state.json'
    state = json.loads(state_path.read_text())
    state.update(best_metric=.42, best_model_checkpoint=str(checkpoint))
    state_path.write_text(json.dumps(state))
    policy = dict(POLICY, start_learning_rate=explicit_lr)
    resumed, updates = trainer(8, policy, stop=saved_step + 2)
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert updates[0] == pytest.approx((first_lr, saved_step))
    assert resumed.state.global_step == saved_step + 2
    assert resumed.state.best_metric == .42
    assert resumed.state.best_model_checkpoint == str(checkpoint)
    later = tmp_path / 'checkpoints' / f'checkpoint-{saved_step + 2}'
    second, updates2 = trainer(8, policy)
    second.train(resume_from_checkpoint=str(later))
    assert updates2[0] == pytest.approx((second_lr, saved_step + 2))
    assert second.state.global_step == 16
    assert second.optimizer.param_groups[0]['lr'] == 0
    assert second.state.best_metric == .42


def test_production_config_accepts_policy_but_requires_resume():
    from scripts.train.shared.run_config import load_run, compile_run
    config = load_run(Path(__file__).parents[1] / 'configs/runs/gemma12b_sft_stabilized_medical.yaml')
    config['sft']['training']['continuation'] = POLICY
    with pytest.raises(ValueError, match='resume'):
        compile_run(config)
    config['resume'] = '/tmp/run/checkpoints/checkpoint-10'
    plan = compile_run(config)
    assert plan['experiment']['training']['continuation'] == POLICY
    from scripts.train.sft.launch import build_training_config
    training = build_training_config(Path('/tmp/test-continuation-config'), plan['experiment']['training'])
    assert not hasattr(training, 'continuation')


def full_checkpoint(tmp_path, *, step=5):
    cp = tmp_path / 'checkpoints' / f'checkpoint-{step}'
    cp.mkdir(parents=True)
    (cp / 'trainer_state.json').write_text(json.dumps({'global_step': step}))
    p, opt, sched = old_optimizer(step)
    torch.save(opt.state_dict(), cp / 'optimizer.pt')
    torch.save(sched.state_dict(), cp / 'scheduler.pt')
    return cp


def test_checkpoint_rejects_mismatched_scheduler_step(tmp_path):
    cp = full_checkpoint(tmp_path)
    state = torch.load(cp / 'scheduler.pt', weights_only=True)
    state['last_epoch'] = 4
    torch.save(state, cp / 'scheduler.pt')
    with pytest.raises(ValueError, match='step'):
        api().validate_checkpoint(cp)


def test_checkpoint_requires_every_rank_rng_and_policy_health(tmp_path):
    cp = full_checkpoint(tmp_path)
    for rank in range(2):
        with pytest.raises(ValueError, match='RNG'):
            api().validate_checkpoint(cp, world_size=2)
        torch.save({'cpu': torch.get_rng_state()}, cp / f'rng_state_{rank}.pth')
    api().validate_checkpoint(cp, world_size=2)
    with pytest.raises(ValueError, match='health'):
        api().validate_checkpoint(cp, world_size=2, require_health=True)


def test_extension_rejects_non_cosine_original_schedule():
    old = {'recipe': 'sft', 'sft': {'training': {'num_train_epochs': 5, 'lr_scheduler_type': 'linear'}}}
    new = copy.deepcopy(old)
    new['sft']['training'].update(num_train_epochs=8, continuation=POLICY)
    with pytest.raises(ValueError, match='cosine'):
        api().validate_launch_extension(old, new)


def test_scheduler_rejects_corrupted_resume_lr_instead_of_silently_repairing():
    p, opt, tail = restored_tail()
    opt.step(); tail.step()
    state = copy.deepcopy(tail.state_dict())
    optimizer_state = copy.deepcopy(opt.state_dict())
    optimizer_state['param_groups'][0]['lr'] = 1e-3
    again = api().make_scheduler(opt, 15, POLICY)
    opt.load_state_dict(optimizer_state)
    with pytest.raises(ValueError, match='LR|learning rate'):
        again.load_state_dict(state)


def test_ordinary_resume_remains_numerically_identical(tmp_path):
    from transformers import Trainer, TrainingArguments
    class ContinuedTrainer(api().ContinuationMixin, Trainer):
        pass
    class DropoutModel(TinyModel):
        def forward(self, input_ids, labels=None):
            logits = torch.nn.functional.dropout(self.weight * input_ids.float(), .25, self.training)
            return {'loss': ((logits - labels.float()) ** 2).mean(), 'logits': logits}
    rows = [{'input_ids': torch.tensor([float(i + 1)]), 'labels': torch.tensor([0.])} for i in range(2)]
    def args(path):
        return TrainingArguments(output_dir=str(path), use_cpu=True, num_train_epochs=4,
                                 per_device_train_batch_size=1, learning_rate=.01,
                                 lr_scheduler_type='cosine', save_strategy='epoch',
                                 report_to='none', disable_tqdm=True)
    baseline = Trainer(model=DropoutModel(), args=args(tmp_path / 'baseline'), train_dataset=rows)
    baseline.train()
    resumed = ContinuedTrainer(model=DropoutModel(), args=args(tmp_path / 'resumed'), train_dataset=rows)
    resumed.train(resume_from_checkpoint=str(tmp_path / 'baseline/checkpoint-4'))
    torch.testing.assert_close(resumed.model.weight, baseline.model.weight, rtol=0, atol=0)
    assert resumed.state.global_step == baseline.state.global_step == 8
    assert resumed.lr_scheduler.state_dict() == baseline.lr_scheduler.state_dict()


def test_direct_metadata_entrypoint_rejects_unrelated_change_and_keeps_original(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.train.shared.runtime import save_run_metadata
    monkeypatch.delenv('TRIAGE_RUN_CONFIG', raising=False)
    for name in ('catalog.yaml', 'prompts.yaml', 'train.jsonl'):
        (tmp_path / name).write_text('')
    args = SimpleNamespace(run_location={'recipe': 'sft_only'}, resume=None,
                           catalog=tmp_path / 'catalog.yaml', prompts=tmp_path / 'prompts.yaml', split_dir=tmp_path)
    train = {'num_train_epochs': 5, 'learning_rate': 2e-5, 'lr_scheduler_type': 'cosine'}
    config = {'training': train, 'evaluation': {'max_new_tokens': 10}}
    output = tmp_path / 'output'
    save_run_metadata(args, output, {'generation': dict(do_sample=False, temperature=1., top_p=1., top_k=0, min_p=0., presence_penalty=0., repetition_penalty=1.)}, config, train, {}, {'train': []})
    original = (output / 'run.json').read_bytes()
    args.resume = full_checkpoint(output)
    changed = dict(train, num_train_epochs=8, continuation=POLICY)
    record = save_run_metadata(args, output, {'generation': dict(do_sample=False, temperature=1., top_p=1., top_k=0, min_p=0., presence_penalty=0., repetition_penalty=1.)}, dict(config, training=changed), changed, {}, {'train': []})
    assert record['metadata']['training_arguments']['num_train_epochs'] == 8
    assert (output / 'run.json').read_bytes() == original
    with pytest.raises(ValueError):
        save_run_metadata(args, output, {'generation': dict(do_sample=False, temperature=1., top_p=1., top_k=0, min_p=0., presence_penalty=0., repetition_penalty=1.)}, dict(config, training=changed), changed, {}, {'train': [{'label': 'different'}]})


def test_tracking_continuation_requires_original_id(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.train.shared.tracking import configure_tracking
    import os
    monkeypatch.setenv('WANDB_MODE', 'offline')
    args = SimpleNamespace(model='gemma', run_name='medical', run_date='2026-11-04',
                           resume=tmp_path / 'checkpoint-5', continuation_policy=POLICY)
    with pytest.raises(ValueError, match='run_id'):
        configure_tracking(args, 'sft_only', tmp_path)
    (tmp_path / 'tracking').mkdir()
    (tmp_path / 'tracking/run_id.json').write_text('{"id":"original-run"}')
    configure_tracking(args, 'sft_only', tmp_path)
    assert os.environ['WANDB_RUN_ID'] == 'original-run'


def test_pbs_prepare_wires_resume_and_allows_only_ephemeral_endpoint(tmp_path, monkeypatch):
    import sys
    import yaml
    from scripts.train.shared.run_config import load_run, compile_run
    from scripts.train.run import public_config, select_resume_stages
    from scripts.train.psc import launch
    source = Path(__file__).parents[1] / 'configs/runs/gemma12b_sft_stabilized_medical.yaml'
    current = load_run(source)
    current['paths']['output_root'] = str(tmp_path / 'outputs')
    current['sft']['training']['num_train_epochs'] = 5
    current['execution']['vllm_server_url'] = 'http://old-node:8000'
    old_plan = compile_run(current)
    output = Path(old_plan['stages'][0]['output'])
    cp = full_checkpoint(output)
    (cp / 'adapter_model.safetensors').write_bytes(b'fixture')
    for rank in range(8):
        torch.save({}, cp / f'rng_state_{rank}.pth')
    (cp / 'health_state.json').write_text('{"stopped": false}')
    (output / 'run.json').write_text(json.dumps({'launch_config': public_config(old_plan['run']),
                                              'run_location': {'recipe': 'sft_only'}}))
    current['resume'] = str(cp)
    current['sft']['training'].update(num_train_epochs=8, continuation=POLICY)
    config_path = tmp_path / 'continue.yaml'
    config_path.write_text(yaml.safe_dump(current))
    # External dataset, cloud-weight and cluster-runtime checks are separate contracts.
    monkeypatch.setattr('scripts.train.run.preflight', lambda plan: None)
    monkeypatch.setattr('dotenv.dotenv_values', lambda path: {'WANDB_API_KEY': 'test-placeholder'})
    monkeypatch.setattr(launch, 'verify_server_runtime', lambda run: None)
    monkeypatch.setattr('scripts.train.storage.cloud_models.verify_local_source', lambda *a, **kw: None)
    job = tmp_path / 'job'
    monkeypatch.setattr(sys, 'argv', ['launch.py', 'prepare', '--config', str(config_path), '--scheduler', 'pbs',
                                    '--job-root', str(job), '--master', 'new-node', '--allocated-nodes', '1'])
    launch.main()
    saved = json.loads((job / 'plan.json').read_text())
    assert saved['stages'][0]['resume'] == str(cp)
    assert saved['run']['execution']['vllm_server_url'] == 'http://new-node:8000'
    saved['run']['execution']['vllm_model_name'] = 'different-model'
    with pytest.raises(ValueError):
        select_resume_stages(saved)


def test_missing_best_weights_fails_before_training(tmp_path):
    cp = full_checkpoint(tmp_path)
    best = tmp_path / 'checkpoints/checkpoint-2'
    best.mkdir()
    (best / 'trainer_state.json').write_text('{"global_step":2}')
    state = json.loads((cp / 'trainer_state.json').read_text())
    state['best_model_checkpoint'] = str(best)
    (cp / 'trainer_state.json').write_text(json.dumps(state))
    with pytest.raises(ValueError, match='weights'):
        api().validate_checkpoint(cp)
    (best / 'adapter_model.safetensors').write_bytes(b'fixture')
    api().validate_checkpoint(cp)


def test_explicit_restart_lr_rejects_checkpoint_before_original_end(tmp_path):
    cp = full_checkpoint(tmp_path)
    state_path = cp / 'trainer_state.json'
    state_path.write_text('{"global_step":5,"max_steps":10}')
    with pytest.raises(ValueError, match='exhausted|completed'):
        api().validate_checkpoint(cp, policy=dict(POLICY, start_learning_rate=2e-6))


def test_selected_checkpoint_requires_weights_when_requested(tmp_path):
    cp = full_checkpoint(tmp_path)
    with pytest.raises(ValueError, match='weights'):
        api().validate_checkpoint(cp, require_weights=True)
    (cp / 'model.safetensors.index.json').write_text('{"weight_map":{"weight":"model-00001-of-00001.safetensors"}}')
    with pytest.raises(ValueError, match='weights|shard'):
        api().validate_checkpoint(cp, require_weights=True)
    (cp / 'model-00001-of-00001.safetensors').write_bytes(b'fixture')
    api().validate_checkpoint(cp, require_weights=True)

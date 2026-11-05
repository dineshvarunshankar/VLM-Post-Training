"""A completed continuation can extend again with an explicit stopping rule."""
import copy
import json
from pathlib import Path

import pytest
import torch
from scripts.train.sft import continuation as api

POLICY = {'scheduler': 'cosine_tail', 'start_learning_rate': 2e-6,
          'early_stopping': {'patience': 3, 'threshold': 0.0}}


def test_tail_metadata_allows_only_explicit_epoch_and_stop_changes(tmp_path):
    train = {'num_train_epochs': 5, 'learning_rate': 2e-5, 'lr_scheduler_type': 'cosine'}
    old = {'config': {'training': copy.deepcopy(train)}, 'training_arguments': copy.deepcopy(train),
           'data_sha256': {'train': 'unchanged'}}
    path = tmp_path / 'run.json'
    path.write_text(json.dumps(old))
    original = path.read_bytes()
    new = copy.deepcopy(old)
    for target in (new['config']['training'], new['training_arguments']):
        target.update(num_train_epochs=12, continuation=POLICY, early_stopping=POLICY['early_stopping'])
    cp = tmp_path / 'checkpoints/checkpoint-8'
    cp.mkdir(parents=True)
    (cp / 'trainer_state.json').write_text('{"global_step": 8}')
    api.save_continuation_metadata(path, new, cp)
    assert path.read_bytes() == original
    for field, value in [('learning_rate', 1e-4), ('early_stopping', {'patience': 4})]:
        bad = copy.deepcopy(new)
        bad['config']['training'][field] = value
        with pytest.raises(ValueError):
            api.validate_metadata_extension(old, bad)
    bad = copy.deepcopy(new)
    bad['data_sha256']['train'] = 'changed'
    with pytest.raises(ValueError):
        api.validate_metadata_extension(old, bad)


def test_completed_tail_reanchors_only_explicit_zero_lr_and_increased_horizon():
    from scripts.train.tests.test_sft_continuation import restored_tail
    p, opt, tail = restored_tail(saved_step=10, end=16, start=2e-6)
    for _ in range(6):
        opt.step(); tail.step()
    saved_opt, saved_sched = copy.deepcopy(opt.state_dict()), copy.deepcopy(tail.state_dict())
    for end, policy, corrupt_lr in [(20, POLICY, False), (20, dict(POLICY, start_learning_rate=None), False),
                                    (20, POLICY, True), (16, POLICY, False)]:
        next_tail = api.make_scheduler(opt, end, policy)
        opt.load_state_dict(saved_opt)
        if corrupt_lr:
            opt.param_groups[0]['lr'] = 1e-8
        if end == 20 and policy['start_learning_rate'] is not None and not corrupt_lr:
            next_tail.load_state_dict(saved_sched)
            assert next_tail.anchor_step == 16
            assert opt.param_groups[0]['lr'] == 2e-6
            assert opt.state[p]['step'].item() == 16
        else:
            with pytest.raises(ValueError):
                next_tail.load_state_dict(saved_sched)
    active = copy.deepcopy(saved_sched)
    active['last_epoch'] = 15
    next_tail = api.make_scheduler(opt, 20, POLICY)
    opt.load_state_dict(saved_opt)
    with pytest.raises(ValueError):
        next_tail.load_state_dict(active)


def test_real_trainer_five_eight_twelve_preserves_state_and_stops_on_third_miss(tmp_path):
    from transformers import Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
        def forward(self, input_ids, labels=None):
            logits = torch.nn.functional.dropout(self.weight * input_ids.float(), .25, self.training)
            return {'loss': ((logits - labels.float()) ** 2).mean(), 'logits': logits}
    class Continued(api.ContinuationMixin, Trainer):
        pass
    class StopAt(TrainerCallback):
        def __init__(self, step):
            self.step = step
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == self.step:
                control.should_training_stop = True
                control.should_save = True
    rows = [{'input_ids': torch.tensor([1.]), 'labels': torch.tensor([0.])}]
    scores = [.1, .2, .3, .4, .5, .6, .7, .8, .75, .74, .73, .72]
    def make(epochs, policy=None, stop=None):
        args = TrainingArguments(output_dir=str(tmp_path/'run/checkpoints'), use_cpu=True,
                  num_train_epochs=epochs, per_device_train_batch_size=1,
                  learning_rate=2e-5, lr_scheduler_type='cosine', warmup_steps=0,
                  eval_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True,
                  metric_for_best_model='macro_f1', greater_is_better=True,
                  restore_callback_states_from_checkpoint=True, report_to='none', disable_tqdm=True)
        callbacks = [StopAt(stop)] if stop else []
        if policy and 'early_stopping' in policy:
            callbacks.append(EarlyStoppingCallback(policy['early_stopping']['patience']))
        obj = Continued(model=Model(), args=args, train_dataset=rows, eval_dataset=rows, callbacks=callbacks)
        obj.continuation_policy = policy
        obj.compute_metrics = lambda _: {'macro_f1': scores[round(obj.state.epoch)-1]}
        updates = []
        original = obj.create_optimizer
        def create(*a, **kw):
            result = original(*a, **kw)
            def record(opt, a, kw):
                p = opt.param_groups[0]['params'][0]
                cbs = [c for c in obj.callback_handler.callbacks if isinstance(c, EarlyStoppingCallback)]
                updates.append((opt.param_groups[0]['lr'], float(opt.state.get(p, {}).get('step', 0)),
                                obj.state.best_metric, cbs[0].early_stopping_patience_counter if cbs else None))
            obj.optimizer.register_step_pre_hook(record)
            return result
        obj.create_optimizer = create
        return obj, updates
    first, _ = make(5)
    first.train()
    cp5 = tmp_path/'run/checkpoints/checkpoint-5'
    second, updates2 = make(8, {'scheduler': 'cosine_tail', 'start_learning_rate': 2e-6})
    second.train(resume_from_checkpoint=str(cp5))
    assert updates2[0][:3] == pytest.approx((2e-6, 5, .5))
    assert second.state.global_step == 8
    cp8 = tmp_path/'run/checkpoints/checkpoint-8'
    original = {p.name: p.read_bytes() for p in cp8.iterdir() if p.is_file()}
    third, updates3 = make(12, POLICY, stop=9)
    third.train(resume_from_checkpoint=str(cp8))
    assert updates3[0] == pytest.approx((2e-6, 8, .8, 0))
    assert third.state.global_step == 9
    assert all((cp8/name).read_bytes() == value for name, value in original.items())
    again, updates4 = make(12, POLICY)
    again.train(resume_from_checkpoint=str(tmp_path/'run/checkpoints/checkpoint-9'))
    assert updates4[0] == pytest.approx((2e-6 * (1 + 2**-.5) / 2, 9, .8, 1))
    assert again.state.global_step == 11
    assert again.state.best_metric == .8
    assert Path(again.state.best_model_checkpoint).name == 'checkpoint-8'
    state = json.loads((tmp_path/'run/checkpoints/checkpoint-11/trainer_state.json').read_text())
    cb = state['stateful_callbacks']['EarlyStoppingCallback']
    assert cb['args']['early_stopping_patience'] == 3
    assert cb['attributes']['early_stopping_patience_counter'] == 3
    # A second uninterrupted continuation from the identical checkpoint must
    # reproduce the dropout-dependent updates despite the intermediate resume.
    from safetensors.torch import load_file
    final_weights = tmp_path/'run/checkpoints/checkpoint-11/model.safetensors'
    expected_weights = {name: tensor.clone() for name, tensor in load_file(final_weights).items()}
    uninterrupted, full_updates = make(12, POLICY)
    uninterrupted.train(resume_from_checkpoint=str(cp8))
    assert full_updates == updates3 + updates4
    actual_weights = load_file(final_weights)
    assert actual_weights.keys() == expected_weights.keys()
    for name in expected_weights:
        torch.testing.assert_close(actual_weights[name], expected_weights[name], rtol=0, atol=0)
    assert uninterrupted.lr_scheduler.state_dict() == again.lr_scheduler.state_dict()
    with pytest.raises(ValueError):
        api.validate_checkpoint(tmp_path/'run/checkpoints/checkpoint-11', policy=POLICY)


def test_compiled_tail_stop_enables_native_callback_restore():
    from scripts.train.shared.run_config import load_run, compile_run
    from scripts.train.sft.launch import build_training_config, build_callbacks
    config = load_run(Path(__file__).parents[1]/'configs/runs/gemma12b_sft_stabilized_medical.yaml')
    config['sft']['training'].update(num_train_epochs=12, continuation=POLICY,
                                     early_stopping=POLICY['early_stopping'])
    config['resume'] = '/tmp/run/checkpoints/checkpoint-5264'
    plan = compile_run(config)
    training = build_training_config(Path('/tmp/tail-test'), plan['experiment']['training'])
    assert training.restore_callback_states_from_checkpoint
    assert training.num_train_epochs == 12
    assert build_callbacks(plan['experiment']['training'])[0].early_stopping_patience == 3


def test_tail_checkpoint_rejects_lost_callback_counter_and_inconsistent_horizon(tmp_path):
    cp = tmp_path/'checkpoints/checkpoint-9'
    cp.mkdir(parents=True)
    state = {'global_step': 9, 'max_steps': 12, 'stateful_callbacks': {
        'EarlyStoppingCallback': {'args': {'early_stopping_patience': 3, 'early_stopping_threshold': 0.0},
                                 'attributes': {'early_stopping_patience_counter': 1}}}}
    torch.save({}, cp/'optimizer.pt')
    scheduler = {'continuation_schema': 1, 'end_step': 12, 'last_epoch': 9, 'policy': POLICY}
    torch.save(scheduler, cp/'scheduler.pt')
    (cp/'trainer_state.json').write_text(json.dumps(state))
    assert api.validate_checkpoint(cp, policy=POLICY) == 9
    for mutate in [lambda s: s.update(max_steps=13), lambda s: s.update(stateful_callbacks={}),
                   lambda s: s['stateful_callbacks']['EarlyStoppingCallback']['args'].update(early_stopping_threshold=.1),
                   lambda s: s['stateful_callbacks']['EarlyStoppingCallback']['args'].update(early_stopping_patience=4),
                   lambda s: s['stateful_callbacks']['EarlyStoppingCallback']['attributes'].update(early_stopping_patience_counter=3)]:
        bad = copy.deepcopy(state)
        mutate(bad)
        (cp/'trainer_state.json').write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            api.validate_checkpoint(cp, policy=POLICY)


def test_saved_tail_stopping_rule_cannot_be_removed(tmp_path):
    cp = tmp_path/'checkpoints/checkpoint-8'
    cp.mkdir(parents=True)
    (cp/'trainer_state.json').write_text(json.dumps({'global_step': 8, 'max_steps': 8}))
    torch.save({}, cp/'optimizer.pt')
    torch.save({'continuation_schema': 1, 'end_step': 8, 'last_epoch': 8, 'policy': POLICY}, cp/'scheduler.pt')
    with pytest.raises(ValueError, match='stopping'):
        api.validate_checkpoint(cp, policy={'scheduler': 'cosine_tail', 'start_learning_rate': 2e-6})

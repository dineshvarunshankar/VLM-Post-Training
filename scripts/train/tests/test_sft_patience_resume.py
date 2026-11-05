"""Same-horizon patience overrides preserve the native training trajectory."""
import copy
import json
from pathlib import Path

import pytest
import torch
from scripts.train.sft import continuation as api

POLICY = {'scheduler': 'preserve', 'early_stopping': {'patience': 3, 'threshold': 0.0}}


def test_same_horizon_allows_only_explicit_patience_increase():
    old = {'recipe': 'sft', 'sft': {'training': {'num_train_epochs': 8, 'learning_rate': 1e-4,
           'lr_scheduler_type': 'cosine', 'early_stopping': {'patience': 2, 'threshold': 0.0}}}}
    new = copy.deepcopy(old)
    new['sft']['training'].update(continuation=POLICY, early_stopping=POLICY['early_stopping'])
    api.validate_launch_extension(old, new)
    for key, value in [('num_train_epochs', 9), ('learning_rate', 2e-4),
                       ('early_stopping', {'patience': 3, 'threshold': .01})]:
        bad = copy.deepcopy(new)
        bad['sft']['training'][key] = value
        with pytest.raises(ValueError):
            api.validate_launch_extension(old, bad)
    bad = copy.deepcopy(new)
    bad['shared'] = {'seed': 17}
    with pytest.raises(ValueError):
        api.validate_launch_extension(old, bad)


def test_real_stopped_trainer_preserves_schedule_counter_best_rng_and_second_resume(tmp_path):
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
    class Observe(TrainerCallback):
        def __init__(self, stop=None):
            self.stop, self.begin, self.updates = stop, None, []
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == self.stop:
                control.should_training_stop = True
                control.should_save = True
    scores = [.3, .4, .35, .38, .45, .43, .42, .41]
    rows = [{'input_ids': torch.tensor([1.]), 'labels': torch.tensor([0.])}]
    def make(path, patience, *, policy=None, stop=None, cls=Continued):
        args = TrainingArguments(output_dir=str(path), use_cpu=True, num_train_epochs=8,
                  per_device_train_batch_size=1, learning_rate=.01, lr_scheduler_type='cosine',
                  warmup_steps=0, eval_strategy='epoch', save_strategy='epoch',
                  load_best_model_at_end=True, metric_for_best_model='macro_f1', greater_is_better=True,
                  restore_callback_states_from_checkpoint=True, report_to='none', disable_tqdm=True,
                  seed=42)
        observe = Observe(stop)
        obj = cls(model=Model(), args=args, train_dataset=rows, eval_dataset=rows,
                  callbacks=[EarlyStoppingCallback(patience), observe])
        obj.compute_metrics = lambda _: {'macro_f1': scores[int(round(obj.state.epoch)) - 1]}
        obj.continuation_policy = policy
        original = obj.create_optimizer
        def create(*a, **kw):
            result = original(*a, **kw)
            def record(opt, a, kw):
                cb = next(c for c in obj.callback_handler.callbacks if isinstance(c, EarlyStoppingCallback))
                p = opt.param_groups[0]['params'][0]
                observe.updates.append((opt.param_groups[0]['lr'], float(opt.state.get(p, {}).get('step', 0)),
                                        cb.early_stopping_patience, cb.early_stopping_patience_counter,
                                        obj.state.best_metric))
            obj.optimizer.register_step_pre_hook(record)
            return result
        obj.create_optimizer = create
        return obj, observe
    initial, _ = make(tmp_path/'run/checkpoints', 2)
    initial.train()
    assert initial.state.global_step == 4
    cp = tmp_path/'run/checkpoints/checkpoint-4'
    original = {p.name: p.read_bytes() for p in cp.iterdir() if p.is_file()}
    resumed, observed = make(tmp_path/'run/checkpoints', 3, policy=POLICY, stop=5)
    resumed.train(resume_from_checkpoint=str(cp))
    assert observed.updates[0] == pytest.approx((.005, 4, 3, 2, .4))
    assert resumed.state.global_step == 5
    assert resumed.state.best_metric == .45
    assert all((cp/name).read_bytes() == data for name, data in original.items())
    again, observed2 = make(tmp_path/'run/checkpoints', 3, policy=POLICY)
    again.train(resume_from_checkpoint=str(tmp_path/'run/checkpoints/checkpoint-5'))
    assert observed2.updates[0][2:4] == (3, 0)
    assert again.state.global_step == 8
    baseline, baseline_observed = make(tmp_path/'baseline', 3, cls=Trainer)
    baseline.train()
    assert [x[:2] for x in observed.updates + observed2.updates] == [x[:2] for x in baseline_observed.updates[4:]]
    torch.testing.assert_close(again.model.weight, baseline.model.weight, rtol=0, atol=0)
    assert again.lr_scheduler.state_dict() == baseline.lr_scheduler.state_dict()
    assert Path(again.state.best_model_checkpoint).name == 'checkpoint-5'


def test_preserved_policy_records_override_without_rewriting_original(tmp_path):
    train = {'num_train_epochs': 8, 'learning_rate': .01, 'lr_scheduler_type': 'cosine',
             'early_stopping': {'patience': 2, 'threshold': 0.0}}
    old = {'config': {'training': train}, 'training_arguments': train,
           'data_sha256': {'train': 'original'}, 'mixture': {'manifest': {'sha256': 'pinned'}}}
    path = tmp_path/'run.json'
    path.write_text(json.dumps(old))
    raw = path.read_bytes()
    cp = tmp_path/'checkpoints/checkpoint-4'
    cp.mkdir(parents=True)
    (cp/'trainer_state.json').write_text('{"global_step": 4}')
    new = copy.deepcopy(old)
    for target in [new['config']['training'], new['training_arguments']]:
        target.update(continuation=POLICY, early_stopping=POLICY['early_stopping'])
    record = api.save_continuation_metadata(path, new, cp)
    assert path.read_bytes() == raw
    assert record['metadata']['training_arguments']['early_stopping']['patience'] == 3
    for field in ['data_sha256', 'mixture']:
        bad = copy.deepcopy(new)
        bad[field] = {'changed': True}
        with pytest.raises(ValueError):
            api.save_continuation_metadata(path, bad, cp)
    (tmp_path/'test_results.json').write_text('{}')
    with pytest.raises(ValueError, match='Archive'):
        api.save_continuation_metadata(path, new, cp)


def test_preserved_checkpoint_rejects_exhausted_schedule_or_incompatible_callback(tmp_path):
    cp = tmp_path/'checkpoints/checkpoint-4'
    cp.mkdir(parents=True)
    state = {'global_step': 4, 'max_steps': 8, 'stateful_callbacks': {
        'EarlyStoppingCallback': {'args': {'early_stopping_patience': 2, 'early_stopping_threshold': 0.0},
                                 'attributes': {'early_stopping_patience_counter': 2}}}}
    torch.save({}, cp/'optimizer.pt')
    torch.save({'last_epoch': 4}, cp/'scheduler.pt')
    (cp/'trainer_state.json').write_text(json.dumps(state))
    assert api.validate_checkpoint(cp, policy=POLICY) == 4
    for mutate in [lambda s: s.update(max_steps=4),
                   lambda s: s.update(stateful_callbacks={}),
                   lambda s: s['stateful_callbacks']['EarlyStoppingCallback']['args'].update(early_stopping_patience=4),
                   lambda s: s['stateful_callbacks']['EarlyStoppingCallback']['attributes'].update(early_stopping_patience_counter=3)]:
        bad = copy.deepcopy(state)
        mutate(bad)
        (cp/'trainer_state.json').write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            api.validate_checkpoint(cp, policy=POLICY)


def test_compiled_preserve_policy_keeps_callback_restore_and_native_schedule():
    from scripts.train.shared.run_config import load_run, compile_run
    from scripts.train.sft.launch import build_training_config
    config = load_run(Path(__file__).parents[1]/'configs/cot/qwen3.5-9b.yaml')
    config['sft']['training'].update(continuation=POLICY, early_stopping=POLICY['early_stopping'])
    with pytest.raises(ValueError, match='resume'):
        compile_run(config)
    config['resume'] = '/tmp/run/checkpoints/checkpoint-2828'
    plan = compile_run(config)
    training = build_training_config(Path('/tmp/preserve-test'), plan['experiment']['training'])
    assert training.restore_callback_states_from_checkpoint
    assert training.num_train_epochs == 8
    assert training.lr_scheduler_type == 'cosine'

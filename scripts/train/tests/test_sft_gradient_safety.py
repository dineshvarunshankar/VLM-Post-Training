"""Reject invalid updates on every rank; retain finite optimizer behavior."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def guard_module():
    return importlib.import_module('scripts.train.sft.gradient_safety')


def worker(rank, rendezvous, output):
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        module=guard_module()
        p=torch.nn.Parameter(torch.ones(2));p.grad=torch.ones(2)
        if rank==1:p.grad[0]=float('nan')
        optimizer=torch.optim.SGD([p],lr=.1)
        failed=False
        try:
            module.require_finite_gradients([p],torch.tensor(1.0),device=torch.device('cpu'))
            optimizer.step()
        except FloatingPointError:
            failed=True
        Path(output,f'{rank}.json').write_text(json.dumps({'failed':failed,'parameter':p.tolist()}))
    finally:dist.destroy_process_group()


class GradientSafetyTests(unittest.TestCase):
    def module(self):
        try:return guard_module()
        except ModuleNotFoundError:self.fail('SFT pre-update gradient protection is missing')

    def test_finite_gradient_is_unchanged_and_can_update(self):
        m=self.module();p=torch.nn.Parameter(torch.ones(2));p.grad=torch.tensor([.5,-.5])
        m.require_finite_gradients([p],torch.tensor(.7071),device=p.device)
        torch.optim.SGD([p],lr=.1).step()
        torch.testing.assert_close(p,torch.tensor([.95,1.05]))

    def test_rejects_nonfinite_gradient_or_overflowed_norm(self):
        m=self.module()
        for value,norm in [(float('nan'),1.),(float('inf'),1.),(1.,float('inf'))]:
            with self.subTest(value=value,norm=norm):
                p=torch.nn.Parameter(torch.ones(2));p.grad=torch.full_like(p,value)
                with self.assertRaises(FloatingPointError):
                    m.require_finite_gradients([p],torch.tensor(norm),device=p.device)

    def test_one_bad_rank_blocks_both_updates(self):
        self.module()
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(worker,args=(str(Path(tmp,'rendezvous')),tmp),nprocs=2,join=True)
            for rank in range(2):
                result=json.loads(Path(tmp,f'{rank}.json').read_text())
                self.assertTrue(result['failed'])
                self.assertEqual(result['parameter'],[1.,1.])

    def test_real_trainer_blocks_optimizer_before_corruption(self):
        m=self.module()
        from transformers import Trainer,TrainingArguments
        class Guarded(m.FiniteGradientMixin,Trainer):pass
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.ones(1))
                self.weight.register_hook(lambda grad:grad*float('nan'))
            def forward(self,input_ids,labels=None):
                return {'loss':self.weight.square().sum(),'logits':self.weight*input_ids.float()}
        with tempfile.TemporaryDirectory() as tmp:
            model=Model()
            trainer=Guarded(model=model,args=TrainingArguments(output_dir=tmp,use_cpu=True,
                max_steps=1,report_to='none',save_strategy='no',max_grad_norm=1.,disable_tqdm=True),
                train_dataset=[{'input_ids':torch.ones(1),'labels':torch.ones(1)}])
            with self.assertRaises(FloatingPointError):trainer.train()
            torch.testing.assert_close(model.weight,torch.ones(1))
            self.assertEqual(len(trainer.optimizer.state),0)

if __name__=='__main__':unittest.main()

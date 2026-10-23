"""Offline P0 checks reject coherent saved-result and checkpoint tampering."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from safetensors.torch import save_file

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p0_analysis',ROOT/'experiments/02_supervised/finish_p0.py')
module=importlib.util.module_from_spec(spec) if Path(spec.origin).is_file() else None
if module is not None:spec.loader.exec_module(module)
import sys
sys.path.insert(0,str(ROOT/'src'))
from jev.evaluation.metrics import evaluate_predictions
from jev.models.pointer import PointerHead
from jev.training.pointer_runner import save_head_checkpoint
from jev.training.runner import make_scheduler


class P0AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(module,'P0 offline verification tool must exist')
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)
        self.expected=[dict(case_id='c1',workflow='invoice',question_id='q1',type='choice',option_ids=['n','y'],
                            option_texts=['No','Yes'],gold=[.2,.8],hard_target=1,score_levels=None)]
        self.rows=[dict(self.expected[0],label_logits=[0.,1.],probabilities=[.2689414322376251,.7310585975646973],candidate_letters=['A','B'])]

    def fixture(self):
        p=self.path
        (p/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        np.savez_compressed(p/'logits.npz',logits=np.array([[0,1]],np.float32),valid_mask=np.array([[1,1]],bool),
                            option_ids=np.array([['n','y']]),case_ids=np.array(['c1']),question_ids=np.array(['q1']))
        metrics=evaluate_predictions(self.rows,bootstrap_samples=5,seed=43)
        (p/'metrics.json').write_text(json.dumps(metrics))
        summary=dict(n_cases=1,n_decisions=1,metrics=metrics['overall'],head_sha256='h',feature_manifest_sha256='f',
                     mode='cached frozen features; five independent question rows per case; no latency claim',
                     step_zero='random Pointer Head; not reused from B0')
        (p/'summary.json').write_text(json.dumps(summary))
        (p/'started.json').write_text(json.dumps(dict(head_sha256='h',feature_manifest_sha256='f')))

    def audit(self):
        return module.audit_cached_dev(self.path,self.expected,5,43,0,'h','f')

    def test_cached_random_step_zero_is_recomputed_without_timing(self):
        self.fixture();result=self.audit()
        self.assertEqual(result['metrics']['overall']['accuracy'],1.)
        self.assertFalse((self.path/'timing.json').exists())

    def test_consistently_forged_gold_rejected(self):
        self.rows[0]['gold']=[.8,.2];self.rows[0]['hard_target']=0;self.fixture()
        with self.assertRaisesRegex(ValueError,'gold|hard_target'):self.audit()

    def test_probabilities_must_follow_saved_logits(self):
        self.rows[0]['probabilities']=[.1,.9];self.fixture()
        with self.assertRaisesRegex(ValueError,'probabilit'):self.audit()

    def test_step_zero_cannot_claim_baseline_reuse(self):
        self.fixture();value=module.read_json(self.path/'summary.json');value['step_zero']='B0 reused'
        (self.path/'summary.json').write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'random'):self.audit()

    def test_wrong_checkpoint_or_feature_generation_rejected(self):
        self.fixture();value=module.read_json(self.path/'started.json');value['head_sha256']='forged'
        (self.path/'started.json').write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'Head'):self.audit()

    def test_real_head_checkpoint_includes_trainer_and_rejects_backbone(self):
        head=PointerHead();opt=torch.optim.AdamW(head.parameters(),lr=.001)
        scheduler=make_scheduler(opt,450,.05);dest=self.path/'checkpoints/step000000'
        manifest=save_head_checkpoint(head,opt,scheduler,dest,dict(epoch=0,group=0,step=0))
        selection=dict(epoch=0,step=0,checkpoint='checkpoints/step000000',head_sha256=manifest['head_sha256'])
        checked=module.audit_head(self.path,selection)
        self.assertEqual(checked['stored_parameters'],262144)
        self.assertIn('trainer.pt',checked['files_sha256'])
        weights={**head.state_dict(),'model.embed_tokens.weight':torch.zeros(1)}
        save_file(weights,str(dest/'head.safetensors'))
        manifest['files_sha256']['head.safetensors']=module.sha256_file(dest/'head.safetensors')
        (dest/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'keys|Head|head'):module.audit_head(self.path,selection)

    def test_history_checks_pointer_learning_rate_and_finite_updates(self):
        rows=[dict(step=i,epoch=(i-1)//150,group=(i-1)%150+1,questions=32,loss=1.,grad_norm=.2,seconds=.01,
                   learning_rate=.001*((i-1)/23 if i<=23 else .5*(1+math.cos(math.pi*(i-24)/427)))) for i in range(1,451)]
        file=self.path/'train.jsonl';file.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.assertEqual(module.audit_history(file)['optimizer_steps'],450)
        rows[100]['learning_rate']*=.02;file.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError,'learning rate'):module.audit_history(file)

    def test_baseline_cost_uses_saved_whole_run_memory_without_inventing_split_peak(self):
        (self.path/'test').mkdir()
        (self.path/'test/timing.json').write_text(json.dumps(dict(p50_case_seconds=.3,p95_case_seconds=.4)))
        (self.path/'metadata.json').write_text(json.dumps(dict(memory=dict(peak_allocated_bytes=1000,definition='whole-run peak'))))
        value=module.reference_cost(self.path,'b0',42)
        self.assertEqual(value['p50_case_seconds'],.3)
        self.assertEqual(value['memory']['peak_allocated_bytes'],1000)
        self.assertEqual(value['memory_scope'],'whole baseline run; dev and test')

    def test_failed_run_rejected_before_publishing(self):
        (self.path/'metadata.json').write_text(json.dumps(dict(status='failed',mode='train')))
        with self.assertRaisesRegex(ValueError,'complete'):module.finish(self.path)
        self.assertFalse((self.path/'analysis.json').exists())
        self.assertFalse((self.path/'verification.json').exists())

if __name__=='__main__':unittest.main()

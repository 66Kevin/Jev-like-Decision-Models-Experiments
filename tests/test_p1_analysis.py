"""Offline P1 audits detect coherent result forgery and joint-generation drift."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'experiments/02_supervised/finish_p1.py'
module=None
if path.is_file():
    spec=importlib.util.spec_from_file_location('p1_analysis',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
from jev.evaluation.metrics import evaluate_predictions
from jev.training.pointer_lora import create_joint_optimizer,save_joint_checkpoint
from jev.training.runner import make_scheduler
from test_pointer_lora import tiny_pair


class P1AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(module,'P1 offline verifier must exist')
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)
        self.expected=[dict(case_id='c1',workflow='invoice',question_id='q1',type='choice',option_ids=['n','y'],
                            option_texts=['No','Yes'],gold=[.2,.8],hard_target=1,score_levels=None)]
        self.rows=[dict(self.expected[0],label_logits=[0.,1.],probabilities=[.2689414322376251,.7310585975646973],
                        candidate_letters=['A','B'],input_tokens=20,case_latency_seconds=.4)]

    def fixture(self):
        p=self.p
        (p/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        np.savez_compressed(p/'logits.npz',logits=np.array([[0,1]],np.float32),valid_mask=np.array([[1,1]],bool),
                            option_ids=np.array([['n','y']]),case_ids=np.array(['c1']),question_ids=np.array(['q1']))
        metrics=evaluate_predictions(self.rows,bootstrap_samples=5,seed=43)
        (p/'metrics.json').write_text(json.dumps(metrics))
        timing=dict(n_cases=1,n_decisions=1,total_seconds=.4,mean_case_seconds=.4,p50_case_seconds=.4,p95_case_seconds=.4,
                    cases=[dict(case_id='c1',workflow='invoice',total_seconds=.4,tokenization_seconds=.1,
                                inference_seconds=.3,input_lengths=[20],padded_tokens=20)])
        (p/'timing.json').write_text(json.dumps(timing))
        (p/'summary.json').write_text(json.dumps(dict(n_cases=1,n_decisions=1,metrics=metrics['overall'],p50_case_seconds=.4,p95_case_seconds=.4)))

    def test_prediction_audit_and_each_npz_array_decompressed_once(self):
        self.fixture();original=np.load;counts={}
        class CountLoad:
            def __init__(self,*args,**kwargs):self.a=original(*args,**kwargs);self.files=self.a.files
            def __enter__(self):return self
            def __exit__(self,*args):self.a.close()
            def __getitem__(self,key):counts[key]=counts.get(key,0)+1;return self.a[key]
        with patch.object(module.np,'load',side_effect=CountLoad):
            result=module.audit_prediction_split(self.p,self.expected,5,43)
        self.assertEqual(result['metrics']['overall']['accuracy'],1.)
        self.assertEqual(counts,dict.fromkeys(['logits','valid_mask','option_ids','case_ids','question_ids'],1))

    def test_coherent_gold_tamper_rejected(self):
        self.rows[0].update(gold=[.8,.2],hard_target=0);self.fixture()
        with self.assertRaisesRegex(ValueError,'gold|hard_target'):module.audit_prediction_split(self.p,self.expected,5,43)

    def test_probabilities_cannot_disagree_with_logits(self):
        self.rows[0]['probabilities']=[.1,.9];self.fixture()
        with self.assertRaisesRegex(ValueError,'probabilit'):module.audit_prediction_split(self.p,self.expected,5,43)

    def test_coherent_logit_change_cannot_hide_npz_disagreement(self):
        self.rows[0].update(label_logits=[0.,2.],probabilities=[.11920292,.88079708]);self.fixture()
        with self.assertRaisesRegex(ValueError,'logits differ'):module.audit_prediction_split(self.p,self.expected,5,43)

    def test_history_checks_both_learning_rates(self):
        rows=[]
        for i in range(1,451):
            s=i-1;factor=s/23 if s<23 else .5*(1+math.cos(math.pi*(s-23)/427))
            rows.append(dict(step=i,epoch=s//150,group=s%150+1,questions=32,loss=1.,grad_norm=.2,seconds=.01,
                             learning_rate=2e-4*factor,lora_learning_rate=2e-4*factor,head_learning_rate=.001*factor))
        path=self.p/'train.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.assertEqual(module.audit_history(path)['optimizer_steps'],450)
        rows[100]['head_learning_rate']*=.2;path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError,'learning rate'):module.audit_history(path)

    def test_selection_must_minimize_dev_nll_with_earlier_tie(self):
        values=[dict(epoch=i,step=i*150,hard_nll=n,checkpoint=f'checkpoints/step{i*150:06d}',head_sha256=f'h{i}',adapter_sha256=f'a{i}') for i,n in enumerate([4.,1.,1.,2.])]
        module.audit_selection(values[1],values)
        with self.assertRaisesRegex(ValueError,'selection'):module.audit_selection(values[2],values)

    def test_all_seeds_must_be_locked_before_official_test(self):
        (self.p/'selections.json').write_text(json.dumps(dict(locked=True,criterion='minimum dev hard NLL; earlier step breaks ties',seeds={'42':{}})))
        with self.assertRaisesRegex(ValueError,'three|seed'):module.audit_lock(self.p,dict(completed_seeds={'42':{}}))

    def test_real_joint_checkpoint_detects_mixed_head_adapter_generation(self):
        torch.manual_seed(42);model,head=tiny_pair();opt=create_joint_optimizer(model,head);sched=make_scheduler(opt,450,.05)
        target=self.p/'checkpoints/step000000'
        m=save_joint_checkpoint(model,head,opt,sched,target,dict(epoch=0,group=0,step=0))
        selected=dict(epoch=0,step=0,checkpoint='checkpoints/step000000',head_sha256=m['head_sha256'],adapter_sha256=m['adapter_sha256'])
        result=module.audit_joint(self.p,selected,expected_lora=64,expected_head=48,hidden=8,projection=3)
        self.assertEqual(result['stored_parameters'],112)
        state=torch.load(target/'trainer.pt',map_location='cpu',weights_only=False);state['head_sha256']='other';torch.save(state,target/'trainer.pt')
        m['files_sha256']['trainer.pt']=module.sha256_file(target/'trainer.pt');(target/'manifest.json').write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'Mixed'):module.audit_joint(self.p,selected,expected_lora=64,expected_head=48,hidden=8,projection=3)

    def test_failed_run_rejects_before_publication(self):
        (self.p/'metadata.json').write_text(json.dumps(dict(status='failed',mode='train')))
        with self.assertRaisesRegex(ValueError,'complete'):module.finish(self.p)
        self.assertFalse((self.p/'analysis.json').exists());self.assertFalse((self.p/'verification.json').exists())


if __name__=='__main__':unittest.main()

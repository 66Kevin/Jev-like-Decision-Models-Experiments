import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from jev.models.pointer import PointerHead
from jev.training.pointer_runner import (
    cache_features, load_feature_cache, head_train_update, save_head_checkpoint,
    restore_head_checkpoint, validate_head_checkpoint, score_cached, require_test_lock,
    assert_frozen, feature_records, build_cache_key,
)
from jev.training.runner import make_scheduler, lock_selections


def fixture():
    decisions=[]
    for i in range(5):
        decisions.append({'case_id':'case1','question_id':f'q{i}','workflow':'w',
                          'option_ids':['a','b'],'option_texts':['one','two'],
                          'hard_target':i%2,'gold':[.8,.2], 'state':{},'instructions':'pick',
                          'type':'categorical'})
    return [{'case_id':'case1','workflow':'w','official_split':'train','decisions':decisions}]


def prepared(*args, **kwargs):
    return {'input_ids':torch.ones(5,3,dtype=torch.long),'query_positions':torch.tensor([2]*5),'candidate_positions':torch.tensor([[0,1,0,0,0]]*5),
            'input_lengths':[3]*5, 'prompts':['prompt']*5,
            'candidate_character_spans':[[(0,1),(1,2)]]*5}


def extracted(*args):
    torch.manual_seed(1)
    candidates=torch.randn(5,5,4);candidates[:,2:]=0
    return {'query_features':torch.randn(5,4), 'candidate_features':candidates,
            'candidate_mask':torch.tensor([[True,True,False,False,False]]*5),
            'option_counts':[2]*5,'query_positions':torch.tensor([2]*5),
            'candidate_positions':torch.tensor([[0,1,0,0,0]]*5)}


class PointerRunnerTests(unittest.TestCase):
    def cache(self,directory,split='train',key=None):
        with patch('jev.training.pointer_runner.prepare_pointer_case',side_effect=prepared), \
             patch('jev.training.pointer_runner.extract_pointer_features',side_effect=extracted):
            return cache_features(directory,split,fixture(),None,None,key or {'protocol':'v1'})

    def test_cache_roundtrip_identity_tamper_and_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'train'
            result=self.cache(path)
            loaded=load_feature_cache(path,{'protocol':'v1'},fixture())
            self.assertTrue(torch.equal(result['query_features'],loaded['query_features']))
            self.assertFalse(loaded['query_features'].is_inference())
            self.assertEqual(loaded['records'][0]['question_id'],'q0')
            with self.assertRaisesRegex(ValueError,'key'):
                load_feature_cache(path,{'protocol':'v2'},fixture())
            other=copy.deepcopy(fixture());other[0]['decisions'][0]['option_ids'].reverse()
            with self.assertRaisesRegex(ValueError,'identit'):
                load_feature_cache(path,{'protocol':'v1'},other)
            with (path/'features.npz').open('ab') as stream:stream.write(b'tamper')
            with self.assertRaisesRegex(ValueError,'checksum'):
                load_feature_cache(path,{'protocol':'v1'},fixture())

    def test_partial_dev_fails_closed_and_no_test_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'dev';path.mkdir();(path/'started.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'Incomplete'):
                self.cache(path,'dev')
            with self.assertRaisesRegex(ValueError,'train/dev'):
                self.cache(Path(tmp)/'test','test')

    def test_head_only_update_random_zero_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached=self.cache(Path(tmp)/'train')
            torch.manual_seed(42);head=PointerHead(4,3)
            opt=torch.optim.AdamW(head.parameters(),lr=.01,foreach=False)
            scheduler=make_scheduler(opt,8,.05)
            zero=score_cached(head,cached)
            self.assertEqual(len(zero),5)
            self.assertNotIn('reused_from_b0',zero[0])
            for _ in range(2):head_train_update(head,cached,list(range(5)),opt,scheduler,1.)
            checkpoint=Path(tmp)/'checkpoint';save_head_checkpoint(head,opt,scheduler,checkpoint,{'step':2})
            before={k:v.clone() for k,v in head.state_dict().items()}
            expected=head_train_update(head,cached,list(range(5)),opt,scheduler,1.)
            expected_weights=copy.deepcopy(head.state_dict())
            ids=[id(p) for p in head.parameters()]
            progress=restore_head_checkpoint(head,opt,scheduler,checkpoint)
            self.assertEqual(progress,{'step':2});self.assertEqual(ids,[id(p) for p in head.parameters()])
            self.assertTrue(all(torch.equal(head.state_dict()[k],v) for k,v in before.items()))
            actual=head_train_update(head,cached,list(range(5)),opt,scheduler,1.)
            self.assertEqual(expected,actual)
            self.assertTrue(all(torch.equal(head.state_dict()[k],v) for k,v in expected_weights.items()))
            (checkpoint/'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'checksum'):validate_head_checkpoint(checkpoint)

    def test_all_seeds_locked_before_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'selections.json'
            with self.assertRaises(ValueError):require_test_lock(path,[42,43,44])
            picks={str(seed):{'step':0} for seed in [42,43,44]}
            lock_selections(path,picks,[42,43,44])
            self.assertEqual(require_test_lock(path,[42,43,44]),picks)
            picks['42']['step']=2
            with self.assertRaises(ValueError):require_test_lock(path,[42,43,44],picks)

    def test_frozen_checks_include_buffers_and_grad_flags(self):
        model=torch.nn.Linear(4,4).eval().requires_grad_(False)
        model.register_buffer('rotary',torch.tensor([1.],dtype=torch.float32))
        first=assert_frozen(model)
        self.assertEqual(first,assert_frozen(model,first))
        model.rotary.add_(1)
        with self.assertRaisesRegex(RuntimeError,'changed'):assert_frozen(model,first)

    def test_cache_key_changes_with_protocol_and_has_no_gold(self):
        key=build_cache_key({'model':{'dtype':'bfloat16'},'head':{'hidden_size':4,'projection_dim':3,'max_candidates':5},
                             'evaluation':{'max_input_tokens':8192}}, {'source':'x'}, {'weights':'y'}, 'split')
        self.assertIn('batch_policy',key)
        self.assertNotIn('gold',json.dumps(key))
        records=feature_records(fixture())
        self.assertNotIn('gold',records[0]);self.assertNotIn('hard_target',records[0])


    def entrypoint(self):
        spec=importlib.util.spec_from_file_location('p0_test_entrypoint',Path(__file__).resolve().parents[1]/'experiments/02_supervised/run_p0.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def tiny_config(self):
        import yaml
        config=yaml.safe_load((Path(__file__).resolve().parents[1]/'experiments/02_supervised/configs/p0.yaml').read_text())
        config['model']['device']='cpu'
        config['head'].update(hidden_size=4,projection_dim=3,expected_parameters=24)
        config['evaluation']['bootstrap_samples']=2
        return config

    def test_smoke_has_no_dev_or_test_and_real_step2_resume(self):
        module=self.entrypoint();config=self.tiny_config()
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'seed42'
            cached=self.cache(Path(tmp)/'features')
            with patch.object(module,'cached_dev',side_effect=AssertionError('heldout touched')), \
                 patch.object(module,'verify_train_probe') as probe:
                outcome=module.train_seed(config,42,{'train':fixture(),'dev':None,'test':None},
                                          {'train':cached},directory,None,None,True)
                self.assertEqual(outcome,{'smoke':True})
                saved=json.loads((directory/'smoke.json').read_text())
                self.assertTrue(saved['resumed_third_update'])
                self.assertEqual(json.loads((directory/'latest.json').read_text())['step'],2)
                first_loss=saved['after_resume_loss']
                module.train_seed(config,42,{'train':fixture(),'dev':None,'test':None},
                                  {'train':cached},directory,None,None,True)
                self.assertEqual(first_loss,json.loads((directory/'smoke.json').read_text())['after_resume_loss'])
                self.assertEqual(probe.call_count,2)
                self.assertFalse(list(directory.glob('dev*')))
                self.assertFalse((directory/'test').exists())

    def test_epoch_zero_scores_random_head_for_each_seed(self):
        module=self.entrypoint();config=self.tiny_config()
        with tempfile.TemporaryDirectory() as tmp:
            cached=self.cache(Path(tmp)/'features')
            metrics={'overall':{'accuracy':.5,'hard_nll':1.}}
            for seed in (42,43):
                with patch.object(module,'evaluate_predictions',return_value=metrics), \
                     patch.object(module,'verify_train_probe'):
                    module.train_seed(config,seed,{'train':fixture(),'dev':fixture()},
                                      {'train':cached,'dev':cached},Path(tmp)/f'seed{seed}',None,None,False)
            first=json.loads((Path(tmp)/'seed42/dev_epoch0/summary.json').read_text())
            second=json.loads((Path(tmp)/'seed43/dev_epoch0/summary.json').read_text())
            self.assertNotEqual(first['head_sha256'],second['head_sha256'])
            self.assertIn('random Pointer Head',first['step_zero'])
            self.assertEqual(json.loads((Path(tmp)/'seed42/selection.json').read_text())['step'],0)

    def test_uncached_test_measures_preparation_and_backbone_and_refuses_partial(self):
        module=self.entrypoint();config=self.tiny_config();head=PointerHead(4,3)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);seed_dir=root/'seed42';seed_dir.mkdir()
            picks={str(seed):{'head_sha256':module.head_hash(head)} for seed in config['seeds']}
            lock_selections(root/'selections.json',picks,config['seeds'])
            train=[]
            for i in range(3):
                case=copy.deepcopy(fixture()[0]);case['case_id']=f'c{i}';train.append(case)
            result={'probabilities':[[.5,.5]]*5,'label_logits':[[0.,0.]]*5,'inference_seconds':1.}
            clock=iter(range(100))
            with patch.object(module,'prepare_pointer_case',side_effect=prepared) as prepare_call, \
                 patch.object(module,'read_pointer_case',return_value=result) as infer, \
                 patch.object(module.time,'perf_counter',side_effect=lambda:next(clock)), \
                 patch.object(module,'evaluate_predictions',return_value={'overall':{'accuracy':.5,'hard_nll':1.}}), \
                 patch.object(module.torch.cuda,'empty_cache'), \
                 patch.object(module.torch.cuda,'reset_peak_memory_stats'), \
                 patch.object(module.torch.cuda,'max_memory_allocated',return_value=1), \
                 patch.object(module.torch.cuda,'max_memory_reserved',return_value=1), \
                 patch.object(module.torch.cuda,'memory_allocated',return_value=1):
                summary=module.evaluate_uncached(None,head,None,fixture(),train,seed_dir,config,42,root/'selections.json')
                self.assertEqual(prepare_call.call_count,4);self.assertEqual(infer.call_count,4)
                timing=json.loads((seed_dir/'test/timing.json').read_text())
                self.assertGreater(timing['cases'][0]['total_seconds'],timing['cases'][0]['tokenization_seconds'])
                self.assertIn('uncached',timing['definition'])
                self.assertEqual(module.evaluate_uncached(None,head,None,fixture(),train,seed_dir,config,42,root/'selections.json'),summary)
                self.assertEqual(infer.call_count,4)
                (seed_dir/'test/summary.json').unlink()
                with self.assertRaisesRegex(RuntimeError,'Partial test'):
                    module.evaluate_uncached(None,head,None,fixture(),train,seed_dir,config,42,root/'selections.json')

if __name__=='__main__':unittest.main()

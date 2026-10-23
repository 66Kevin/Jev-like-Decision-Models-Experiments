"""P1 orchestration guards and tiny real joint training/recovery."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from jev.training.pointer_lora import cache_pointer_decisions
from jev.training.pointer_runner import head_hash
from jev.training.runner import lock_selections
from test_pointer import FastCharacterTokenizer
from test_pointer_lora import tiny_pair
from test_readout import decision

ROOT=Path(__file__).resolve().parents[1]


def entry():
    path=ROOT/'experiments/02_supervised/run_p1.py'
    if not path.is_file():return None
    spec=importlib.util.spec_from_file_location('p1_test_entry',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def cases():
    rows=[]
    for i in range(5):
        row=decision('pick',2);row['question_id']=f'q{i}';rows.append(row)
    return [{'case_id':rows[0]['case_id'],'workflow':rows[0]['workflow'],'decisions':rows}]


class P1RunnerTests(unittest.TestCase):
    def setUp(self):
        self.run=entry();self.assertIsNotNone(self.run,'P1 runner must exist')
        self.config=yaml.safe_load((ROOT/'experiments/02_supervised/configs/p1.yaml').read_text())

    def tiny(self):
        config=copy.deepcopy(self.config);config['model']['device']='cpu'
        config['head'].update(hidden_size=8,projection_dim=3,expected_parameters=48)
        config['lora'].update(rank=2,alpha=4,expected_target_modules=2)
        config['training'].update(expected_frozen_parameters=524544,expected_lora_parameters=64,
                                  expected_trainable_parameters=112)
        config['evaluation']['bootstrap_samples']=2
        return config

    def test_fixed_config_profile_and_gpu_guard(self):
        self.run.validate_config(self.config)
        for group,key,value in [('training','lora_learning_rate',.001),('head','projection_dim',64),
                                ('training','microbatch_questions',2),('evaluation','kernel_backend','fla')]:
            other=copy.deepcopy(self.config);other[group][key]=value
            with self.assertRaises(ValueError):self.run.validate_config(other)
        profile={'status':'passed','train_only':True,'hostname':'alg26','physical_gpu':0,
                 'zero_lora_pointer_native_bitexact':True,'lm_head_projection_calls':0,
                 'selected_microbatch':8,'selected_accumulation':4,'head_and_lora_updated':True,
                 'hidden_feature_cache':False,'lora_a_updated_after_nonzero_b':True,
                 'frozen_backbone_before':{'state_sha256':'a'*64,'unique_parameters':752393024},
                 'frozen_backbone_after':{'state_sha256':'a'*64,'unique_parameters':752393024},
                 'parameters':{'lora_parameters':10822656,'head_parameters':262144,'trainable_parameters':11084800,
                               'frozen_parameters':752393024},
                 'gradient_agreement':{'head':{'cosine':.999,'relative_l2':.03},'lora':{'cosine':.999,'relative_l2':.03}}}
        self.run.validate_profile(profile,self.config)
        bad=copy.deepcopy(profile);bad['frozen_backbone_after']['state_sha256']='b'*64
        with self.assertRaises(ValueError):self.run.validate_profile(bad,self.config)
        with patch.object(self.run.socket,'gethostname',return_value='alg20'):
            with self.assertRaisesRegex(ValueError,'alg26'):self.run.gpu_guard()

    def test_physical_gpu_uuid_must_match_torch_logical_gpu(self):
        import types
        with patch.object(self.run.subprocess,'check_output',return_value='GPU-aaa, NVIDIA RTX 4090, 0000:01:00.0, 535.54.03'), \
             patch.object(self.run.torch.cuda,'get_device_properties',return_value=types.SimpleNamespace(uuid='GPU-aaa')):
            evidence=self.run.device_provenance()
            self.assertEqual(evidence['torch_gpu_uuid'],'GPU-aaa')
        with patch.object(self.run.subprocess,'check_output',return_value='GPU-bbb, NVIDIA RTX 4090, bus, driver'), \
             patch.object(self.run.torch.cuda,'get_device_properties',return_value=types.SimpleNamespace(uuid='GPU-aaa')):
            with self.assertRaisesRegex(ValueError,'UUID'):self.run.device_provenance()

    def test_resume_drift_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);source=directory/'source.py';source.write_text('original')
            (directory/'source').mkdir();(directory/'source/source.py').write_text('original')
            profile=directory/'profile.json';profile.write_text('{}')
            (directory/'preflight_train.json').write_text('{}')
            (directory/'config.yaml').write_text(yaml.safe_dump(self.config))
            digest=self.run.sha256_file(source)
            meta={'mode':'train','status':'failed','source_sha256':{'source.py':digest},'runtime_fingerprint':{'x':1},
                  'profile_sha256':self.run.sha256_file(profile),'config_sha256':self.run.sha256_file(directory/'config.yaml'),'model_files_sha256':{'w':'x'},
                  'raw_manifest_sha256':'raw','split_manifest_sha256':'split'}
            with patch.object(self.run,'ROOT',directory):
                self.run.validate_resume(directory,self.config,meta,{'x':1},profile,{'w':'x'},'raw','split',False)
                before={str(p):p.read_bytes() for p in directory.rglob('*') if p.is_file()}
                with self.assertRaisesRegex(ValueError,'runtime'):
                    self.run.validate_resume(directory,self.config,meta,{'x':2},profile,{'w':'x'},'raw','split',False)
                self.assertEqual(before,{str(p):p.read_bytes() for p in directory.rglob('*') if p.is_file()})
                changed=copy.deepcopy(self.config);changed['output_root']='changed'
                (directory/'config.yaml').write_text(yaml.safe_dump(changed))
                with self.assertRaisesRegex(ValueError,'config'):
                    self.run.validate_resume(directory,changed,meta,{'x':1},profile,{'w':'x'},'raw','split',False)
                (directory/'config.yaml').write_text(yaml.safe_dump(self.config))
                source.write_text('changed')
                with self.assertRaisesRegex(ValueError,'source'):
                    self.run.validate_resume(directory,self.config,meta,{'x':1},profile,{'w':'x'},'raw','split',False)

    def test_smoke_real_joint_step2_recovery_third_update_no_heldout(self):
        config=self.tiny();tokenizer=FastCharacterTokenizer();data=cases()
        cached=cache_pointer_decisions(data[0]['decisions'],tokenizer)
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'seed42'
            def pair(*args):
                torch.manual_seed(42);model,head=tiny_pair();return model,head,tokenizer,{}
            with patch.object(self.run,'create_pair',side_effect=pair), \
                 patch.object(self.run,'backend'), \
                 patch.object(self.run,'validate_counts'), \
                 patch.object(self.run,'evaluate_native',side_effect=AssertionError('heldout touched')):
                self.assertEqual(self.run.train_seed(config,42,{'train':data},directory,True),{'smoke':True})
                first=json.loads((directory/'smoke.json').read_text())
                self.assertEqual(first['optimizer_updates'],3);self.assertTrue(first['native_reload_bitexact'])
                self.assertTrue(first['resumed_third_update']);self.assertTrue(first['no_heldout_inference'])
                self.assertEqual(json.loads((directory/'latest.json').read_text())['step'],2)
                self.run.train_seed(config,42,{'train':data},directory,True)
                self.assertEqual(first['after_resume_loss'],json.loads((directory/'smoke.json').read_text())['after_resume_loss'])
                self.assertFalse(list(directory.glob('dev*')));self.assertFalse((directory/'test').exists())
                tok=json.loads((directory/'tokenization.json').read_text());self.assertFalse(tok['hidden_features_cached'])

    def test_epoch_zero_is_actual_random_joint_model_each_seed_and_selection_hashes(self):
        config=self.tiny();tokenizer=FastCharacterTokenizer();data=cases();hashes=[]
        def pair(config,seed):
            torch.manual_seed(seed);model,head=tiny_pair();return model,head,tokenizer,{}
        def evaluate(name,cases,model,head,tokenizer,directory,config,seed,train,lock_path=None):
            hashes.append((seed,name,head_hash(head)))
            target=directory/name;target.mkdir()
            result={'metrics':{'hard_nll':1.,'accuracy':.5},'head_sha256':head_hash(head),
                    'adapter_sha256':self.run.adapter_hash(model)}
            self.run.write_json(target/'summary.json',result);return result
        with tempfile.TemporaryDirectory() as tmp, patch.object(self.run,'create_pair',side_effect=pair), \
             patch.object(self.run,'backend'),patch.object(self.run,'validate_counts'), \
             patch.object(self.run,'evaluate_native',side_effect=evaluate):
            picks=[]
            for seed in (42,43):picks.append(self.run.train_seed(config,seed,{'train':data,'dev':data},Path(tmp)/f'seed{seed}',False))
            zeros=[row for row in hashes if row[1]=='dev_epoch0']
            self.assertEqual(len(zeros),2);self.assertNotEqual(zeros[0][2],zeros[1][2])
            self.assertTrue(all(p['step']==0 and 'adapter_sha256' in p and 'head_sha256' in p for p in picks))
            self.assertFalse(list(Path(tmp).rglob('reused_from_b0.json')))

    def test_uncached_native_test_scores_once_with_full_timing_and_complete_reuse(self):
        model,head=tiny_pair();config=self.tiny();tokenizer=FastCharacterTokenizer();data=cases()
        train=[]
        for i in range(3):
            row=copy.deepcopy(data[0]);row['case_id']=f'warm{i}';row['official_split']='train';train.append(row)
        with tempfile.TemporaryDirectory() as tmp, patch.object(self.run,'backend'):
            directory=Path(tmp);lock=directory/'selections.json'
            picks={str(seed):{'head_sha256':head_hash(head),'adapter_sha256':self.run.adapter_hash(model)} for seed in config['seeds']}
            lock_selections(lock,picks,config['seeds'])
            with patch.object(self.run,'prepare_pointer_case',wraps=self.run.prepare_pointer_case) as prepare_call, \
                 patch.object(self.run,'read_pointer',wraps=self.run.read_pointer) as read:
                summary=self.run.evaluate_native('test',data,model,head,tokenizer,directory,config,42,train,lock)
                self.assertEqual(prepare_call.call_count,4);self.assertEqual(read.call_count,4)
                timing=json.loads((directory/'test/timing.json').read_text())
                self.assertGreater(timing['cases'][0]['total_seconds'],timing['cases'][0]['tokenization_seconds'])
                self.assertIn('uncached',timing['definition']);self.assertIn('4090',timing['definition'])
                self.assertEqual(len((directory/'test/predictions.jsonl').read_text().splitlines()),5)
                self.assertEqual(summary,self.run.evaluate_native('test',data,model,head,tokenizer,directory,config,42,train,lock))
                self.assertEqual(read.call_count,4)
                head.query_projection.weight.data.add_(1)
                with self.assertRaisesRegex(ValueError,'locked'):self.run.evaluate_native('test',data,model,head,tokenizer,directory,config,42,train,lock)

    def test_test_requires_all_seeds_and_partial_dev_fails_closed(self):
        model,head=tiny_pair();config=self.tiny();tokenizer=FastCharacterTokenizer();data=cases()
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);lock=directory/'selections.json'
            with self.assertRaises(ValueError):self.run.evaluate_native('test',data,model,head,tokenizer,directory,config,42,data,lock)
            picks={str(seed):{'head_sha256':head_hash(head),'adapter_sha256':self.run.adapter_hash(model)} for seed in config['seeds']}
            lock_selections(lock,picks,config['seeds'])
            (directory/'dev_epoch0').mkdir();(directory/'dev_epoch0/started.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'Partial'):
                self.run.evaluate_native('dev_epoch0',data,model,head,tokenizer,directory,config,42,data)
            (directory/'test').mkdir();(directory/'test/started.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'Partial'):
                self.run.evaluate_native('test',data,model,head,tokenizer,directory,config,42,data,lock)

if __name__=='__main__':unittest.main()

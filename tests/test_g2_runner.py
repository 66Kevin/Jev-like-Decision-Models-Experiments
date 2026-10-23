"""G2 fixed recipe and durable full-weight pipeline contracts."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import yaml
import torch
from test_lora import TinyLM
from jev.training.full import enable_full_training
from jev.training.runner import make_scheduler

ROOT = Path(__file__).resolve().parents[1]
try:
    runner = importlib.import_module('jev.training.full_runner')
except ModuleNotFoundError:
    runner = None

class G2RunnerContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner)

    def test_fixed_recipe_rejects_soft_loss_lora_precision_and_batch_drift(self):
        config = yaml.safe_load((ROOT / 'experiments/02_supervised/configs/g2.yaml').read_text())
        runner.validate_config(config)
        for group, key, value in [('training', 'loss', 'soft_ce'), ('training', 'master_dtype', 'bfloat16'),
                                  ('training', 'learning_rate', 2e-4), ('training', 'fused', False),
                                  ('evaluation', 'temperature', .7), ('training', 'gradient_accumulation', 16)]:
            changed = copy.deepcopy(config); changed[group][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): runner.validate_config(changed)
        changed = copy.deepcopy(config); changed['lora'] = {}
        with self.assertRaises(ValueError): runner.validate_config(changed)

    def test_latest_published_after_complete_generation_and_prune_keeps_epochs(self):
        model = enable_full_training(TinyLM()); opt = torch.optim.AdamW(model.parameters())
        sched = make_scheduler(opt, 450, .05)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for step in [2, 50, 150, 200]:
                runner.publish_checkpoint(model, opt, sched, directory, {'epoch': 0, 'group': step, 'step': step}, 150)
            self.assertEqual(json.loads((directory/'latest.json').read_text())['step'], 200)
            self.assertEqual(sorted(p.name for p in (directory/'checkpoints').iterdir()), ['step000150', 'step000200'])
            before = (directory/'latest.json').read_bytes()
            with patch.object(runner, 'save_full_checkpoint', side_effect=RuntimeError('disk failed')):
                with self.assertRaises(RuntimeError):
                    runner.publish_checkpoint(model, opt, sched, directory, {'step': 250}, 150)
            self.assertEqual((directory/'latest.json').read_bytes(), before)

    def test_history_reconcile_archives_only_uncommitted_tail_and_rejects_missing_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'train.jsonl'
            path.write_text(''.join(json.dumps({'step': s})+'\n' for s in [1,2,3]))
            runner.reconcile_history(path, 2)
            self.assertEqual([json.loads(x)['step'] for x in path.read_text().splitlines()], [1,2])
            self.assertEqual(len(list(path.parent.glob('train.uncommitted.*.jsonl'))), 1)
            with self.assertRaisesRegex(ValueError, 'history'):
                runner.reconcile_history(path, 3)

    def test_case_bootstrap_compares_mean_seed_correctness_not_probability_ensemble(self):
        def row(case, p): return {'case_id':case,'question_id':'q','option_ids':['a','b'], 'hard_target':0,'probabilities':p}
        reference = [[row('a',[.51,.49]),row('b',[.2,.8])], [row('a',[.01,.99]),row('b',[.8,.2])]]
        candidates = [[row('a',[.6,.4]),row('b',[.9,.1])]]
        result = runner.paired_seed_bootstrap(reference,candidates,samples=100)
        self.assertEqual(result['mean_accuracy_delta'], .5)
        bad = copy.deepcopy(candidates); bad[0][0]['case_id']='z'
        with self.assertRaises(ValueError): runner.paired_seed_bootstrap(reference,bad,samples=100)

    def test_profile_must_be_passed_train_only_and_match_microbatch(self):
        config = yaml.safe_load((ROOT / 'experiments/02_supervised/configs/g2.yaml').read_text())
        profile = {'status':'passed','train_only':True,'selected_microbatch':4,'selected_accumulation':8}
        runner.validate_profile(profile,config)
        for key,value in [('status','failed'),('train_only',False),('selected_microbatch',2)]:
            changed=dict(profile); changed[key]=value
            with self.assertRaises(ValueError): runner.validate_profile(changed,config)

    def test_full_update_accumulation_matches_and_original_head_changes(self):
        torch.manual_seed(12)
        a=enable_full_training(TinyLM()); b=copy.deepcopy(a)
        rows=[{'input_ids':[i+2,7],'hard_target':i%2,'option_count':2,'label_token_ids':[11,12], 'identity':{'case_id':str(i)}} for i in range(5)]
        original=a.lm_head.weight.detach().clone()
        for model,micro in [(a,2),(b,5)]:
            opt=torch.optim.AdamW(model.parameters(),lr=.01)
            runner.train_update(model,rows,list(range(5)),0,opt,make_scheduler(opt,10,0),micro,1.)
        for x,y in zip(a.parameters(),b.parameters()): self.assertTrue(torch.allclose(x,y,atol=1e-6))
        self.assertFalse(torch.equal(a.lm_head.weight,original))

    def test_smoke_and_smoke_resume_use_train_only_and_resume_before_third_update(self):
        spec=importlib.util.spec_from_file_location('g2_smoke_test',ROOT/'experiments/02_supervised/run_g2.py')
        entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        config=yaml.safe_load((ROOT/'experiments/02_supervised/configs/g2.yaml').read_text())
        config['training']['fused']=False
        config['training']['expected_parameters']=sum(p.numel() for p in TinyLM().parameters())
        config['model']['device']='cpu'
        rows=[{'input_ids':[3,7],'hard_target':0,'option_count':2,'label_token_ids':[11,12], 'identity':{'case_id':'a'}}]*4800
        from types import SimpleNamespace
        def load(_):
            model=TinyLM().bfloat16().requires_grad_(False)
            model.get_input_embeddings=lambda:model.embed_tokens
            model.get_output_embeddings=lambda:model.lm_head
            return model,SimpleNamespace(pad_token_id=0),{}
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(entry,'load_local_model',side_effect=load), \
             patch.object(entry,'probe_logits',return_value=[[1.,2.]]), \
             patch.object(entry,'isolated_probe',return_value=[[1.,2.]]), \
             patch.object(entry,'cache_decisions',return_value=rows), \
             patch.object(entry,'backend',return_value={}), \
             patch.object(entry,'isolated_dev',side_effect=AssertionError('smoke accessed dev')), \
             patch.object(entry,'evaluate',side_effect=AssertionError('smoke accessed test')), \
             patch.object(torch.cuda,'reset_peak_memory_stats'), \
             patch.object(torch.cuda,'max_memory_allocated',return_value=0):
            directory=Path(temporary)/'seed42';splits={'train':[{'case_id':'a','decisions':[]}]}
            for _ in range(2):
                self.assertEqual(entry.train_seed(config,42,splits,directory,{'model_files_sha256':{}},True),{'smoke':True})
                report=json.loads((directory/'smoke.json').read_text())
                self.assertTrue(report['fused_resume_update_passed'])
                self.assertEqual(report['optimizer_updates'],3)
                self.assertEqual(len((directory/'train.jsonl').read_text().splitlines()),2)
            self.assertFalse((directory/'dev_epoch0').exists())

if __name__=='__main__': unittest.main()

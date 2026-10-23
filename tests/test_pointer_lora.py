"""P1 differentiable Pointer + LoRA and atomic joint recovery contracts."""
import copy
import importlib
import json
from pathlib import Path
import random
import tempfile
import types
import unittest

import numpy as np
import torch
from transformers import PreTrainedModel, PretrainedConfig

from jev.models.pointer import PointerHead, prepare_pointer_case
from jev.training.lora import inject_lora, hard_candidate_loss
from jev.training.runner import make_scheduler
from test_pointer import FastCharacterTokenizer
from test_readout import decision

try:
    p1 = importlib.import_module('jev.training.pointer_lora')
except ModuleNotFoundError:
    p1 = None


class TinyText(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(65538, 8)
        self.q_proj = torch.nn.Linear(8, 8, bias=False)
        self.down_proj = torch.nn.Linear(8, 8, bias=False)
        self.register_buffer('rotary_fp32', torch.arange(8).float())
        self.calls = []

    def forward(self, input_ids, attention_mask, position_ids, use_cache=False, return_dict=True):
        self.calls.append({'grad_enabled': torch.is_grad_enabled(), 'use_cache': use_cache})
        h = self.embed_tokens(input_ids)
        # Context-sensitive differentiable backbone makes query/candidate gradients real.
        h = h + (h * attention_mask[..., None]).cumsum(1) / (position_ids[..., None] + 1)
        return types.SimpleNamespace(last_hidden_state=self.down_proj(torch.tanh(self.q_proj(h))))


class TinyPointerLM(PreTrainedModel):
    config_class = PretrainedConfig
    def __init__(self):
        super().__init__(PretrainedConfig(vocab_size=65538, hidden_size=8, tie_word_embeddings=True))
        self.model = TinyText()
        self.lm_head = torch.nn.Linear(8, 65538, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
    def forward(self, **kwargs):
        raise AssertionError('P1 must never call LM projection')
    def get_input_embeddings(self):
        return self.model.embed_tokens
    def get_output_embeddings(self):
        return self.lm_head


def tiny_pair(alpha=4):
    base = TinyPointerLM().to(torch.bfloat16)
    base.model.rotary_fp32 = base.model.rotary_fp32.float()
    return inject_lora(base, rank=2, alpha=alpha, expected_count=2), PointerHead(8, 3)


class PointerLoRAContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(p1, 'P1 primitives must exist')
        torch.manual_seed(29)
        self.tokenizer = FastCharacterTokenizer()
        self.rows = [decision('short', 2), decision('Long instruction ' * 4, 5)]
        self.rows[1]['hard_target'] = 4
        self.rows[1]['question_id'] = 'second'

    def test_cache_tokens_only_rebases_positions_matches_b0_and_rejects_bad_rows(self):
        cached = p1.cache_pointer_decisions(self.rows, self.tokenizer)
        prepared = p1.collate_pointer_decisions(cached, self.tokenizer.pad_token_id)
        expected = prepare_pointer_case(self.rows, self.tokenizer)
        for name in ('input_ids', 'attention_mask', 'position_ids', 'candidate_positions', 'query_positions', 'candidate_mask'):
            self.assertTrue(torch.equal(prepared[name], expected[name]), name)
        self.assertEqual(prepared['hard_targets'].tolist(), [0, 4])
        self.assertEqual(prepared['identities'][1]['question_id'], 'second')
        self.assertTrue(all('gold' not in r and 'query_features' not in r and 'candidate_features' not in r for r in cached))
        changed = copy.deepcopy(self.rows)
        changed[0]['gold'] = [999]
        self.assertEqual(cached, p1.cache_pointer_decisions(changed, self.tokenizer))
        single = p1.collate_pointer_decisions(cached[:1], self.tokenizer.pad_token_id)
        self.assertTrue(single['attention_mask'].all())
        bad = copy.deepcopy(cached); bad[0]['candidate_positions'][0] = -1
        with self.assertRaises(ValueError): p1.collate_pointer_decisions(bad, 0)
        bad = copy.deepcopy(self.rows); bad[0]['hard_target'] = 1.5
        with self.assertRaises(ValueError): p1.cache_pointer_decisions(bad, self.tokenizer)
        with self.assertRaises(ValueError): p1.cache_pointer_decisions(self.rows, self.tokenizer, 10)

    def test_direct_backbone_recomputes_gradients_both_branches_without_lm_head(self):
        model, head = tiny_pair()
        state = p1.frozen_backbone_state(model)
        batch = p1.collate_pointer_decisions(p1.cache_pointer_decisions(self.rows, self.tokenizer), 0)
        adapter_before = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}
        head_before = {n: p.clone() for n, p in head.named_parameters()}
        optimizer = p1.create_joint_optimizer(model, head)
        model.train(); head.train()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            logits = p1.pointer_lora_logits(model, head, batch)
        self.assertEqual(logits.dtype, torch.float32)
        hard_candidate_loss(logits, batch['option_counts'], batch['hard_targets']).backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertEqual(bool(parameter.grad.abs().sum()), 'lora_B.' in name)
            else: self.assertIsNone(parameter.grad)
        self.assertTrue(all(p.grad.abs().sum() > 0 for p in head.parameters()))
        optimizer.step(); optimizer.zero_grad()
        hard_candidate_loss(p1.pointer_lora_logits(model, head, batch), batch['option_counts'], batch['hard_targets']).backward()
        self.assertTrue(all(p.grad.abs().sum() > 0 for n, p in model.named_parameters() if 'lora_A.' in n))
        self.assertTrue(all(not torch.equal(v, dict(model.named_parameters())[n]) for n,v in adapter_before.items() if 'lora_B.' in n))
        self.assertTrue(all(not torch.equal(v, dict(head.named_parameters())[n]) for n,v in head_before.items()))
        self.assertEqual(p1.frozen_backbone_state(model), state)
        self.assertEqual(len(model.get_base_model().model.calls), 2)
        self.assertTrue(all(c['grad_enabled'] and not c['use_cache'] for c in model.get_base_model().model.calls))
        self.assertTrue(all(p.dtype == torch.float32 for p in head.parameters()))
        model.get_base_model().model.rotary_fp32.add_(1)
        self.assertNotEqual(p1.frozen_backbone_state(model), state)

    def test_trainability_optimizer_groups_and_accumulation(self):
        model, head = tiny_pair()
        report = p1.assert_pointer_lora_trainable(model, head)
        self.assertEqual(report['head_parameters'], 48)
        self.assertEqual(report['trainable_parameters'], 112)
        optimizer = p1.create_joint_optimizer(model, head)
        self.assertEqual([g['lr'] for g in optimizer.param_groups], [2e-4, 1e-3])
        self.assertTrue(all(g['foreach'] is False for g in optimizer.param_groups))
        cached = p1.cache_pointer_decisions(self.rows, self.tokenizer)
        scheduler = make_scheduler(optimizer, 10, .1)
        result = p1.joint_train_update(model, head, cached, [0, 1], 0, optimizer, scheduler, 1, 1.)
        self.assertEqual(result['questions'], 2)
        self.assertTrue(np.isfinite(result['loss']))
        self.assertEqual(len(model.get_base_model().model.calls), 2)
        head.query_projection.weight.requires_grad_(False)
        with self.assertRaises(ValueError): p1.assert_pointer_lora_trainable(model, head)
        head.query_projection.weight.requires_grad_(True)
        model.get_base_model().lm_head.weight.requires_grad_(True)
        with self.assertRaises(ValueError): p1.create_joint_optimizer(model, head)

    def test_checkpoint_joint_exact_next_update_rng_and_immutable_generations(self):
        model, head = tiny_pair()
        original = copy.deepcopy(model.get_base_model().state_dict())
        cached = p1.cache_pointer_decisions(self.rows, self.tokenizer)
        optimizer = p1.create_joint_optimizer(model, head)
        scheduler = make_scheduler(optimizer, 10, 0.)
        p1.joint_train_update(model, head, cached, [0,1], 0, optimizer, scheduler, 2, 1.)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'step000001'
            progress = {'epoch': 0, 'group': 1, 'step': 1}
            p1.save_joint_checkpoint(model, head, optimizer, scheduler, path, progress)
            rng_expected = (random.random(), np.random.random(), torch.rand(3))
            p1.joint_train_update(model, head, cached, [1,0], 0, optimizer, scheduler, 2, 1.)
            expected_model = copy.deepcopy(model.state_dict()); expected_head = copy.deepcopy(head.state_dict())
            restored, restored_head = tiny_pair()
            restored.get_base_model().load_state_dict(original)
            restored_optimizer = p1.create_joint_optimizer(restored, restored_head)
            restored_scheduler = make_scheduler(restored_optimizer, 10, 0.)
            self.assertEqual(p1.restore_joint_checkpoint(restored, restored_head, restored_optimizer, restored_scheduler, path), progress)
            self.assertEqual(random.random(), rng_expected[0]); self.assertEqual(np.random.random(), rng_expected[1])
            self.assertTrue(torch.equal(torch.rand(3), rng_expected[2]))
            p1.joint_train_update(restored, restored_head, cached, [1,0], 0, restored_optimizer, restored_scheduler, 2, 1.)
            self.assertTrue(all(torch.equal(v, restored.state_dict()[n]) for n,v in expected_model.items()))
            self.assertTrue(all(torch.equal(v, restored_head.state_dict()[n]) for n,v in expected_head.items()))
            with self.assertRaises(ValueError): p1.save_joint_checkpoint(model, head, optimizer, scheduler, path, progress)
            # Mixing another generation's head is rejected by the same joint manifest.
            (path/'head.safetensors').write_bytes(b'tampered')
            with self.assertRaises(ValueError): p1.validate_joint_checkpoint(path)

    def test_eval_native_dtype_and_zero_adapter_matches_frozen_pointer_route(self):
        from jev.models.pointer import extract_pointer_features
        model,head = tiny_pair()
        model.eval(); head.eval()
        prepared=prepare_pointer_case(self.rows,self.tokenizer)
        base=model.get_base_model()
        # P0's extractor intentionally rejects trainable parameters, so disable
        # gradients only for this reference and restore the exact adapter flags.
        adapters=[p for p in model.parameters() if p.requires_grad]
        for p in adapters: p.requires_grad_(False)
        features=extract_pointer_features(base,prepared)
        for p in adapters: p.requires_grad_(True)
        expected=head(features['query_features'],features['candidate_features'],features['candidate_mask'])
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
            actual=p1.pointer_lora_logits(model,head,prepared)
        self.assertTrue(torch.equal(actual,expected))
        self.assertFalse(model.training)
        self.assertFalse(base.model.calls[-1]['grad_enabled'])

    def test_restore_rejects_same_shape_adapter_with_different_scaling(self):
        model,head=tiny_pair()
        optimizer=p1.create_joint_optimizer(model,head);scheduler=make_scheduler(optimizer,10,0.)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'joint'
            p1.save_joint_checkpoint(model,head,optimizer,scheduler,path,{'step':0})
            other,other_head=tiny_pair(alpha=8)
            with self.assertRaisesRegex(ValueError,'config|scal|architecture'):
                p1.load_joint_weights(other,other_head,path)

    def test_manifest_rejects_mixed_head_even_if_file_checksum_is_updated(self):
        from safetensors.torch import save_file
        from jev.data.typed_decisions import sha256_file
        model,head=tiny_pair()
        optimizer=p1.create_joint_optimizer(model,head);scheduler=make_scheduler(optimizer,10,0.)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'joint'
            p1.save_joint_checkpoint(model,head,optimizer,scheduler,path,{'step':0})
            other_head=PointerHead(8,3)
            save_file(other_head.state_dict(),str(path/'head.safetensors'))
            manifest=json.loads((path/'manifest.json').read_text())
            manifest['files_sha256']['head.safetensors']=sha256_file(path/'head.safetensors')
            (path/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'Head|head|generation'):
                p1.validate_joint_checkpoint(path)

    def test_publish_keeps_initial_and_epoch_generations_and_latest_pair(self):
        model, head = tiny_pair()
        optimizer = p1.create_joint_optimizer(model, head); scheduler = make_scheduler(optimizer,10,0.)
        with tempfile.TemporaryDirectory() as temp:
            for step in (0,2,3,4):
                p1.publish_joint_checkpoint(model, head, optimizer, scheduler, temp, {'step':step}, 3)
            self.assertEqual(sorted(p.name for p in Path(temp,'checkpoints').iterdir()), ['step000000','step000003','step000004'])
            latest=json.loads(Path(temp,'latest.json').read_text())
            self.assertEqual(latest['step'],4)
            manifest=p1.validate_joint_checkpoint(Path(temp)/latest['checkpoint'])
            self.assertEqual(latest['head_sha256'],manifest['head_sha256'])
            self.assertEqual(latest['adapter_sha256'],manifest['adapter_sha256'])


if __name__ == '__main__': unittest.main()

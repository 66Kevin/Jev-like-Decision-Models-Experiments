"""Lossless full-text-model training, checkpoint and evaluation contracts."""
import copy
import importlib
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from test_lora import TinyLM, prepared_tiny
from jev.training.lora import candidate_logits, hard_candidate_loss, inject_lora
from jev.training.runner import make_scheduler

try:
    full = importlib.import_module('jev.training.full')
except ModuleNotFoundError:
    full = None


class FullTrainingContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(full, 'G2 full training helpers must exist')
        torch.manual_seed(17)

    def trained(self):
        model = full.enable_full_training(TinyLM().bfloat16().requires_grad_(False))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.005)
        scheduler = make_scheduler(optimizer, 5, 0.)
        self.update(model, optimizer, scheduler)
        return model, optimizer, scheduler

    def update(self, model, optimizer, scheduler):
        batch = prepared_tiny()
        hard_candidate_loss(candidate_logits(model, batch), [2, 3], [1, 0]).backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)

    def test_full_fp32_trainability_tied_head_and_actual_updates(self):
        model = TinyLM().bfloat16().requires_grad_(False)
        count = sum(p.numel() for p in model.parameters())
        before = {n: p.detach().clone().float() for n, p in model.named_parameters()}
        self.assertIs(full.enable_full_training(model, expected_count=count), model)
        report = full.assert_full_trainable(model, expected_count=count)
        self.assertEqual(report['total_parameters'], count)
        self.assertEqual(report['trainable_parameters'], count)
        self.assertEqual(report['frozen_parameters'], 0)
        self.assertTrue(all(p.dtype == torch.float32 and p.requires_grad for p in model.parameters()))
        self.assertIs(model.lm_head.weight, model.embed_tokens.weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.005)
        self.update(model, optimizer, make_scheduler(optimizer, 5, 0.))
        for n, p in model.named_parameters():
            self.assertFalse(torch.equal(p, before[n]), n)
        with self.assertRaisesRegex(ValueError, 'count'):
            full.assert_full_trainable(model, expected_count=count + 1)
        model.backbone.norm.bias.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, 'trainable'):
            full.assert_full_trainable(model)

    def test_rejects_adapter_models_and_broken_tying(self):
        with self.assertRaisesRegex(ValueError, 'adapter|LoRA'):
            full.enable_full_training(inject_lora(TinyLM(), expected_count=2))
        model = TinyLM()
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        with self.assertRaisesRegex(ValueError, 'tied|tying'):
            full.enable_full_training(model)

    def test_eval_copy_does_not_round_master_or_change_mode_and_preserves_tie(self):
        model, _, _ = self.trained()
        model.train()
        before = copy.deepcopy(model.state_dict())
        evaluation = full.create_eval_copy(model, device='cpu')
        self.assertTrue(model.training)
        self.assertFalse(evaluation.training)
        self.assertIs(evaluation.lm_head.weight, evaluation.embed_tokens.weight)
        self.assertTrue(all(p.dtype == torch.bfloat16 and not p.requires_grad for p in evaluation.parameters()))
        self.assertTrue(any(not torch.equal(t, t.bfloat16().float()) for t in before.values()))
        for n, t in model.state_dict().items():
            self.assertTrue(torch.equal(t, before[n]), n)
        self.assertNotEqual(model.embed_tokens.weight.data_ptr(), evaluation.embed_tokens.weight.data_ptr())

    def test_eval_copy_preserves_native_fp32_rotary_buffers(self):
        model = TinyLM().bfloat16().requires_grad_(False)
        native = torch.tensor([0.12345678, 0.003579246], dtype=torch.float32)
        self.assertFalse(torch.equal(native, native.bfloat16().float()))
        model.register_buffer('rotary_inv_freq', native.clone())
        model.register_buffer('original_inv_freq', native.clone(), persistent=False)
        full.enable_full_training(model)
        # Re-enabling masters must retain their pre-promotion native dtypes.
        full.enable_full_training(model)
        original = copy.deepcopy(model.state_dict())
        evaluation = full.create_eval_copy(model, device='cpu')
        self.assertEqual(evaluation.rotary_inv_freq.dtype, torch.float32)
        self.assertEqual(evaluation.original_inv_freq.dtype, torch.float32)
        self.assertTrue(torch.equal(evaluation.rotary_inv_freq, native))
        self.assertTrue(torch.equal(evaluation.original_inv_freq, native))
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in evaluation.parameters()))
        for name, value in original.items():
            self.assertTrue(torch.equal(model.state_dict()[name], value), name)
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))

    def test_checkpoint_bitexact_fp32_optimizer_rng_and_next_update(self):
        model, optimizer, scheduler = self.trained()
        expected_hash = full.parameter_hash(model)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'step1'
            manifest = full.save_full_checkpoint(model, optimizer, scheduler, checkpoint, {'step': 1, 'epoch': 0})
            self.assertEqual(manifest['parameter_sha256'], expected_hash)
            self.assertTrue((checkpoint / 'model.safetensors').is_file())
            self.assertTrue((checkpoint / 'config.json').is_file())
            expected_random = (random.random(), np.random.random(), torch.rand(4))
            self.update(model, optimizer, scheduler)
            expected_weights = copy.deepcopy(model.state_dict())
            resumed = full.enable_full_training(TinyLM())
            opt = torch.optim.AdamW(resumed.parameters(), lr=.005)
            schedule = make_scheduler(opt, 5, 0.)
            identities = [id(p) for p in resumed.parameters()]
            progress = full.restore_full_checkpoint(resumed, opt, schedule, checkpoint)
            self.assertEqual(progress, {'step': 1, 'epoch': 0})
            self.assertEqual(full.parameter_hash(resumed), expected_hash)
            self.assertEqual([id(p) for p in resumed.parameters()], identities)
            self.assertIs(resumed.lm_head.weight, resumed.embed_tokens.weight)
            self.assertEqual(random.random(), expected_random[0]); self.assertEqual(np.random.random(), expected_random[1])
            self.assertTrue(torch.equal(torch.rand(4), expected_random[2]))
            self.update(resumed, opt, schedule)
            for name, value in expected_weights.items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[name]), name)
            with self.assertRaisesRegex((ValueError, FileExistsError), 'exist|immutable'):
                full.save_full_checkpoint(resumed, opt, schedule, checkpoint, {'step': 2})

    def test_checkpoint_keeps_native_keys_despite_hf_legacy_conversion(self):
        from transformers.core_model_loading import WeightRenaming
        model, optimizer, scheduler = self.trained()
        model._weight_conversions = [WeightRenaming('legacy_backbone.', 'backbone.')]
        with tempfile.TemporaryDirectory() as tmp:
            # Exercise the actual installed HF conversion path rather than a mock.
            default = Path(tmp) / 'hf_default'
            state = {n: t for n, t in model.state_dict().items() if n != 'lm_head.weight'}
            model.save_pretrained(default, state_dict=state, safe_serialization=True)
            default_keys = set(load_file(default / 'model.safetensors'))
            self.assertIn('legacy_backbone.q_proj.weight', default_keys)
            self.assertNotIn('backbone.q_proj.weight', default_keys)
            checkpoint = Path(tmp) / 'step1'
            full.save_full_checkpoint(model, optimizer, scheduler, checkpoint, {'step': 1})
            actual_keys = set(load_file(checkpoint / 'model.safetensors'))
            self.assertEqual(actual_keys, set(state))
            self.update(model, optimizer, scheduler)
            expected = copy.deepcopy(model.state_dict())
            restored = full.enable_full_training(TinyLM().bfloat16())
            opt = torch.optim.AdamW(restored.parameters(), lr=.005)
            schedule = make_scheduler(opt, 5, 0.)
            full.restore_full_checkpoint(restored, opt, schedule, checkpoint)
            self.update(restored, opt, schedule)
            for name, value in expected.items():
                self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)

    def test_corruption_and_missing_untied_weights_rejected(self):
        model, optimizer, scheduler = self.trained()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'step1'
            full.save_full_checkpoint(model, optimizer, scheduler, checkpoint, {'step': 1})
            weights = checkpoint / 'model.safetensors'
            with weights.open('ab') as handle:
                handle.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'hash|checksum'):
                full.load_full_weights(model, checkpoint)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'step1'
            full.save_full_checkpoint(model, optimizer, scheduler, checkpoint, {'step': 1})
            weights = checkpoint / 'model.safetensors'
            state = load_file(weights)
            del state['backbone.norm.bias']
            save_file(state, weights)
            # Even if the file manifest is rewritten, incomplete native weights fail closed.
            import hashlib
            manifest_path = checkpoint / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['files_sha256']['model.safetensors'] = hashlib.sha256(weights.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'missing|Missing|complete'):
                full.load_full_weights(model, checkpoint)

    def test_cpu_offload_restore_preserves_parameter_references_moments_and_dtype(self):
        model, optimizer, scheduler = self.trained()
        before = full.parameter_hash(model)
        identities = [id(p) for p in model.parameters()]
        states = {id(p): {k: v.clone() if torch.is_tensor(v) else v for k, v in state.items()}
                  for p, state in optimizer.state.items()}
        token = full.offload_training_state(model, optimizer)
        full.restore_training_state(model, optimizer, token)
        self.assertEqual(full.parameter_hash(model), before)
        self.assertEqual([id(p) for p in model.parameters()], identities)
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
        for p, state in optimizer.state.items():
            for key, value in state.items():
                if torch.is_tensor(value):
                    self.assertTrue(torch.equal(value, states[id(p)][key]))
        self.update(model, optimizer, scheduler)


if __name__ == '__main__':
    unittest.main()

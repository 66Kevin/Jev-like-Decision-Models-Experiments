"""Hard-label G1 adapter, masking and frozen-backbone contracts."""
import copy
import importlib
from pathlib import Path
import tempfile
import types
import unittest

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from test_readout import CharacterTokenizer, decision
from jev.models.lm_readout import prepare_case

try:
    lora = importlib.import_module('jev.training.lora')
except ModuleNotFoundError:
    lora = None


class TinyLM(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self):
        super().__init__(PretrainedConfig(vocab_size=256, hidden_size=8, tie_word_embeddings=True))
        self.embed_tokens = nn.Embedding(256, 8)
        self.backbone = nn.Module()
        self.backbone.q_proj = nn.Linear(8, 8, bias=False)
        self.backbone.down_proj = nn.Linear(8, 8, bias=False)
        self.backbone.norm = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 256, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.last_forward = None

    def forward(self, input_ids, attention_mask, position_ids, use_cache=False, logits_to_keep=1):
        self.last_forward = {'use_cache': use_cache, 'logits_to_keep': logits_to_keep}
        hidden = self.embed_tokens(input_ids)
        hidden = self.backbone.norm(self.backbone.down_proj(torch.tanh(self.backbone.q_proj(hidden))))
        return types.SimpleNamespace(logits=self.lm_head(hidden[:, -logits_to_keep:]))


def prepared_tiny():
    return {'input_ids': torch.tensor([[3, 4], [5, 6]]),
            'attention_mask': torch.ones(2, 2, dtype=torch.long),
            'position_ids': torch.tensor([[0, 1], [0, 1]]),
            'input_lengths': [2, 2], 'option_counts': [2, 3], 'label_token_ids': [11, 12, 13]}


class LoRAContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(lora, 'G1 shared LoRA module must exist')
        torch.manual_seed(17)

    def test_discovery_exact_whitelist_count_and_exclusions(self):
        model = TinyLM()
        model.lm_head.q_proj = nn.Linear(8, 8)
        model.embed_tokens.v_proj = nn.Linear(8, 8)
        self.assertEqual(lora.discover_lora_targets(model, expected_count=2),
                         ['backbone.down_proj', 'backbone.q_proj'])
        with self.assertRaisesRegex(ValueError, 'count'):
            lora.discover_lora_targets(model, expected_count=3)
        model.backbone.o_proj = nn.Identity()
        with self.assertRaisesRegex(ValueError, 'Linear'):
            lora.discover_lora_targets(model)

    def test_noop_fp32_adapters_gradients_frozen_weights_and_tying(self):
        base = TinyLM().to(torch.bfloat16)
        before = {n: p.detach().clone() for n, p in base.named_parameters()}
        batch = prepared_tiny()
        base.eval()
        expected = lora.candidate_logits(base, batch).detach()
        model = lora.inject_lora(base, expected_count=2)
        model.eval()
        self.assertTrue(torch.equal(lora.candidate_logits(model, batch), expected))
        report = lora.assert_adapter_only_trainable(model)
        self.assertGreater(report['trainable_parameters'], 0)
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(all(('lora_A.' in n or 'lora_B.' in n) and p.dtype == torch.float32
                            for n, p in trainable))
        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=.05)
        model.train()
        loss = lora.hard_candidate_loss(lora.candidate_logits(model, batch), [2, 3], [1, 0])
        loss.backward()
        for name, param in trainable:
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertEqual(bool(param.grad.abs().sum() > 0), 'lora_B.' in name)
        optimizer.step()
        optimizer.zero_grad()
        lora.hard_candidate_loss(lora.candidate_logits(model, batch), [2, 3], [1, 0]).backward()
        self.assertTrue(all(p.grad.abs().sum() > 0 for n, p in trainable if 'lora_A.' in n))
        for name, original in before.items():
            if name.endswith('.weight') and name.rsplit('.', 1)[0] in ('backbone.q_proj', 'backbone.down_proj'):
                actual = base.get_submodule(name.rsplit('.', 1)[0]).base_layer.weight
            else:
                actual = base.get_parameter(name)
            self.assertTrue(torch.equal(actual.detach(), original), name)
        self.assertIs(base.embed_tokens.weight, base.lm_head.weight)
        self.assertEqual(base.last_forward, {'use_cache': False, 'logits_to_keep': 1})

    def test_trainability_checker_rejects_base_leak(self):
        model = lora.inject_lora(TinyLM(), expected_count=2)
        model.get_base_model().lm_head.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'trainable'):
            lora.assert_adapter_only_trainable(model)

    def test_adapter_save_reload_preserves_outputs(self):
        from peft import PeftModel
        base = TinyLM()
        state = copy.deepcopy(base.state_dict())
        model = lora.inject_lora(base, expected_count=2)
        for name, parameter in model.named_parameters():
            if 'lora_B.' in name:
                parameter.data.fill_(.02)
        model.eval()
        expected = lora.candidate_logits(model, prepared_tiny()).detach()
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            self.assertTrue(Path(tmp, 'adapter_config.json').exists())
            original = TinyLM()
            original.load_state_dict(state)
            original.requires_grad_(False)
            loaded = PeftModel.from_pretrained(original, tmp, is_trainable=True)
            loaded.eval()
            self.assertTrue(torch.equal(expected, lora.candidate_logits(loaded, prepared_tiny())))
            lora.assert_adapter_only_trainable(loaded)

    def test_hard_loss_excludes_invalid_candidates_and_has_no_invalid_gradient(self):
        logits = torch.tensor([[1., 2., 1000.], [3., 1., 2.]], dtype=torch.float64, requires_grad=True)
        loss = lora.hard_candidate_loss(logits, [2, 3], [1, 2])
        expected = (torch.nn.functional.cross_entropy(logits[:1, :2].float(), torch.tensor([1])) +
                    torch.nn.functional.cross_entropy(logits[1:].float(), torch.tensor([2]))) / 2
        self.assertEqual(loss.dtype, torch.float32)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(logits.grad[0, 2], 0)

    def test_hard_loss_rejects_bad_shapes_counts_targets_and_nonfinite_valid_logits(self):
        for logits, counts, targets in [
            (torch.ones(2, 3), [2], [0, 0]),
            (torch.ones(2, 3), [2, 3], [0]),
            (torch.ones(2, 3), [2, 4], [0, 0]),
            (torch.ones(2, 3), [2, 1], [0, 0]),
            (torch.ones(2, 3), [2, 3], [2, 0]),
            (torch.ones(2, 3), [2, 3], [-1, 0]),
            (torch.ones(2, 3), [2, 3], [1.5, 0]),
            (torch.tensor([[float('nan'), 0.]]), [2], [0]),
            (torch.ones(1, 2, 3), [2], [0]),
        ]:
            with self.subTest(counts=counts, targets=targets), self.assertRaises(ValueError):
                lora.hard_candidate_loss(logits, counts, targets)

    def test_cache_reuses_exact_b0_tokens_and_never_retains_soft_gold(self):
        tokenizer = CharacterTokenizer()
        decisions = [decision('Short.', 2), decision('Longer instruction.', 5)]
        decisions[1]['question_id'] = 'second'
        decisions[1]['hard_target'] = 4
        expected = prepare_case(decisions, tokenizer)
        cached = lora.cache_decisions(decisions, tokenizer)
        self.assertEqual(len(cached), 2)
        self.assertTrue(all('gold' not in row for row in cached))
        self.assertEqual(cached[1]['hard_target'], 4)
        decisions[1]['gold'] = [999]
        self.assertEqual(cached, lora.cache_decisions(decisions, tokenizer))
        batch = lora.collate_decisions(cached, tokenizer.pad_token_id)
        for name in ('input_ids', 'attention_mask', 'position_ids'):
            self.assertTrue(torch.equal(batch[name], expected[name]))
        for name in ('input_lengths', 'option_counts', 'label_token_ids'):
            self.assertEqual(batch[name], expected[name])
        self.assertEqual(batch['hard_targets'].tolist(), [0, 4])
        self.assertEqual(batch['identities'][1]['question_id'], 'second')
        singleton = lora.collate_decisions(cached[:1], tokenizer.pad_token_id)
        self.assertTrue(torch.all(singleton['attention_mask'] == 1))

    def test_cache_and_collation_reject_invalid_targets_and_truncation(self):
        row = decision()
        row['hard_target'] = 3
        with self.assertRaisesRegex(ValueError, 'target'):
            lora.cache_decisions([row], CharacterTokenizer())
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            lora.cache_decisions([decision()], CharacterTokenizer(), max_input_tokens=10)
        with self.assertRaises(ValueError):
            lora.collate_decisions([], 0)

    def test_native_candidate_logits_rejects_padded_singleton(self):
        batch = prepared_tiny()
        for key in ('input_ids', 'attention_mask', 'position_ids'):
            batch[key] = batch[key][:1]
        batch['attention_mask'][0, 0] = 0
        with self.assertRaisesRegex(ValueError, 'padded singleton'):
            lora.candidate_logits(TinyLM(), batch)


if __name__ == '__main__':
    unittest.main()

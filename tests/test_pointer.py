"""Contracts for frozen native features and the shared Pointer decision Head."""
import copy
import importlib
import io
import math
import types
import unittest

import torch

from jev.models.lm_readout import prepare_case, build_messages
from jev.training.lora import hard_candidate_loss
from test_readout import CharacterTokenizer, decision

try:
    pointer = importlib.import_module('jev.models.pointer')
except ModuleNotFoundError:
    pointer = None


class FastCharacterTokenizer(CharacterTokenizer):
    is_fast = True

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {'input_ids': self.encode(text, add_special_tokens=add_special_tokens),
                'offset_mapping': [(i, i + 1) for i in range(len(text))]}


class TinyTextBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(65538, 4)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(65538 * 4).reshape(65538, 4) / 1000)
        self.calls = []

    def forward(self, input_ids, attention_mask, position_ids, use_cache=False,
                return_dict=True):
        self.calls.append({'use_cache': use_cache, 'grad_enabled': torch.is_grad_enabled()})
        return types.SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TinyFrozenLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyTextBackbone()
        self.lm_head = torch.nn.Linear(4, 65538, bias=False)
        self.lm_head.weight = self.model.embedding.weight
        self.requires_grad_(False)
        self.eval()

    def forward(self, **kwargs):
        raise AssertionError('Causal LM / vocabulary projection must be bypassed')


class PointerContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(pointer, 'Pointer primitives must exist')
        self.tokenizer = FastCharacterTokenizer()

    def test_preserves_b0_and_anchors_repeated_multiline_candidate_text(self):
        first = decision('Description 0 also occurs here. Candidates:\nnoise', count=2)
        first['state']['repeated'] = 'Description 0'
        first['option_texts'] = ['Description 0\n内部重复\t ', 'Description 0\n内部重复\n']
        second = decision('Long instruction ' * 4, count=5)
        rows = [first, second]
        expected = prepare_case(rows, self.tokenizer)
        actual = pointer.prepare_pointer_case(rows, self.tokenizer)
        for key in ('input_ids', 'attention_mask', 'position_ids'):
            self.assertTrue(torch.equal(expected[key], actual[key]), key)
        self.assertEqual(expected['prompts'], actual['prompts'])
        self.assertEqual(actual['candidate_mask'].tolist(), [[True, True, False, False, False], [True] * 5])
        self.assertEqual(actual['query_positions'].tolist(), [actual['input_ids'].shape[1] - 1] * 2)
        for i, row in enumerate(rows):
            shift = actual['input_ids'].shape[1] - actual['input_lengths'][i]
            for j, text in enumerate(row['option_texts']):
                start, stop = actual['candidate_character_spans'][i][j]
                self.assertEqual(actual['prompts'][i][start:stop], text)
                pos = int(actual['candidate_positions'][i, j])
                self.assertEqual(pos, shift + start + len(text.rstrip()) - 1)
                self.assertEqual(int(actual['input_ids'][i, pos]), ord(text.rstrip()[-1]) + 2)
        self.assertEqual(actual['candidate_positions'][0, 2:].tolist(), [0, 0, 0])
        changed = copy.deepcopy(rows)
        for row in changed:
            row.update(gold=[0.] * len(row['option_ids']), hard_target=1, factors='SECRET')
        self.assertEqual(pointer.prepare_pointer_case(changed, self.tokenizer)['prompts'], actual['prompts'])

    def test_overlapping_unicode_offsets_choose_last_covering_token(self):
        class OverlappingTokenizer(FastCharacterTokenizer):
            def encode(self, text, **kwargs):
                return [value for c in text for value in ([700, 701] if c == '你' else [ord(c) + 2])]
            def __call__(self, text, **kwargs):
                return {'input_ids': self.encode(text),
                        'offset_mapping': [span for i, c in enumerate(text) for span in
                                           ([(i, i + 1)] * 2 if c == '你' else [(i, i + 1)])]}
        row = decision(count=2)
        row['option_texts'] = ['你  ', 'ordinary']
        prepared = pointer.prepare_pointer_case([row], OverlappingTokenizer())
        self.assertEqual(int(prepared['input_ids'][0, prepared['candidate_positions'][0, 0]]), 701)

    def test_rejects_missing_offsets_mismatched_encoding_empty_and_ambiguous_spans(self):
        with self.assertRaisesRegex(ValueError, 'fast|offset'):
            pointer.prepare_pointer_case([decision()], CharacterTokenizer())
        class WrongEncoding(FastCharacterTokenizer):
            def __call__(self, text, **kwargs):
                value = super().__call__(text, **kwargs)
                value['input_ids'][0] += 1
                return value
        with self.assertRaisesRegex(ValueError, 'token|encod'):
            pointer.prepare_pointer_case([decision()], WrongEncoding())
        class MissingOffsets(FastCharacterTokenizer):
            def __call__(self, text, **kwargs):
                value = super().__call__(text, **kwargs)
                value['offset_mapping'] = [(0, 0)] * len(value['input_ids'])
                return value
        with self.assertRaisesRegex(ValueError, 'offset|cover'):
            pointer.prepare_pointer_case([decision()], MissingOffsets())
        row = decision()
        row['option_texts'][0] = ' \n\t'
        with self.assertRaisesRegex(ValueError, 'empty|whitespace'):
            pointer.prepare_pointer_case([row], self.tokenizer)
        class DuplicatedUser(FastCharacterTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return super().apply_chat_template(messages, **kwargs) + messages[1]['content']
        with self.assertRaisesRegex(ValueError, 'unique|ambiguous'):
            pointer.prepare_pointer_case([decision()], DuplicatedUser())
        with self.assertRaisesRegex(ValueError, 'candidate'):
            pointer.prepare_pointer_case([decision(count=6)], self.tokenizer)
        self.assertEqual(pointer.prepare_pointer_case([decision(count=6)], self.tokenizer,
                                                       max_candidates=6)['candidate_mask'].shape, (1, 6))

    def test_frozen_native_features_bypass_lm_head_keep_padding_and_support_head_backward(self):
        model = TinyFrozenLM()
        state = {k: v.clone() for k, v in model.state_dict().items()}
        prepared = pointer.prepare_pointer_case([decision('short', count=2), decision('long ' * 10)],
                                                self.tokenizer)
        result = pointer.extract_pointer_features(model, prepared)
        self.assertEqual(result['query_features'].shape, (2, 4))
        self.assertEqual(result['candidate_features'].shape, (2, 5, 4))
        self.assertEqual(result['query_features'].dtype, torch.float32)
        self.assertEqual(result['query_features'].device.type, 'cpu')
        self.assertFalse(result['query_features'].is_inference())
        self.assertTrue(torch.equal(result['candidate_features'][0, 2:], torch.zeros(3, 4)))
        for i, count in enumerate(prepared['option_counts']):
            for j in range(count):
                tok = prepared['input_ids'][i, prepared['candidate_positions'][i, j]]
                self.assertTrue(torch.equal(result['candidate_features'][i, j], model.model.embedding.weight[tok]))
        self.assertEqual(model.model.calls, [{'use_cache': False, 'grad_enabled': False}])
        head = pointer.PointerHead(hidden_size=4, projection_dim=3)
        loss = hard_candidate_loss(head(result['query_features'], result['candidate_features'],
                                        result['candidate_mask']), prepared['option_counts'], [0, 1])
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in state.items()))
        self.assertIs(model.model.embedding.weight, model.lm_head.weight)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_extract_rejects_unfrozen_or_training_backbone_and_inference_mode(self):
        prepared = pointer.prepare_pointer_case([decision()], self.tokenizer)
        model = TinyFrozenLM()
        model.train()
        with self.assertRaisesRegex(ValueError, 'eval|training'):
            pointer.extract_pointer_features(model, prepared)
        model.eval()
        next(model.parameters()).requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'frozen|grad'):
            pointer.extract_pointer_features(model, prepared)
        model.requires_grad_(False)
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, 'inference'):
            pointer.extract_pointer_features(model, prepared)

    def test_head_formula_parameter_budget_gradients_masking_and_state_roundtrip(self):
        torch.manual_seed(42)
        head = pointer.PointerHead()
        self.assertEqual(sum(p.numel() for p in head.parameters()), 262144)
        self.assertTrue(all(p.dtype == torch.float32 and p.requires_grad for p in head.parameters()))
        self.assertIsNone(head.query_projection.bias)
        self.assertIsNone(head.candidate_projection.bias)
        self.assertFalse(torch.equal(head.query_projection.weight, head.candidate_projection.weight))
        query = torch.randn(3, 1024)
        candidates = torch.randn(3, 5, 1024)
        mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
        candidates[~mask] = float('nan')
        scores = head(query, candidates, mask)
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(torch.equal(scores[~mask], torch.zeros_like(scores[~mask])))
        expected = (head.query_projection(query)[:, None] * head.candidate_projection(candidates.masked_fill(~mask[..., None], 0))).sum(-1) / math.sqrt(128)
        self.assertTrue(torch.allclose(scores[mask], expected[mask]))
        before = [p.clone() for p in head.parameters()]
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
        hard_candidate_loss(scores, [2, 5, 3], [1, 3, 0]).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in head.parameters()))
        optimizer.step()
        self.assertTrue(all(not torch.equal(old, new) for old, new in zip(before, head.parameters())))
        with io.BytesIO() as buffer:
            torch.save(head.state_dict(), buffer)
            buffer.seek(0)
            restored = pointer.PointerHead()
            restored.load_state_dict(torch.load(buffer, weights_only=True))
        self.assertTrue(torch.equal(head(query, candidates, mask), restored(query, candidates, mask)))

    def test_head_shared_matching_is_equivariant_for_fixed_candidate_features(self):
        head = pointer.PointerHead(hidden_size=4, projection_dim=3)
        query, candidates = torch.randn(2, 4), torch.randn(2, 6, 4)
        mask = torch.ones(2, 6, dtype=torch.bool)
        permutation = torch.tensor([4, 2, 0, 5, 1, 3])
        self.assertTrue(torch.equal(head(query, candidates[:, permutation], mask),
                                    head(query, candidates, mask)[:, permutation]))

    def test_head_rejects_invalid_shapes_masks_precision_and_nonfinite_valid_features(self):
        head = pointer.PointerHead(hidden_size=4, projection_dim=3)
        query, candidates = torch.randn(2, 4), torch.randn(2, 5, 4)
        mask = torch.ones(2, 5, dtype=torch.bool)
        for q, k, m in [(query, candidates[:, :, :3], mask), (query, candidates, mask[:, :4]),
                        (query.bfloat16(), candidates, mask), (query, candidates, mask.long()),
                        (query, candidates, torch.tensor([[1, 0, 1, 0, 0], [1] * 5], dtype=torch.bool)),
                        (query, candidates, torch.zeros_like(mask))]:
            with self.assertRaises(ValueError):
                head(q, k, m)
        candidates[0, 0, 0] = float('inf')
        with self.assertRaisesRegex(ValueError, 'finite'):
            head(query, candidates, mask)


if __name__ == '__main__':
    unittest.main()

"""B0 contracts: gold-free prompts, exact continuation tokens and padded readout."""
import copy
import importlib
import types
import unittest

import torch

try:
    readout = importlib.import_module('jev.models.lm_readout')
except ModuleNotFoundError:
    readout = None


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 2 for char in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        assert not enable_thinking and add_generation_prompt and not tokenize
        return ''.join(m['role'] + ':' + m['content'] + '\n' for m in messages) + 'assistant:'


def decision(text='Choose a route.', count=3):
    return {'case_id': 'case1', 'workflow': 'workflow1', 'question_id': 'route',
            'state': {'message': 'Need help', 'count': 2}, 'type': 'choice',
            'instructions': text, 'option_ids': ['route_' + str(i) for i in range(count)],
            'option_texts': ['Description ' + str(i) for i in range(count)],
            'gold': [1 / count] * count, 'hard_target': 0, 'score_levels': None}


class ReadoutContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(readout, 'The B0 readout module must exist')
        self.tokenizer = CharacterTokenizer()

    def test_prompt_only_uses_whitelisted_task_fields(self):
        first = decision()
        second = copy.deepcopy(first)
        second.update(gold=[0, 0, 1], hard_target=2, factors='SECRET',
                      label_agreement='SECRET', case_id='SECRET')
        prompt = readout.build_messages(first)
        self.assertEqual(prompt, readout.build_messages(second))
        self.assertIn('A. route_0: Description 0', prompt[1]['content'])
        self.assertIn('"count":2,"message":"Need help"', prompt[1]['content'])
        self.assertIn('Choose a route.', prompt[1]['content'])

    def test_variable_candidates_get_exact_label_mapping(self):
        result = readout.prepare_case([decision(count=2), decision(count=5)], self.tokenizer, 8192)
        self.assertEqual(result['label_token_ids'], [ord(x) + 2 for x in 'ABCDE'])
        self.assertEqual(result['option_counts'], [2, 5])

    def test_left_padding_position_ids_preserve_each_sequence(self):
        batch = readout.prepare_case([decision('Short.'), decision('A longer instruction.')], self.tokenizer, 8192)
        mask, positions = batch['attention_mask'], batch['position_ids']
        self.assertTrue(torch.all(mask[:, -1] == 1))
        self.assertEqual(int(mask[0, 0]), 0)
        for i, length in enumerate(batch['input_lengths']):
            self.assertEqual(positions[i][mask[i].bool()].tolist(), list(range(length)))
        self.assertTrue(torch.all(batch['input_ids'][mask == 0] == 0))

    def test_rejects_overlength_without_truncation(self):
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            readout.prepare_case([decision()], self.tokenizer, 10)

    def test_rejects_multitoken_letter(self):
        tokenizer = CharacterTokenizer()
        encode = tokenizer.encode
        tokenizer.encode = lambda text, **kw: [50, 51] if text == 'A' else encode(text, **kw)
        with self.assertRaisesRegex(ValueError, 'single token'):
            readout.prepare_case([decision()], tokenizer, 8192)

    def test_rejects_context_sensitive_continuation(self):
        tokenizer = CharacterTokenizer()
        encode = tokenizer.encode
        def changed(text, **kw):
            ids = encode(text, **kw)
            return ids[:-2] + [999] if text.endswith('assistant:A') else ids
        tokenizer.encode = changed
        with self.assertRaisesRegex(ValueError, 'continuation'):
            readout.prepare_case([decision()], tokenizer, 8192)

    def test_softmax_is_over_valid_candidate_labels_only(self):
        logits = torch.tensor([[1., 2., 100.], [-3., -3., -3.]])
        probabilities = readout.label_probabilities(logits, [2, 3])
        self.assertTrue(torch.allclose(torch.tensor(probabilities[0]), torch.softmax(logits[0, :2], 0)))
        self.assertTrue(torch.allclose(torch.tensor(probabilities[1]), torch.ones(3) / 3))

    def test_native_last_position_readout_matches_unpadded_rows(self):
        class LastPositionModel:
            device = torch.device('cpu')
            def __call__(self, input_ids, attention_mask, position_ids, use_cache, logits_to_keep):
                self_test.assertFalse(use_cache)
                self_test.assertEqual(logits_to_keep, 1)
                # Vary output with each sequence's unpadded positions to detect bad position IDs.
                value = position_ids[:, -1].float()
                logits = torch.zeros((len(value), 1, 300))
                logits[:, 0, ord('A') + 2] = value / 1000
                return types.SimpleNamespace(logits=logits)
        self_test = self
        rows = [decision('Short.'), decision('Longer instruction text.')]
        prepared = readout.prepare_case(rows, self.tokenizer, 8192)
        batched = readout.read_case(LastPositionModel(), prepared)
        for i, row in enumerate(rows):
            single = readout.read_case(LastPositionModel(), readout.prepare_case([row], self.tokenizer, 8192))
            self.assertEqual(batched['probabilities'][i], single['probabilities'][0])
        self.assertEqual(len(batched['label_logits']), 2)


class RunnerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / 'experiments/01_baseline/run.py'
        cls.runner = None
        if path.exists():
            spec = importlib.util.spec_from_file_location('b0_runner', path)
            cls.runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.runner)

    def setUp(self):
        self.assertIsNotNone(self.runner, 'The B0 runner must exist')

    def test_smoke_uses_only_train_and_selects_distinct_workflows(self):
        splits = {'train': [{'case_id': str(i), 'workflow': w, 'official_split': 'train'}
                            for i, w in enumerate(['a', 'a', 'b', 'b', 'c', 'd'])],
                  'test': [{'case_id': 'test', 'workflow': 'z', 'official_split': 'test'}]}
        cases = self.runner.select_warmup(splits['train'], 3)
        self.assertEqual([c['workflow'] for c in cases], ['a', 'b', 'c'])
        self.assertTrue(all(c['official_split'] == 'train' for c in cases))

    def test_npz_preserves_dynamic_option_identity_and_mask(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        rows = [decision(count=2), decision(count=4)]
        rows[1]['question_id'] = 'other'
        for row in rows:
            row['label_logits'] = list(range(len(row['option_ids'])))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'logits.npz'
            self.runner.save_logits(path, rows)
            data = np.load(path, allow_pickle=False)
            self.assertEqual(data['valid_mask'].tolist(), [[True, True, False, False], [True] * 4])
            self.assertEqual(data['option_ids'][0].tolist(), ['route_0', 'route_1', '', ''])
            self.assertEqual(data['question_ids'].tolist(), ['route', 'other'])
            self.assertEqual(data['logits'][1].tolist(), [0, 1, 2, 3])

    def test_protocol_rejects_unsupported_temperature_and_calibration_inference(self):
        import yaml
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / 'experiments/01_baseline/configs/b0.yaml'
        config = yaml.safe_load(path.read_text())
        self.runner.validate_config(config)
        config['evaluation']['temperature'] = 2
        with self.assertRaisesRegex(ValueError, 'temperature'):
            self.runner.validate_config(config)
        config['evaluation']['temperature'] = 1
        config['evaluation']['splits'] = ['calibration']
        with self.assertRaisesRegex(ValueError, 'splits'):
            self.runner.validate_config(config)

    def test_source_provenance_must_match_config_revision(self):
        import tempfile
        import json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'download_manifest.json').write_text(json.dumps({'dataset': 'example', 'revision': 'wrong'}))
            with self.assertRaisesRegex(ValueError, 'revision'):
                self.runner.validate_source(root, {'name': 'example', 'revision': 'expected'})

    def test_run_metadata_serializes_huggingface_loading_sets(self):
        import tempfile
        import json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'metadata.json'
            self.runner.write_json(path, {'loading_info': {'missing_keys': set(), 'unexpected_keys': {'b', 'a'}}})
            value = json.loads(path.read_text())
            self.assertEqual(value['loading_info'], {'missing_keys': [], 'unexpected_keys': ['a', 'b']})
            with self.assertRaises(TypeError):
                self.runner.write_json(path, {'unsupported': object()})


class NativePaddingGuard(unittest.TestCase):
    def test_padded_singleton_rejected_for_native_qwen_mask_contract(self):
        self.assertIsNotNone(readout)
        batch = readout.prepare_case([decision()], CharacterTokenizer())
        for key in ('input_ids', 'attention_mask', 'position_ids'):
            batch[key] = torch.cat([torch.zeros((1, 1), dtype=torch.long), batch[key]], dim=1)
        with self.assertRaisesRegex(ValueError, 'padded singleton'):
            readout.read_case(types.SimpleNamespace(device=torch.device('cpu')), batch)


if __name__ == '__main__':
    unittest.main()

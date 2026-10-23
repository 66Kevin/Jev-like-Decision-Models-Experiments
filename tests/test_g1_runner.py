"""Durable G1 training and held-out selection contracts (CPU only)."""
import copy
import importlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from test_lora import TinyLM
from jev.training.lora import inject_lora

try:
    runner = importlib.import_module('jev.training.runner')
except ModuleNotFoundError:
    runner = None

ROOT = Path(__file__).resolve().parents[1]


class G1RunnerContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner, 'G1 training runner must exist')

    def test_metadata_serializes_huggingface_loading_sets_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'model.json'
            value = {'loading_info': {'missing_keys': set(), 'unexpected_keys': {'z', 'a'}}}
            runner.write_json(path, value)
            self.assertEqual(json.loads(path.read_text()),
                             {'loading_info': {'missing_keys': [], 'unexpected_keys': ['a', 'z']}})
            self.assertFalse(path.with_suffix('.json.tmp').exists())

    def test_runtime_fingerprint_detects_local_fla_code_and_metadata_changes(self):
        self.assertTrue(hasattr(runner, 'runtime_fingerprint'), 'Runtime fingerprint must be collected')
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / 'fla_package'
            (package / 'fla/__pycache__').mkdir(parents=True)
            code = package / 'fla/__init__.py'
            code.write_text('__version__ = "test"\n')
            info = package / 'fla_core-test.dist-info'
            info.mkdir()
            metadata = info / 'METADATA'
            metadata.write_text('Version: test\n')
            first = runner.runtime_fingerprint(package)
            self.assertEqual(first['fla']['package_dir'], str(package.resolve()))
            self.assertEqual(set(first['fla']['files_sha256']),
                             {'fla/__init__.py', 'fla_core-test.dist-info/METADATA'})
            self.assertIn('torch', first['versions'])
            self.assertIn('cuda', first['torch_build'])
            (package / 'fla/__pycache__/ignored.py').write_text('cache\n')
            self.assertEqual(first, runner.runtime_fingerprint(package))
            code.write_text('__version__ = "changed"\n')
            self.assertNotEqual(first, runner.runtime_fingerprint(package))
            second = runner.runtime_fingerprint(package)
            metadata.write_text('Version: changed\n')
            self.assertNotEqual(second, runner.runtime_fingerprint(package))

    def test_runtime_fingerprint_detects_dependency_version_drift(self):
        self.assertTrue(hasattr(runner, 'runtime_fingerprint'), 'Runtime fingerprint must be collected')
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            (package / 'fla').mkdir()
            (package / 'fla/__init__.py').write_text('')
            with patch('importlib.metadata.version', return_value='original'):
                first = runner.runtime_fingerprint(package)
            with patch('importlib.metadata.version', return_value='changed'):
                second = runner.runtime_fingerprint(package)
            self.assertNotEqual(first, second)

    def test_resume_preserves_original_evidence_and_rejects_runtime_or_profile_drift(self):
        self.assertTrue(hasattr(runner, 'preserve_runtime_provenance'), 'Resume must validate immutable evidence')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'run'
            directory.mkdir()
            profile = root / 'kernel_profile_train.json'
            profile.write_text('{"validated": true}\n')
            metadata = {'status': 'failed'}
            fingerprint = {'python': 'original', 'versions': {'torch': 'original'},
                           'fla': {'files_sha256': {'fla/__init__.py': 'original'}}}
            runner.preserve_runtime_provenance(directory, profile, fingerprint, metadata, resume=False)
            saved = directory / 'metadata.json'
            runner.write_json(saved, metadata)
            original_metadata = saved.read_bytes()
            copied = directory / profile.name
            original_profile = copied.read_bytes()
            runner.preserve_runtime_provenance(directory, profile, fingerprint, metadata, resume=True)
            for key, value in [('python', 'changed'), ('versions', {'torch': 'changed'}),
                               ('fla', {'files_sha256': {'fla/__init__.py': 'changed'}})]:
                changed = copy.deepcopy(fingerprint)
                changed[key] = value
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'runtime'):
                    runner.preserve_runtime_provenance(directory, profile, changed, metadata, resume=True)
                self.assertEqual(saved.read_bytes(), original_metadata)
                self.assertEqual(copied.read_bytes(), original_profile)
            profile.write_text('{"validated": false}\n')
            with self.assertRaisesRegex(ValueError, 'profile'):
                runner.preserve_runtime_provenance(directory, profile, fingerprint, metadata, resume=True)
            self.assertEqual(copied.read_bytes(), original_profile)
            self.assertEqual(saved.read_bytes(), original_metadata)
            profile.write_bytes(original_profile)
            copied.write_text('tampered copy\n')
            with self.assertRaisesRegex(ValueError, 'profile'):
                runner.preserve_runtime_provenance(directory, profile, fingerprint, metadata, resume=True)
            self.assertEqual(copied.read_text(), 'tampered copy\n')
            self.assertEqual(saved.read_bytes(), original_metadata)
            self.assertEqual(metadata, json.loads(original_metadata))

    def test_resume_requires_existing_runtime_fingerprint(self):
        self.assertTrue(hasattr(runner, 'preserve_runtime_provenance'), 'Resume must fail closed')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'runtime'):
                runner.preserve_runtime_provenance(directory, directory / 'profile.json', {}, {}, resume=True)

    def test_entrypoint_rejects_resume_drift_before_mutating_run_or_loading_model(self):
        spec = importlib.util.spec_from_file_location('g1_resume_test', ROOT / 'experiments/02_supervised/run.py')
        entrypoint = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entrypoint)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'run'
            directory.mkdir()
            config = {'output_root': 'outputs', 'training': {'fla_package_dir': '.runtime/fla'}}
            (directory / 'config.yaml').write_text(yaml.safe_dump(config))
            profile = root / 'outputs/kernel_profile_train.json'
            profile.parent.mkdir()
            profile.write_text('{"validated": true}\n')
            metadata = {'source_sha256': {}, 'mode': 'train', 'status': 'failed'}
            fingerprint = {'versions': {'torch': 'original'}}
            runner.preserve_runtime_provenance(directory, profile, fingerprint, metadata)
            runner.write_json(directory / 'metadata.json', metadata)
            (directory / 'run.log').write_text('original log\n')
            (directory / 'pid').write_text('original pid\n')
            before = {path.name: path.read_bytes() for path in directory.iterdir()}
            for drift in ('runtime', 'profile'):
                current = {'versions': {'torch': 'changed'}} if drift == 'runtime' else fingerprint
                if drift == 'profile':
                    profile.write_text('changed profile\n')
                with patch.object(entrypoint, 'ROOT', root), \
                     patch.object(entrypoint, 'runtime_fingerprint', return_value=current), \
                     patch.object(entrypoint.b0, 'model_file_hashes') as model_hashes:
                    with self.subTest(drift=drift), self.assertRaisesRegex(ValueError, drift):
                        entrypoint.execute(config, directory / 'config.yaml', {}, {}, resume=directory)
                    model_hashes.assert_not_called()
                self.assertEqual(before, {path.name: path.read_bytes() for path in directory.iterdir()})

    def test_protocol_rejects_soft_loss_calibration_or_changed_seeds(self):
        config = yaml.safe_load((ROOT / 'experiments/02_supervised/configs/g1.yaml').read_text())
        runner.validate_config(config)
        for path, value in [('loss', 'soft_ce'), ('effective_batch_questions', 64)]:
            changed = copy.deepcopy(config)
            changed['training'][path] = value
            with self.assertRaises(ValueError):
                runner.validate_config(changed)
        changed = copy.deepcopy(config)
        changed['evaluation']['temperature'] = .7
        with self.assertRaises(ValueError):
            runner.validate_config(changed)
        changed = copy.deepcopy(config)
        changed['seeds'] = [42]
        with self.assertRaises(ValueError):
            runner.validate_config(changed)

    def test_epoch_groups_cover_each_question_once_and_reproduce_resume(self):
        a = runner.epoch_groups(67, 32, 42, 0)
        self.assertEqual([len(group) for group in a], [32, 32, 3])
        self.assertEqual(sorted(sum(a, [])), list(range(67)))
        self.assertEqual(a, runner.epoch_groups(67, 32, 42, 0))
        self.assertNotEqual(a, runner.epoch_groups(67, 32, 42, 1))

    def test_frozen_hash_changes_only_when_original_parameters_change(self):
        model = inject_lora(TinyLM(), rank=2, expected_count=2)
        before = runner.frozen_hash(model)
        with torch.no_grad():
            next(p for n, p in model.named_parameters() if 'lora_B.' in n).add_(1)
        self.assertEqual(before, runner.frozen_hash(model))
        with torch.no_grad():
            next(p for p in model.parameters() if not p.requires_grad).add_(1)
        self.assertNotEqual(before, runner.frozen_hash(model))

    def test_checkpoint_restores_optimizer_rng_and_adapter_outputs(self):
        torch.manual_seed(7)
        model = inject_lora(TinyLM(), rank=2, expected_count=2)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        scheduler = runner.make_scheduler(optimizer, 10, .05)
        sum(p.square().sum() for p in model.parameters() if p.requires_grad).backward()
        optimizer.step(); scheduler.step(); optimizer.zero_grad()
        expected = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'checkpoint'
            runner.save_checkpoint(model, optimizer, scheduler, destination, {'epoch': 1, 'group': 2})
            random_expected, numpy_expected, torch_expected = random.random(), np.random.rand(), torch.rand(3)
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:
                        p.zero_()
            state = runner.restore_checkpoint(model, optimizer, scheduler, destination)
            self.assertEqual(state, {'epoch': 1, 'group': 2})
            self.assertEqual(random.random(), random_expected)
            self.assertEqual(np.random.rand(), numpy_expected)
            self.assertTrue(torch.equal(torch.rand(3), torch_expected))
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    self.assertTrue(torch.equal(parameter, expected[name]))
            self.assertTrue(optimizer.state)

    def test_dev_nll_selection_earlier_step_wins_ties_and_nan_rejected(self):
        candidates = [{'step': 150, 'hard_nll': 1.1}, {'step': 0, 'hard_nll': 1.1},
                      {'step': 300, 'hard_nll': 1.2}]
        self.assertEqual(runner.select_checkpoint(candidates)['step'], 0)
        with self.assertRaises(ValueError):
            runner.select_checkpoint([{'step': 0, 'hard_nll': float('nan')}])

    def test_lock_requires_every_seed_and_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'selections.json'
            selections = {'42': {'step': 0}, '43': {'step': 150}}
            with self.assertRaises(ValueError):
                runner.lock_selections(path, selections, [42, 43, 44])
            selections['44'] = {'step': 300}
            runner.lock_selections(path, selections, [42, 43, 44])
            runner.lock_selections(path, selections, [42, 43, 44])
            selections['44']['step'] = 450
            with self.assertRaises(ValueError):
                runner.lock_selections(path, selections, [42, 43, 44])
            self.assertTrue(json.loads(path.read_text())['locked'])

    def test_paired_bootstrap_averages_seeds_and_requires_row_identity(self):
        def row(case, correct):
            return {'case_id': case, 'question_id': 'q', 'option_ids': ['a', 'b'],
                    'hard_target': 0, 'probabilities': [.8, .2] if correct else [.2, .8]}
        base = [row('a', False), row('b', True)]
        seeds = [[row('a', True), row('b', True)], [row('a', False), row('b', True)]]
        delta = runner.paired_bootstrap(base, seeds, samples=100, seed=42)
        self.assertEqual(delta['mean_accuracy_delta'], .25)
        self.assertEqual(delta['n_cases'], 2)
        with self.assertRaises(ValueError):
            runner.paired_bootstrap(base, [[row('x', True), row('b', True)]])

    def test_gradient_accumulation_matches_full_effective_batch(self):
        torch.manual_seed(10)
        first = inject_lora(TinyLM(), rank=2, expected_count=2)
        second = copy.deepcopy(first)
        rows = [{'input_ids': [i + 2, 7], 'hard_target': i % 2, 'option_count': 2,
                 'label_token_ids': [11, 12], 'identity': {'case_id': str(i)}} for i in range(5)]
        results = []
        for model, micro in [(first, 2), (second, 5)]:
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
            scheduler = runner.make_scheduler(optimizer, 10, 0)
            results.append(runner.train_update(model, rows, list(range(5)), 0, optimizer, scheduler, micro, 1.))
        self.assertAlmostEqual(results[0]['loss'], results[1]['loss'], places=6)
        self.assertEqual(results[0]['questions'], 5)
        for (name, left), (_, right) in zip(first.named_parameters(), second.named_parameters()):
            self.assertTrue(torch.allclose(left, right, atol=1e-6), name)

    def test_prepare_only_entrypoint_validates_fixed_b0_provenance_without_gpu(self):
        result = subprocess.run([sys.executable, str(ROOT / 'experiments/02_supervised/run.py'), '--prepare-only'],
                                cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        counts = json.loads(result.stdout)
        self.assertEqual(counts['train_decisions'], 4800)
        self.assertEqual(counts['counts']['test'], 400)
        self.assertEqual(counts['optimizer_steps_per_seed'], 450)


if __name__ == '__main__':
    unittest.main()

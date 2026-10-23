"""Training acceleration must not change the frozen native evaluation backend."""
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as native


def small_model():
    layer = native.Qwen3_5GatedDeltaNet.__new__(native.Qwen3_5GatedDeltaNet)
    torch.nn.Module.__init__(layer)
    layer.norm = native.Qwen3_5RMSNormGated(8, eps=1e-6)
    layer.chunk_gated_delta_rule = native.torch_chunk_gated_delta_rule
    layer.recurrent_gated_delta_rule = native.torch_recurrent_gated_delta_rule
    layer.causal_conv1d_fn = None
    layer.causal_conv1d_update = native.torch_causal_conv1d_update
    return torch.nn.Sequential(layer)


class KernelTests(unittest.TestCase):
    def setUp(self):
        try:
            self.kernels = importlib.import_module('jev.training.kernels')
        except ModuleNotFoundError:
            self.kernels = None
        self.assertIsNotNone(self.kernels, 'G1 kernel control module must exist')

    def test_native_evaluation_restores_training_kernel_without_changing_norm(self):
        model = small_model()
        norm = model[0].norm
        def accelerated(*args, **kwargs):
            return None
        with patch.object(self.kernels, '_load_fla', return_value=(accelerated, 'test-version')):
            info = self.kernels.set_delta_backend(model, 'fla', '.runtime/fla')
        self.assertIs(model[0].chunk_gated_delta_rule, accelerated)
        self.assertEqual(info['delta_layers'], 1)
        self.kernels.set_delta_backend(model, 'native')
        self.assertIs(model[0].chunk_gated_delta_rule, native.torch_chunk_gated_delta_rule)
        self.assertIs(model[0].norm, norm)
        self.assertIsNone(model[0].causal_conv1d_fn)

    def test_native_path_does_not_load_optional_package(self):
        with patch.object(self.kernels, '_load_fla', side_effect=AssertionError('optional import')):
            info = self.kernels.set_delta_backend(small_model(), 'native')
        self.assertEqual(info['backend'], 'native')

    def test_unexpected_norm_or_backend_is_rejected(self):
        model = small_model()
        model[0].norm = torch.nn.LayerNorm(8)
        with self.assertRaisesRegex(ValueError, 'normalization'):
            self.kernels.set_delta_backend(model, 'native')
        with self.assertRaises(ValueError):
            self.kernels.set_delta_backend(small_model(), 'unknown')

    def test_optional_package_must_resolve_inside_requested_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'fla').mkdir()
            package = types.SimpleNamespace(__file__='/tmp/unrelated_fla/__init__.py', __version__='other',
                                            chunk_gated_delta_rule=lambda *args: None)
            original_path = list(sys.path)
            try:
                with patch.object(self.kernels.importlib, 'import_module', return_value=package):
                    with self.assertRaisesRegex(ValueError, 'outside'):
                        self.kernels._load_fla(tmp)
            finally:
                sys.path[:] = original_path


if __name__ == '__main__':
    unittest.main()

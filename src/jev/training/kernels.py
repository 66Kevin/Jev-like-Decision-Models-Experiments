"""Optional training kernels with an explicit, B0-compatible native eval path."""
import importlib
from pathlib import Path
import sys

from transformers.models.qwen3_5 import modeling_qwen3_5 as native


def _load_fla(package_dir):
    # Import after the native model is constructed: do not replace its RMSNorm.
    path = Path(package_dir).resolve()
    if not (path / 'fla').is_dir():
        raise ValueError(f'Project-local FLA package missing: {path}')
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    package = importlib.import_module('fla')
    operation = importlib.import_module('fla.ops.gated_delta_rule')
    for module in (package, operation):
        location = getattr(module, '__file__', None)
        if location is None or not Path(location).resolve().is_relative_to(path):
            raise ValueError(f'FLA resolved outside requested project package directory: {location}')
    return operation.chunk_gated_delta_rule, package.__version__


def set_delta_backend(model, backend, package_dir=None):
    """Switch only the DeltaNet chunk operation; keep convolution/norm native."""
    if backend not in ('native', 'fla'):
        raise ValueError(f'Unsupported DeltaNet backend: {backend}')
    layers = [module for module in model.modules()
              if isinstance(module, native.Qwen3_5GatedDeltaNet)]
    if not layers:
        raise ValueError('No Qwen3.5 DeltaNet layers found')
    if any(type(layer.norm) is not native.Qwen3_5RMSNormGated for layer in layers):
        raise ValueError('Native B0 normalization must be retained for comparable evaluation')
    function, version = (native.torch_chunk_gated_delta_rule, None)
    if backend == 'fla':
        if package_dir is None:
            raise ValueError('FLA requires its explicit project-local package directory')
        function, version = _load_fla(package_dir)
    for layer in layers:
        layer.chunk_gated_delta_rule = function
        layer.recurrent_gated_delta_rule = native.torch_recurrent_gated_delta_rule
        layer.causal_conv1d_fn = None
        layer.causal_conv1d_update = native.torch_causal_conv1d_update
    return {'backend': backend, 'delta_layers': len(layers), 'fla_version': version,
            'fla_package_dir': str(Path(package_dir).resolve()) if backend == 'fla' else None,
            'chunk_function': f'{function.__module__}.{function.__name__}',
            'normalization': 'native', 'convolution': 'native'}

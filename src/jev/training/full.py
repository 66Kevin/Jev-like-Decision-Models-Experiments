"""G2 full-model FP32 masters, lossless generations and isolated BF16 evaluation.

The objective/tokenization is shared with G1's generic functions; this module
never injects adapters or changes the original pretrained model directory.
"""
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch
from safetensors.torch import load_file

from jev.training.runner import rng_state, write_json


def _reject_adapters(model):
    if (getattr(model, 'peft_config', None)
            or any('lora_' in n.lower() for n, _ in model.named_parameters())):
        raise ValueError('Full training rejects LoRA/adapter models')


def _check_tying(model):
    config = getattr(model, 'config', None)
    text = getattr(config, 'text_config', config)
    if not getattr(text, 'tie_word_embeddings', False):
        return
    try:
        inputs = model.get_input_embeddings()
    except (AttributeError, NotImplementedError):
        inputs = getattr(model, 'embed_tokens', None)
    try:
        outputs = model.get_output_embeddings()
    except (AttributeError, NotImplementedError):
        outputs = None
    outputs = outputs if outputs is not None else getattr(model, 'lm_head', None)
    if inputs is None or outputs is None or inputs.weight is not outputs.weight:
        raise ValueError('Expected embedding/LM Head tied parameter identity')


def assert_full_trainable(model, expected_count=None):
    """Verify all unique original text parameters are FP32 and trainable."""
    _reject_adapters(model)
    _check_tying(model)
    parameters = list(model.named_parameters())
    if not parameters:
        raise ValueError('No full-model trainable parameters')
    for name, p in parameters:
        if not p.requires_grad or p.dtype != torch.float32:
            raise ValueError(f'Full-model parameter must be trainable FP32: {name}, {p.dtype}')
    count = sum(p.numel() for _, p in parameters)
    if expected_count is not None and count != expected_count:
        raise ValueError(f'Full-model parameter count {count} differs from expected count {expected_count}')
    return {'trainable_parameters': count, 'frozen_parameters': 0,
            'total_parameters': count, 'trainable_tensors': len(parameters)}


def enable_full_training(model, expected_count=None):
    """Promote the loaded base to FP32 masters, preserving tied identities."""
    _reject_adapters(model)
    _check_tying(model)
    # The native BF16 loader intentionally retains FP32 rotary buffers (and
    # may retain selected FP32 parameters). Remember that policy before
    # promoting masters; repeated enable calls must not replace the policy.
    if not hasattr(model, '_jev_native_eval_dtypes'):
        model._jev_native_eval_dtypes = {
            'parameters': {n: p.dtype for n, p in model.named_parameters()},
            'buffers': {n: b.dtype for n, b in model.named_buffers()}}
    model.float().requires_grad_(True)
    assert_full_trainable(model, expected_count)
    return model


def parameter_hash(model):
    """Hash unique named parameters without retaining a full CPU weight copy."""
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        digest.update(f'{name}:{p.dtype}:{list(p.shape)}'.encode())
        flat = p.detach().reshape(-1)
        for start in range(0, flat.numel(), 1024 * 1024):
            digest.update(flat[start:start + 1024 * 1024].contiguous().cpu()
                          .view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def create_eval_copy(master, device='cpu'):
    """Create native evaluation dtypes; never cast FP32 masters in place.

    Copy parameters/buffers to CPU through deepcopy's memo first, so even a
    GPU-resident master does not temporarily need a second GPU FP32 model.
    """
    assert_full_trainable(master)
    memo = {}
    for p in master.parameters():
        memo[id(p)] = torch.nn.Parameter(p.detach().to('cpu', copy=True), requires_grad=False)
    for b in master.buffers():
        memo[id(b)] = b.detach().to('cpu', copy=True)
    evaluation = copy.deepcopy(master, memo)
    native = master._jev_native_eval_dtypes
    for name, p in evaluation.named_parameters():
        p.data = p.data.to(device=device, dtype=native['parameters'][name])
    for name, b in evaluation.named_buffers():
        b.data = b.data.to(device=device, dtype=native['buffers'][name])
    evaluation.requires_grad_(False).eval()
    _check_tying(evaluation)
    return evaluation


def _aliases(model):
    canonical, aliases = {}, {}
    for name, value in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False)):
        if id(value) in canonical:
            aliases[name] = canonical[id(value)]
        else:
            canonical[id(value)] = name
    return aliases


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_full_checkpoint(model, optimizer, scheduler, destination, progress):
    """Publish a complete immutable full-weight+trainer generation atomically."""
    assert_full_trainable(model)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'Checkpoint generations are immutable; destination exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.full_checkpoint_', dir=destination.parent))
    try:
        aliases = _aliases(model)
        state = {name: tensor for name, tensor in model.state_dict().items() if name not in aliases}
        # Explicit alias removal also supports HF subclasses without tied-key metadata.
        # Save canonical text-model keys, not the original multimodal load
        # namespace that Transformers otherwise restores through conversions.
        model.save_pretrained(temporary, state_dict=state, safe_serialization=True,
                              save_original_format=False)
        torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'rng': rng_state(), 'progress': progress}, temporary / 'trainer.pt')
        files = {str(p.relative_to(temporary)): _sha256(p) for p in sorted(temporary.rglob('*')) if p.is_file()}
        manifest = {'format_version': 1, 'kind': 'full_fp32', 'progress': progress,
                    'parameter_sha256': parameter_hash(model), 'aliases': aliases,
                    'files_sha256': files}
        write_json(temporary / 'manifest.json', manifest)
        temporary.replace(destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_full_checkpoint(directory):
    """Check the complete generation before any state is mutated."""
    directory = Path(directory)
    try:
        manifest = json.loads((directory / 'manifest.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError('Missing or invalid complete checkpoint manifest') from error
    if manifest.get('format_version') != 1 or manifest.get('kind') != 'full_fp32':
        raise ValueError('Unsupported full checkpoint format')
    files = manifest.get('files_sha256', {})
    if 'trainer.pt' not in files or 'config.json' not in files or not any(n.endswith('.safetensors') for n in files):
        raise ValueError('Incomplete checkpoint file manifest')
    for name, expected in files.items():
        path = directory / name
        if Path(name).is_absolute() or '..' in Path(name).parts or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f'Checkpoint checksum/hash mismatch: {name}')
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    if actual != set(files):
        raise ValueError('Checkpoint file manifest is incomplete or contains extra files')
    return manifest


def load_full_weights(model, directory):
    """Load complete FP32 weights into existing parameter objects, including ties."""
    assert_full_trainable(model)
    directory = Path(directory)
    manifest = validate_full_checkpoint(directory)
    expected = model.state_dict()
    aliases = _aliases(model)
    if manifest.get('aliases') != aliases:
        raise ValueError('Checkpoint tied aliases differ from destination model')
    state = {}
    for filename in sorted(manifest['files_sha256']):
        if filename.endswith('.safetensors'):
            shard = load_file(directory / filename, device='cpu')
            if set(state).intersection(shard):
                raise ValueError('Duplicate checkpoint weight keys')
            state.update(shard)
    missing = set(expected) - set(state)
    if missing != set(aliases) or set(state) - set(expected):
        raise ValueError(f'Incomplete weights: missing {sorted(missing - set(aliases))}, unexpected {sorted(set(state) - set(expected))}')
    for name, value in state.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f'Checkpoint shape/dtype mismatch: {name}')
    for alias, canonical in aliases.items():
        if canonical not in state:
            raise ValueError(f'Missing canonical tied weight: {canonical}')
        state[alias] = state[canonical]
    model.load_state_dict(state, strict=True, assign=False)
    _check_tying(model)
    if parameter_hash(model) != manifest['parameter_sha256']:
        raise ValueError('Restored parameter hash mismatch')
    return manifest


def _move_state(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_state(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_state(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_state(v, device) for v in value)
    return value


def restore_full_checkpoint(model, optimizer, scheduler, directory):
    """Restore lossless masters, Adam/scheduler and all RNGs for exact continuation."""
    manifest = load_full_weights(model, directory)
    state = torch.load(Path(directory) / 'trainer.pt', map_location='cpu', weights_only=False)
    if state['progress'] != manifest['progress']:
        raise ValueError('Trainer/model checkpoint generation mismatch')
    optimizer.load_state_dict(state['optimizer'])
    # PyTorch keeps ordinary Adam's step on CPU, but fused/capturable requires
    # it on the parameter device. Moment states always follow their parameter.
    for group in optimizer.param_groups:
        for p in group['params']:
            for key, value in optimizer.state.get(p, {}).items():
                device = p.device if key != 'step' or group.get('fused') or group.get('capturable') else 'cpu'
                optimizer.state[p][key] = _move_state(value, device)
    scheduler.load_state_dict(state['scheduler'])
    random.setstate(state['rng']['python'])
    np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['torch'])
    if state['rng']['cuda']:
        torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state['progress']


def offload_training_state(model, optimizer):
    """Move FP32 masters and optimizer to CPU without changing parameter identity."""
    assert_full_trainable(model)
    token = {'parameters': [(p, p.device) for p in model.parameters()],
             'buffers': [(b, b.device) for b in model.buffers()], 'optimizer': {}}
    for p, device in token['parameters']:
        p.data = p.data.cpu()
        if p.grad is not None:
            p.grad.data = p.grad.data.cpu()
    for b, device in token['buffers']:
        b.data = b.data.cpu()
    for p, state in optimizer.state.items():
        devices = {}
        def capture(value, path=()):
            if torch.is_tensor(value):
                devices[path] = value.device
            elif isinstance(value, dict):
                for key, item in value.items():
                    capture(item, path + (key,))
            elif isinstance(value, (tuple, list)):
                for key, item in enumerate(value):
                    capture(item, path + (key,))
        capture(state)
        token['optimizer'][p] = devices
        optimizer.state[p] = _move_state(state, 'cpu')
    return token


def restore_training_state(model, optimizer, token):
    """Restore offloaded devices exactly, including fused Adam step tensors."""
    for p, device in token['parameters']:
        p.data = p.data.to(device)
        if p.grad is not None:
            p.grad.data = p.grad.data.to(device)
    for b, device in token['buffers']:
        b.data = b.data.to(device)
    for p, devices in token['optimizer'].items():
        def restore(value, path=()):
            if torch.is_tensor(value):
                return value.to(devices[path])
            if isinstance(value, dict):
                return {k: restore(v, path + (k,)) for k, v in value.items()}
            if isinstance(value, list):
                return [restore(v, path + (k,)) for k, v in enumerate(value)]
            if isinstance(value, tuple):
                return tuple(restore(v, path + (k,)) for k, v in enumerate(value))
            return value
        optimizer.state[p] = restore(optimizer.state[p])
    assert_full_trainable(model)

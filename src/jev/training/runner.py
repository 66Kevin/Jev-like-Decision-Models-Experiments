"""G1 fixed-protocol utilities: hard supervision, durable state and case pairing."""
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import platform
import sys
import tempfile

import numpy as np
import torch

from jev.data.typed_decisions import stable_argmax
from jev.training.lora import candidate_logits, collate_decisions, hard_candidate_loss


def runtime_fingerprint(fla_package_dir):
    """Identify runtime versions/build and the explicit local FLA source tree."""
    package = Path(fla_package_dir).resolve()
    if not (package / 'fla').is_dir():
        raise ValueError(f'Project-local FLA runtime missing: {package}')
    versions = {}
    for name in ('torch', 'transformers', 'peft', 'accelerate', 'numpy', 'PyYAML',
                 'tokenizers', 'safetensors', 'huggingface-hub', 'triton', 'einops', 'packaging'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    files = {}
    for path in sorted(package.rglob('*')):
        relative = path.relative_to(package)
        if any(part in ('__pycache__', '.cache', 'cache', 'caches', '.pytest_cache') for part in relative.parts):
            continue
        if path.is_file() and (path.suffix == '.py' or any(part.endswith('.dist-info') for part in relative.parts)):
            files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'python': {'version': sys.version, 'implementation': platform.python_implementation(),
                       'executable': str(Path(sys.executable).resolve())},
            'versions': versions,
            'torch_build': {'version': str(torch.__version__), 'cuda': torch.version.cuda,
                            'git_version': torch.version.git_version, 'config': torch.__config__.show()},
            'fla': {'package_dir': str(package), 'files_sha256': files}}


def preserve_runtime_provenance(directory, profile, fingerprint, metadata, resume=False):
    """Validate immutable evidence before resume; only fresh runs create a copy."""
    directory, profile = Path(directory), Path(profile)
    copied = directory / profile.name
    if resume and ('runtime_fingerprint' not in metadata or metadata['runtime_fingerprint'] != fingerprint):
        raise ValueError('Resume runtime fingerprint differs or is missing')
    if not profile.is_file():
        raise ValueError('Train-only kernel profile evidence missing')
    current_hash = hashlib.sha256(profile.read_bytes()).hexdigest()
    if resume:
        expected = metadata.get('kernel_profile_sha256')
        if (not copied.is_file() or hashlib.sha256(copied.read_bytes()).hexdigest() != expected
                or current_hash != expected):
            raise ValueError('Resume kernel profile evidence differs from original hash')
        return
    if copied.exists():
        raise ValueError('Fresh run kernel profile evidence already exists')
    shutil.copyfile(profile, copied)
    metadata['runtime_fingerprint'] = fingerprint
    metadata['kernel_profile_sha256'] = current_hash


def _json_default(value):
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f'Unsupported metadata type: {type(value).__name__}')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                                    default=_json_default) + '\n')
    temporary.replace(path)


def validate_config(config):
    fixed = {'experiment_id': 'g1', 'seeds': [42, 43, 44]}
    groups = {'lora': {'rank': 16, 'alpha': 32, 'dropout': 0., 'expected_target_modules': 186},
              'training': {'epochs': 3, 'effective_batch_questions': 32, 'learning_rate': 2e-4,
                           'weight_decay': .01, 'max_grad_norm': 1., 'warmup_fraction': .05,
                           'scheduler': 'cosine', 'gradient_checkpointing': False,
                           'loss': 'hard_candidate_cross_entropy', 'select_metric': 'dev_hard_nll',
                           'eval_each_epoch': True, 'include_step_zero': True, 'kernel_backend': 'fla'},
              'evaluation': {'splits': ['dev', 'test'], 'max_input_tokens': 8192, 'warmup_cases': 3,
                             'case_batch_size': 1, 'ece_bins': 15, 'temperature': 1., 'use_cache': False,
                             'prompt_version': 'b0_v1', 'kernel_backend': 'native', 'merge_adapter': False}}
    for key, value in fixed.items():
        if config.get(key) != value:
            raise ValueError(f'G1 requires {key}={value!r}')
    for group, mapping in groups.items():
        for key, value in mapping.items():
            if config[group].get(key) != value:
                raise ValueError(f'G1 requires {group}.{key}={value!r}')
    micro, accumulation = (config['training'][key] for key in
                           ('microbatch_questions', 'gradient_accumulation'))
    if (micro, accumulation) not in ((8, 4), (4, 8)):
        raise ValueError('Only preflight memory fallback micro4/acc8 is allowed')


def epoch_groups(count, effective_batch, seed, epoch):
    indices = list(range(count))
    random.Random(f'{seed}:epoch:{epoch}').shuffle(indices)
    return [indices[start:start + effective_batch] for start in range(0, count, effective_batch)]


def frozen_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            continue
        digest.update(f'{name}:{parameter.dtype}:{list(parameter.shape)}'.encode())
        flat = parameter.detach().reshape(-1)
        for start in range(0, flat.numel(), 1024 * 1024):
            digest.update(flat[start:start + 1024 * 1024].contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def make_scheduler(optimizer, total_steps, warmup_fraction):
    warmup = math.ceil(total_steps * warmup_fraction)
    def factor(step):
        if step < warmup:
            return step / max(1, warmup)
        return .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, total_steps - warmup))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def save_checkpoint(model, optimizer, scheduler, destination, progress):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.checkpoint_', dir=destination.parent))
    model.save_pretrained(temporary, safe_serialization=True)
    torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'rng': rng_state(), 'progress': progress}, temporary / 'trainer.pt')
    backup = destination.with_name(destination.name + '.old')
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.replace(backup)
    temporary.replace(destination)
    if backup.exists():
        shutil.rmtree(backup)


def load_adapter(model, directory):
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    result = set_peft_model_state_dict(model, load_peft_weights(str(directory), device=str(model.device)))
    if result.unexpected_keys:
        raise ValueError(f'Unexpected adapter keys: {result.unexpected_keys}')


def restore_checkpoint(model, optimizer, scheduler, directory):
    directory = Path(directory)
    if not directory.exists() and directory.with_name(directory.name + '.old').exists():
        directory = directory.with_name(directory.name + '.old')
    load_adapter(model, directory)
    state = torch.load(directory / 'trainer.pt', map_location='cpu', weights_only=False)
    optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
    random.setstate(state['rng']['python']); np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['torch'])
    if state['rng']['cuda']:
        torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state['progress']


def train_update(model, cached, indices, pad, optimizer, scheduler, microbatch, max_grad_norm):
    model.train(); optimizer.zero_grad(set_to_none=True)
    total_loss = 0.
    context = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if model.device.type == 'cuda' else nullcontext
    for start in range(0, len(indices), microbatch):
        rows = [cached[i] for i in indices[start:start + microbatch]]
        batch = collate_decisions(rows, pad)
        with context():
            loss = hard_candidate_loss(candidate_logits(model, batch), batch['option_counts'], batch['hard_targets'])
        weight = len(rows) / len(indices)
        (loss * weight).backward()
        total_loss += float(loss.detach()) * weight
    parameters = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise RuntimeError('Missing/nonfinite adapter gradient')
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
    lr = optimizer.param_groups[0]['lr']
    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    return {'loss': total_loss, 'grad_norm': float(norm), 'learning_rate': lr, 'questions': len(indices)}


def select_checkpoint(candidates):
    if not candidates or any(not math.isfinite(row['hard_nll']) for row in candidates):
        raise ValueError('All checkpoint dev NLL values must be finite')
    return min(candidates, key=lambda row: (row['hard_nll'], row['step']))


def lock_selections(path, selections, seeds):
    if set(selections) != {str(seed) for seed in seeds}:
        raise ValueError('All preregistered seed selections must exist before test')
    value = {'locked': True, 'criterion': 'minimum dev hard NLL; earlier step breaks ties', 'seeds': selections}
    if Path(path).exists():
        if json.loads(Path(path).read_text()) != value:
            raise ValueError('Locked selections cannot change after test is unlocked')
    else:
        write_json(path, value)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def paired_bootstrap(baseline, seeds, samples=2000, seed=42):
    def keyed(rows):
        result = {}
        for row in rows:
            key = (row['case_id'], row['question_id'])
            if key in result:
                raise ValueError('Duplicate prediction identity')
            result[key] = row
        return result
    base = keyed(baseline); candidates = [keyed(rows) for rows in seeds]
    if not candidates or any(set(rows) != set(base) for rows in candidates):
        raise ValueError('Prediction identities differ between baseline and seeds')
    cases = {}
    for key, row in base.items():
        matched = [mapping[key] for mapping in candidates]
        if any((other['option_ids'], other['hard_target']) != (row['option_ids'], row['hard_target']) for other in matched):
            raise ValueError('Candidate or target identity differs between baseline and seeds')
        def correct(item):
            return float(stable_argmax(item['probabilities'], item['option_ids']) == item['hard_target'])
        delta = float(np.mean([correct(item) for item in matched])) - correct(row)
        cases.setdefault(key[0], []).append(delta)
    values = np.asarray([sum(cases[key]) for key in sorted(cases)])
    counts = np.asarray([len(cases[key]) for key in sorted(cases)])
    rng = np.random.default_rng(seed); distribution = []
    for start in range(0, samples, 100):
        indices = rng.integers(0, len(values), (min(100, samples - start), len(values)))
        distribution.extend((values[indices].sum(1) / counts[indices].sum(1)).tolist())
    low, high = np.quantile(distribution, [.025, .975])
    return {'mean_accuracy_delta': float(values.sum() / counts.sum()), 'low': float(low), 'high': float(high),
            'n_cases': len(values), 'samples': samples, 'seed': seed, 'unit': 'paired case',
            'definition': 'Mean correctness over all registered seeds minus B0; percentile 95% CI'}

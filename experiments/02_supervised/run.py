#!/usr/bin/env python3
"""G1: fixed hard-label Backbone LoRA; train three seeds, lock dev choices, then test."""
import argparse
from datetime import datetime
import gc
import fcntl
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
import yaml
from jev.data.typed_decisions import iter_decisions, load_cases, prepare_splits, sha256_file
from jev.models.lm_readout import load_local_model, prepare_case, read_case, synchronize
from jev.training.lora import assert_adapter_only_trainable, cache_decisions, inject_lora
from jev.training.runner import (epoch_groups, frozen_hash, load_adapter, lock_selections, make_scheduler,
                                 paired_bootstrap, preserve_runtime_provenance, read_rows, restore_checkpoint,
                                 runtime_fingerprint, save_checkpoint,
                                 select_checkpoint, train_update, validate_config, write_json)

spec = importlib.util.spec_from_file_location('jev_b0_runner', ROOT / 'experiments/01_baseline/run.py')
b0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b0)


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def prepare(config):
    validate_config(config)
    raw = ROOT / config['dataset']['raw_dir']
    b0.validate_source(raw, config['dataset'])
    splits = prepare_splits(load_cases(raw), ROOT / config['dataset']['split_dir'], seed=42,
                            source_manifest=raw / 'download_manifest.json')
    baseline = ROOT / config['baseline_run']
    previous = json.loads((baseline / 'metadata.json').read_text())
    baseline_config = yaml.safe_load((baseline / 'config.yaml').read_text())
    if previous['status'] != 'complete' or previous['dataset'] != config['dataset']:
        raise ValueError('G1 source must match the completed pinned B0 run')
    if baseline_config['model'] != config['model']:
        raise ValueError('G1 model configuration differs from B0')
    if previous['split_manifest_sha256'] != sha256_file(ROOT / config['dataset']['split_dir'] / 'manifest.json'):
        raise ValueError('G1/B0 split provenance differs')
    for source in ('src/jev/data/typed_decisions.py', 'src/jev/models/lm_readout.py', 'src/jev/evaluation/metrics.py'):
        if previous['source_sha256'][source] != sha256_file(ROOT / source):
            raise ValueError(f'B0 shared scientific code changed: {source}')
    if {key: len(value) for key, value in splits.items()} != {'train': 960, 'dev': 120, 'calibration': 120, 'test': 400}:
        raise ValueError('Unexpected split sizes')
    return splits, previous


def sources(directory):
    paths = sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'tests').glob('test_*.py'))
    paths += [Path(__file__), ROOT / 'experiments/01_baseline/run.py',
              ROOT / 'experiments/02_supervised/README.md', ROOT / 'experiments/02_supervised/configs/g1.yaml']
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT); destination = directory / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, destination)
        hashes[str(relative)] = sha256_file(destination)
    return hashes


def memory(device):
    return {'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
            'current_allocated_bytes': torch.cuda.memory_allocated(device)}


def backend(model, config, kind):
    # Import project-local FLA only after construction of a native base model.
    from jev.training.kernels import set_delta_backend
    package = (ROOT / config['training']['fla_package_dir']).resolve()
    return {**set_delta_backend(model, kind, package), 'package_dir': str(package) if kind == 'fla' else None}


def evaluate(name, cases, model, tokenizer, directory, config, seed, warmup):
    backend(model, config, 'native'); model.eval()
    for case in warmup:
        read_case(model, prepare_case(case['decisions'], tokenizer))
    synchronize(model.device); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(model.device)
    value = b0.run_split(name, cases, model, tokenizer, directory, {**config, 'seed': seed})
    value['memory'] = memory(model.device)
    value['memory_definition'] = 'Native evaluation allocator including base plus unmerged adapters; optimizer off-device.'
    write_json(directory / name / 'summary.json', value)
    return value


def roundtrip(model, tokenizer, probe, checkpoint, config, expected=None):
    backend(model, config, 'native'); model.eval()
    batch = prepare_case(probe['decisions'], tokenizer)
    before = read_case(model, batch)['label_logits'] if expected is None else expected
    load_adapter(model, checkpoint)
    after = read_case(model, batch)['label_logits']
    difference = max(float(np.max(np.abs(np.asarray(a) - b))) for a, b in zip(before, after))
    if difference != 0:
        raise RuntimeError(f'Adapter reload changed native train logits: {difference}')
    value = {'train_case_id': probe['case_id'], 'max_absolute_logit_difference': difference, 'label_logits': after}
    write_json(checkpoint / 'reload_check_train.json', value)
    return value


def train_seed(config, seed, splits, directory, baseline, smoke):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.fp32_precision = 'ieee'; torch.backends.cudnn.conv.fp32_precision = 'ieee'
    directory.mkdir(exist_ok=True)
    model, tokenizer, loading = load_local_model(config['model'])
    probe = splits['train'][0]
    original_logits = read_case(model, prepare_case(probe['decisions'], tokenizer))['label_logits']
    lora = config['lora']
    model = inject_lora(model, lora['rank'], lora['alpha'], lora['dropout'], lora['expected_target_modules'])
    model.eval()
    zero_logits = read_case(model, prepare_case(probe['decisions'], tokenizer))['label_logits']
    if original_logits != zero_logits:
        raise RuntimeError('Zero-adapter native B0 parity failed on train')
    frozen_before = frozen_hash(model)
    counts = assert_adapter_only_trainable(model)
    write_json(directory / 'model.json', {**counts, 'loading_info': loading, 'frozen_sha256': frozen_before,
                                          'target_modules': list(model._jev_lora_target_modules),
                                          'zero_adapter_train_parity': True})
    train = list(iter_decisions(splits['train']))
    cached = cache_decisions(train, tokenizer, config['evaluation']['max_input_tokens'])
    write_json(directory / 'tokenization.json', {'decisions': len(cached),
               'max_tokens': max(len(row['input_ids']) for row in cached),
               'cache_identity_sha256': __import__('hashlib').sha256(json.dumps(cached, sort_keys=True).encode()).hexdigest()})
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    training = config['training']; steps_per_epoch = math.ceil(len(cached) / training['effective_batch_questions'])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=training['learning_rate'], weight_decay=training['weight_decay'])
    scheduler = make_scheduler(optimizer, training['epochs'] * steps_per_epoch, training['warmup_fraction'])
    progress = {'epoch': 0, 'group': 0, 'step': 0}
    latest = directory / 'latest.json'
    if latest.exists():
        checkpoint = directory / json.loads(latest.read_text())['checkpoint']
        progress = restore_checkpoint(model, optimizer, scheduler, checkpoint)
        if frozen_hash(model) != frozen_before:
            raise RuntimeError('Checkpoint modified frozen base')
    else:
        checkpoint = directory / 'checkpoints/step000000'
        save_checkpoint(model, optimizer, scheduler, checkpoint, progress)
        write_json(latest, {'checkpoint': str(checkpoint.relative_to(directory)), **progress})
        roundtrip(model, tokenizer, probe, checkpoint, config)
    if not smoke and not (directory / 'dev_epoch0').exists():
        shutil.copytree(ROOT / config['baseline_run'] / 'dev', directory / 'dev_epoch0')
        write_json(directory / 'dev_epoch0/reused_from_b0.json', {
            'baseline_run': config['baseline_run'], 'zero_adapter_train_parity': True,
            'metrics_sha256': sha256_file(ROOT / config['baseline_run'] / 'dev/metrics.json')})
    kernel = backend(model, config, training['kernel_backend'])
    write_json(directory / 'training_backend.json', kernel)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(model.device)
    train_memory_path = directory / 'training_memory.json'
    training_peak = json.loads(train_memory_path.read_text())['peak_allocated_bytes'] if train_memory_path.exists() else 0
    history = directory / 'train.jsonl'
    for epoch in range(progress['epoch'], training['epochs']):
        groups = epoch_groups(len(cached), training['effective_batch_questions'], seed, epoch)
        start = progress['group'] if epoch == progress['epoch'] else 0
        for group in range(start, len(groups)):
            started = time.perf_counter()
            stats = train_update(model, cached, groups[group], pad, optimizer, scheduler,
                                 training['microbatch_questions'], training['max_grad_norm'])
            synchronize(model.device)
            progress = {'epoch': epoch, 'group': group + 1, 'step': epoch * steps_per_epoch + group + 1}
            stats.update({**progress, 'seconds': time.perf_counter() - started})
            with history.open('a') as stream:
                stream.write(json.dumps(stats, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
            old = checkpoint
            checkpoint = directory / f'checkpoints/step{progress["step"]:06d}'
            save_checkpoint(model, optimizer, scheduler, checkpoint, progress)
            write_json(latest, {'checkpoint': str(checkpoint.relative_to(directory)), **progress})
            if old.name != 'step000000' and int(old.name[4:]) % steps_per_epoch != 0:
                shutil.rmtree(old)
            training_peak = max(training_peak, torch.cuda.max_memory_allocated(model.device))
            write_json(train_memory_path, {'peak_allocated_bytes': training_peak, 'step': progress['step']})
            if progress['step'] % 10 == 0 or smoke:
                logging.info('seed%d epoch%d step%d/%d loss=%.5f seconds=%.2f', seed, epoch + 1,
                             progress['step'], training['epochs'] * steps_per_epoch, stats['loss'], stats['seconds'])
            if smoke and progress['step'] >= 2:
                roundtrip(model, tokenizer, probe, checkpoint, config)
                if frozen_hash(model) != frozen_before:
                    raise RuntimeError('Frozen original weights changed')
                write_json(directory / 'smoke.json', {'optimizer_updates': 2, 'train_only': True,
                            'frozen_weights_unchanged': True, 'reload_verified': True,
                            'training_peak_allocated_bytes': training_peak})
                del optimizer, scheduler, model; gc.collect(); torch.cuda.empty_cache()
                return {'smoke': True}
        # Move optimizer states off GPU to keep native evaluation memory comparable.
        saved_optimizer = optimizer.state_dict()
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.cpu()
        del saved_optimizer
        roundtrip(model, tokenizer, probe, checkpoint, config)
        dev = directory / f'dev_epoch{epoch + 1}'
        if not dev.exists():
            evaluate(dev.name, splits['dev'], model, tokenizer, directory, config, seed,
                     b0.select_warmup(splits['train'], config['evaluation']['warmup_cases']))
        elif not (dev / 'summary.json').exists():
            raise RuntimeError(f'Incomplete held-out evaluation; refusing implicit replay: {dev}')
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != 'step':
                    state[key] = value.to(model.device)
        backend(model, config, training['kernel_backend']); model.train()
        torch.cuda.reset_peak_memory_stats(model.device)
        progress = {'epoch': epoch + 1, 'group': 0, 'step': (epoch + 1) * steps_per_epoch}
        # latest trainer epoch/group still denotes the just-finished epoch; on resume
        # its complete dev directory is recognized, then training continues safely.
    after = frozen_hash(model)
    if after != frozen_before:
        raise RuntimeError('Frozen original weights changed during G1')
    candidates = []
    for epoch in range(training['epochs'] + 1):
        metric = json.loads((directory / f'dev_epoch{epoch}/metrics.json').read_text())['overall']['hard_nll']
        candidates.append({'epoch': epoch, 'step': epoch * steps_per_epoch, 'hard_nll': metric,
                           'checkpoint': f'checkpoints/step{epoch * steps_per_epoch:06d}'})
    selected = dict(select_checkpoint(candidates))
    selected['adapter_sha256'] = sha256_file(directory / selected['checkpoint'] / 'adapter_model.safetensors')
    write_json(directory / 'selection.json', selected)
    write_json(directory / 'training_summary.json', {'frozen_before': frozen_before, 'frozen_after': after,
               'optimizer_steps': training['epochs'] * steps_per_epoch,
               'training_peak_allocated_bytes': training_peak, 'dev_candidates': candidates})
    del optimizer, scheduler, model; gc.collect(); torch.cuda.empty_cache()
    return selected


def execute(config, config_path, splits, baseline, smoke=False, resume=None):
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    directory = Path(resume).resolve() if resume else ROOT / config['output_root'] / f'{"smoke_train" if smoke else "train"}_{stamp}'
    fingerprint = runtime_fingerprint(ROOT / config['training']['fla_package_dir'])
    profile = ROOT / config['output_root'] / 'kernel_profile_train.json'
    if resume:
        metadata = json.loads((directory / 'metadata.json').read_text())
        if yaml.safe_load((directory / 'config.yaml').read_text()) != config:
            raise ValueError('Resume config differs from saved run')
        for source, expected in metadata['source_sha256'].items():
            if sha256_file(ROOT / source) != expected:
                raise ValueError(f'Resume source differs from snapshot: {source}')
        if metadata['mode'] != ('smoke_train' if smoke else 'train'):
            raise ValueError('Resume mode differs')
        if metadata['status'] == 'complete':
            raise ValueError('Run is already complete')
        # Reject drift before lock/log/pid creation or any metadata modification.
        preserve_runtime_provenance(directory, profile, fingerprint, metadata, resume=True)
    else:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        metadata = {'status': 'running', 'mode': 'smoke_train' if smoke else 'train', 'started_at': now(),
                    'source_sha256': sources(directory), 'completed_seeds': {}, 'test': {},
                    'baseline_run': config['baseline_run'], 'dataset': config['dataset'],
                    'split_manifest_sha256': baseline['split_manifest_sha256'],
                    'python': platform.python_version(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                    'versions': {name: importlib.metadata.version(name) for name in
                                 ('torch', 'transformers', 'peft', 'accelerate', 'numpy', 'PyYAML')}}
        shutil.copytree(ROOT / config['dataset']['split_dir'], directory / 'splits')
        preserve_runtime_provenance(directory, profile, fingerprint, metadata)
    lease = (directory / '.run.lock').open('a')
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        raise RuntimeError('This run already has a live worker') from None
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(directory / 'run.log'), logging.StreamHandler()], force=True)
    (directory / 'pid').write_text(str(os.getpid()) + '\n')
    print(f'RUN_DIR={directory}', flush=True)
    try:
        files = b0.model_file_hashes(config['model']['path'])
        if files != baseline['model_files_sha256']:
            raise ValueError('Local model source hashes differ from B0')
        metadata['model_files_sha256'] = files
        metadata['config_sha256'] = sha256_file(directory / 'config.yaml')
        metadata['config_source'] = str(config_path.resolve())
        metadata['raw_manifest_sha256'] = sha256_file(ROOT / config['dataset']['raw_dir'] / 'download_manifest.json')
        metadata['device'] = {'name': torch.cuda.get_device_name(config['model']['device']),
                              'logical': config['model']['device']}
        metadata['status'] = 'training'; write_json(directory / 'metadata.json', metadata)
        for seed in config['seeds'][:1] if smoke else config['seeds']:
            seed_dir = directory / f'seed{seed}'
            if str(seed) not in metadata['completed_seeds']:
                metadata['completed_seeds'][str(seed)] = train_seed(config, seed, splits, seed_dir, baseline, smoke)
                write_json(directory / 'metadata.json', metadata)
        if not smoke:
            lock_selections(directory / 'selections.json', metadata['completed_seeds'], config['seeds'])
            metadata['status'] = 'selections_locked'; write_json(directory / 'metadata.json', metadata)
            for seed in config['seeds']:
                if str(seed) in metadata['test']:
                    continue
                seed_dir = directory / f'seed{seed}'; selection = metadata['completed_seeds'][str(seed)]
                checkpoint = seed_dir / selection['checkpoint']
                if sha256_file(checkpoint / 'adapter_model.safetensors') != selection['adapter_sha256']:
                    raise ValueError('Locked selected adapter hash changed')
                if (seed_dir / 'test').exists():
                    if (seed_dir / 'test/summary.json').exists():
                        metadata['test'][str(seed)] = json.loads((seed_dir / 'test/summary.json').read_text())
                        continue
                    raise RuntimeError('Partial test exists; refusing to repeat official test inference')
                model, tokenizer, _ = load_local_model(config['model'])
                lora = config['lora']
                model = inject_lora(model, lora['rank'], lora['alpha'], lora['dropout'], lora['expected_target_modules'])
                expected = json.loads((checkpoint / 'reload_check_train.json').read_text())['label_logits']
                roundtrip(model, tokenizer, splits['train'][0], checkpoint, config, expected)
                metadata['status'] = 'evaluating'; write_json(directory / 'metadata.json', metadata)
                metadata['test'][str(seed)] = evaluate('test', splits['test'], model, tokenizer, seed_dir, config, seed,
                                                      b0.select_warmup(splits['train'], config['evaluation']['warmup_cases']))
                write_json(directory / 'metadata.json', metadata)
                del model; gc.collect(); torch.cuda.empty_cache()
            metrics = [metadata['test'][str(seed)]['metrics'] for seed in config['seeds']]
            aggregate = {key: {'mean': float(np.mean([row[key] for row in metrics])),
                              'sample_std': float(np.std([row[key] for row in metrics], ddof=1))}
                         for key in metrics[0] if isinstance(metrics[0][key], (int, float)) and not key.startswith('n_')}
            delta = paired_bootstrap(read_rows(ROOT / config['baseline_run'] / 'test/predictions.jsonl'),
                                     [read_rows(directory / f'seed{seed}/test/predictions.jsonl') for seed in config['seeds']],
                                     config['evaluation']['bootstrap_samples'])
            write_json(directory / 'summary.json', {'per_seed': metadata['test'], 'aggregate': aggregate,
                                                    'paired_accuracy_delta_vs_b0': delta})
        metadata['status'] = 'complete'
    except BaseException as error:
        metadata['status'] = 'failed'; metadata['error'] = {'type': type(error).__name__,
            'message': str(error), 'traceback': traceback.format_exc()}
        logging.exception('G1 interrupted/failed; preserving checkpoints')
        raise
    finally:
        metadata['updated_at'] = now(); write_json(directory / 'metadata.json', metadata)
        fcntl.flock(lease, fcntl.LOCK_UN); lease.close()
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/02_supervised/configs/g1.yaml')
    parser.add_argument('--prepare-only', action='store_true', help='Verify source/splits without loading a model')
    parser.add_argument('--smoke', action='store_true', help='Train-only two optimizer updates; no dev/test inference')
    parser.add_argument('--resume', type=Path, help='Resume the exact snapshotted config and source')
    args = parser.parse_args()
    if args.prepare_only and (args.smoke or args.resume):
        parser.error('--prepare-only cannot be combined with --smoke/--resume')
    config_path = args.resume / 'config.yaml' if args.resume else args.config
    config = yaml.safe_load(config_path.read_text()); splits, baseline = prepare(config)
    if args.prepare_only:
        print(json.dumps({'counts': {key: len(value) for key, value in splits.items()}, 'train_decisions': 4800,
                          'optimizer_steps_per_seed': 450, 'baseline_run': config['baseline_run']}))
        return
    execute(config, config_path, splits, baseline, args.smoke, args.resume)


if __name__ == '__main__':
    main()

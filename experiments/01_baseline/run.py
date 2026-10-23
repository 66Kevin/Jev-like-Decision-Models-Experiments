#!/usr/bin/env python3
"""Reproducible B0 evaluation. --smoke touches train only; default evaluates dev/test."""
import argparse
from datetime import datetime
import json
import importlib.util
import logging
import os
from pathlib import Path
import platform
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

from jev.data.typed_decisions import load_cases, prepare_splits, sha256_file
from jev.evaluation.metrics import evaluate_predictions
from jev.models.lm_readout import load_local_model, prepare_case, read_case, synchronize


def _json_default(value):
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f'Unsupported metadata type: {type(value).__name__}')


def write_json(path, value):
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default) + '\n'
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(content)
    temporary.replace(path)


def validate_config(config):
    fixed = {'temperature': 1.0, 'use_cache': False, 'case_batch_size': 1,
             'ece_bins': 15, 'prompt_version': 'b0_v1'}
    for key, expected in fixed.items():
        if config['evaluation'].get(key) != expected:
            raise ValueError(f'B0 requires {key}={expected!r}')
    if config['evaluation']['splits'] != ['dev', 'test']:
        raise ValueError('B0 production splits must be [dev, test]; calibration is reserved')
    if config['experiment_id'] != 'b0':
        raise ValueError('This runner implements experiment b0 only')
    if config['evaluation']['max_input_tokens'] != 8192:
        raise ValueError('B0 protocol requires max_input_tokens=8192')


def validate_source(raw_root, dataset):
    manifest = json.loads((raw_root / "download_manifest.json").read_text())
    if manifest.get("dataset") != dataset["name"] or manifest.get("revision") != dataset["revision"]:
        raise ValueError("Raw manifest dataset/revision differs from configured pinned source")


def select_warmup(train, count):
    selected, workflows = [], set()
    for case in train:
        if case['official_split'] != 'train':
            raise ValueError('Warmup and smoke may only access official train cases')
        if case['workflow'] not in workflows:
            selected.append(case); workflows.add(case['workflow'])
        if len(selected) == count:
            return selected
    for case in train:
        if case not in selected:
            selected.append(case)
        if len(selected) == count:
            return selected
    raise ValueError('Not enough train cases for warmup')


def save_logits(path, rows):
    width = max(len(row['option_ids']) for row in rows)
    logits = np.zeros((len(rows), width), dtype=np.float32)
    valid = np.zeros_like(logits, dtype=bool)
    identities = [[''] * width for _ in rows]
    for i, row in enumerate(rows):
        count = len(row['option_ids'])
        logits[i, :count] = row['label_logits']
        valid[i, :count] = True
        identities[i][:count] = row['option_ids']
    np.savez_compressed(path, logits=logits, valid_mask=valid,
                        option_ids=np.asarray(identities, dtype=str),
                        case_ids=np.asarray([row['case_id'] for row in rows], dtype=str),
                        question_ids=np.asarray([row['question_id'] for row in rows], dtype=str))


def snapshot_sources(run_dir):
    files = sorted((ROOT / 'src').rglob('*.py'))
    files += sorted((ROOT / 'tests').glob('test_*.py'))
    files += [Path(__file__).resolve(), ROOT / 'experiments/01_baseline/README.md']
    hashes = {}
    for source in files:
        relative = source.relative_to(ROOT)
        destination = run_dir / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        hashes[str(relative)] = sha256_file(destination)
    return hashes


def model_file_hashes(model_dir):
    root = Path(model_dir)
    names = ('*config*.json', '*token*', '*vocab*', '*merges*', '*chat_template*',
             '*.safetensors', '*.safetensors.index.json')
    files = sorted({path for pattern in names for path in root.glob(pattern) if path.is_file()})
    if not any(path.suffix == '.safetensors' for path in files):
        raise ValueError('Local model directory contains no safetensors weights')
    return {path.name: sha256_file(path) for path in files}


def check_padding(model, tokenizer, case, max_tokens):
    """Train-only numerical check: five-question batch versus five unpadded forwards."""
    batch = read_case(model, prepare_case(case['decisions'], tokenizer, max_tokens))
    max_logit = max_prob = 0.0
    argmax_changes = 0
    for i, decision in enumerate(case['decisions']):
        single = read_case(model, prepare_case([decision], tokenizer, max_tokens))
        max_logit = max(max_logit, float(np.max(np.abs(
            np.asarray(batch['label_logits'][i]) - single['label_logits'][0]))))
        max_prob = max(max_prob, float(np.max(np.abs(
            np.asarray(batch['probabilities'][i]) - single['probabilities'][0]))))
        argmax_changes += int(np.argmax(batch['probabilities'][i]) != np.argmax(single['probabilities'][0]))
    result = {'case_id': case['case_id'], 'split': 'train', 'max_absolute_logit_difference': max_logit,
              'max_absolute_probability_difference': max_prob, 'argmax_changes': argmax_changes,
              'logit_atol': 0.5, 'probability_atol': 0.05,
              'note': 'Empirical BF16 numerical guard after same-weight FP32 train-only padding validation; '
                      'batched and single-row kernels can differ and near-tie argmax may change. No label-based tuning.'}
    if max_logit > result['logit_atol'] or max_prob > result['probability_atol']:
        raise RuntimeError(f'Batched/unpadded readout mismatch exceeds BF16 tolerance: {result}')
    return result


def run_split(name, cases, model, tokenizer, run_dir, config):
    directory = run_dir / name
    directory.mkdir()
    records, times = [], []
    with (directory / 'predictions.jsonl').open('x') as stream:
        for index, case in enumerate(cases):
            synchronize(model.device)
            started = time.perf_counter()
            prepared = prepare_case(case['decisions'], tokenizer, config['evaluation']['max_input_tokens'])
            tokenize_seconds = time.perf_counter() - started
            result = read_case(model, prepared)
            elapsed = time.perf_counter() - started
            timing = {'case_id': case['case_id'], 'workflow': case['workflow'],
                      'tokenization_seconds': tokenize_seconds,
                      'inference_seconds': result['inference_seconds'], 'total_seconds': elapsed,
                      'input_lengths': prepared['input_lengths'],
                      'padded_tokens': int(prepared['input_ids'].numel())}
            times.append(timing)
            for i, decision in enumerate(case['decisions']):
                # Keep task identity, candidates, targets and scores; do not duplicate the state text.
                row = {key: value for key, value in decision.items() if key not in ('state', 'instructions')}
                row.update(probabilities=result['probabilities'][i], label_logits=result['label_logits'][i],
                           candidate_letters=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(decision['option_ids'])]),
                           input_tokens=prepared['input_lengths'][i], case_latency_seconds=elapsed)
                records.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            if (index + 1) % 20 == 0 or index + 1 == len(cases):
                stream.flush()
                logging.info('%s: %d/%d cases', name, index + 1, len(cases))
    save_logits(directory / 'logits.npz', records)
    metrics = evaluate_predictions(records, bootstrap_samples=config['evaluation']['bootstrap_samples'],
                                   seed=config['seed'])
    write_json(directory / 'metrics.json', metrics)
    elapsed = np.asarray([row['total_seconds'] for row in times])
    timing = {'n_cases': len(cases), 'n_decisions': len(records),
              'total_seconds': float(elapsed.sum()), 'mean_case_seconds': float(elapsed.mean()),
              'p50_case_seconds': float(np.percentile(elapsed, 50)),
              'p95_case_seconds': float(np.percentile(elapsed, 95)),
              'definition': 'Per case, five independent questions batched; tokenization, CPU-to-GPU transfer, '
                            'synchronized inference and CPU logits/probability conversion. Excludes model load, '
                            'preflight, warmup, correctness checks, metrics and file I/O.',
              'cases': times}
    write_json(directory / 'timing.json', timing)
    logging.info('%s Accuracy=%.4f HardNLL=%.4f ECE15=%.4f', name,
                 metrics['overall']['accuracy'], metrics['overall']['hard_nll'], metrics['overall']['ece_15'])
    return {'n_cases': len(cases), 'n_decisions': len(records),
            'metrics': metrics['overall'], 'p50_case_seconds': timing['p50_case_seconds'],
            'p95_case_seconds': timing['p95_case_seconds']}


def execute(config, config_path, splits, smoke=False):
    import pyarrow
    import transformers

    mode = 'smoke_train' if smoke else 'eval'
    timestamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    run_dir = ROOT / config['output_root'] / f'{mode}_{timestamp}_seed{config["seed"]}'
    run_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(run_dir / 'run.log'), logging.StreamHandler()], force=True)
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    metadata = {'status': 'running', 'mode': mode, 'experiment_id': 'b0',
                'started_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                'config_source': str(config_path), 'config_sha256': sha256_file(run_dir / 'config.yaml'),
                'seed': config['seed'],
                'python': platform.python_version(), 'versions': {'torch': torch.__version__,
                'transformers': transformers.__version__, 'numpy': np.__version__,
                'pyarrow': pyarrow.__version__, 'PyYAML': yaml.__version__},
                'dataset': config['dataset'], 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'optional_kernels': {name: importlib.util.find_spec(name) is not None
                                     for name in ('fla', 'causal_conv1d')},
                'protocol': {'prompt_version': 'b0_v1', 'candidate_order': 'original',
                             'generation': False, 'enable_thinking': False, 'use_cache': False,
                             'readout': 'native LM forward logits_to_keep=1; candidate-letter logits only',
                             'temperature': 1, 'model_dtype': 'bfloat16', 'softmax_dtype': 'float32',
                             'padding': 'left; attention-mask cumulative position IDs',
                             'max_input_tokens': 8192, 'truncation': False,
                             'calibration_inference': False}, 'completed_splits': {}}
    metadata_path = run_dir / 'metadata.json'
    write_json(metadata_path, metadata)
    print(f'RUN_DIR={run_dir}', flush=True)
    try:
        metadata['source_sha256'] = snapshot_sources(run_dir)
        split_dir = ROOT / config['dataset']['split_dir']
        shutil.copytree(split_dir, run_dir / 'splits')
        metadata['split_manifest_sha256'] = sha256_file(split_dir / 'manifest.json')
        metadata['model_files_sha256'] = model_file_hashes(config['model']['path'])
        diagnostic = ROOT / config['output_root'] / 'padding_diagnostic_train.json'
        if diagnostic.exists():
            shutil.copyfile(diagnostic, run_dir / diagnostic.name)
            metadata['train_padding_diagnostic'] = {'file': diagnostic.name,
                                                    'sha256': sha256_file(run_dir / diagnostic.name)}
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        torch.backends.cudnn.conv.fp32_precision = 'ieee'
        logging.info('Loading local frozen text model; no training or generation')
        model, tokenizer, loading = load_local_model(config['model'])
        metadata['loading_info'] = loading
        metadata['n_parameters'] = sum(p.numel() for p in model.parameters())
        metadata['n_trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        device = torch.device(config['model']['device'])
        metadata['device'] = {'logical': str(device), 'name': torch.cuda.get_device_name(device),
                              'capability': list(torch.cuda.get_device_capability(device)),
                              'total_memory_bytes': torch.cuda.get_device_properties(device).total_memory}
        warmup = select_warmup(splits['train'], config['evaluation']['warmup_cases'])
        requested = {'smoke_train': warmup} if smoke else {key: splits[key] for key in config['evaluation']['splits']}
        # Validate all selected inputs before any measured evaluation; no truncation or silent skips.
        metadata['preflight'] = {}
        for name, group in [('warmup_train', warmup), *requested.items()]:
            lengths = []
            for case in group:
                batch = prepare_case(case['decisions'], tokenizer, config['evaluation']['max_input_tokens'])
                lengths.extend(batch['input_lengths'])
            metadata['preflight'][name] = {'n_cases': len(group), 'n_decisions': len(lengths),
                                           'min_tokens': min(lengths), 'max_tokens': max(lengths)}
        max_candidates = max(len(d['option_ids']) for group in requested.values()
                             for case in group for d in case['decisions'])
        metadata['label_token_ids'] = {letter: tokenizer.encode(letter, add_special_tokens=False)[0]
                                       for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:max_candidates]}
        for case in warmup:
            read_case(model, prepare_case(case['decisions'], tokenizer))
        metadata['padding_equivalence'] = check_padding(model, tokenizer, warmup[0],
                                                       config['evaluation']['max_input_tokens'])
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        write_json(metadata_path, metadata)
        for name, group in requested.items():
            metadata['completed_splits'][name] = run_split(name, group, model, tokenizer, run_dir, config)
            metadata['memory'] = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                                  'current_allocated_bytes': torch.cuda.memory_allocated(device),
                                  'definition': 'PyTorch CUDA allocator including model, after warmup; '
                                                'reserved includes reusable warmup allocations.'}
            write_json(metadata_path, metadata)
        metadata['status'] = 'complete'
        logging.info('All requested splits complete: %s', ', '.join(requested))
    except BaseException as error:
        metadata['status'] = 'failed'
        metadata['error'] = {'type': type(error).__name__, 'message': str(error),
                             'traceback': traceback.format_exc()}
        logging.exception('Run failed; preserving all artifacts')
        raise
    finally:
        metadata['finished_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        write_json(metadata_path, metadata)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/01_baseline/configs/b0.yaml')
    parser.add_argument('--prepare-only', action='store_true', help='Validate local data and persist splits; no model')
    parser.add_argument('--smoke', action='store_true', help='Evaluate three train cases only in a separate run')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    validate_config(config)
    raw_root = ROOT / config['dataset']['raw_dir']
    validate_source(raw_root, config['dataset'])
    splits = prepare_splits(load_cases(raw_root), ROOT / config['dataset']['split_dir'],
                            seed=config['seed'], source_manifest=raw_root / 'download_manifest.json')
    if args.prepare_only:
        print(json.dumps({key: len(value) for key, value in splits.items()}))
        return
    execute(config, args.config.resolve(), splits, smoke=args.smoke)


if __name__ == '__main__':
    main()

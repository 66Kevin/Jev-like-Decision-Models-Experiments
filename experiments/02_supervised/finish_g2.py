#!/usr/bin/env python3
"""Verify a completed G2 run and reconstruct its report without model inference.

Usage: python experiments/02_supervised/finish_g2.py --run RUN [--update-docs]
A failed/incomplete run is rejected before any output is written. Training,
checkpoints and original predictions are read-only inputs to this tool.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import yaml
from safetensors import safe_open
from jev.data.typed_decisions import iter_decisions, load_cases, prepare_splits, sha256_file, stable_argmax
from jev.evaluation.metrics import evaluate_predictions
from jev.training.full import validate_full_checkpoint
from jev.training.full_runner import paired_seed_bootstrap, validate_config, validate_profile
from jev.training.runner import runtime_fingerprint, write_json

DEFAULT_RUN = ROOT/'outputs/02_supervised/g2/train_20261022_104739_026131'
SEEDS = [42, 43, 44]
IDENTITY_FIELDS = ('case_id', 'workflow', 'question_id', 'type', 'option_ids', 'option_texts',
                   'gold', 'hard_target', 'score_levels')


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal_tree(actual, expected, label, tolerance=1e-10):
    """Compare nested metrics, including nulls, counts and bootstrap definitions."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f'{label}: keys differ')
        for key, value in expected.items():
            equal_tree(actual[key], value, f'{label}.{key}', tolerance)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), f'{label}: length differs')
        for index, value in enumerate(expected):
            equal_tree(actual[index], value, f'{label}[{index}]', tolerance)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        require(isinstance(actual, (float, int)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), f'{label}: value differs')
    else:
        require(actual == expected, f'{label}: value differs')


def audit_prediction_split(directory, expected_decisions, bootstrap_samples=2000, seed=42):
    """Check raw identities, saved logits, probabilities, metrics and case timing."""
    directory = Path(directory)
    rows = read_rows(directory/'predictions.jsonl')
    require(len(rows) == len(expected_decisions) and bool(rows), f'{directory}: prediction identity count differs')
    seen = set()
    maximum_error = 0.
    with np.load(directory/'logits.npz', allow_pickle=False) as arrays:
        required = {'logits', 'valid_mask', 'option_ids', 'case_ids', 'question_ids'}
        require(set(arrays.files) == required, f'{directory}: NPZ arrays differ')
        width = max(len(row['option_ids']) for row in expected_decisions)
        shape = (len(rows), width)
        require(arrays['logits'].shape == shape and arrays['logits'].dtype == np.float32, 'NPZ logits shape/dtype differs')
        require(arrays['valid_mask'].shape == shape and arrays['valid_mask'].dtype == np.bool_, 'NPZ mask shape/dtype differs')
        require(arrays['option_ids'].shape == shape, 'NPZ option identity shape differs')
        require(arrays['case_ids'].shape == (len(rows),) and arrays['question_ids'].shape == (len(rows),),
                'NPZ case/question identity shape differs')
        for index, (row, source) in enumerate(zip(rows, expected_decisions)):
            key = (row['case_id'], row['question_id'])
            require(key not in seen, f'Duplicate prediction identity: {key}')
            seen.add(key)
            for field in IDENTITY_FIELDS:
                require(row.get(field) == source.get(field), f'{directory}: source {field} identity differs at {key}')
            count = len(source['option_ids'])
            require(row.get('candidate_letters') == list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:count]), 'Candidate letters differ')
            require(arrays['case_ids'][index] == row['case_id'] and arrays['question_ids'][index] == row['question_id'],
                    'NPZ prediction identity differs')
            require(np.array_equal(arrays['valid_mask'][index], np.arange(width) < count), 'NPZ valid mask differs')
            require(list(arrays['option_ids'][index, :count]) == row['option_ids'], 'NPZ candidate identity differs')
            require(np.all(arrays['option_ids'][index, count:] == '') and np.all(arrays['logits'][index, count:] == 0),
                    'NPZ padding must contain empty identities and zero logits')
            logits = np.asarray(row['label_logits'], dtype=np.float32)
            require(logits.shape == (count,) and np.isfinite(logits).all(), 'Nonfinite or malformed label logits')
            require(np.array_equal(arrays['logits'][index, :count], logits), 'NPZ/JSONL logits differ')
            probabilities = np.asarray(row['probabilities'], dtype=np.float64)
            require(probabilities.shape == (count,) and np.isfinite(probabilities).all()
                    and np.all((probabilities >= 0) & (probabilities <= 1))
                    and abs(probabilities.sum()-1) <= 1e-6, 'Invalid probability vector')
            rebuilt = np.exp(logits.astype(np.float64)-float(logits.max()))
            rebuilt /= rebuilt.sum()
            error = float(np.max(np.abs(probabilities-rebuilt)))
            maximum_error = max(maximum_error, error)
            require(error <= 2e-7, 'Saved probabilities disagree with candidate logits')
            require(isinstance(row.get('input_tokens'), int) and 0 < row['input_tokens'] <= 8192,
                    'Invalid input token count')
    metrics = evaluate_predictions(rows, bootstrap_samples=bootstrap_samples, seed=seed)
    equal_tree(read_json(directory/'metrics.json'), metrics, str(directory/'metrics.json'))
    timing = read_json(directory/'timing.json')
    expected_cases = {}
    for row in rows:
        expected_cases.setdefault(row['case_id'], []).append(row)
    require(timing['n_cases'] == len(expected_cases) and timing['n_decisions'] == len(rows), 'Timing count differs')
    cases = timing['cases']
    require([row['case_id'] for row in cases] == list(expected_cases), 'Timing case identity/order differs')
    for case in cases:
        group = expected_cases[case['case_id']]
        require(case['workflow'] == group[0]['workflow'], 'Timing workflow differs')
        for field in ('total_seconds', 'tokenization_seconds', 'inference_seconds'):
            require(math.isfinite(case[field]) and case[field] >= 0, f'Invalid timing {field}')
        require(case['total_seconds'] > 0, 'Case latency must be positive')
        require(case['input_lengths'] == [row['input_tokens'] for row in group], 'Timing input lengths differ')
        require(case['padded_tokens'] == max(case['input_lengths'])*len(group), 'Timing padded token count differs')
        for row in group:
            equal_tree(row['case_latency_seconds'], case['total_seconds'], 'Case latency')
    seconds = np.asarray([row['total_seconds'] for row in cases])
    for key, value in {'total_seconds': seconds.sum(), 'mean_case_seconds': seconds.mean(),
                       'p50_case_seconds': np.percentile(seconds, 50), 'p95_case_seconds': np.percentile(seconds, 95)}.items():
        equal_tree(timing[key], float(value), f'Timing {key}')
    if (directory/'summary.json').exists():
        summary = read_json(directory/'summary.json')
        equal_tree(summary['metrics'], metrics['overall'], 'Split summary metrics')
        for key in ('n_cases', 'n_decisions', 'p50_case_seconds', 'p95_case_seconds'):
            equal_tree(summary[key], timing[key], f'Split summary {key}')
    return {'rows': rows, 'metrics': metrics, 'timing': timing, 'max_probability_error': maximum_error}


def audit_train_history(path):
    rows = read_rows(path)
    require(len(rows) == 450 and [row['step'] for row in rows] == list(range(1, 451)), 'Training must cover exactly 450 steps')
    for row in rows:
        require(row['epoch'] == (row['step']-1)//150 and row['group'] == (row['step']-1)%150+1,
                'Training epoch/group differs')
        require(row['questions'] == 32, 'Training effective batch must be 32 questions')
        for key in ('loss', 'grad_norm', 'learning_rate', 'seconds'):
            require(math.isfinite(row[key]) and row[key] >= 0, f'Training {key} must be finite and nonnegative')
        scheduler_step = row['step']-1
        factor = scheduler_step/23 if scheduler_step < 23 else .5*(1+math.cos(math.pi*(scheduler_step-23)/427))
        require(math.isclose(row['learning_rate'], 2e-5*factor, rel_tol=1e-10, abs_tol=1e-15),
                'Training learning rate differs from fixed warmup/cosine recipe')
    return {'optimizer_steps': 450, 'questions_presented': 14400, 'update_seconds': sum(row['seconds'] for row in rows),
            'maximum_loss': max(row['loss'] for row in rows), 'maximum_grad_norm': max(row['grad_norm'] for row in rows)}


def aggregate_metrics(metrics):
    return {key: {'mean': float(np.mean([row[key] for row in metrics])),
                  'sample_std': float(np.std([row[key] for row in metrics], ddof=1))}
            for key, value in metrics[0].items() if isinstance(value, (int, float)) and not key.startswith('n_')}


def checked_path(base, relative):
    path = (Path(base)/relative).resolve()
    require(path.is_relative_to(Path(base).resolve()), f'Path escapes provenance root: {relative}')
    return path


def check_hash(path, expected):
    require(Path(path).is_file() and sha256_file(path) == expected, f'Checksum mismatch: {path}')


def audit_provenance(run, config, metadata):
    validate_config(config)
    check_hash(run/'config.yaml', metadata['config_sha256'])
    require(config['dataset'] == metadata['dataset'], 'Dataset configuration differs from metadata')
    require(len(metadata['source_sha256']) >= 25, 'Incomplete scientific source archive')
    for name, digest in metadata['source_sha256'].items():
        check_hash(checked_path(ROOT, name), digest)
        check_hash(checked_path(run/'source', name), digest)
    raw = checked_path(ROOT, config['dataset']['raw_dir'])
    split = checked_path(ROOT, config['dataset']['split_dir'])
    check_hash(raw/'download_manifest.json', metadata['raw_manifest_sha256'])
    check_hash(split/'manifest.json', metadata['split_manifest_sha256'])
    for name in ('train.jsonl', 'dev.jsonl', 'calibration.jsonl', 'test.jsonl', 'manifest.json'):
        require((split/name).is_file(), f'Missing immutable split: {name}')
        check_hash(run/'splits'/name, sha256_file(split/name))
    model = Path(config['model']['path'])
    for name, digest in metadata['model_files_sha256'].items():
        check_hash(checked_path(model, name), digest)
    base = checked_path(ROOT, config['baseline_run'])
    g1 = checked_path(ROOT, config['g1_run'])
    for other in (base, g1):
        prior = read_json(other/'metadata.json')
        require(prior['status'] == 'complete' and prior['dataset'] == config['dataset'], 'Reference run incomplete or dataset differs')
        require(prior['split_manifest_sha256'] == metadata['split_manifest_sha256'], 'Reference split differs')
        require(prior['model_files_sha256'] == metadata['model_files_sha256'], 'Reference original model differs')
        prior_config = yaml.safe_load((other/'config.yaml').read_text())
        require(prior_config['model'] == config['model'], 'Reference model/prompt input configuration differs')
    fingerprint = runtime_fingerprint(ROOT/config['training']['fla_package_dir'])
    equal_tree(fingerprint, metadata['runtime_fingerprint'], 'Runtime fingerprint')
    for profile in (run/'kernel_profile_train.json', ROOT/config['output_root']/'kernel_profile_train.json'):
        check_hash(profile, metadata['kernel_profile_sha256'])
        validate_profile(read_json(profile), config)
    cases = load_cases(raw)  # Verifies all eight pinned Parquet file checksums.
    splits = prepare_splits(cases, split, seed=42, source_manifest=raw/'download_manifest.json')
    require({key:len(value) for key,value in splits.items()} == {'train':960,'dev':120,'calibration':120,'test':400},
            'Pinned split sizes differ')
    return splits, base, g1


def audit_checkpoint(directory, selection, model_files):
    if selection['step'] == 0:
        reference = read_json(directory/'original_reference.json')
        require(reference['kind'] == 'original_model_reference' and reference['native_bf16_train_parity'] is True,
                'Original model parity/reference missing')
        require(reference['parameter_sha256'] == selection['parameter_sha256']
                and reference['model_files_sha256'] == model_files, 'Selected original reference differs')
        return {'kind':'original_model_reference', 'path':str(directory/'original_reference.json'),
                'parameter_sha256':reference['parameter_sha256'], 'model_files_sha256':model_files}
    require(selection['checkpoint'] == f'checkpoints/step{selection["step"]:06d}', 'Selected checkpoint path differs')
    checkpoint = checked_path(directory, selection['checkpoint'])
    manifest = validate_full_checkpoint(checkpoint)
    require(manifest['parameter_sha256'] == selection['parameter_sha256'] and manifest['progress']['step'] == selection['step'],
            'Selected checkpoint parameter hash/progress differs')
    require(manifest['aliases'].get('lm_head.weight') == 'model.embed_tokens.weight', 'Checkpoint Head/embedding tie missing')
    count = 0
    dtypes = set()
    for name in manifest['files_sha256']:
        if name.endswith('.safetensors'):
            with safe_open(checkpoint/name, framework='np') as weights:
                for key in weights.keys():
                    tensor = weights.get_slice(key)
                    dtypes.add(tensor.get_dtype())
                    count += math.prod(tensor.get_shape())
    require(dtypes == {'F32'} and count == 752393024, 'Selected checkpoint is not the complete FP32 text model')
    return {'kind':'full_fp32', 'path':str(checkpoint), 'parameter_sha256':manifest['parameter_sha256'],
            'files_sha256':manifest['files_sha256'], 'aliases':manifest['aliases'], 'stored_parameters':count,
            'evaluation_policy':'Native BF16 parameter copy; original native FP32 rotary buffers retained; native DeltaNet; temperature 1'}


def error_changes(reference, candidate):
    def correct(row):
        return stable_argmax(row['probabilities'], row['option_ids']) == row['hard_target']
    fixed = regressed = 0
    for base, current in zip(reference, candidate):
        old, new = correct(base), correct(current)
        fixed += int(not old and new)
        regressed += int(old and not new)
    return {'fixed_errors':fixed, 'new_errors':regressed, 'net_correct_change':fixed-regressed}


def finish(run, update_docs=False):
    """Perform every check before publishing any success artifact or documentation."""
    run = Path(run).resolve()
    metadata = read_json(run/'metadata.json')
    require(metadata.get('status') == 'complete' and metadata.get('mode') == 'train',
            'G2 run is not complete; refusing final analysis and documentation')
    require(set(metadata.get('completed_seeds', {})) == set(map(str, SEEDS))
            and set(metadata.get('test', {})) == set(map(str, SEEDS)), 'All three seeds must complete before analysis')
    config = yaml.safe_load((run/'config.yaml').read_text())
    splits, baseline_run, g1_run = audit_provenance(run, config, metadata)
    require(not any('calibration' in path.parts for path in run.rglob('predictions.jsonl')), 'Unexpected calibration inference artifacts')
    expected_dev = list(iter_decisions(splits['dev']))
    expected_test = list(iter_decisions(splits['test']))
    samples = config['evaluation']['bootstrap_samples']
    baseline = audit_prediction_split(baseline_run/'test', expected_test, samples, 42)
    g1 = [audit_prediction_split(g1_run/f'seed{seed}/test', expected_test, samples, seed) for seed in SEEDS]
    locked = read_json(run/'selections.json')
    require(locked.get('locked') is True and locked['seeds'] == metadata['completed_seeds'], 'All selections must be locked before test')
    require(locked.get('criterion') == 'minimum dev hard NLL; earlier step breaks ties', 'Selection criterion differs')
    lock_time = (run/'selections.json').stat().st_mtime_ns
    tests, curves, histories, chosen, hashes = [], {}, {}, {}, {}
    checked_count = 0
    max_error = 0.
    for seed in SEEDS:
        directory = run/f'seed{seed}'
        histories[str(seed)] = audit_train_history(directory/'train.jsonl')
        model_info = read_json(directory/'model.json')
        equal_tree(read_json(directory/'tokenization.json'), read_json(g1_run/f'seed{seed}/tokenization.json'),
                   'G2/G1 training tokenization/cache identity')
        require(model_info['trainable_parameters'] == 752393024 and model_info['frozen_parameters'] == 0
                and model_info['master_dtype'] == 'float32' and model_info['autocast_dtype'] == 'bfloat16'
                and model_info['head_and_embedding_tied'] is True and model_info['zero_step_train_parity'] is True,
                'Full-model training policy differs')
        candidates = []
        curves[str(seed)] = []
        for epoch in range(4):
            dev = directory/f'dev_epoch{epoch}'
            checked = audit_prediction_split(dev, expected_dev, samples, 42 if epoch == 0 else seed)
            if epoch == 0:
                reuse = read_json(dev/'reused_from_b0.json')
                require(reuse['native_bf16_train_parity'] is True and reuse['baseline_run'] == config['baseline_run'],
                        'Step-zero baseline reuse lacks parity evidence')
                check_hash(baseline_run/'dev/metrics.json', reuse['metrics_sha256'])
                for name in ('predictions.jsonl', 'logits.npz', 'metrics.json', 'timing.json'):
                    check_hash(dev/name, sha256_file(baseline_run/'dev'/name))
            else:
                integrity = read_json(directory/f'eval_integrity_epoch{epoch}.json')
                require(integrity['master_sha256_before'] == integrity['master_sha256_after']
                        and integrity['optimizer_parameter_identities_preserved'] is True,
                        'Dev evaluation modified FP32 training state')
                require((dev/'summary.json').stat().st_mtime_ns <= lock_time, 'Dev evaluation finished after selection lock')
            metrics = checked['metrics']['overall']
            candidates.append({'epoch':epoch, 'step':epoch*150, 'hard_nll':metrics['hard_nll']})
            curves[str(seed)].append({'epoch':epoch, **metrics})
            checked_count += len(checked['rows'])
            max_error = max(max_error, checked['max_probability_error'])
        selected = read_json(directory/'selection.json')
        require(selected == locked['seeds'][str(seed)], 'Per-seed and locked selections differ')
        best = min(candidates, key=lambda row:(row['hard_nll'], row['step']))
        for key, value in best.items():
            equal_tree(selected[key], value, f'Seed {seed} selection {key}')
        training_summary = read_json(directory/'training_summary.json')
        require(training_summary['optimizer_steps'] == 450 and training_summary['original_parameter_sha256'] != training_summary['final_parameter_sha256'],
                'Full-model training completion/change evidence missing')
        require([value['epoch'] for value in training_summary['dev_candidates']] == list(range(4)),
                'Training summary dev candidates must cover ordered epochs 0..3')
        for candidate in training_summary['dev_candidates']:
            matching = candidates[candidate['epoch']]
            for key in ('epoch','step','hard_nll'):
                equal_tree(candidate[key], matching[key], 'Training summary dev candidate')
        reference = read_json(directory/'original_reference.json')
        require(reference['kind'] == 'original_model_reference' and reference['native_bf16_train_parity'] is True
                and reference['model_files_sha256'] == metadata['model_files_sha256'], 'Original model reference differs')
        original_hash = training_summary['original_parameter_sha256']
        require(original_hash == model_info['original_parameter_sha256'] == reference['parameter_sha256'],
                'Original FP32 parameter identity differs')
        final_hash = training_summary['final_parameter_sha256']
        require(read_json(directory/'eval_integrity_epoch3.json')['master_sha256_after'] == final_hash,
                'Final dev/master parameter identity differs')
        latest = read_json(directory/'latest.json')
        require(latest['step'] == 450 and latest['parameter_sha256'] == final_hash
                and latest['checkpoint'] == 'checkpoints/step000450', 'Final committed generation differs')
        for candidate in training_summary['dev_candidates']:
            epoch = candidate['epoch']
            expected_path = f'checkpoints/step{epoch*150:06d}' if epoch else 'original_reference.json'
            expected_hash = original_hash if epoch == 0 else read_json(directory/f'eval_integrity_epoch{epoch}.json')['master_sha256_before']
            if epoch:
                manifest = read_json(checked_path(directory, expected_path)/'manifest.json')
                require(manifest['parameter_sha256'] == expected_hash and manifest['progress']['step'] == epoch*150,
                        'Dev generation differs from retained epoch checkpoint manifest')
            require(candidate['checkpoint'] == expected_path and candidate['parameter_sha256'] == expected_hash,
                    'Dev candidate FP32 generation differs')
        selected_candidate = training_summary['dev_candidates'][selected['epoch']]
        require(selected == selected_candidate, 'Selected model differs from complete dev candidate evidence')
        chosen[str(seed)] = {**selected, **audit_checkpoint(directory, selected, metadata['model_files_sha256'])}
        test_directory = directory/'test'
        require(lock_time <= (test_directory/'predictions.jsonl').stat().st_mtime_ns, 'Test predictions predate selection lock')
        checked = audit_prediction_split(test_directory, expected_test, samples, seed)
        equal_tree(metadata['test'][str(seed)], read_json(test_directory/'summary.json'), 'Metadata test summary')
        tests.append(checked)
        checked_count += len(checked['rows'])
        max_error = max(max_error, checked['max_probability_error'])
        for folder in [directory/f'dev_epoch{epoch}' for epoch in range(4)] + [test_directory]:
            for name in ('predictions.jsonl','logits.npz','metrics.json','timing.json'):
                hashes[str((folder/name).relative_to(run))] = sha256_file(folder/name)
    aggregate = aggregate_metrics([value['metrics']['overall'] for value in tests])
    paired = {'paired_accuracy_delta_vs_b0':paired_seed_bootstrap([baseline['rows']], [value['rows'] for value in tests], samples),
              'paired_accuracy_delta_vs_g1':paired_seed_bootstrap([value['rows'] for value in g1], [value['rows'] for value in tests], samples)}
    original_summary = read_json(run/'summary.json')
    equal_tree(original_summary['aggregate'], aggregate, 'Production aggregate')
    equal_tree(original_summary['per_seed'], metadata['test'], 'Production per seed')
    for name, value in paired.items():
        equal_tree(original_summary[name], value, name)
    groups = {}
    for name in ('by_workflow','by_type','by_question'):
        groups[name] = {group:{'b0':baseline['metrics'][name][group],
                               'g1':aggregate_metrics([value['metrics'][name][group] for value in g1]),
                               'g2':aggregate_metrics([value['metrics'][name][group] for value in tests])}
                        for group in baseline['metrics'][name]}
    costs = {str(seed): {'training':histories[str(seed)],
                         'training_peak_allocated_bytes':read_json(run/f'seed{seed}/training_memory.json')['peak_allocated_bytes'],
                         'test':{key:value for key,value in checked['timing'].items() if key != 'cases'},
                         'test_memory':read_json(run/f'seed{seed}/test/summary.json')['memory']}
             for seed,checked in zip(SEEDS,tests)}
    for value in costs.values():
        memory = value['test_memory']
        require(all(isinstance(memory.get(key), int) and memory[key] > 0
                    for key in ('peak_allocated_bytes','peak_reserved_bytes','current_allocated_bytes')),
                'Evaluation memory must be positive integer byte counts')
        require(isinstance(value['training_peak_allocated_bytes'], int) and value['training_peak_allocated_bytes'] > 0,
                'Training memory must be a positive integer byte count')
    reference_costs = {'b0':{key:value for key,value in baseline['timing'].items() if key != 'cases'},
                       'g1':{str(seed):{key:value for key,value in checked['timing'].items() if key != 'cases'}
                             for seed,checked in zip(SEEDS,g1)}}
    analysis = {'run':str(run),'seeds':SEEDS,'b0':baseline['metrics']['overall'],
                'g1':aggregate_metrics([value['metrics']['overall'] for value in g1]),'g2':aggregate,
                **paired,**groups,'dev_curves':curves,'per_seed_costs':costs,
                'per_seed_test':{str(seed):value['metrics']['overall'] for seed,value in zip(SEEDS,tests)},
                'selected':locked['seeds'],
                'error_changes':{str(seed):{'vs_b0':error_changes(baseline['rows'],checked['rows']),
                                           'vs_same_seed_g1':error_changes(g1[index]['rows'],checked['rows'])}
                                 for index,(seed,checked) in enumerate(zip(SEEDS,tests))},
                'reference_costs':reference_costs,'started_at':metadata['started_at'],'completed_at':metadata['updated_at']}
    recovery = run/'recovery/allocator_20261022'
    if (recovery/'metadata_before.json').exists():
        profile = read_json(recovery/'profile.json')
        launch = read_json(recovery/'launch.json')
        require(profile['status'] == 'passed' and profile['train_only'] is True
                and profile['allocator'] == launch['allocator'] == 'expandable_segments:True'
                and launch['resume_step'] == read_json(recovery/'latest_before.json')['step'] == 150,
                'Allocator recovery evidence differs')
        require(read_json(recovery/'metadata_before.json')['status'] == 'failed', 'Original failure evidence missing')
        check_hash(recovery/'profile.json', launch['profile_sha256'])
        check_hash(recovery/'continue.py', launch['script_sha256'])
        analysis['recovery'] = {'allocator':profile['allocator'],'resumed_from_step':150,
                                'full_train_only_updates':profile['full_updates'],
                                'native_eval_train_roundtrips':profile['eval_train_roundtrips'],
                                'profile_sha256':launch['profile_sha256'],
                                'peak_probe_allocated_bytes':max(value['peak_allocated_bytes'] for value in profile['rounds'])}
    verification = {'status':'passed','verified_at':datetime.now(timezone.utc).isoformat(),
                    'method':'Offline CPU reconstruction; no model inference; original results unchanged',
                    'prediction_records_checked':checked_count,'reused_step_zero_records':1800,
                    'reference_test_records_checked':8000,'source_files_checked':len(metadata['source_sha256']),
                    'maximum_probability_absolute_error':max_error,'record_file_sha256':hashes,
                    'checkpoint_integrity':'Selected complete full generations, including trainer.pt, verified against every manifest file checksum',
                    'selection_chronology':'All epoch dev summaries precede immutable selections.json; test prediction files follow lock (filesystem mtime evidence)',
                    'error_changes':analysis['error_changes']}
    require(checked_count == 13200, 'Unexpected total G2 saved prediction count')
    # Publishing starts only after every input, summary and selected checkpoint passed.
    write_json(run/'analysis.json', analysis)
    write_json(run/'selected_models.json', {'seeds':chosen,'original_model':config['model']['path'],
                                          'loading_note':'Full weights use canonical text-model keys. Use run_g2.selected_eval_model for original native dtype/buffer policy.'})
    write_json(run/'verification.json', verification)
    if update_docs:
        update_documentation(run, config, analysis, verification)
    return verification


def create_figures(analysis):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = ROOT/'reports/figures'
    directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for seed in SEEDS:
        values = analysis['dev_curves'][str(seed)]
        chosen = analysis['selected'][str(seed)]['epoch']
        for axis, metric, title in zip(axes, ('accuracy','hard_nll','soft_ce'), ('Dev Accuracy','Dev Hard NLL','Dev Soft CE')):
            series = [row[metric] for row in values]
            line, = axis.plot(range(4), series, marker='o', label=f'Seed {seed}')
            axis.scatter([chosen], [series[chosen]], marker='*', s=140, color=line.get_color(), zorder=4)
            axis.set(title=title, xlabel='Epoch', xticks=range(4))
            axis.grid(alpha=.2)
    axes[0].legend()
    figure.tight_layout()
    for extension in ('svg','png'):
        figure.savefig(directory/f'g2_dev_curves.{extension}', dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for axis, metric, title in zip(axes, ('accuracy','hard_nll','ece_15'), ('Test Accuracy','Test Hard NLL','Test ECE15')):
        means = [analysis['b0'][metric], analysis['g1'][metric]['mean'], analysis['g2'][metric]['mean']]
        errors = [0, analysis['g1'][metric]['sample_std'], analysis['g2'][metric]['sample_std']]
        axis.bar(['B0','G1','G2'], means, yerr=errors, capsize=4, color=['#8899aa','#4a82bd','#3c9f79'])
        axis.set_title(title)
        axis.grid(axis='y', alpha=.2)
    figure.tight_layout()
    for extension in ('svg','png'):
        figure.savefig(directory/f'g2_comparison.{extension}', dpi=160)
    plt.close(figure)


def update_documentation(run, config, analysis, verification):
    create_figures(analysis)
    stats = analysis['g2']
    accuracy = stats['accuracy']
    def estimate(group, metric, percent=False):
        value = analysis[group][metric]
        factor = 100 if percent else 1
        suffix = ' pp' if percent else ''
        return f'{value["mean"]*factor:.2f}{"%" if percent else ""} ± {value["sample_std"]*factor:.2f}{suffix}' if percent else f'{value["mean"]:.4f} ± {value["sample_std"]:.4f}'
    relative_run = run.relative_to(ROOT)
    lines = ['# G2 结果：文本模型全量微调 + 原 LM Head', '',
             '**状态：三个 seed 的 450 步训练、dev 选模及官方 test 已完成，离线产物复核通过。**', '',
             f'G2 的 test Accuracy 为 **{estimate("g2", "accuracy", True)}**。本实验保留 B0／G1 的候选字母读出、输入和划分，训练全部 **752,393,024** 个文本参数（包括绑定的 embedding／LM Head）。', '',
             'Accuracy 衡量与数据集教师 gold argmax 的一致性，不能替代人工业务真值。± 表示三个 seed 的样本标准差，结果没有进行温度校准。', '',
             '## 总体结果', '', '| 指标 | B0 | G1（三 seed） | G2（三 seed） |', '| --- | --- | --- | --- |']
    for metric, title in [('accuracy','Accuracy'),('hard_nll','Hard NLL'),('soft_ce','Soft CE'),('kl_q_p','KL(gold ‖ prediction)'),
                          ('brier_soft','Soft Brier'),('brier_hard','Hard Brier'),('ece_15','ECE15'),('score_mae','Score 期望 MAE')]:
        base = f'{analysis["b0"][metric]*100:.2f}%' if metric == 'accuracy' else f'{analysis["b0"][metric]:.4f}'
        lines.append(f'| {title} | {base} | {estimate("g1",metric,metric=="accuracy")} | {estimate("g2",metric,metric=="accuracy")} |')
    for name, title in [('paired_accuracy_delta_vs_b0','B0'),('paired_accuracy_delta_vs_g1','G1')]:
        value = analysis[name]
        lines += ['', f'相对 {title} 的平均 Accuracy 差为 **{value["mean_accuracy_delta"]*100:+.2f} pp**，配对 case bootstrap 95% 区间为 **{value["low"]*100:+.2f}～{value["high"]*100:+.2f} pp**。']
    lines += ['', '这里按每个已训练模型的正确性取 seed 均值，再对完整 case 重采样；没有平均概率构造集成模型。置信区间反映固定模型下测试 case 的不确定性，三 seed 标准差反映本次重新训练的波动，两者含义不同。', '',
              '![G2 总体比较](figures/g2_comparison.png)', '', '误差棒为 seed 样本标准差；B0 为单个固定模型。[SVG](figures/g2_comparison.svg)。', '',
              '## 选模与各 seed 结果', '', '每个 seed 完成 3 epochs／450 步。只按 dev Hard NLL 最小值选模，同分取更早步骤，包含 step 0。三个选择锁定后才开始 test。', '',
              '| Seed | 选中 epoch／step | Dev NLL | Test Accuracy | Test NLL | Test ECE15 |', '| --- | --- | --- | --- | --- | --- |']
    for seed in SEEDS:
        chosen = analysis['selected'][str(seed)]; test = analysis['per_seed_test'][str(seed)]
        lines.append(f'| {seed} | {chosen["epoch"]}／{chosen["step"]} | {chosen["hard_nll"]:.4f} | {test["accuracy"]*100:.2f}% | {test["hard_nll"]:.4f} | {test["ece_15"]:.4f} |')
    lines += ['', '![G2 Dev 曲线](figures/g2_dev_curves.png)', '', '星号为 dev NLL 选择点，epoch 0 复用经训练样本原生读出一致性验证的 B0 dev。[SVG](figures/g2_dev_curves.svg)。Hard NLL、Accuracy 与 Soft CE 衡量不同目标；没有根据 test 改选 checkpoint。', '', '## 分工作流与题型', '']
    for grouping, title in [('by_workflow','工作流'),('by_type','类型')]:
        lines += [f'| {title} | B0 Accuracy | G1 Accuracy | G2 Accuracy | G2 − G1 |', '| --- | --- | --- | --- | --- |']
        for name, values in analysis[grouping].items():
            g2, g1 = values['g2']['accuracy'], values['g1']['accuracy']
            lines.append(f'| {name} | {values["b0"]["accuracy"]*100:.2f}% | {g1["mean"]*100:.2f}% | {g2["mean"]*100:.2f}% ± {g2["sample_std"]*100:.2f} pp | {(g2["mean"]-g1["mean"])*100:+.2f} pp |')
        lines.append('')
    lines += ['上述分组为描述性比较，未做逐组多重检验。完整逐问题指标见 analysis.json。', '', '| Seed | 修正 B0 错误／新增错误 | 修正同 seed G1 错误／新增错误 |', '| --- | --- | --- |']
    for seed in SEEDS:
        change = analysis['error_changes'][str(seed)]
        lines.append(f'| {seed} | {change["vs_b0"]["fixed_errors"]}／{change["vs_b0"]["new_errors"]} | {change["vs_same_seed_g1"]["fixed_errors"]}／{change["vs_same_seed_g1"]["new_errors"]} |')
    costs = analysis['per_seed_costs']
    p50 = [value['test']['p50_case_seconds'] for value in costs.values()]
    p95 = [value['test']['p95_case_seconds'] for value in costs.values()]
    train_peak = [value['training_peak_allocated_bytes']/2**30 for value in costs.values()]
    eval_peak = [value['test_memory']['peak_allocated_bytes']/2**30 for value in costs.values()]
    b0_latency = analysis['reference_costs']['b0']['p50_case_seconds']*1000
    g1_latency = np.mean([value['p50_case_seconds'] for value in analysis['reference_costs']['g1'].values()])*1000
    lines += ['', '## 训练设置、成本与限制', '',
              '| 项目 | B0 | G1 | G2 |', '| --- | --- | --- | --- |',
              f'| 每 case p50 | {b0_latency:.2f} ms | {g1_latency:.2f} ms | {np.mean(p50)*1000:.2f} ms |', '',
              f'- 原始模型：`{config["model"]["path"]}`；数据版本：`{config["dataset"]["revision"]}`。',
              '- train／dev／calibration／test 分别为 960／120／120／400 cases；训练只用 gold argmax 硬候选 CE，校准集未使用。',
              '- 三 seed 从相同原始模型独立初始化，不继承 G1 adapter。全量 FP32 参数、梯度和 Adam 状态，BF16 autocast；AdamW LR 2e-5、weight decay 0.01、clip 1.0、23 步 warmup 后 cosine。',
              f'- 有效 batch 32 = microbatch {config["training"]["microbatch_questions"]} × 累积 {config["training"]["gradient_accumulation"]}。每 seed 14,400 次训练题呈现。训练只切换 FLA DeltaNet chunk；评估为独立 BF16 副本、原生内核，保留原生 FP32 rotary buffers，不损伤 FP32 训练权重。',
              f'- G2 每 case p50：{np.mean(p50)*1000:.2f} ± {np.std(p50,ddof=1)*1000:.2f} ms；p95 均值：{np.mean(p95)*1000:.2f} ms。各 seed 训练峰值 allocated：{min(train_peak):.2f}～{max(train_peak):.2f} GiB；评估峰值：{min(eval_peak):.2f}～{max(eval_peak):.2f} GiB。',
              '- 延迟是每 case 五道独立问题，包含分词、传输、同步原生推理和概率搬回 CPU，排除模型加载、预热和文件写入；p50 是各 seed p50 的均值。G1 当前未合并 adapter，其运行栈开销不能当作 LoRA 的固有倍率。',
              f'- 流程开始：{analysis["started_at"]}；完成：{analysis["completed_at"]}。墙钟跨度包含 checkpoint IO、dev／test、故障和暂停；各 seed 纯更新耗时保存在 analysis.json，不含评估和保存。',
              '- G1 使用 LR 2e-4，冻结 BF16 原参数并训练 FP32 LoRA；G2 使用 LR 2e-5，全部原参数为 FP32 master。本结果比较两套固定配方，不代表两种方法分别充分调参后的最优结果。', '',
              '本轮围绕 Q1 的固定 LM Head 比较，尚不能回答专用 Pointer／Candidate Head 的价值、未知候选泛化或软标签蒸馏收益。', '']
    recovery = run/'recovery/allocator_20261022'
    if (recovery/'metadata_before.json').is_file():
        lines += ['## 中断与恢复记录', '',
                  '原运行在 seed 42 第 154 步因 CUDA OOM 中断，step 150 的完整 FP32 模型／Adam／scheduler／RNG 检查点保留，第 1 轮 dev 也已完成。', '',
                  '恢复前仅用训练集补测：完整 32 题梯度累积更新、原故障批次、最长输入与三次独立原生评估切换。通过后使用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 恢复内存分配；原训练源码、配置、microbatch 和选模协议不变。', '',
                  '从 step 150 恢复，旧的未提交 151～153 步另存档后重放，最终 train.jsonl 仍为连续 450 个已提交步骤。旧故障元数据、预检脚本／结果和启动记录保存在运行目录的 recovery/allocator_20261022，未把部分 dev 当作正式 test 结论。', '']
    lines += ['## 复核和复用', '',
              f'离线复核 {verification["prediction_records_checked"]:,} 条 G2 保存预测（其中 1,800 条复用 B0 dev），另核对 B0／G1 的 8,000 条 test 预测；没有重新进行模型推理。身份、候选、gold、hard target、原始 logits／mask 和概率与固定原始数据一一对应。重建概率最大绝对差为 {verification["maximum_probability_absolute_error"]:.3g}。', '',
              '总体／分组／bootstrap 指标、三 seed 统计与配对比较均复算一致；源代码、配置、模型、原始数据、划分、FLA 和内核预检哈希核对通过；所选完整 checkpoint 包括 trainer.pt 在内的所有文件校验通过。文件 mtime 支持先锁定全部选择再开始 test 的顺序，这属于文件系统证据，不是外部防篡改时间戳。', '',
              f'- [正式汇总](../{relative_run}/summary.json)', f'- [分组、曲线与成本](../{relative_run}/analysis.json)',
              f'- [复核记录](../{relative_run}/verification.json)', f'- [所选完整模型与 SHA256](../{relative_run}/selected_models.json)',
              f'- [训练日志](../{relative_run}/run.log)', '',
              '完整权重使用独立文本模型的 canonical 参数名，不能直接按原多模态 checkpoint 名称加载。复用时使用 `experiments/02_supervised/run_g2.py` 中的 `selected_eval_model`，以保留绑定权重和原生 BF16／FP32 buffer 策略。完成的 run 不需要再调用 --resume。', '',
              '可用以下 CPU 命令重复复核并更新报告；不会进行模型推理：', '',
              '```bash', f'python experiments/02_supervised/finish_g2.py --run {relative_run} --update-docs', '```', '']
    report = ROOT/'reports/02_supervised_g2.md'
    temporary = report.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines)); temporary.replace(report)
    readme = ROOT/'README.md'
    text = readme.read_text()
    status = '**当前进度：B0、G1、G2 均已完成评估与复核。G1／G2 各完成三个 seed 的训练和 dev 选模，结果见 [G1 报告](reports/02_supervised_g1.md) 与 [G2 报告](reports/02_supervised_g2.md)。P0／P1／P2／C1 及阶段 3～4 尚未开展。**'
    text, count = re.subn(r'^\*\*当前进度：.*?\*\*$', lambda _:status, text, count=1, flags=re.MULTILINE)
    require(count == 1, 'README current status anchor missing')
    row = f'| G2（三 seed） | {accuracy["mean"]*100:.2f}% ± {accuracy["sample_std"]*100:.2f} pp | {stats["hard_nll"]["mean"]:.4f} | {stats["soft_ce"]["mean"]:.4f} | {stats["brier_soft"]["mean"]:.4f} | {stats["ece_15"]["mean"]:.4f} | {np.mean(p50)*1000:,.2f} ms/case | {np.mean(eval_peak):.2f} GiB |'
    text, count = re.subn(r'^\| G2（[^\n]*$', lambda _:row, text, count=1, flags=re.MULTILINE)
    require(count == 1, 'README G2 result row anchor missing')
    paragraph = f'G2 官方 test 的三 seed Accuracy 为 **{accuracy["mean"]*100:.2f}% ± {accuracy["sample_std"]*100:.2f} 个百分点**，相对 G1 均值变化 **{analysis["paired_accuracy_delta_vs_g1"]["mean_accuracy_delta"]*100:+.2f} 个百分点**；NLL 均值 **{stats["hard_nll"]["mean"]:.4f}**，ECE15 均值 **{stats["ece_15"]["mean"]:.4f}**。完整配对区间、全量训练成本与恢复记录见 [G2 报告](reports/02_supervised_g2.md)。'
    if re.search(r'^G2 官方 test .*$', text, re.MULTILINE):
        text = re.sub(r'^G2 官方 test .*$', lambda _:paragraph, text, count=1, flags=re.MULTILINE)
    else:
        text = text.replace('## 1. 项目整体路线', paragraph+'\n\n## 1. 项目整体路线', 1)
    text = text.replace('G2 已启动，状态见 [G2 记录](reports/02_supervised_g2.md)', 'G2 已完成，结果见 [G2 报告](reports/02_supervised_g2.md)')
    temporary = readme.with_suffix('.md.tmp')
    temporary.write_text(text); temporary.replace(readme)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--update-docs', action='store_true')
    args = parser.parse_args()
    try:
        result = finish(args.run, update_docs=args.update_docs)
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(1, f'G2 offline review rejected: {error}\n')
    print(json.dumps({'status':result['status'],'records':result['prediction_records_checked'],
                      'run':str(args.run.resolve())}, ensure_ascii=False))


if __name__ == '__main__':
    main()

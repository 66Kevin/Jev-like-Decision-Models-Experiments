"""Pinned typed-decisions loading, gold-free inputs and persistent case splits."""

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random


WORKFLOWS = ('agent_trace_observability', 'customer_service',
             'invoice_processing', 'security_incidents')
SPLIT_COUNTS = {'train': 240, 'dev': 30, 'calibration': 30, 'test': 100}


def _decoded(value):
    return json.loads(value) if isinstance(value, str) else value


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def stable_argmax(values, option_ids):
    """Choose the lexicographically smallest option identity on an exact tie."""
    return min(range(len(values)), key=lambda i: (-values[i], option_ids[i]))


def normalize_case(raw):
    """Retain original task content; keep gold separate from the model input."""
    import math

    state = _decoded(raw['state'])
    questions, gold = _decoded(raw['questions']), _decoded(raw['gold'])
    if not isinstance(state, dict) or not isinstance(questions, dict):
        raise ValueError('State and questions must be JSON objects')
    if len(questions) != 5 or raw.get('n_questions', 5) != 5 or set(questions) != set(gold):
        raise ValueError('Expected the same five questions in input and gold')
    case_id = raw['id']
    if not isinstance(case_id, str) or not case_id or raw['split'] not in ('train', 'test'):
        raise ValueError('Invalid case identity or official split')
    decisions = []
    for question_id, question in questions.items():
        kind = question['type']
        criteria = question.get('criteria')
        if kind == 'score':
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise ValueError('Score question requires an ordered rubric')
            option_ids = [str(i) for i in range(len(criteria))]
            texts = list(criteria)
            levels = [float(i) for i in range(len(criteria))]
        elif kind in ('choice', 'noul'):
            if criteria is None and kind == 'noul':
                criteria = {'false': 'False', 'true': 'True'}
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError('Choice question requires candidate criteria')
            option_ids, texts = list(criteria), list(criteria.values())
            levels = None
        else:
            raise ValueError(f'Unsupported decision type: {kind}')
        if not all(isinstance(x, str) and x for x in option_ids + texts):
            raise ValueError('Candidate identities and texts must be nonempty strings')
        target = gold[question_id]
        probabilities = target['probabilities']
        if target.get('type', kind) != kind or set(probabilities) != set(option_ids):
            raise ValueError('Gold candidate identities/type differ from input')
        values = [float(probabilities[identity]) for identity in option_ids]
        total = sum(values)
        if (not all(math.isfinite(x) and 0 <= x <= 1 for x in values)
                or abs(total - 1) > 1e-5):
            raise ValueError('Gold probabilities must sum to one within rounding tolerance 1e-5')
        values = [x / total for x in values]
        decisions.append({
            'case_id': case_id, 'workflow': raw['workflow'], 'question_id': question_id,
            'type': kind, 'state': state, 'instructions': question['instructions'],
            'option_ids': option_ids, 'option_texts': texts, 'gold': values,
            'hard_target': stable_argmax(values, option_ids), 'score_levels': levels,
        })
    return {'case_id': case_id, 'workflow': raw['workflow'], 'official_split': raw['split'],
            'state': state, 'decisions': decisions}


def model_input(decision):
    """Explicit whitelist prevents labels and dataset construction metadata entering prompts."""
    return copy.deepcopy({key: decision[key] for key in
                          ('state', 'instructions', 'option_ids', 'option_texts', 'type')})


def iter_decisions(cases):
    for case in cases:
        yield from case['decisions']


def load_cases(raw_root):
    """Verify pinned source checksums, then read the eight official Parquet files."""
    import pyarrow.parquet as parquet

    root = Path(raw_root)
    manifest = json.loads((root / 'download_manifest.json').read_text())
    expected = {f'{workflow}/{split}-00000-of-00001.parquet'
                for workflow in WORKFLOWS for split in ('train', 'test')}
    files = manifest['files']
    if len(files) != 8 or {item['file'] for item in files} != expected:
        raise ValueError('Source manifest must contain exactly the eight expected files')
    cases = []
    for item in sorted(files, key=lambda row: row['file']):
        path = root / item['file']
        if sha256_file(path) != item['sha256']:
            raise ValueError(f'Source checksum mismatch: {path}')
        workflow, filename = item['file'].split('/')
        split = filename.split('-')[0]
        rows = parquet.read_table(path).to_pylist()
        if len(rows) != {'train': 300, 'test': 100}[split]:
            raise ValueError(f'Unexpected number of source cases: {path}')
        for row in rows:
            if row['workflow'] != workflow or row['split'] != split:
                raise ValueError(f'Source row disagrees with source filename: {row["id"]}')
            cases.append(normalize_case(row))
    make_splits(cases)  # Validate counts, unique identities and canonical states before use.
    return cases


def make_splits(cases, seed=42):
    """Split complete states within each workflow; never split individual questions."""
    if not cases:
        raise ValueError('No cases supplied')
    ids, states = set(), set()
    for case in cases:
        identity = case['case_id']
        canonical_state = _json(case['state'])
        if identity in ids or canonical_state in states:
            raise ValueError(f'Duplicate case identity or canonical state: {identity}')
        ids.add(identity); states.add(canonical_state)
        if case['official_split'] not in ('train', 'test') or len(case['decisions']) != 5:
            raise ValueError(f'Unexpected official split or question count: {identity}')
    splits = {key: [] for key in SPLIT_COUNTS}
    for workflow in sorted({case['workflow'] for case in cases}):
        group = [case for case in cases if case['workflow'] == workflow]
        counts = Counter(case['official_split'] for case in group)
        if counts != {'train': 300, 'test': 100}:
            raise ValueError(f'Unexpected case counts for {workflow}: {counts}')
        train = sorted((c for c in group if c['official_split'] == 'train'), key=lambda c: c['case_id'])
        random.Random(f'{seed}:{workflow}').shuffle(train)
        splits['train'].extend(train[:240])
        splits['dev'].extend(train[240:270])
        splits['calibration'].extend(train[270:])
        splits['test'].extend(sorted((c for c in group if c['official_split'] == 'test'),
                                     key=lambda c: c['case_id']))
    return splits


def prepare_splits(cases, split_dir, seed=42, source_manifest=None):
    """Create ID-only files once; reject stale/partial/tampered files instead of replacing."""
    splits = make_splits(cases, seed)
    root = Path(split_dir)
    if isinstance(source_manifest, (str, Path)):
        source_manifest = json.loads(Path(source_manifest).read_text())
    manifest = {
        'split_id': f'seed{seed}', 'seed': seed,
        'algorithm': 'sorted-case-ids; random.Random(str(seed)+":"+workflow).shuffle; 240/30/30',
        'source_manifest': source_manifest,
        'source_manifest_sha256': hashlib.sha256(_json(source_manifest).encode()).hexdigest(),
        'gold_normalization': 'divide by original sum; require original sum within 1e-5 of 1',
        'hard_label_ties': 'lexicographically smallest original option identity among exact maxima',
        'counts': {key: len(value) for key, value in splits.items()},
        'by_workflow': {key: dict(Counter(c['workflow'] for c in value)) for key, value in splits.items()},
        'case_state_sha256': hashlib.sha256(_json(sorted(
            (c['case_id'], _json(c['state'])) for c in cases)).encode()).hexdigest(),
    }
    expected = {f'{split}.jsonl': ''.join(json.dumps({'case_id': c['case_id']})+'\n' for c in group)
                for split, group in splits.items()}
    expected['manifest.json'] = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    exists = [root.joinpath(name).exists() for name in expected]
    if any(exists):
        if not all(exists):
            raise ValueError('Incomplete existing split files; refusing to overwrite')
        for name, content in expected.items():
            if root.joinpath(name).read_text() != content:
                raise ValueError(f'Existing split differs from requested protocol/provenance: {name}')
    else:
        root.mkdir(parents=True, exist_ok=True)
        for name, content in expected.items():
            with root.joinpath(name).open('x') as stream:
                stream.write(content)
    return splits

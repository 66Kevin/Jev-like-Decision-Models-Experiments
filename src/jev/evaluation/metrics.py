"""Uncalibrated decision metrics with explicit hard/soft target definitions."""

from collections import defaultdict
import math

import numpy as np

from jev.data.typed_decisions import stable_argmax


def _validated(record):
    p = np.asarray(record['probabilities'], dtype=np.float64)
    q = np.asarray(record['gold'], dtype=np.float64)
    ids = record['option_ids']
    if (p.ndim != 1 or q.ndim != 1 or len(p) < 2 or len(p) != len(q)
            or len(ids) != len(p) or len(set(ids)) != len(ids)):
        raise ValueError('Expected aligned, distinct candidate identities and probability vectors')
    for values in (p, q):
        if (not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1)
                or not math.isclose(float(values.sum()), 1., abs_tol=1e-6, rel_tol=0)):
            raise ValueError('Probability vectors must be finite, nonnegative and normalized')
    target = stable_argmax(q, ids)
    if 'hard_target' in record and record['hard_target'] != target:
        raise ValueError('Stored hard target disagrees with stable gold argmax')
    predicted = stable_argmax(p, ids)
    return p, q, target, predicted


def compute_metrics(records):
    """Equal weight per decision; ECE15 uses hard-target correctness and max p."""
    rows = list(records)
    if not rows:
        raise ValueError('Cannot evaluate an empty set of decisions')
    totals = defaultdict(float)
    bins = [[] for _ in range(15)]
    score_errors = []
    gold_ties = prediction_ties = 0
    for row in rows:
        p, q, target, predicted = _validated(row)
        correct = float(predicted == target)
        log_p = np.log(np.maximum(p, 1e-12))
        totals['accuracy'] += correct
        totals['hard_nll'] += -log_p[target]
        totals['soft_ce'] += -float(q @ log_p)
        positive = q > 0
        totals['kl_q_p'] += float(q[positive] @ (np.log(q[positive]) - log_p[positive]))
        totals['brier_soft'] += float(np.square(p - q).sum())
        onehot = np.zeros_like(p); onehot[target] = 1
        totals['brier_hard'] += float(np.square(p-onehot).sum())
        confidence = float(p.max())
        bins[min(14, int(confidence * 15))].append((confidence, correct))
        gold_ties += int(np.sum(q == q.max()) > 1)
        prediction_ties += int(np.sum(p == p.max()) > 1)
        if row['type'] == 'score':
            levels = np.asarray(row['score_levels'], dtype=np.float64)
            if levels.shape != p.shape or not np.all(np.isfinite(levels)) or np.any(np.diff(levels) <= 0):
                raise ValueError('Score levels must be finite, aligned and strictly increasing')
            score_errors.append(abs(float(p @ levels) - float(q @ levels)))
    result = {key: float(value / len(rows)) for key, value in totals.items()}
    result.update({
        'n_decisions': len(rows), 'n_score_decisions': len(score_errors),
        'score_mae': float(np.mean(score_errors)) if score_errors else None,
        'ece_15': float(sum(abs(sum(correct-confidence for confidence, correct in bucket))
                            for bucket in bins) / len(rows)),
        'gold_tie_count': gold_ties, 'prediction_tie_count': prediction_ties,
    })
    return result


def evaluate_predictions(records, bootstrap_samples=2000, seed=42):
    """Overall/group metrics and percentile Accuracy interval resampling complete cases."""
    rows = list(records)
    overall = compute_metrics(rows)
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise ValueError('bootstrap_samples must be a positive integer')
    groups = {key: defaultdict(list) for key in ('workflow', 'type', 'question')}
    cases = defaultdict(list)
    for row in rows:
        groups['workflow'][row['workflow']].append(row)
        groups['type'][row['type']].append(row)
        groups['question'][f'{row["workflow"]}/{row["question_id"]}'].append(row)
        _, _, target, predicted = _validated(row)
        cases[row['case_id']].append(float(target == predicted))
    case_values = [cases[key] for key in sorted(cases)]
    correct = np.array([sum(values) for values in case_values])
    count = np.array([len(values) for values in case_values])
    rng = np.random.default_rng(seed)
    # Batch resamples to bound memory even for future, larger datasets.
    values = []
    for start in range(0, bootstrap_samples, 100):
        indices = rng.integers(0, len(case_values), size=(min(100, bootstrap_samples-start), len(case_values)))
        values.extend((correct[indices].sum(axis=1) / count[indices].sum(axis=1)).tolist())
    low, high = np.quantile(values, [.025, .975])
    return {
        'overall': overall,
        **{f'by_{key}': {name: compute_metrics(group) for name, group in sorted(mapping.items())}
           for key, mapping in groups.items()},
        'accuracy_bootstrap': {'low': float(low), 'high': float(high), 'confidence_level': .95,
                               'samples': bootstrap_samples, 'seed': seed, 'n_cases': len(cases),
                               'unit': 'case', 'method': 'percentile'},
    }

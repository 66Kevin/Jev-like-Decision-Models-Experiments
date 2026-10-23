import math
import unittest

from jev.evaluation.metrics import compute_metrics, evaluate_predictions


def record(p, q, i=0, ids=None, kind='choice', levels=None):
    return {'case_id': str(i), 'workflow': 'example', 'question_id': str(i),
            'type': kind, 'option_ids': ids or [str(j) for j in range(len(p))],
            'probabilities': p, 'gold': q, 'score_levels': levels}


class MetricsTests(unittest.TestCase):
    def test_known_values(self):
        rows = [record([.8,.2], [.7,.3]), record([.1,.9], [1.,0.], 1)]
        m = compute_metrics(rows)
        self.assertEqual(m['n_decisions'], 2)
        self.assertAlmostEqual(m['accuracy'], .5)
        self.assertAlmostEqual(m['hard_nll'], -(math.log(.8)+math.log(.1))/2)
        soft = -(.7*math.log(.8)+.3*math.log(.2)+math.log(.1))/2
        self.assertAlmostEqual(m['soft_ce'], soft)
        self.assertAlmostEqual(m['kl_q_p'], soft-(-.7*math.log(.7)-.3*math.log(.3))/2)
        self.assertAlmostEqual(m['brier_soft'], (.02+1.62)/2)
        self.assertAlmostEqual(m['brier_hard'], (.08+1.62)/2)
        self.assertAlmostEqual(m['ece_15'], .55)
        self.assertIsNone(m['score_mae'])

    def test_score_mae_uses_expectations(self):
        m = compute_metrics([record([.5,0,.5], [0,1,0], kind='score', levels=[0,1,2])])
        self.assertEqual(m['score_mae'], 0)
        m = compute_metrics([record([1,0,0], [0,.5,.5], kind='score', levels=[0,1,2])])
        self.assertEqual(m['score_mae'], 1.5)

    def test_ties_resolve_by_identity_and_confidence_one_is_binned(self):
        m = compute_metrics([record([.5,.5], [.5,.5], ids=['z','a']),
                             record([0,0,1], [0,0,1], 1)])
        self.assertEqual(m['accuracy'], 1)
        self.assertAlmostEqual(m['ece_15'], .25)
        self.assertTrue(math.isfinite(m['soft_ce']))

    def test_invalid_probabilities_and_empty_rejected(self):
        for p in [[-.1,1.1], [float('nan'), 1], [.2,.2], [1], [float('inf'),0]]:
            with self.assertRaises(ValueError):
                compute_metrics([record(p,[1,0])])
        with self.assertRaises(ValueError):
            compute_metrics([])

    def test_groups_and_case_bootstrap_deterministic(self):
        rows = [record([1,0],[1,0], 0), record([0,1],[1,0], 1)]
        rows[1]['workflow'] = 'other'
        m = evaluate_predictions(rows, bootstrap_samples=100, seed=42)
        self.assertEqual(m['by_workflow']['example']['accuracy'], 1)
        self.assertEqual(m['by_workflow']['other']['accuracy'], 0)
        self.assertIn('choice', m['by_type'])
        self.assertEqual(m['accuracy_bootstrap']['n_cases'], 2)
        self.assertEqual(m, evaluate_predictions(rows, bootstrap_samples=100, seed=42))

    def test_bootstrap_preserves_all_questions_of_a_case(self):
        rows = [record([1,0], [1,0], 0), record([0,1], [1,0], 1)]
        rows[1]['case_id'] = rows[0]['case_id']
        interval = evaluate_predictions(rows, bootstrap_samples=100)['accuracy_bootstrap']
        self.assertEqual(interval['n_cases'], 1)
        self.assertEqual(interval['low'], .5)
        self.assertEqual(interval['high'], .5)


if __name__ == '__main__':
    unittest.main()

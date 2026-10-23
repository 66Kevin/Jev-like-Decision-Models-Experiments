import copy
import json
from pathlib import Path
import tempfile
import unittest

from jev.data.typed_decisions import normalize_case, model_input, make_splits, prepare_splits


def raw_case(i=0, split='train', workflow='example'):
    questions = {
        'choose': {'type': 'choice', 'instructions': 'Choose.',
                   'criteria': {'zebra': 'Z', 'ant': 'A'}},
        'bool': {'type': 'noul', 'instructions': 'Is it valid?'},
        'score': {'type': 'score', 'instructions': 'Rate.', 'criteria': ['Low', 'High']},
        'other': {'type': 'choice', 'instructions': 'Again.', 'criteria': {'a': 'A', 'b': 'B'}},
        'last': {'type': 'noul', 'instructions': 'Final?'},
    }
    gold = {q: {'type': spec['type'], 'probabilities':
            ({'zebra': .5, 'ant': .5} if q == 'choose' else
             {'0': .25, '1': .75} if q == 'score' else
             {'a': .4, 'b': .599999} if q == 'other' else {'false': .6, 'true': .4})}
            for q, spec in questions.items()}
    return {'id': f'{workflow}-{split}-{i}', 'workflow': workflow, 'split': split,
            'state': json.dumps({'i': i, 'split': split, 'workflow': workflow}),
            'questions': json.dumps(questions), 'gold': json.dumps(gold),
            'n_questions': 5, 'factors': 'secret', 'label_agreement': 'secret'}


class DataTests(unittest.TestCase):
    def test_normalization_preserves_order_tie_identity_and_input_whitelist(self):
        case = normalize_case(raw_case())
        choice = case['decisions'][0]
        self.assertEqual(choice['option_ids'], ['zebra', 'ant'])
        self.assertEqual(choice['hard_target'], 1)
        self.assertEqual(case['decisions'][1]['option_ids'], ['false', 'true'])
        self.assertEqual(case['decisions'][2]['score_levels'], [0., 1.])
        self.assertAlmostEqual(sum(case['decisions'][3]['gold']), 1)
        inp = model_input(choice)
        self.assertEqual(set(inp), {'state', 'instructions', 'option_ids', 'option_texts', 'type'})
        for forbidden in ['gold', 'hard_target', 'factors', 'label_agreement']:
            self.assertNotIn(forbidden, inp)
        reverse = raw_case()
        q = json.loads(reverse['questions'])
        q['choose']['criteria'] = {'ant': 'A', 'zebra': 'Z'}
        reverse['questions'] = json.dumps(q)
        new = normalize_case(reverse)['decisions'][0]
        self.assertEqual(new['option_ids'][new['hard_target']], 'ant')

    def test_invalid_gold_rejected(self):
        for probs in [{'zebra': -.1, 'ant': 1.1}, {'zebra': 0, 'ant': 0},
                      {'zebra': .4, 'missing': .6}, {'zebra': .2, 'ant': .2}]:
            raw = raw_case(); gold = json.loads(raw['gold'])
            gold['choose']['probabilities'] = probs; raw['gold'] = json.dumps(gold)
            with self.assertRaises(ValueError):
                normalize_case(raw)

    def test_full_size_splits_reproducible_and_disjoint(self):
        cases = [normalize_case(raw_case(i, s)) for s, n in [('train', 300), ('test', 100)] for i in range(n)]
        splits = make_splits(cases)
        self.assertEqual({k: len(v) for k, v in splits.items()},
                         {'train': 240, 'dev': 30, 'calibration': 30, 'test': 100})
        self.assertEqual(splits, make_splits(list(reversed(cases))))
        ids = [c['case_id'] for group in splits.values() for c in group]
        self.assertEqual(len(ids), len(set(ids)))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            self.assertEqual(splits, prepare_splits(cases, p, source_manifest={'revision': 'fake'}))
            self.assertEqual(splits, prepare_splits(cases, p, source_manifest={'revision': 'fake'}))
            self.assertEqual(set(json.loads((p/'train.jsonl').read_text().splitlines()[0])), {'case_id'})
            self.assertIn('source_manifest', json.loads((p/'manifest.json').read_text()))
            (p/'dev.jsonl').write_text('{"case_id":"unknown"}\n')
            with self.assertRaises(ValueError):
                prepare_splits(cases, p, source_manifest={'revision': 'fake'})

    def test_bad_counts_duplicates_and_canonical_overlap_rejected(self):
        cases = [normalize_case(raw_case(i, s)) for s, n in [('train', 300), ('test', 100)] for i in range(n)]
        for bad in [cases[:-1], cases + [cases[0]]]:
            with self.assertRaises(ValueError):
                make_splits(bad)
        bad = copy.deepcopy(cases)
        bad[-1]['state'] = copy.deepcopy(bad[0]['state'])
        with self.assertRaises(ValueError):
            make_splits(bad)


if __name__ == '__main__':
    unittest.main()

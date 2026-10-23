"""Offline G2 review checks reject plausible, internally consistent tampering."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import os
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('g2_analysis', ROOT/'experiments/02_supervised/finish_g2.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from jev.evaluation.metrics import evaluate_predictions


class OfflineReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.expected = [dict(case_id='case1', workflow='invoice', question_id='approve', type='choice',
                              option_ids=['no', 'yes'], option_texts=['Reject', 'Approve'], gold=[.25, .75],
                              hard_target=1, score_levels=None),
                         dict(case_id='case1', workflow='invoice', question_id='risk', type='score',
                              option_ids=['0', '1', '2'], option_texts=['Low', 'Medium', 'High'],
                              gold=[.1, .7, .2], hard_target=1, score_levels=[0., 1., 2.])]
        self.rows = []
        for row, logits in zip(self.expected, [[0., 1.], [0., 2., -1.]]):
            values = np.asarray(logits, dtype=np.float32)
            p = np.exp(values-values.max()); p /= p.sum()
            self.rows.append(dict(row, label_logits=logits, probabilities=p.tolist(),
                                  candidate_letters=list('ABC'[:len(logits)]), input_tokens=10,
                                  case_latency_seconds=.2))
        self.write_fixture()

    def write_fixture(self):
        d = self.directory
        (d/'predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
        logits = np.zeros((2, 3), np.float32); valid = np.zeros((2, 3), bool)
        ids = np.full((2, 3), '', dtype='<U3')
        for i, row in enumerate(self.rows):
            n = len(row['option_ids']); logits[i, :n] = row['label_logits']; valid[i, :n] = True
            ids[i, :n] = row['option_ids']
        np.savez_compressed(d/'logits.npz', logits=logits, valid_mask=valid, option_ids=ids,
                            case_ids=np.array(['case1']*2), question_ids=np.array(['approve', 'risk']))
        (d/'metrics.json').write_text(json.dumps(evaluate_predictions(self.rows, bootstrap_samples=20, seed=42)))
        timing = dict(n_cases=1, n_decisions=2, total_seconds=.2, mean_case_seconds=.2,
                      p50_case_seconds=.2, p95_case_seconds=.2,
                      cases=[dict(case_id='case1', workflow='invoice', total_seconds=.2,
                                  inference_seconds=.15, tokenization_seconds=.04,
                                  input_lengths=[10,10], padded_tokens=20)])
        (d/'timing.json').write_text(json.dumps(timing))

    def audit(self):
        return module.audit_prediction_split(self.directory, self.expected, bootstrap_samples=20, seed=42)

    def test_valid_fixture_reconstructs_metrics(self):
        checked = self.audit()
        self.assertEqual(len(checked['rows']), 2)
        self.assertEqual(checked['metrics']['overall']['accuracy'], 1.)
        self.assertLess(checked['max_probability_error'], 2e-7)

    def test_incomplete_run_cannot_publish(self):
        (self.directory/'metadata.json').write_text(json.dumps(dict(status='failed', mode='train')))
        with self.assertRaisesRegex(ValueError, 'complete'):
            module.finish(self.directory)
        self.assertFalse((self.directory/'verification.json').exists())
        self.assertFalse((self.directory/'analysis.json').exists())

    def test_consistent_forged_gold_rejected_against_source(self):
        self.rows[0]['gold'] = [.75, .25]; self.rows[0]['hard_target'] = 0
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'gold|hard_target'):
            self.audit()

    def test_consistent_forged_identity_rejected_against_source(self):
        self.rows[0]['workflow'] = 'security'
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'workflow'):
            self.audit()

    def test_probabilities_must_follow_logits_even_with_matching_metrics(self):
        self.rows[0]['probabilities'] = [.1, .9]
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'probabilit'):
            self.audit()

    def test_npz_mask_and_padding_cannot_hide_logits(self):
        path = self.directory/'logits.npz'
        with np.load(path) as source:
            arrays = {key:source[key] for key in source.files}
        arrays['valid_mask'][0, 2] = True
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, 'mask'):
            self.audit()
        self.write_fixture()
        with np.load(path) as source:
            arrays = {key:source[key] for key in source.files}
        arrays['logits'][0, 2] = 5
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, 'padding'):
            self.audit()

    def test_duplicate_or_missing_question_rejected(self):
        self.rows[1] = copy.deepcopy(self.rows[0]); self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'identity|Duplicate'):
            self.audit()

    def test_nonfinite_training_history_rejected(self):
        rows = [dict(step=i, epoch=(i-1)//150, group=(i-1)%150+1,
                     loss=1., grad_norm=.1, learning_rate=(2e-5*((i-1)/23 if i<=23 else .5*(1+math.cos(math.pi*(i-24)/427)))), seconds=1., questions=32)
                for i in range(1, 451)]
        path = self.directory/'train.jsonl'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        self.assertEqual(module.audit_train_history(path)['optimizer_steps'], 450)
        rows[153]['loss'] = float('nan')
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        with self.assertRaisesRegex(ValueError, 'finite'):
            module.audit_train_history(path)


    def test_completed_run_reconstruction_and_tampered_summary_fail_closed(self):
        run = self.directory/'run'
        baseline = self.directory/'b0'
        g1 = self.directory/'g1'
        run.mkdir()
        expected = {}

        def source_rows(prefix, count):
            result = []
            for index in range(count):
                row = copy.deepcopy(self.expected[index % 2])
                row['case_id'] = f'{prefix}{index//5:04d}'
                row['question_id'] = f'q{index%5}'
                result.append(row)
            return result

        expected['dev'] = source_rows('dev', 600)
        expected['test'] = source_rows('test', 2000)

        def write_split(directory, source, seed, strength=1.):
            directory.mkdir(parents=True)
            rows = []
            for row in source:
                logits = np.zeros(len(row['option_ids']), np.float32)
                logits[row['hard_target']] = strength
                probabilities = np.exp(logits-logits.max()); probabilities /= probabilities.sum()
                rows.append(dict(row, label_logits=logits.tolist(), probabilities=probabilities.tolist(),
                                 candidate_letters=list('ABCDE'[:len(logits)]), input_tokens=10,
                                 case_latency_seconds=.2))
            (directory/'predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            width = max(len(row['option_ids']) for row in rows)
            logits = np.zeros((len(rows), width), np.float32)
            valid = np.zeros_like(logits, dtype=bool)
            identities = np.full(logits.shape, '', dtype='<U3')
            for index,row in enumerate(rows):
                count = len(row['option_ids'])
                logits[index,:count] = row['label_logits']; valid[index,:count] = True
                identities[index,:count] = row['option_ids']
            np.savez_compressed(directory/'logits.npz', logits=logits, valid_mask=valid,
                                option_ids=identities, case_ids=np.array([r['case_id'] for r in rows]),
                                question_ids=np.array([r['question_id'] for r in rows]))
            metrics = evaluate_predictions(rows, bootstrap_samples=5, seed=seed)
            cases = []
            for index in range(0,len(rows),5):
                cases.append(dict(case_id=rows[index]['case_id'], workflow='invoice', total_seconds=.2,
                                  tokenization_seconds=.04, inference_seconds=.15,
                                  input_lengths=[10]*5,padded_tokens=50))
            times = np.array([.2]*len(cases))
            timing = dict(n_cases=len(cases),n_decisions=len(rows),total_seconds=float(times.sum()),
                          mean_case_seconds=float(times.mean()),p50_case_seconds=.2,p95_case_seconds=.2,cases=cases)
            summary = dict(n_cases=len(cases),n_decisions=len(rows),metrics=metrics['overall'],
                           p50_case_seconds=.2,p95_case_seconds=.2,
                           memory=dict(peak_allocated_bytes=100,peak_reserved_bytes=120,current_allocated_bytes=80))
            for name,value in [('metrics',metrics),('timing',timing),('summary',summary)]:
                (directory/f'{name}.json').write_text(json.dumps(value))
            return rows,summary

        write_split(baseline/'dev', expected['dev'],42)
        baseline_rows,_ = write_split(baseline/'test',expected['test'],42)
        metadata = dict(status='complete',mode='train',completed_seeds={},test={},
                        model_files_sha256={},source_sha256={},started_at='2026-01-01T00:00:00Z',
                        updated_at='2026-01-01T01:00:00Z')
        config = dict(baseline_run=str(baseline),evaluation=dict(bootstrap_samples=5),
                      model=dict(path='/original/model'), dataset=dict(revision='fixture_revision'),
                      training=dict(microbatch_questions=4,gradient_accumulation=8))
        (run/'config.yaml').write_text(module.yaml.safe_dump(config))
        g1_rows = []
        candidate_rows = []
        histories = [dict(step=i,epoch=(i-1)//150,group=(i-1)%150+1,
                          loss=1.,grad_norm=.1,learning_rate=(2e-5*((i-1)/23 if i<=23 else .5*(1+math.cos(math.pi*(i-24)/427)))),seconds=1.,questions=32)
                     for i in range(1,451)]
        for seed in [42,43,44]:
            seed_dir = run/f'seed{seed}'
            seed_dir.mkdir()
            candidate_list = []
            for epoch in range(4):
                _,summary = write_split(seed_dir/f'dev_epoch{epoch}',expected['dev'],42 if epoch==0 else seed,epoch+1.)
                candidate_list.append(dict(epoch=epoch,step=epoch*150,hard_nll=summary['metrics']['hard_nll'],
                                           checkpoint=f'checkpoints/step{epoch*150:06d}' if epoch else 'original_reference.json',
                                           parameter_sha256=f'epoch{epoch}' if epoch else 'original'))
                if epoch:
                    (seed_dir/f'eval_integrity_epoch{epoch}.json').write_text(json.dumps(dict(
                        master_sha256_before=f'epoch{epoch}',master_sha256_after=f'epoch{epoch}',optimizer_parameter_identities_preserved=True)))
                else:
                    # The real runner copies B0 bytes for epoch zero.
                    import shutil
                    for name in ['predictions.jsonl','logits.npz','metrics.json','timing.json']:
                        shutil.copyfile(baseline/'dev'/name,seed_dir/'dev_epoch0'/name)
                    (seed_dir/'dev_epoch0/reused_from_b0.json').write_text(json.dumps(dict(
                        native_bf16_train_parity=True,baseline_run=str(baseline),
                        metrics_sha256=module.sha256_file(baseline/'dev/metrics.json'))))
                if epoch:
                    checkpoint = seed_dir/f'checkpoints/step{epoch*150:06d}'
                    checkpoint.mkdir(parents=True)
                    (checkpoint/'manifest.json').write_text(json.dumps(dict(parameter_sha256=f'epoch{epoch}',progress=dict(step=epoch*150))))
                os.utime(seed_dir/f'dev_epoch{epoch}/summary.json',ns=(1000000000,1000000000))
            selected = dict(candidate_list[3])
            metadata['completed_seeds'][str(seed)] = selected
            for name,value in [('selection',selected),('model',dict(trainable_parameters=752393024,frozen_parameters=0,
                    master_dtype='float32',autocast_dtype='bfloat16',head_and_embedding_tied=True,zero_step_train_parity=True,
                    original_parameter_sha256='original')),
                    ('training_summary',dict(optimizer_steps=450,original_parameter_sha256='original',
                    final_parameter_sha256='epoch3',dev_candidates=candidate_list)),
                    ('original_reference',dict(kind='original_model_reference',native_bf16_train_parity=True,
                     model_files_sha256={},parameter_sha256='original')),
                    ('latest',dict(step=450,parameter_sha256='epoch3',checkpoint='checkpoints/step000450')),
                    ('training_memory',dict(peak_allocated_bytes=100))]:
                (seed_dir/f'{name}.json').write_text(json.dumps(value))
            (seed_dir/'train.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in histories))
            rows,summary = write_split(seed_dir/'test',expected['test'],seed,4.)
            candidate_rows.append(rows); metadata['test'][str(seed)] = summary
            rows,_ = write_split(g1/f'seed{seed}/test',expected['test'],seed,2.)
            g1_rows.append(rows)
            for directory in [seed_dir,g1/f'seed{seed}']:
                (directory/'tokenization.json').write_text(json.dumps(dict(decisions=4800,max_tokens=666,cache_identity_sha256='fixture')))
            os.utime(seed_dir/'test/predictions.jsonl',ns=(3000000000,3000000000))
        (run/'selections.json').write_text(json.dumps(dict(locked=True,
             criterion='minimum dev hard NLL; earlier step breaks ties',seeds=metadata['completed_seeds'])))
        os.utime(run/'selections.json',ns=(2000000000,2000000000))
        (run/'metadata.json').write_text(json.dumps(metadata))
        summary = dict(per_seed=metadata['test'],aggregate=module.aggregate_metrics([metadata['test'][str(s)]['metrics'] for s in [42,43,44]]),
                       paired_accuracy_delta_vs_b0=module.paired_seed_bootstrap([baseline_rows],candidate_rows,5),
                       paired_accuracy_delta_vs_g1=module.paired_seed_bootstrap(g1_rows,candidate_rows,5))
        (run/'summary.json').write_text(json.dumps(summary))
        splits = {name:[dict(decisions=rows)] for name,rows in expected.items()}
        # File and full-checkpoint hashing are tested separately; fixture scope is the
        # complete saved-prediction/selection/history/aggregate publication workflow.
        with mock.patch.object(module,'audit_provenance',return_value=(splits,baseline,g1)), \
             mock.patch.object(module,'audit_checkpoint',return_value=dict(kind='full_fp32')):
            summary['aggregate']['accuracy']['mean'] = .5
            (run/'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,'Production aggregate'):
                module.finish(run)
            self.assertFalse((run/'verification.json').exists())
            summary['aggregate']['accuracy']['mean'] = 1.
            (run/'summary.json').write_text(json.dumps(summary))
            (self.directory/'reports').mkdir()
            original_readme = (ROOT/'README.md').read_text()
            (self.directory/'README.md').write_text(original_readme)
            with mock.patch.object(module,'ROOT',self.directory):
                checked = module.finish(run,update_docs=True)
            rendered = (self.directory/'reports/02_supervised_g2.md').read_text()
            self.assertIn('离线产物复核通过',rendered)
            self.assertIn('## 训练设置、成本与限制',rendered)
            self.assertIn('G2（三 seed）', (self.directory/'README.md').read_text())
            for stem in ['g2_dev_curves','g2_comparison']:
                for extension in ['png','svg']:
                    self.assertGreater((self.directory/f'reports/figures/{stem}.{extension}').stat().st_size,1000)
            self.assertEqual(checked['prediction_records_checked'],13200)
            self.assertEqual(checked['status'],'passed')
            self.assertTrue((run/'selected_models.json').is_file())
            self.assertEqual(module.read_json(run/'analysis.json')['g2']['accuracy']['mean'],1.)


if __name__ == '__main__':
    unittest.main()

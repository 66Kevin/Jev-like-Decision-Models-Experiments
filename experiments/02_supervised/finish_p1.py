#!/usr/bin/env python3
"""Offline CPU reconstruction of completed P1 evidence; never performs inference.

Usage: python experiments/02_supervised/finish_p1.py --run RUN
Run in the original alg26 Python environment to verify its runtime fingerprint.
Original predictions, metrics, summaries and checkpoints remain read-only inputs.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import math
from pathlib import Path
import sys
import numpy as np
import torch
import yaml
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from jev.data.typed_decisions import iter_decisions,load_cases,prepare_splits,sha256_file
from jev.evaluation.metrics import evaluate_predictions
from jev.training.pointer_lora import validate_joint_checkpoint
from jev.training.full_runner import paired_seed_bootstrap
from jev.training.runner import write_json,runtime_fingerprint

def import_entry(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'experiments/02_supervised'/filename)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

g2=import_entry('p1_g2_offline','finish_g2.py')
entry=import_entry('p1_original_entry','run_p1.py')
read_json=g2.read_json;read_rows=g2.read_rows;require=g2.require;equal_tree=g2.equal_tree
checked_path=g2.checked_path;check_hash=g2.check_hash;aggregate_metrics=g2.aggregate_metrics
IDENTITY_FIELDS=g2.IDENTITY_FIELDS
SEEDS=[42,43,44]
DEFAULT_RUN=ROOT/'outputs/02_supervised/p1/train_20261023_102132_592199'

def audit_prediction_split(directory, expected_decisions, bootstrap_samples=2000, seed=42):
    """Check raw identities, saved logits, probabilities, metrics and case timing."""
    directory = Path(directory)
    rows = read_rows(directory/'predictions.jsonl')
    require(len(rows) == len(expected_decisions) and bool(rows), f'{directory}: prediction identity count differs')
    seen = set()
    maximum_error = 0.
    with np.load(directory/'logits.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
        required = {'logits', 'valid_mask', 'option_ids', 'case_ids', 'question_ids'}
        require(set(arrays) == required, f'{directory}: NPZ arrays differ')
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


def audit_history(path):
    rows=read_rows(path)
    require(len(rows)==450 and [r['step'] for r in rows]==list(range(1,451)),'P1 must complete exactly 450 updates')
    for row in rows:
        step=row['step'];require(row['epoch']==(step-1)//150 and row['group']==(step-1)%150+1,'Training epoch/group differs')
        require(row['questions']==32,'P1 effective batch must be 32')
        for key in ('loss','grad_norm','seconds','learning_rate','lora_learning_rate','head_learning_rate'):
            require(math.isfinite(row[key]) and row[key]>=0,f'Training {key} must be finite and nonnegative')
        s=step-1;factor=s/23 if s<23 else .5*(1+math.cos(math.pi*(s-23)/427))
        for key,base in [('learning_rate',2e-4),('lora_learning_rate',2e-4),('head_learning_rate',.001)]:
            require(math.isclose(row[key],base*factor,rel_tol=1e-10,abs_tol=1e-15),f'{key} learning rate differs')
    return dict(optimizer_steps=450,questions_presented=14400,update_seconds=sum(r['seconds'] for r in rows),
                maximum_loss=max(r['loss'] for r in rows),maximum_grad_norm=max(r['grad_norm'] for r in rows),
                epoch_mean_loss=[float(np.mean([r['loss'] for r in rows if r['epoch']==e])) for e in range(3)])


def audit_selection(selected,candidates):
    require(len(candidates)==4 and [v['epoch'] for v in candidates]==[0,1,2,3], 'Four actual dev generations required')
    equal_tree(selected,min(candidates,key=lambda x:(x['hard_nll'],x['step'])),'Minimum dev NLL selection')


def audit_lock(run,metadata):
    locked=read_json(Path(run)/'selections.json')
    require(locked.get('locked') is True and set(locked.get('seeds',{}))==set(map(str,SEEDS)), 'All three seeds must be locked')
    equal_tree(locked['seeds'],metadata['completed_seeds'],'Completed selections')
    require(locked['criterion']=='minimum dev hard NLL; earlier step breaks ties','Selection criterion differs')
    return locked


def audit_joint(directory,selection,expected_lora=10822656,expected_head=262144,hidden=1024,projection=128):
    require(selection['checkpoint']==f'checkpoints/step{selection["step"]:06d}','Joint checkpoint path differs')
    path=checked_path(directory,selection['checkpoint']);manifest=validate_joint_checkpoint(path)
    for key in ('head_sha256','adapter_sha256'):require(manifest[key]==selection[key],f'Selected joint {key} differs')
    step=selection['step'];require(manifest['progress']['step']==step,'Joint checkpoint progress differs')
    expected_progress=dict(epoch=0,group=0,step=0) if step==0 else dict(epoch=step//150-1,group=150,step=step)
    equal_tree(manifest['progress'],expected_progress,'Joint epoch progress')
    cfg=read_json(path/'joint_config.json')
    for key,value in dict(hidden_size=hidden,projection_dim=projection,head_parameters=expected_head,lora_parameters=expected_lora,dtype='float32').items():
        require(cfg[key]==value,f'Joint {key} differs')
    count=0;tensor_counts={}
    for relative,expected in [('head.safetensors',expected_head),('adapter/adapter_model.safetensors',expected_lora)]:
        with safe_open(path/relative,framework='np') as weights:
            tensor_counts[relative]=len(list(weights.keys()))
            total=0
            for name in weights.keys():
                tensor=weights.get_slice(name);require(tensor.get_dtype()=='F32','Joint weights must be FP32')
                require(np.isfinite(weights.get_tensor(name)).all(),'Nonfinite joint weights')
                if step==0 and name.endswith('.lora_B.weight'):
                    require(np.count_nonzero(weights.get_tensor(name))==0,'Epoch zero LoRA B must be zero')
                total+=math.prod(tensor.get_shape())
            require(total==expected,'Joint stored parameter count differs');count+=total
    trainer=torch.load(path/'trainer.pt',map_location='cpu',weights_only=False)
    require(set(trainer)=={'optimizer','scheduler','rng','progress','adapter_sha256','head_sha256'},'Joint trainer keys differ')
    require(trainer['progress']==manifest['progress'],'Trainer progress differs')
    optimizer=trainer['optimizer'];groups=optimizer['param_groups']
    require(len(groups)==2 and len(groups[0]['params'])==tensor_counts['adapter/adapter_model.safetensors'] and len(groups[1]['params'])==2,
            'Joint optimizer must have LoRA and two Head tensor groups')
    ids=[p for group in groups for p in group['params']]
    require(len(set(ids))==len(ids),'Joint optimizer parameter aliases')
    factor=step/23 if step<23 else .5*(1+math.cos(math.pi*(step-23)/427))
    for group,rate in zip(groups,[2e-4,.001]):
        require(group['initial_lr']==rate and group['weight_decay']==.01 and list(group['betas'])==[.9,.999]
                and group['eps']==1e-8 and group['foreach'] is False,'Joint optimizer recipe differs')
        require(math.isclose(group['lr'],rate*factor,rel_tol=1e-10,abs_tol=1e-15),'Joint optimizer current learning rate differs')
    require(trainer['scheduler']['last_epoch']==step and trainer['scheduler']['base_lrs']==[2e-4,.001],'Joint scheduler progress/rates differ')
    require(isinstance(trainer['rng'],dict) and set(trainer['rng'])=={'python','numpy','torch','cuda'},'Joint RNG state missing')
    require(set(optimizer['state'])==(set(ids) if step else set()),'Joint optimizer state coverage differs')
    for state in optimizer['state'].values():
        require(float(state['step'])==step and state['exp_avg'].dtype==torch.float32
                and state['exp_avg_sq'].dtype==torch.float32 and torch.isfinite(state['exp_avg']).all().item()
                and torch.isfinite(state['exp_avg_sq']).all().item(),'Joint optimizer state differs')
    adapter=read_json(path/'adapter/adapter_config.json')
    if expected_lora==10822656:
        require(len(groups[0]['params'])==372 and adapter['r']==16 and adapter['lora_alpha']==32 and adapter['lora_dropout']==0
                and adapter['bias']=='none' and not adapter['use_dora'] and not adapter['use_rslora'],'LoRA recipe/targets differ')
    return dict(kind='pointer_head_and_unmerged_lora_fp32',path=str(path),stored_parameters=count,
                head_sha256=manifest['head_sha256'],adapter_sha256=manifest['adapter_sha256'],files_sha256=manifest['files_sha256'])


def audit_provenance(run,config,metadata):
    entry.validate_config(config);check_hash(run/'config.yaml',metadata['config_sha256'])
    require(config['dataset']==metadata['dataset'],'P1 dataset differs')
    require(len(metadata['source_sha256'])==34,'Expected original 34-file scientific source snapshot')
    for name,digest in metadata['source_sha256'].items():
        check_hash(checked_path(ROOT,name),digest);check_hash(checked_path(run/'source',name),digest)
    fingerprint=runtime_fingerprint(ROOT/config['training']['fla_package_dir'])
    equal_tree(fingerprint,{k:v for k,v in metadata['runtime_fingerprint'].items() if k!='device'},'P1 original runtime fingerprint')
    device=metadata['device'];equal_tree(device,metadata['runtime_fingerprint']['device'],'Device runtime metadata')
    require(device['hostname']=='alg26' and device['physical_gpu']==0 and 'RTX 4090' in device['nvidia_smi']
            and device['cuda_visible_devices']=='0' and device['cuda_device_order']=='PCI_BUS_ID','P1 alg26 GPU0 provenance differs')
    require(device['torch_gpu_uuid'].lower().removeprefix('gpu-')==device['nvidia_smi'].split(',')[0].strip().lower().removeprefix('gpu-'),'Saved GPU UUID differs')
    raw=ROOT/config['dataset']['raw_dir'];split=ROOT/config['dataset']['split_dir']
    check_hash(raw/'download_manifest.json',metadata['raw_manifest_sha256']);check_hash(split/'manifest.json',metadata['split_manifest_sha256'])
    for name in ('manifest.json','train.jsonl','dev.jsonl','calibration.jsonl','test.jsonl'):
        check_hash(run/'splits'/name,sha256_file(split/name))
    for name,digest in metadata['model_files_sha256'].items():check_hash(checked_path(config['model']['path'],name),digest)
    for path in [run/'preflight_train.json',ROOT/config['output_root']/'preflight_train.json']:check_hash(path,metadata['profile_sha256'])
    profile=read_json(run/'preflight_train.json');entry.validate_profile(profile,config)
    equal_tree(profile['runtime_fingerprint'],fingerprint,'Preflight runtime fingerprint')
    frozen=profile['frozen_backbone_before']
    require(frozen['unique_parameters']==752393024 and frozen['head_and_embedding_tied'] is True
            and frozen['parameter_dtypes']==['torch.bfloat16'] and frozen['buffer_dtypes']==['torch.float32'],'Frozen original dtype/tie differs')
    require(metadata['protocol']['hidden_feature_cache'] is False and metadata['protocol']['calibration_inference'] is False,'Forbidden hidden cache/calibration inference')
    require(not (run/'features').exists() and not any('calibration' in p.parts for p in run.rglob('predictions.jsonl')),'Unexpected feature cache/calibration results')
    references={}
    for name,field in [('b0','baseline_run'),('g1','g1_run'),('g2','g2_run'),('p0','p0_run')]:
        directory=ROOT/config[field];prior=read_json(directory/'metadata.json')
        require(prior['status']=='complete' and prior['dataset']==config['dataset'],'Reference data/status differs')
        require(prior['model_files_sha256']==metadata['model_files_sha256'] and prior['split_manifest_sha256']==metadata['split_manifest_sha256'], 'Reference original model/split differs')
        require(yaml.safe_load((directory/'config.yaml').read_text())['model']==config['model'],'Reference input/prompt policy differs')
        references[name]=directory
    cases=load_cases(raw);splits=prepare_splits(cases,split,seed=42,source_manifest=raw/'download_manifest.json')
    require({k:len(v) for k,v in splits.items()}==dict(train=960,dev=120,calibration=120,test=400),'Fixed split size differs')
    return splits,references,profile

def finish(run):
    run=Path(run).resolve();metadata=read_json(run/'metadata.json')
    require(metadata.get('status')=='complete' and metadata.get('mode')=='train','P1 run is not complete; refusing publication')
    require(set(metadata.get('completed_seeds',{}))==set(map(str,SEEDS)) and set(metadata.get('test',{}))==set(map(str,SEEDS)), 'All three P1 seeds must complete')
    print('Checking frozen Backbone, 34 source archives, runtime, raw/split files and profile...',flush=True)
    config=yaml.safe_load((run/'config.yaml').read_text());splits,references,profile=audit_provenance(run,config,metadata)
    samples=config['evaluation']['bootstrap_samples'];dev_expected=list(iter_decisions(splits['dev']));test_expected=list(iter_decisions(splits['test']))
    locked=audit_lock(run,metadata);lock_time=(run/'selections.json').stat().st_mtime_ns;lock_hash=sha256_file(run/'selections.json')
    print('Checking 20,000 reference test predictions (NPZ arrays decompressed once)...',flush=True)
    refs={name:[audit_prediction_split(directory/('test' if name=='b0' else f'seed{seed}/test'),test_expected,samples,seed)
                 for seed in ([42] if name=='b0' else SEEDS)] for name,directory in references.items()}
    tests=[];curves={};histories={};chosen={};hashes={};checkpoints={};checked_count=0
    maximum=max(v['max_probability_error'] for values in refs.values() for v in values);token_digests=[]
    frozen=profile['frozen_backbone_before']
    for seed in SEEDS:
        print(f'Checking P1 seed {seed}: four dev splits, joint checkpoints, actual initialization and test...',flush=True)
        s=str(seed);directory=run/f'seed{seed}';histories[s]=audit_history(directory/'train.jsonl');curves[s]=[];checkpoints[s]=[]
        training=read_json(directory/'training_summary.json');require(training['optimizer_steps']==450,'Incomplete joint training')
        model=read_json(directory/'model.json')
        for key,value in dict(lora_parameters=10822656,head_parameters=262144,trainable_parameters=11084800,
                              frozen_parameters=752393024,lora_trainable_tensors=372).items():require(model[key]==value,f'Model {key} differs')
        require(len(model['target_modules'])==186 and len(set(model['target_modules']))==186,'LoRA target modules differ')
        require(all(not value for value in model['loading_info'].values()),'Original model loading errors')
        for value in [model['frozen_backbone'],training['frozen_before'],training['frozen_after']]:equal_tree(value,frozen,'Frozen original Backbone')
        require(model['initial_head_sha256']==training['initial_head_sha256'] and model['initial_adapter_sha256']==training['initial_adapter_sha256'],'Initial joint generation differs')
        p0initial=read_json(references['p0']/f'seed{seed}/checkpoints/step000000/manifest.json')
        require(model['initial_head_sha256']==p0initial['head_sha256'],'P1 fresh random Head must match per-seed P0 initialization')
        token=read_json(directory/'tokenization.json')
        require(token['decisions']==4800 and token['max_tokens']==profile['max_train_tokens'] and token['hidden_features_cached'] is False
                and len(token['cache_identity_sha256'])==64,'Train token-only cache evidence differs')
        token_digests.append(token['cache_identity_sha256'])
        candidates=[]
        require(len(training['dev_candidates'])==4,'Four actual dev evaluations required')
        for epoch in range(4):
            candidate=training['dev_candidates'][epoch]
            require(candidate['epoch']==epoch and candidate['step']==150*epoch,'Dev candidate order/step differs')
            info=audit_joint(directory,candidate);checkpoints[s].append(info)
            target=directory/f'dev_epoch{epoch}';checked=audit_prediction_split(target,dev_expected,samples,seed)
            summary=read_json(target/'summary.json');started=read_json(target/'started.json')
            for value in (summary,started):
                for field in ('head_sha256','adapter_sha256'):require(value[field]==candidate[field],'Dev joint checkpoint generation differs')
                require(value['selections_sha256'] is None,'Dev must precede selections lock')
            require((target/'summary.json').stat().st_mtime_ns<=lock_time,'Dev completed after all-seed selections lock')
            equal_tree(candidate['hard_nll'],checked['metrics']['overall']['hard_nll'],'Candidate dev NLL')
            equal_tree(summary['memory'],read_json(target/'memory.json'),'Dev memory')
            if epoch==0:
                require(candidate['head_sha256']==training['initial_head_sha256'] and candidate['adapter_sha256']==training['initial_adapter_sha256'],'Random epoch-zero joint generation differs')
                require(summary['step_zero']=='Actual random Pointer Head + zero LoRA; no reused B0/P0 predictions','Actual random Head dev0 evidence missing')
                require(not (target/'reused_from_b0.json').exists(),'Random dev0 cannot reuse baseline')
            else:require(summary['step_zero'] is None,'Trained dev must not claim step zero')
            candidates.append(candidate)
            probabilities=[np.asarray(row['probabilities'],dtype=np.float64) for row in checked['rows']]
            curves[s].append(dict(epoch=epoch,**checked['metrics']['overall'],
                                 mean_max_probability=float(np.mean([p.max() for p in probabilities])),
                                 mean_prediction_entropy=float(np.mean([-np.sum(p[p>0]*np.log(p[p>0])) for p in probabilities]))))
            checked_count+=len(checked['rows']);maximum=max(maximum,checked['max_probability_error'])
        selected=read_json(directory/'selection.json');equal_tree(selected,locked['seeds'][s],'Locked per-seed selection');audit_selection(selected,candidates)
        chosen[s]={**selected,**checkpoints[s][selected['epoch']]}
        latest=read_json(directory/'latest.json');final=candidates[3]
        for field in ('step','checkpoint','head_sha256','adapter_sha256'):require(latest[field]==final[field],'Latest joint generation differs')
        require(final['head_sha256']!=training['initial_head_sha256'] and final['adapter_sha256']!=training['initial_adapter_sha256'],'Joint trainable weights did not change')
        reload=read_json(directory/'reload_check_train.json')
        require(reload['train_only'] is True and reload['native_reload_bitexact'] is True and reload['case_id']==splits['train'][0]['case_id'],'Train-only joint reload parity missing')
        for field in ('checkpoint','head_sha256','adapter_sha256'):require(reload[field]==final[field],'Final joint reload generation differs')
        training_memory=read_json(directory/'training_memory.json')
        require(training_memory['step']==450 and training_memory['peak_allocated_bytes']==training['training_peak_allocated_bytes']>0,'Training memory progress differs')
        target=directory/'test';started=read_json(target/'started.json');summary=read_json(target/'summary.json')
        for value in (started,summary):
            for field in ('head_sha256','adapter_sha256'):require(value[field]==selected[field],'Test joint selection differs')
            require(value['selections_sha256']==lock_hash,'Test all-seed lock differs')
        require((target/'started.json').stat().st_mtime_ns>=lock_time and (target/'predictions.jsonl').stat().st_mtime_ns>=lock_time,'Official test predates all-seed lock')
        checked=audit_prediction_split(target,test_expected,samples,seed);equal_tree(metadata['test'][s],summary,'Metadata test summary')
        equal_tree(summary['memory'],read_json(target/'memory.json'),'Test memory summary')
        require(all(isinstance(v,int) and v>0 for v in summary['memory'].values()),'Invalid test memory counters')
        tests.append(checked);checked_count+=len(checked['rows']);maximum=max(maximum,checked['max_probability_error'])
        for folder in [directory/f'dev_epoch{e}' for e in range(4)]+[target]:
            for path in folder.iterdir():
                if path.is_file():hashes[str(path.relative_to(run))]=sha256_file(path)
        for filename in ('train.jsonl','model.json','training_summary.json','training_memory.json','tokenization.json','selection.json','latest.json','reload_check_train.json'):
            hashes[str((directory/filename).relative_to(run))]=sha256_file(directory/filename)
    require(len(set(token_digests))==1,'Train token/position ordering must match across seeds')
    aggregate=aggregate_metrics([value['metrics']['overall'] for value in tests]);paired={}
    for name,values in refs.items():
        value=paired_seed_bootstrap([v['rows'] for v in values],[v['rows'] for v in tests],samples)
        value['definition']='Mean P1 seed correctness minus mean reference seed correctness; no probability ensemble; paired case percentile 95% CI'
        paired[f'paired_accuracy_delta_vs_{name}']=value
    original=read_json(run/'summary.json');equal_tree(original['aggregate'],aggregate,'Production aggregate');equal_tree(original['per_seed'],metadata['test'],'Production per seed')
    for name,value in paired.items():equal_tree(original[name],value,name)
    groups={group:{key:{name:(values[0]['metrics'][group][key] if name=='b0' else aggregate_metrics([v['metrics'][group][key] for v in values]))
                        for name,values in dict(**refs,p1=tests).items()}
                   for key in refs['b0'][0]['metrics'][group]} for group in ('by_workflow','by_type','by_question')}
    start=datetime.fromisoformat(metadata['started_at']);end=datetime.fromisoformat(metadata['updated_at'])
    costs={str(seed):dict(training=histories[str(seed)],test={k:v for k,v in checked['timing'].items() if k!='cases'},
                         test_memory=metadata['test'][str(seed)]['memory'],
                         training_peak_allocated_bytes=read_json(run/f'seed{seed}/training_memory.json')['peak_allocated_bytes']) for seed,checked in zip(SEEDS,tests)}
    reference_costs={name:{str(seed):reference_cost(directory,name,seed) for seed in ([42] if name=='b0' else SEEDS)} for name,directory in references.items()}
    analysis=dict(run=str(run),seeds=SEEDS,b0=refs['b0'][0]['metrics']['overall'],
                  **{name:aggregate_metrics([v['metrics']['overall'] for v in refs[name]]) for name in ('g1','g2','p0')},
                  p1=aggregate,**paired,**groups,dev_curves=curves,per_seed_costs=costs,reference_costs=reference_costs,
                  per_seed_test={str(s):v['metrics']['overall'] for s,v in zip(SEEDS,tests)},selected=locked['seeds'],
                  started_at=metadata['started_at'],completed_at=metadata['updated_at'],wall_seconds=(end-start).total_seconds(),
                  frozen_backbone=frozen,device=metadata['device'],
                  cost_interpretation='P1 measured on alg26 RTX 4090; B0/G1/G2/P0 on RTX 5090D v2; hardware/build differ, so no causal architecture speed comparison. B0 allocator peak covers entire dev/test run.',
                  error_changes={str(seed):{f'vs_{name}':g2.error_changes(values[0 if name=='b0' else i]['rows'],tests[i]['rows'])
                                           for name,values in refs.items()} for i,seed in enumerate(SEEDS)})
    require(checked_count==13200,'Unexpected P1 prediction count')
    for name in ('metadata.json','summary.json','config.yaml','preflight_train.json','selections.json'):hashes[name]=sha256_file(run/name)
    reference_hashes={str(path.relative_to(ROOT)):sha256_file(path) for name,directory in references.items()
                      for seed in ([42] if name=='b0' else SEEDS)
                      for path in (directory/('test' if name=='b0' else f'seed{seed}/test')).iterdir() if path.is_file()}
    verifier_files=('experiments/02_supervised/finish_p1.py','experiments/02_supervised/finish_g2.py','tests/test_p1_analysis.py')
    verification=dict(status='passed',verified_at=datetime.now(timezone.utc).isoformat(),
                      method='Offline CPU reconstruction; no model inference; original evidence unchanged; each NPZ array decompressed exactly once per split',
                      prediction_records_checked=checked_count,reference_test_records_checked=20000,source_files_checked=34,
                      maximum_probability_absolute_error=maximum,record_file_sha256=hashes,reference_file_sha256=reference_hashes,
                      verifier_source_sha256={name:sha256_file(ROOT/name) for name in verifier_files},checkpoint_generations=checkpoints,
                      checkpoint_integrity='All 12 joint epoch checkpoints: adapter/Head/trainer/config SHA256, FP32 weights and state, progress, optimizer/scheduler/RNG and same-step generation verified',
                      frozen_backbone='Original native BF16 parameters and FP32 buffers before/after equal in profile and all seeds; original model files verified',
                      selection_chronology='All dev summaries predate all-three-seed lock; official test started/prediction files follow lock (filesystem mtime evidence)',
                      tokenization_integrity='All three seeds report identical train-only tokens/positions cache digest, 4800 rows and original max length; no hidden features or calibration inference',
                      runtime_fingerprint=metadata['runtime_fingerprint'])
    registry=dict(original_model=config['model']['path'],model_files_sha256=metadata['model_files_sha256'],frozen_backbone=frozen,seeds=chosen,
                  loading_note='Load original BF16 Backbone retaining native FP32 buffers; attach selected unmerged FP32 LoRA and same-generation FP32 Pointer Head together. Native backend, B0 prompt/P0 pooling, temperature 1. These checkpoints contain adapter+Head, not a full model.')
    # No writes above: only publish derived evidence after every consistency check passes.
    create_figures(analysis,run)
    write_json(run/'analysis.json',analysis);write_json(run/'selected_joint_models.json',registry);write_json(run/'verification.json',verification)
    print(f'PASS: {checked_count} P1 and 20000 reference prediction records; all 12 joint generations',flush=True)
    return verification


def reference_cost(directory,name,seed):
    directory=Path(directory)
    if name=='b0':
        value=read_json(directory/'test/timing.json');value.pop('cases',None)
        value['memory']=read_json(directory/'metadata.json')['memory'];value['memory_scope']='whole baseline run; dev and test'
    else:
        value=read_json(directory/f'seed{seed}/test/summary.json');value['memory_scope']='test after warmup'
    value['hardware']='RTX 5090D v2 (prior experiment)'
    return value


def create_figures(analysis,run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=ROOT/'reports/figures';target.mkdir(parents=True,exist_ok=True)
    def save(fig,stem):
        fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(target/f'p1_{stem}.{ext}',dpi=160)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for seed in SEEDS:
        values=analysis['dev_curves'][str(seed)];chosen=analysis['selected'][str(seed)]['epoch']
        for axis,metric in zip(axes,('accuracy','hard_nll','ece_15')):
            series=[v[metric] for v in values];line,=axis.plot(range(4),series,marker='o',label=f'Seed {seed}')
            axis.scatter([chosen],[series[chosen]],marker='*',s=120,color=line.get_color());axis.set(title=f'Dev {metric}',xlabel='Epoch',xticks=range(4));axis.grid(alpha=.2)
            if metric=='hard_nll':axis.set_yscale('log')
    axes[0].legend();save(fig,'dev_curves')
    fig,axes=plt.subplots(1,3,figsize=(12,3.6));names=['b0','g1','g2','p0','p1'];colors=['#8899aa','#4a82bd','#3c9f79','#d98742','#9865bd']
    for axis,metric in zip(axes,('accuracy','hard_nll','ece_15')):
        means=[analysis['b0'][metric]]+[analysis[g][metric]['mean'] for g in names[1:]]
        errors=[0]+[analysis[g][metric]['sample_std'] for g in names[1:]]
        axis.bar([n.upper() for n in names],means,yerr=errors,capsize=4,color=colors);axis.set_title(f'Test {metric}');axis.grid(axis='y',alpha=.2)
    save(fig,'comparison')
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for seed in SEEDS:
        rows=read_rows(run/f'seed{seed}/train.jsonl');steps=[r['step'] for r in rows]
        smooth=np.convolve([r['loss'] for r in rows],np.ones(15)/15,mode='valid')
        axes[0].plot(steps[14:],smooth,label=f'Seed {seed}');axes[1].plot(steps,[r['grad_norm'] for r in rows],label=f'Seed {seed}',alpha=.7)
    rows=read_rows(run/'seed42/train.jsonl')
    axes[2].plot([r['step'] for r in rows],[r['lora_learning_rate'] for r in rows],label='LoRA')
    axes[2].plot([r['step'] for r in rows],[r['head_learning_rate'] for r in rows],label='Head')
    axes[0].set(title='Train hard CE (15-update mean)',xlabel='Update');axes[1].set(title='Pre-clip gradient norm',xlabel='Update',yscale='log')
    axes[2].set(title='Separate learning rates',xlabel='Update');axes[0].legend();axes[2].legend();save(fig,'training_curves')
    fig,axes=plt.subplots(2,2,figsize=(11,6),gridspec_kw={'width_ratios':[4,1.4]})
    for col,methods in enumerate([names[:-1],['p1']]):
        latency=[];memory=[]
        for name in methods:
            values=([read_json(run/f'seed{s}/test/summary.json') for s in SEEDS] if name=='p1'
                    else list(analysis['reference_costs'][name].values()))
            latency.append(float(np.mean([v['p50_case_seconds'] for v in values])))
            memory.append(float(np.mean([v['memory']['peak_allocated_bytes']/2**30 for v in values])))
        axes[0,col].bar([n.upper() for n in methods],latency,color=[colors[names.index(n)] for n in methods])
        axes[1,col].bar([n.upper() for n in methods],memory,color=[colors[names.index(n)] for n in methods])
        axes[0,col].set_title('RTX 5090D v2 (prior runs)' if col==0 else 'RTX 4090 (alg26)')
        axes[0,col].set_ylabel('Uncached test p50 seconds / case');axes[1,col].set_ylabel('Saved allocator peak GiB')
    axes[1,0].set_xlabel('B0 peak: whole run; other methods: test after warmup');axes[1,1].set_xlabel('Test after warmup')
    fig.suptitle('Descriptive costs across different hardware/builds; no architecture speed claim')
    save(fig,'costs')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    args=parser.parse_args();result=finish(args.run)
    print({key:result[key] for key in ('status','prediction_records_checked','reference_test_records_checked','source_files_checked')})

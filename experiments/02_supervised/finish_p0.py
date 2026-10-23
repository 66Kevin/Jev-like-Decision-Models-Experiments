#!/usr/bin/env python3
"""Reconstruct a completed P0 run offline, preserving all original evidence.

Usage: python experiments/02_supervised/finish_p0.py --run RUN
No model inference, parameter changes, or document edits are performed.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import yaml
from safetensors import safe_open
from jev.data.typed_decisions import iter_decisions,load_cases,prepare_splits,sha256_file
from jev.evaluation.metrics import evaluate_predictions
from jev.models.pointer import PointerHead
from jev.training.pointer_runner import (build_cache_key,load_feature_cache,load_head_weights,
                                         validate_config,validate_head_checkpoint,runtime_fingerprint)
from jev.training.full_runner import paired_seed_bootstrap
from jev.training.runner import write_json
spec=importlib.util.spec_from_file_location('p0_g2_offline',ROOT/'experiments/02_supervised/finish_g2.py')
g2=importlib.util.module_from_spec(spec);spec.loader.exec_module(g2)
read_json=g2.read_json;read_rows=g2.read_rows;require=g2.require;equal_tree=g2.equal_tree
checked_path=g2.checked_path;check_hash=g2.check_hash;aggregate_metrics=g2.aggregate_metrics
SEEDS=[42,43,44]
DEFAULT_RUN=ROOT/'outputs/02_supervised/p0/train_20261022_232846_262432'


def audit_cached_dev(directory,expected,samples,seed,epoch,head_hash,feature_hash):
    """Validate cached Head scores against source, never inventing full-model timing."""
    directory=Path(directory);rows=read_rows(directory/'predictions.jsonl')
    require(len(rows)==len(expected) and rows,'Cached dev prediction count differs')
    require(not (directory/'timing.json').exists(),'Cached dev must not claim full-model timing')
    width=max(len(row['option_ids']) for row in expected);maximum=0.;seen=set()
    with np.load(directory/'logits.npz',allow_pickle=False) as arrays:
        require(set(arrays.files)=={'logits','valid_mask','option_ids','case_ids','question_ids'},'Cached NPZ keys differ')
        shape=(len(rows),width)
        require(arrays['logits'].shape==shape and arrays['logits'].dtype==np.float32,'Cached NPZ logits shape/dtype differs')
        require(arrays['valid_mask'].shape==shape and arrays['valid_mask'].dtype==np.bool_,'Cached NPZ mask shape/dtype differs')
        require(arrays['option_ids'].shape==shape and arrays['case_ids'].shape==(len(rows),)
                and arrays['question_ids'].shape==(len(rows),),'Cached NPZ identity shapes differ')
        for i,(row,source) in enumerate(zip(rows,expected)):
            key=(row['case_id'],row['question_id']);require(key not in seen,'Duplicate cached identity');seen.add(key)
            for field in g2.IDENTITY_FIELDS:require(row.get(field)==source.get(field),f'Cached source {field} differs at {key}')
            count=len(source['option_ids']);require(row['candidate_letters']==list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:count]),'Candidate letters differ')
            require(arrays['case_ids'][i]==key[0] and arrays['question_ids'][i]==key[1],'Cached NPZ identity differs')
            require(np.array_equal(arrays['valid_mask'][i],np.arange(width)<count),'Cached NPZ mask differs')
            require(list(arrays['option_ids'][i,:count])==source['option_ids'],'Cached NPZ candidate IDs differ')
            require(np.all(arrays['option_ids'][i,count:]=='') and np.all(arrays['logits'][i,count:]==0),'Cached NPZ padding differs')
            logits=np.asarray(row['label_logits'],dtype=np.float32)
            require(logits.shape==(count,) and np.isfinite(logits).all(),'Invalid cached logits')
            require(np.array_equal(arrays['logits'][i,:count],logits),'Cached NPZ/JSON logits differ')
            probs=np.asarray(row['probabilities'],dtype=np.float64)
            require(probs.shape==(count,) and np.isfinite(probs).all() and np.all((probs>=0)&(probs<=1))
                    and abs(probs.sum()-1)<=1e-6,'Invalid cached probabilities')
            rebuilt=np.exp(logits.astype(np.float64)-float(logits.max()));rebuilt/=rebuilt.sum()
            error=float(np.max(np.abs(rebuilt-probs)));maximum=max(maximum,error)
            require(error<=2e-7,'Cached probabilities disagree with logits')
    metrics=evaluate_predictions(rows,bootstrap_samples=samples,seed=seed)
    equal_tree(read_json(directory/'metrics.json'),metrics,'Cached dev metrics')
    summary=read_json(directory/'summary.json');equal_tree(summary['metrics'],metrics['overall'],'Cached dev summary metrics')
    require(summary['n_decisions']==len(rows) and summary['n_cases']==len({r['case_id'] for r in rows}),'Cached summary count differs')
    for value in (summary,read_json(directory/'started.json')):
        require(value['head_sha256']==head_hash,'Cached dev Head generation differs')
        require(value['feature_manifest_sha256']==feature_hash,'Cached dev feature generation differs')
    require(summary['mode']=='cached frozen features; five independent question rows per case; no latency claim','Cached dev mode differs')
    require(summary['step_zero']==('random Pointer Head; not reused from B0' if epoch==0 else None),
            'Step zero must be actual random Pointer Head scoring')
    require(not (directory/'reused_from_b0.json').exists(),'Random Pointer Head dev cannot reuse B0')
    return dict(rows=rows,metrics=metrics,max_probability_error=maximum)


def audit_history(path):
    rows=read_rows(path)
    require(len(rows)==450 and [r['step'] for r in rows]==list(range(1,451)),'P0 must complete exactly 450 updates')
    for row in rows:
        step=row['step'];require(row['epoch']==(step-1)//150 and row['group']==(step-1)%150+1,'Training epoch/group differs')
        require(row['questions']==32,'P0 effective batch must be 32')
        for key in ('loss','grad_norm','seconds','learning_rate'):
            require(math.isfinite(row[key]) and row[key]>=0,f'Training {key} must be finite and nonnegative')
        s=step-1;factor=s/23 if s<23 else .5*(1+math.cos(math.pi*(s-23)/427))
        require(math.isclose(row['learning_rate'],.001*factor,rel_tol=1e-10,abs_tol=1e-15),'Pointer learning rate differs')
    return dict(optimizer_steps=450,questions_presented=14400,update_seconds=sum(r['seconds'] for r in rows),
                maximum_loss=max(r['loss'] for r in rows),maximum_grad_norm=max(r['grad_norm'] for r in rows),
                epoch_mean_loss=[float(np.mean([r['loss'] for r in rows if r['epoch']==e])) for e in range(3)])


def audit_head(directory,selection):
    require(selection['checkpoint']==f'checkpoints/step{selection["step"]:06d}','Head checkpoint path differs')
    path=checked_path(directory,selection['checkpoint']);manifest=validate_head_checkpoint(path)
    require(manifest['head_sha256']==selection['head_sha256'] and manifest['progress']['step']==selection['step'],
            'Selected Head checkpoint hash/progress differs')
    config=read_json(path/'config.json')
    require(config==dict(hidden_size=1024,projection_dim=128,parameters=262144,dtype='float32'),'Head architecture differs')
    with safe_open(path/'head.safetensors',framework='np') as weights:
        require(set(weights.keys())=={'query_projection.weight','candidate_projection.weight'},'Head keys include unexpected weights')
        count=0
        for name in weights.keys():
            tensor=weights.get_slice(name)
            require(tensor.get_dtype()=='F32' and tensor.get_shape()==[128,1024],'Head shape/dtype differs')
            require(np.isfinite(weights.get_tensor(name)).all(),'Nonfinite Head checkpoint')
            count+=math.prod(tensor.get_shape())
    head=PointerHead();load_head_weights(head,path)  # Hash-only CPU loading; no forward or model inference.
    trainer=torch.load(path/'trainer.pt',map_location='cpu',weights_only=False)
    require(trainer['progress']==manifest['progress'],'Head trainer progress differs')
    require(set(trainer)=={'optimizer','scheduler','rng','progress'},'Head trainer state keys differ')
    opt=trainer['optimizer'];require(len(opt['param_groups'])==1 and len(opt['param_groups'][0]['params'])==2,'Optimizer must contain exactly two Head tensors')
    require(trainer['scheduler']['last_epoch']==selection['step'],'Head scheduler progress differs')
    return dict(kind='pointer_head_fp32',path=str(path),head_sha256=manifest['head_sha256'],
                files_sha256=manifest['files_sha256'],stored_parameters=count)


def audit_provenance(run,config,metadata):
    validate_config(config);check_hash(run/'config.yaml',metadata['config_sha256'])
    require(config['dataset']==metadata['dataset'],'P0 dataset differs')
    require(len(metadata['source_sha256'])==30,'Expected original 30-file scientific source snapshot')
    for name,digest in metadata['source_sha256'].items():
        check_hash(checked_path(ROOT,name),digest);check_hash(checked_path(run/'source',name),digest)
    equal_tree(runtime_fingerprint(),metadata['runtime_fingerprint'],'P0 runtime fingerprint')
    require(metadata['cuda_visible_devices']=='0','P0 physical GPU provenance differs')
    raw=ROOT/config['dataset']['raw_dir'];split=ROOT/config['dataset']['split_dir']
    check_hash(raw/'download_manifest.json',metadata['raw_manifest_sha256']);check_hash(split/'manifest.json',metadata['split_manifest_sha256'])
    for name in ('manifest.json','train.jsonl','dev.jsonl','calibration.jsonl','test.jsonl'):
        check_hash(run/'splits'/name,sha256_file(split/name))
    for name,digest in metadata['model_files_sha256'].items():check_hash(checked_path(config['model']['path'],name),digest)
    profile=run/'preflight_train.json';check_hash(profile,metadata['profile_sha256'])
    check_hash(ROOT/config['output_root']/'preflight_train.json',metadata['profile_sha256'])
    p=read_json(profile)
    require(p['status']=='passed' and p['train_only'] is True and p['physical_gpu']==0,'Train-only GPU0 preflight missing')
    require(p['backbone_sha256_before']==p['backbone_sha256_after'],'Preflight frozen parameters changed')
    before=metadata['frozen_backbone'];equal_tree(metadata['frozen_backbone_after'],before,'Frozen Backbone before/after')
    require(before['unique_parameters']==752393024 and before['head_and_embedding_tied'] is True
            and before['parameter_dtypes']==['torch.bfloat16'] and before['buffer_dtypes']==['torch.float32'],'Original frozen dtype/tie policy differs')
    require(metadata['native_backend']['delta_layers']==18 and metadata['native_backend']['backend']=='native','P0 native backend differs')
    references={}
    for name,field in [('b0','baseline_run'),('g1','g1_run'),('g2','g2_run')]:
        directory=ROOT/config[field];prior=read_json(directory/'metadata.json')
        require(prior['status']=='complete' and prior['dataset']==config['dataset'],'Reference data/status differs')
        require(prior['model_files_sha256']==metadata['model_files_sha256'] and prior['split_manifest_sha256']==metadata['split_manifest_sha256'],
                'Reference original model/split differs')
        require(yaml.safe_load((directory/'config.yaml').read_text())['model']==config['model'],'Reference input/prompt policy differs')
        references[name]=directory
    cases=load_cases(raw);splits=prepare_splits(cases,split,seed=42,source_manifest=raw/'download_manifest.json')
    require({k:len(v) for k,v in splits.items()}==dict(train=960,dev=120,calibration=120,test=400),'Fixed split size differs')
    key=build_cache_key(config,metadata['source_sha256'],metadata['model_files_sha256'],metadata['split_manifest_sha256'])
    require({p.name for p in (run/'features').iterdir()}=={'train','dev'},'Only train/dev feature caches permitted')
    caches={}
    for name in ('train','dev'):
        cache=load_feature_cache(run/'features'/name,key,splits[name])
        require(cache['manifest_sha256']==metadata['feature_caches'][name],'Feature manifest metadata differs')
        caches[name]={k:cache[k] for k in ('manifest_sha256','manifest')}
        del cache
    require(not any('calibration' in path.parts for path in run.rglob('predictions.jsonl')),'Unexpected calibration inference')
    return splits,references,caches


def finish(run):
    run=Path(run).resolve();metadata=read_json(run/'metadata.json')
    require(metadata.get('status')=='complete' and metadata.get('mode')=='train','P0 run is not complete; refusing publication')
    require(set(metadata.get('completed_seeds',{}))==set(map(str,SEEDS)) and set(metadata.get('test',{}))==set(map(str,SEEDS)),
            'All three P0 seeds must complete')
    print('Checking frozen model, source archives, splits and feature caches...',flush=True)
    config=yaml.safe_load((run/'config.yaml').read_text());splits,references,caches=audit_provenance(run,config,metadata)
    samples=config['evaluation']['bootstrap_samples'];dev_expected=list(iter_decisions(splits['dev']));test_expected=list(iter_decisions(splits['test']))
    locked=read_json(run/'selections.json');require(locked.get('locked') is True and locked['seeds']==metadata['completed_seeds'],'Selections differ or not all locked')
    require(locked['criterion']=='minimum dev hard NLL; earlier step breaks ties','Selection criterion differs')
    lock_time=(run/'selections.json').stat().st_mtime_ns;lock_hash=sha256_file(run/'selections.json')
    print('Checking 14,000 reference test predictions...',flush=True)
    refs={name:[g2.audit_prediction_split(directory/('test' if name=='b0' else f'seed{seed}/test'),test_expected,samples,seed)
                 for seed in ([42] if name=='b0' else SEEDS)] for name,directory in references.items()}
    tests=[];curves={};histories={};chosen={};hashes={};checked_count=0;maximum=0.
    for seed in SEEDS:
        print(f'Checking P0 seed {seed}: four dev splits, test and Head generations...',flush=True)
        s=str(seed);directory=run/f'seed{seed}';histories[s]=audit_history(directory/'train.jsonl');curves[s]=[]
        training=read_json(directory/'training_summary.json');require(training['optimizer_steps']==450,'Incomplete Head training')
        require(training['initial_head_sha256']!=training['final_head_sha256'],'Head did not change')
        candidates=[]
        for epoch in range(4):
            candidate=training['dev_candidates'][epoch]
            require(candidate['epoch']==epoch and candidate['step']==150*epoch,'Dev candidate order/step differs')
            info=audit_head(directory,candidate)
            checked=audit_cached_dev(directory/f'dev_epoch{epoch}',dev_expected,samples,seed,epoch,candidate['head_sha256'],caches['dev']['manifest_sha256'])
            equal_tree(candidate['hard_nll'],checked['metrics']['overall']['hard_nll'],'Candidate dev NLL')
            require((directory/f'dev_epoch{epoch}/summary.json').stat().st_mtime_ns<=lock_time,'Dev completed after selections lock')
            if epoch==0:require(candidate['head_sha256']==training['initial_head_sha256'],'Random epoch-zero Head differs')
            if epoch==3:require(candidate['head_sha256']==training['final_head_sha256'],'Final Head differs')
            candidates.append(candidate);curves[s].append(dict(epoch=epoch,**checked['metrics']['overall']))
            checked_count+=len(checked['rows']);maximum=max(maximum,checked['max_probability_error'])
        selected=read_json(directory/'selection.json');equal_tree(selected,locked['seeds'][s],'Locked per-seed selection')
        equal_tree(selected,min(candidates,key=lambda x:(x['hard_nll'],x['step'])),'Minimum dev NLL selection')
        chosen[s]={**selected,**audit_head(directory,selected)}
        latest=read_json(directory/'latest.json')
        require(latest['step']==450 and latest['checkpoint']=='checkpoints/step000450' and latest['head_sha256']==training['final_head_sha256'],'Latest committed Head differs')
        probe=read_json(directory/'train_cached_full_probe.json')
        require(probe['train_only'] is True and probe['cached_full_scores_bitexact'] is True and probe['case_id']==splits['train'][0]['case_id']
                and probe['head_sha256']==selected['head_sha256'],'Train-only cached/full selected Head parity differs')
        target=directory/'test';started=read_json(target/'started.json');summary=read_json(target/'summary.json')
        for value in (started,summary):
            require(value['head_sha256']==selected['head_sha256'] and value['selections_sha256']==lock_hash,'Official test selected Head/lock differs')
        require((target/'started.json').stat().st_mtime_ns>=lock_time and (target/'predictions.jsonl').stat().st_mtime_ns>=lock_time,'Official test predates selections lock')
        checked=g2.audit_prediction_split(target,test_expected,samples,seed);equal_tree(metadata['test'][s],summary,'Metadata test summary')
        equal_tree(summary['memory'],read_json(target/'memory.json'),'Test memory summary')
        require(all(isinstance(v,int) and v>0 for v in summary['memory'].values()),'Invalid test memory counters')
        tests.append(checked);checked_count+=len(checked['rows']);maximum=max(maximum,checked['max_probability_error'])
        for folder in [directory/f'dev_epoch{e}' for e in range(4)]+[target]:
            for path in folder.iterdir():
                if path.is_file():hashes[str(path.relative_to(run))]=sha256_file(path)
    aggregate=aggregate_metrics([value['metrics']['overall'] for value in tests]);paired={}
    for name,values in refs.items():
        value=paired_seed_bootstrap([v['rows'] for v in values],[v['rows'] for v in tests],samples)
        value['definition']='Mean P0 seed correctness minus mean reference seed correctness; no probability ensemble; percentile 95% CI'
        paired[f'paired_accuracy_delta_vs_{name}']=value
    original=read_json(run/'summary.json');equal_tree(original['aggregate'],aggregate,'Production aggregate');equal_tree(original['per_seed'],metadata['test'],'Production per seed')
    for name,value in paired.items():equal_tree(original[name],value,name)
    groups={group:{key:{name:(values[0]['metrics'][group][key] if name=='b0' else aggregate_metrics([v['metrics'][group][key] for v in values]))
                        for name,values in dict(**refs,p0=tests).items()}
                   for key in refs['b0'][0]['metrics'][group]} for group in ('by_workflow','by_type','by_question')}
    start=datetime.fromisoformat(metadata['started_at']);end=datetime.fromisoformat(metadata['updated_at'])
    costs={str(seed):dict(training=histories[str(seed)],test={k:v for k,v in checked['timing'].items() if k!='cases'},
                         test_memory=metadata['test'][str(seed)]['memory']) for seed,checked in zip(SEEDS,tests)}
    reference_costs={name:{str(seed):{k:v for k,v in value['timing'].items() if k!='cases'}
                          for seed,value in zip([42] if name=='b0' else SEEDS,values)} for name,values in refs.items()}
    analysis=dict(run=str(run),seeds=SEEDS,b0=refs['b0'][0]['metrics']['overall'],
                  g1=aggregate_metrics([v['metrics']['overall'] for v in refs['g1']]),g2=aggregate_metrics([v['metrics']['overall'] for v in refs['g2']]),
                  p0=aggregate,**paired,**groups,dev_curves=curves,per_seed_costs=costs,reference_costs=reference_costs,
                  per_seed_test={str(s):v['metrics']['overall'] for s,v in zip(SEEDS,tests)},selected=locked['seeds'],
                  started_at=metadata['started_at'],completed_at=metadata['updated_at'],wall_seconds=(end-start).total_seconds(),
                  feature_caches=caches,frozen_backbone=metadata['frozen_backbone'],
                  error_changes={str(seed):{f'vs_{name}':g2.error_changes(values[0 if name=='b0' else i]['rows'],tests[i]['rows'])
                                           for name,values in refs.items()} for i,seed in enumerate(SEEDS)})
    require(checked_count==13200,'Unexpected P0 prediction count')
    verification=dict(status='passed',verified_at=datetime.now(timezone.utc).isoformat(),
                      method='Offline CPU reconstruction; no model inference; original evidence unchanged',
                      prediction_records_checked=checked_count,reference_test_records_checked=14000,source_files_checked=30,
                      maximum_probability_absolute_error=maximum,record_file_sha256=hashes,
                      verifier_source_sha256={name:sha256_file(ROOT/name) for name in ('experiments/02_supervised/finish_p0.py','experiments/02_supervised/finish_g2.py','tests/test_p0_analysis.py')},
                      checkpoint_integrity='All 12 retained Head epoch generations including trainer checksums, FP32 shapes, parameter hashes and progress verified',
                      frozen_backbone='Original parameters and buffers before/after hashes equal; original model files verified',
                      selection_chronology='All dev summaries predate selections lock; test started/prediction files follow lock (filesystem mtime evidence)',
                      feature_cache_integrity='Train/dev file SHA256, identities, shapes, dtypes, candidate positions/masks and targets checked using strict original loader')
    registry=dict(original_model=config['model']['path'],model_files_sha256=metadata['model_files_sha256'],
                  frozen_backbone=metadata['frozen_backbone'],seeds=chosen,
                  loading_note='Head checkpoints contain only two FP32 Pointer projections. Load the original frozen BF16 Backbone with its native FP32 buffers separately; B0 prompt and P0 candidate/query pooling; native backend; temperature 1.')
    # Publish only after every check has passed; original outputs are immutable inputs.
    write_json(run/'analysis.json',analysis);write_json(run/'selected_heads.json',registry);write_json(run/'verification.json',verification)
    create_figures(analysis,run)
    print(f'PASS: {checked_count} P0 and 14000 reference prediction records',flush=True)
    return verification


def reference_cost(directory,name,seed):
    directory=Path(directory)
    if name=='b0':
        value=read_json(directory/'test/timing.json')
        value['memory']=read_json(directory/'metadata.json')['memory']
        value['memory_scope']='whole baseline run; dev and test'
    else:
        value=read_json(directory/f'seed{seed}/test/summary.json')
        value['memory_scope']='test after warmup'
    return value


def create_figures(analysis,run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=ROOT/'reports/figures';target.mkdir(parents=True,exist_ok=True)
    def save(fig,stem):
        fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(target/f'p0_{stem}.{ext}',dpi=160)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for seed in SEEDS:
        values=analysis['dev_curves'][str(seed)];chosen=analysis['selected'][str(seed)]['epoch']
        for axis,metric in zip(axes,('accuracy','hard_nll','soft_ce')):
            series=[v[metric] for v in values];line,=axis.plot(range(4),series,marker='o',label=f'Seed {seed}')
            axis.scatter([chosen],[series[chosen]],marker='*',s=120,color=line.get_color());axis.set(title=f'Dev {metric}',xlabel='Epoch',xticks=range(4));axis.grid(alpha=.2)
    axes[0].legend();save(fig,'dev_curves')
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for axis,metric in zip(axes,('accuracy','hard_nll','ece_15')):
        means=[analysis['b0'][metric]]+[analysis[g][metric]['mean'] for g in ('g1','g2','p0')]
        errors=[0]+[analysis[g][metric]['sample_std'] for g in ('g1','g2','p0')]
        axis.bar(['B0','G1','G2','P0'],means,yerr=errors,capsize=4,color=['#8899aa','#4a82bd','#3c9f79','#d98742']);axis.set_title(f'Test {metric}');axis.grid(axis='y',alpha=.2)
    save(fig,'comparison')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for seed in SEEDS:
        rows=read_rows(run/f'seed{seed}/train.jsonl');steps=[r['step'] for r in rows]
        smooth=np.convolve([r['loss'] for r in rows],np.ones(15)/15,mode='valid')
        axes[0].plot(steps[14:],smooth,label=f'Seed {seed}');axes[1].plot(steps,[r['grad_norm'] for r in rows],label=f'Seed {seed}',alpha=.7)
    axes[0].set(title='Train hard CE (15-update mean)',xlabel='Update');axes[1].set(title='Pre-clip gradient norm',xlabel='Update',yscale='log')
    axes[0].legend();save(fig,'training_curves')
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    names=['b0','g1','g2','p0'];latencies=[];memory=[]
    for name in names:
        if name=='p0':values=[read_json(run/f'seed{s}/test/summary.json') for s in SEEDS]
        else:
            config=yaml.safe_load((run/'config.yaml').read_text());field='baseline_run' if name=='b0' else f'{name}_run';directory=ROOT/config[field]
            values=[reference_cost(directory,name,s) for s in ([42] if name=='b0' else SEEDS)]
        latencies.append(float(np.mean([v['p50_case_seconds'] for v in values])))
        memory.append(float(np.mean([v['memory']['peak_allocated_bytes']/2**30 for v in values])))
    axes[0].bar([n.upper() for n in names],latencies);axes[0].set(title='Uncached test p50 latency',ylabel='Seconds per 5-question case')
    axes[1].bar([n.upper() for n in names],memory);axes[1].set(title='Saved allocator peak (B0: whole run)',ylabel='GiB')
    save(fig,'costs')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    args=parser.parse_args();print(finish(args.run))

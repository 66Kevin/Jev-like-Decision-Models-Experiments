#!/usr/bin/env python3
"""P0: frozen original native Backbone and FP32 shared Pointer Head."""
import argparse
from datetime import datetime
import fcntl
import gc
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import yaml
from jev.data.typed_decisions import load_cases,prepare_splits,sha256_file
from jev.evaluation.metrics import evaluate_predictions
from jev.models.lm_readout import load_local_model,synchronize
from jev.models.pointer import PointerHead,prepare_pointer_case
from jev.training.pointer_runner import (
    assert_frozen,build_cache_key,cache_features,digest_json,head_hash,head_train_update,
    load_head_weights,publish_head_checkpoint,read_pointer_case,require_test_lock,
    restore_head_checkpoint,runtime_fingerprint,score_cached,validate_config,validate_head_checkpoint,
)
from jev.training.runner import epoch_groups,make_scheduler,lock_selections,read_rows,select_checkpoint,write_json
from jev.training.full_runner import reconcile_history,paired_seed_bootstrap
spec=importlib.util.spec_from_file_location('jev_p0_b0',ROOT/'experiments/01_baseline/run.py')
b0=importlib.util.module_from_spec(spec)
spec.loader.exec_module(b0)


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def prepare(config):
    validate_config(config)
    raw=ROOT/config['dataset']['raw_dir']
    b0.validate_source(raw,config['dataset'])
    splits=prepare_splits(load_cases(raw),ROOT/config['dataset']['split_dir'],seed=42,source_manifest=raw/'download_manifest.json')
    baseline=json.loads((ROOT/config['baseline_run']/'metadata.json').read_text())
    baseline_config=yaml.safe_load((ROOT/config['baseline_run']/'config.yaml').read_text())
    if baseline['status']!='complete' or baseline['dataset']!=config['dataset'] or baseline_config['model']!=config['model']:
        raise ValueError('P0 requires original completed B0 model/data')
    split_hash=sha256_file(ROOT/config['dataset']['split_dir']/'manifest.json')
    if split_hash!=baseline['split_manifest_sha256']:raise ValueError('P0 split provenance differs from B0')
    for source in ('src/jev/data/typed_decisions.py','src/jev/models/lm_readout.py','src/jev/evaluation/metrics.py'):
        if sha256_file(ROOT/source)!=baseline['source_sha256'][source]:raise ValueError(f'B0 scientific source changed: {source}')
    if {key:len(value) for key,value in splits.items()}!={'train':960,'dev':120,'calibration':120,'test':400}:
        raise ValueError('Unexpected P0 split sizes')
    for name in ('g1_run','g2_run'):
        previous=json.loads((ROOT/config[name]/'metadata.json').read_text())
        if (previous['status']!='complete' or previous['dataset']!=config['dataset']
                or previous['split_manifest_sha256']!=split_hash or previous['model_files_sha256']!=baseline['model_files_sha256']):
            raise ValueError(f'Comparison {name} source/model/splits differ')
    return splits,baseline


def snapshot_sources(directory):
    paths=sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'tests').glob('test_*.py'))
    paths += [Path(__file__),ROOT/'experiments/01_baseline/run.py',
              ROOT/'experiments/02_supervised/configs/p0.yaml',ROOT/'experiments/02_supervised/profile_p0.py']
    hashes={}
    for path in paths:
        relative=path.relative_to(ROOT)
        target=directory/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        hashes[str(relative)]=sha256_file(target)
    return hashes


def validate_profile(profile,config):
    required={'status':'passed','train_only':True,'physical_gpu':0,'native_backend':True,
              'reference_features_bitexact':True,'cached_full_scores_bitexact':True,
              'frozen_parameters':config['training']['expected_frozen_parameters'],
              'trainable_head_parameters':config['head']['expected_parameters'],'projection_dim':128}
    for key,value in required.items():
        if profile.get(key)!=value:raise ValueError(f'P0 preflight requires {key}={value!r}')
    if (not isinstance(profile.get('backbone_sha256_before'),str)
            or len(profile['backbone_sha256_before'])!=64
            or profile.get('backbone_sha256_before')!=profile.get('backbone_sha256_after')):
        raise ValueError('Preflight frozen Backbone changed')


def create_head(config,seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    head=PointerHead(config['head']['hidden_size'],config['head']['projection_dim']).to(config['model']['device'])
    if sum(p.numel() for p in head.parameters())!=config['head']['expected_parameters']:
        raise ValueError('Unexpected Pointer Head parameter count')
    return head


def records_for(cases,outputs):
    decisions=[d for case in cases for d in case['decisions']]
    if len(decisions)!=len(outputs):raise ValueError('Decision/output row count differs')
    records=[]
    for decision,result in zip(decisions,outputs):
        row={key:value for key,value in decision.items() if key not in ('state','instructions')}
        row.update(result,candidate_letters=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(decision['option_ids'])]))
        records.append(row)
    return records


def cached_dev(head,cached,cases,directory,epoch,config,seed):
    target=directory/f'dev_epoch{epoch}'
    if target.exists():
        if not (target/'summary.json').is_file():raise RuntimeError(f'Incomplete held-out Head scoring: {target}')
        return json.loads((target/'summary.json').read_text())
    target.mkdir()
    write_json(target/'started.json',{'head_sha256':head_hash(head),'feature_manifest_sha256':cached['manifest_sha256']})
    records=records_for(cases,score_cached(head,cached))
    with (target/'predictions.jsonl').open('x') as stream:
        for row in records:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    b0.save_logits(target/'logits.npz',records)
    metrics=evaluate_predictions(records,bootstrap_samples=config['evaluation']['bootstrap_samples'],seed=seed)
    write_json(target/'metrics.json',metrics)
    summary={'n_cases':len(cases),'n_decisions':len(records),'metrics':metrics['overall'],
             'head_sha256':head_hash(head),'feature_manifest_sha256':cached['manifest_sha256'],
             'mode':'cached frozen features; five independent question rows per case; no latency claim',
             'step_zero':'random Pointer Head; not reused from B0' if epoch==0 else None}
    write_json(target/'summary.json',summary)
    logging.info('seed%d dev_epoch%d Accuracy=%.4f NLL=%.4f',seed,epoch,metrics['overall']['accuracy'],metrics['overall']['hard_nll'])
    return summary


def verify_train_probe(model,head,tokenizer,cached,probe,config,directory):
    prepared=prepare_pointer_case(probe['decisions'],tokenizer,config['evaluation']['max_input_tokens'],config['head']['max_candidates'])
    full=read_pointer_case(model,head,prepared)
    first={key:value for key,value in cached.items() if key not in ('records',)}
    for key in ('query_features','candidate_features','candidate_mask','option_counts','hard_targets'):
        first[key]=cached[key][:5]
    first['records']=cached['records'][:5]
    actual=score_cached(head,first)
    if full['label_logits']!=[row['label_logits'] for row in actual]:
        raise RuntimeError('Cached/full Pointer Head train probe differs')
    write_json(directory/'train_cached_full_probe.json',{'train_only':True,'case_id':probe['case_id'],
               'cached_full_scores_bitexact':True,'head_sha256':head_hash(head)})



def finish_smoke(head,optimizer,scheduler,checkpoint,progress,initial_hash,caches,splits,model,tokenizer,config,directory,seed):
    if progress['step']!=2:raise ValueError('Smoke recovery must be exactly step2')
    before=head_hash(head)
    verify_train_probe(model,head,tokenizer,caches['train'],splits['train'][0],config,directory)
    restore_head_checkpoint(head,optimizer,scheduler,checkpoint)
    if head_hash(head)!=before:raise RuntimeError('Head checkpoint reload is not lossless')
    groups=epoch_groups(len(caches['train']['records']),config['training']['effective_batch_questions'],seed,progress['epoch'])
    next_epoch=progress['epoch'] if progress['group']<len(groups) else progress['epoch']+1
    next_group=progress['group'] if progress['group']<len(groups) else 0
    following=epoch_groups(len(caches['train']['records']),config['training']['effective_batch_questions'],seed,next_epoch)[next_group]
    stats=head_train_update(head,caches['train'],following,optimizer,scheduler,config['training']['max_grad_norm'])
    if head_hash(head)==before or before==initial_hash:raise RuntimeError('Pointer Head failed to update')
    write_json(directory/'smoke.json',{'train_only':True,'optimizer_updates':3,'checkpoint_step':2,
               'head_checkpoint_bitexact':True,'resumed_third_update':True,'after_resume_loss':stats['loss'],
               'no_heldout_inference':True,'trainable_parameters':sum(p.numel() for p in head.parameters())})
    return {'smoke':True}


def train_seed(config,seed,splits,caches,directory,model,tokenizer,smoke):
    directory.mkdir(exist_ok=True)
    head=create_head(config,seed)
    training=config['training']
    optimizer=torch.optim.AdamW(head.parameters(),lr=training['learning_rate'],weight_decay=training['weight_decay'],
                               betas=tuple(training['betas']),eps=training['eps'],foreach=False)
    steps_per_epoch=math.ceil(len(caches['train']['records'])/training['effective_batch_questions'])
    scheduler=make_scheduler(optimizer,training['epochs']*steps_per_epoch,training['warmup_fraction'])
    initial_hash=head_hash(head)
    progress={'epoch':0,'group':0,'step':0}
    latest=directory/'latest.json'
    if latest.exists():
        point=json.loads(latest.read_text());checkpoint=directory/point['checkpoint']
        if checkpoint.resolve().parent!=(directory/'checkpoints').resolve():raise ValueError('Invalid Head recovery path')
        progress=restore_head_checkpoint(head,optimizer,scheduler,checkpoint)
        if progress['step']!=point['step'] or head_hash(head)!=point['head_sha256']:raise ValueError('Head latest generation mismatch')
    else:
        publish_head_checkpoint(head,optimizer,scheduler,directory,progress,steps_per_epoch)
    reconcile_history(directory/'train.jsonl',progress['step'])
    if smoke and progress['step']>=2:
        return finish_smoke(head,optimizer,scheduler,checkpoint,progress,initial_hash,caches,splits,model,tokenizer,config,directory,seed)
    if not smoke:
        if progress['step'] and not (directory/'dev_epoch0/summary.json').is_file():
            raise RuntimeError('Recovery is missing original random Head dev_epoch0 evidence')
        cached_dev(head,caches['dev'],splits['dev'],directory,0,config,seed)
    for epoch in range(progress['epoch'],training['epochs']):
        groups=epoch_groups(len(caches['train']['records']),training['effective_batch_questions'],seed,epoch)
        start=progress['group'] if epoch==progress['epoch'] else 0
        for group in range(start,len(groups)):
            started=time.perf_counter()
            stats=head_train_update(head,caches['train'],groups[group],optimizer,scheduler,training['max_grad_norm'])
            synchronize(config['model']['device'])
            progress={'epoch':epoch,'group':group+1,'step':epoch*steps_per_epoch+group+1}
            stats.update(progress,seconds=time.perf_counter()-started)
            with (directory/'train.jsonl').open('a') as stream:
                stream.write(json.dumps(stats,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
            if progress['step']==2 or progress['step']%training['checkpoint_every_steps']==0 or progress['step']%steps_per_epoch==0:
                checkpoint,_=publish_head_checkpoint(head,optimizer,scheduler,directory,progress,steps_per_epoch)
            if progress['step']<=2 or progress['step']%50==0:
                logging.info('seed%d epoch%d step%d loss=%.5f',seed,epoch+1,progress['step'],stats['loss'])
            if smoke and progress['step']==2:
                return finish_smoke(head,optimizer,scheduler,checkpoint,progress,initial_hash,caches,splits,model,tokenizer,config,directory,seed)
        if not smoke:cached_dev(head,caches['dev'],splits['dev'],directory,epoch+1,config,seed)
        progress={'epoch':epoch+1,'group':0,'step':(epoch+1)*steps_per_epoch}
    candidates=[]
    for epoch in range(training['epochs']+1):
        checkpoint=f'checkpoints/step{epoch*steps_per_epoch:06d}'
        manifest=validate_head_checkpoint(directory/checkpoint)
        dev=json.loads((directory/f'dev_epoch{epoch}/summary.json').read_text())
        if manifest['head_sha256']!=dev['head_sha256']:raise ValueError('Dev score/checkpoint Head differs')
        candidates.append({'epoch':epoch,'step':epoch*steps_per_epoch,'hard_nll':dev['metrics']['hard_nll'],
                           'checkpoint':checkpoint,'head_sha256':manifest['head_sha256']})
    selected=dict(select_checkpoint(candidates))
    write_json(directory/'selection.json',selected)
    write_json(directory/'training_summary.json',{'initial_head_sha256':initial_hash,'final_head_sha256':head_hash(head),
               'optimizer_steps':training['epochs']*steps_per_epoch,'dev_candidates':candidates})
    load_head_weights(head,directory/selected['checkpoint'])
    verify_train_probe(model,head,tokenizer,caches['train'],splits['train'][0],config,directory)
    return selected


def evaluate_uncached(model,head,tokenizer,cases,train,directory,config,seed,lock_path):
    locked=require_test_lock(lock_path,config['seeds'])
    if locked[str(seed)]['head_sha256']!=head_hash(head):
        raise ValueError('Official test Head differs from locked selection')
    target=directory/'test'
    if target.exists():
        if not (target/'summary.json').is_file():raise RuntimeError('Partial test exists; refusing inference replay')
        return json.loads((target/'summary.json').read_text())
    for case in b0.select_warmup(train,config['evaluation']['warmup_cases']):
        read_pointer_case(model,head,prepare_pointer_case(case['decisions'],tokenizer))
    synchronize(config['model']['device']);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(config['model']['device'])
    target.mkdir();write_json(target/'started.json',{'head_sha256':head_hash(head),'selections_sha256':sha256_file(lock_path)})
    records=[];times=[]
    with (target/'predictions.jsonl').open('x') as stream:
        for index,case in enumerate(cases):
            synchronize(config['model']['device']);started=time.perf_counter()
            prepared=prepare_pointer_case(case['decisions'],tokenizer,config['evaluation']['max_input_tokens'],config['head']['max_candidates'])
            tokenization=time.perf_counter()-started
            result=read_pointer_case(model,head,prepared)
            elapsed=time.perf_counter()-started
            outputs=[{'probabilities':probs,'label_logits':logits,'input_tokens':length,'case_latency_seconds':elapsed}
                     for probs,logits,length in zip(result['probabilities'],result['label_logits'],prepared['input_lengths'])]
            rows=records_for([case],outputs);records.extend(rows)
            for row in rows:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            times.append({'case_id':case['case_id'],'workflow':case['workflow'],'tokenization_seconds':tokenization,
                          'inference_seconds':result['inference_seconds'],'total_seconds':elapsed,
                          'input_lengths':prepared['input_lengths'],'padded_tokens':int(prepared['input_ids'].numel())})
            if (index+1)%40==0 or index+1==len(cases):
                stream.flush();logging.info('seed%d test %d/%d cases',seed,index+1,len(cases))
    b0.save_logits(target/'logits.npz',records)
    metrics=evaluate_predictions(records,bootstrap_samples=config['evaluation']['bootstrap_samples'],seed=seed)
    write_json(target/'metrics.json',metrics)
    elapsed=np.array([row['total_seconds'] for row in times])
    timing={'n_cases':len(cases),'n_decisions':len(records),'total_seconds':float(elapsed.sum()),
            'mean_case_seconds':float(elapsed.mean()),'p50_case_seconds':float(np.percentile(elapsed,50)),
            'p95_case_seconds':float(np.percentile(elapsed,95)),
            'definition':'End-to-end uncached native frozen Backbone + Pointer Head per case, five independent questions; '
                         'includes tokenization/offset mapping, transfer, frozen Backbone, feature transfer, FP32 Head, '
                         'normalization and CPU conversion. Excludes warmup, load, metrics and file I/O.', 'cases':times}
    write_json(target/'timing.json',timing)
    memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated(config['model']['device']),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(config['model']['device']),
            'current_allocated_bytes':torch.cuda.memory_allocated(config['model']['device'])}
    write_json(target/'memory.json',memory)
    value={'n_cases':len(cases),'n_decisions':len(records),'metrics':metrics['overall'],
           'p50_case_seconds':timing['p50_case_seconds'],'p95_case_seconds':timing['p95_case_seconds'],
           'memory':memory,'head_sha256':head_hash(head),'selections_sha256':sha256_file(lock_path)}
    write_json(target/'summary.json',value)
    return value


def execute(config,config_path,splits,baseline,smoke=False,resume=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        raise ValueError('P0 physical GPU0 requires CUDA_VISIBLE_DEVICES=0')
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    directory=Path(resume).resolve() if resume else ROOT/config['output_root']/f'{"smoke_train" if smoke else "train"}_{stamp}'
    profile=ROOT/config['output_root']/'preflight_train.json'
    profile_data=json.loads(profile.read_text());validate_profile(profile_data,config)
    fingerprint=runtime_fingerprint();model_files=b0.model_file_hashes(config['model']['path'])
    if model_files!=baseline['model_files_sha256']:raise ValueError('Original model files changed from B0')
    raw_hash=sha256_file(ROOT/config['dataset']['raw_dir']/'download_manifest.json')
    if resume:
        metadata=json.loads((directory/'metadata.json').read_text())
        if yaml.safe_load((directory/'config.yaml').read_text())!=config:raise ValueError('Resume config changed')
        if metadata['mode']!=('smoke_train' if smoke else 'train') or metadata['status']=='complete':raise ValueError('Invalid resume mode/status')
        for source,digest in metadata['source_sha256'].items():
            if sha256_file(ROOT/source)!=digest or sha256_file(directory/'source'/source)!=digest:raise ValueError(f'Resume source changed: {source}')
        if metadata['runtime_fingerprint']!=fingerprint:raise ValueError('Resume runtime changed')
        if metadata['profile_sha256']!=sha256_file(profile) or metadata['profile_sha256']!=sha256_file(directory/profile.name):
            raise ValueError('Resume preflight changed')
        if (metadata['model_files_sha256']!=model_files or metadata['raw_manifest_sha256']!=raw_hash
                or metadata['split_manifest_sha256']!=baseline['split_manifest_sha256']):raise ValueError('Resume source/model/splits changed')
    else:
        directory.mkdir(parents=True,exist_ok=False)
        (directory/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        shutil.copyfile(profile,directory/profile.name)
        shutil.copytree(ROOT/config['dataset']['split_dir'],directory/'splits')
        metadata={'status':'running','mode':'smoke_train' if smoke else 'train','experiment_id':'p0',
                  'started_at':now(),'completed_seeds':{},'test':{},'source_sha256':snapshot_sources(directory),
                  'dataset':config['dataset'],'model_files_sha256':model_files,'raw_manifest_sha256':raw_hash,
                  'split_manifest_sha256':baseline['split_manifest_sha256'],'runtime_fingerprint':fingerprint,
                  'profile_sha256':sha256_file(profile),'config_sha256':sha256_file(directory/'config.yaml'),
                  'config_source':str(config_path.resolve()),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                  'baseline_run':config['baseline_run'],'g1_run':config['g1_run'],'g2_run':config['g2_run'],
                  'protocol':{'backbone':'original frozen BF16; native FP32 rotary buffers; no FLA',
                              'training':'FP32 Head only; cached train/dev; hard argmax labels',
                              'head':'bias-free shared 1024->128 query/key projections; scaled dot match',
                              'test':'uncached end-to-end after all seed selections locked',
                              'calibration_inference':False,'candidate_order':'original; no invariance guarantee'}}
    lease=(directory/'.run.lock').open('a')
    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:lease.close();raise RuntimeError('P0 already has a live worker') from None
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(directory/'run.log'),logging.StreamHandler()],force=True)
    (directory/'pid').write_text(str(os.getpid())+'\n');print(f'RUN_DIR={directory}',flush=True)
    try:
        torch.backends.cuda.matmul.fp32_precision='ieee';torch.backends.cudnn.conv.fp32_precision='ieee'
        metadata['status']='extracting_features';metadata.pop('error',None)
        metadata['device']={'logical':config['model']['device'],'name':torch.cuda.get_device_name(config['model']['device'])}
        write_json(directory/'metadata.json',metadata)
        model,tokenizer,loading=load_local_model(config['model'])
        from jev.training.kernels import set_delta_backend
        metadata['native_backend']=set_delta_backend(model,'native')
        if metadata['native_backend']['delta_layers']!=18:
            raise ValueError('Expected 18 original native DeltaNet layers')
        frozen=assert_frozen(model)
        if frozen['unique_parameters']!=config['training']['expected_frozen_parameters']:raise ValueError('Wrong frozen model size')
        if 'frozen_backbone' in metadata and metadata['frozen_backbone']!=frozen:raise ValueError('Resume frozen model differs')
        metadata['frozen_backbone']=frozen;metadata['loading_info']=loading
        key=build_cache_key(config,metadata['source_sha256'],model_files,metadata['split_manifest_sha256'])
        requested={'train':splits['train'][:8]} if smoke else {'train':splits['train'],'dev':splits['dev']}
        caches={}
        for name,cases in requested.items():
            logging.info('Preparing %s frozen feature cache (%d cases)',name,len(cases))
            caches[name]=cache_features(directory/'features'/name,name,cases,model,tokenizer,key,
                                       config['evaluation']['max_input_tokens'],config['head']['max_candidates'])
        assert_frozen(model,frozen)
        metadata['feature_caches']={name:value['manifest_sha256'] for name,value in caches.items()}
        metadata['status']='training';write_json(directory/'metadata.json',metadata)
        for seed in config['seeds'][:1] if smoke else config['seeds']:
            if str(seed) not in metadata['completed_seeds']:
                metadata['completed_seeds'][str(seed)]=train_seed(config,seed,splits,caches,directory/f'seed{seed}',model,tokenizer,smoke)
                assert_frozen(model,frozen);write_json(directory/'metadata.json',metadata)
        if not smoke:
            lock_selections(directory/'selections.json',metadata['completed_seeds'],config['seeds'])
            require_test_lock(directory/'selections.json',config['seeds'],metadata['completed_seeds'])
            metadata['status']='selections_locked';write_json(directory/'metadata.json',metadata)
            for seed in config['seeds']:
                if str(seed) in metadata['test']:continue
                head=create_head(config,seed);selected=metadata['completed_seeds'][str(seed)]
                manifest=load_head_weights(head,directory/f'seed{seed}'/selected['checkpoint'])
                if manifest['head_sha256']!=selected['head_sha256']:raise ValueError('Locked Head changed')
                verify_train_probe(model,head,tokenizer,caches['train'],splits['train'][0],config,directory/f'seed{seed}')
                metadata['status']='evaluating';write_json(directory/'metadata.json',metadata)
                metadata['test'][str(seed)]=evaluate_uncached(model,head,tokenizer,splits['test'],splits['train'],
                    directory/f'seed{seed}',config,seed,directory/'selections.json')
                assert_frozen(model,frozen);write_json(directory/'metadata.json',metadata)
                del head;gc.collect();torch.cuda.empty_cache()
            metrics=[metadata['test'][str(seed)]['metrics'] for seed in config['seeds']]
            aggregate={key:{'mean':float(np.mean([row[key] for row in metrics])),
                            'sample_std':float(np.std([row[key] for row in metrics],ddof=1))}
                       for key in metrics[0] if isinstance(metrics[0][key],(int,float)) and not key.startswith('n_')}
            candidates=[read_rows(directory/f'seed{seed}/test/predictions.jsonl') for seed in config['seeds']]
            summary={'per_seed':metadata['test'],'aggregate':aggregate}
            for name in ('baseline_run','g1_run','g2_run'):
                paths=[ROOT/config[name]/'test/predictions.jsonl'] if name=='baseline_run' else [ROOT/config[name]/f'seed{seed}/test/predictions.jsonl' for seed in config['seeds']]
                reference=[read_rows(path) for path in paths]
                value=paired_seed_bootstrap(reference,candidates,config['evaluation']['bootstrap_samples'])
                value['definition']='Mean P0 seed correctness minus mean reference seed correctness; no probability ensemble; percentile 95% CI'
                summary[f'paired_accuracy_delta_vs_{"b0" if name=="baseline_run" else name[:2]}']=value
            write_json(directory/'summary.json',summary)
        metadata['frozen_backbone_after']=assert_frozen(model,frozen);metadata['status']='complete'
    except BaseException as error:
        metadata['status']='failed';metadata['error']={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
        logging.exception('P0 stopped; complete Head generations preserved');raise
    finally:
        metadata['updated_at']=now();write_json(directory/'metadata.json',metadata)
        fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
    return directory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'experiments/02_supervised/configs/p0.yaml')
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if args.prepare_only and (args.smoke or args.resume):parser.error('--prepare-only cannot combine with smoke/resume')
    config_path=args.resume/'config.yaml' if args.resume else args.config
    config=yaml.safe_load(config_path.read_text());splits,baseline=prepare(config)
    if args.prepare_only:
        print(json.dumps({'counts':{name:len(cases) for name,cases in splits.items()},'train_decisions':4800,
                          'optimizer_steps_per_seed':450,'trainable_parameters':262144,'backend':'native'}));return
    execute(config,config_path,splits,baseline,args.smoke,args.resume)


if __name__=='__main__':main()

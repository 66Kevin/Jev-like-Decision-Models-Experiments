#!/usr/bin/env python3
"""P1: jointly supervise original-backbone LoRA and shared FP32 Pointer Head."""
import argparse
from contextlib import contextmanager
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
import socket
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import yaml
from jev.data.typed_decisions import iter_decisions,load_cases,prepare_splits,sha256_file
from jev.evaluation.metrics import evaluate_predictions
from jev.models.lm_readout import load_local_model,synchronize
from jev.models.pointer import PointerHead,prepare_pointer_case
from jev.training.lora import inject_lora
from jev.training.pointer_lora import (
    adapter_hash,assert_pointer_lora_trainable,cache_pointer_decisions,collate_pointer_decisions,
    create_joint_optimizer,frozen_backbone_state,joint_train_update,load_joint_weights,
    pointer_lora_logits,publish_joint_checkpoint,restore_joint_checkpoint,validate_joint_checkpoint,
)
from jev.training.pointer_runner import digest_json,head_hash,require_test_lock
from jev.training.runner import epoch_groups,lock_selections,make_scheduler,read_rows,runtime_fingerprint,select_checkpoint,write_json
from jev.training.full_runner import paired_seed_bootstrap,reconcile_history
spec=importlib.util.spec_from_file_location('jev_p1_b0',ROOT/'experiments/01_baseline/run.py')
b0=importlib.util.module_from_spec(spec);spec.loader.exec_module(b0)


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def validate_config(config):
    fixed={'experiment_id':'p1','seeds':[42,43,44]}
    groups={'lora':{'rank':16,'alpha':32,'dropout':0.,'expected_target_modules':186},
            'head':{'hidden_size':1024,'projection_dim':128,'max_candidates':5,'expected_parameters':262144},
            'training':{'epochs':3,'effective_batch_questions':32,'lora_learning_rate':2e-4,'head_learning_rate':1e-3,
                        'weight_decay':.01,'betas':[.9,.999],'eps':1e-8,'max_grad_norm':1.,'warmup_fraction':.05,
                        'scheduler':'cosine','gradient_checkpointing':False,'loss':'hard_candidate_cross_entropy',
                        'select_metric':'dev_hard_nll','kernel_backend':'fla','checkpoint_every_steps':50,
                        'expected_frozen_parameters':752393024,'expected_lora_parameters':10822656,
                        'expected_trainable_parameters':11084800},
            'evaluation':{'splits':['dev','test'],'max_input_tokens':8192,'warmup_cases':3,'case_batch_size':1,
                          'ece_bins':15,'temperature':1.,'use_cache':False,'prompt_version':'b0_v1',
                          'kernel_backend':'native','merge_adapter':False}}
    for key,value in fixed.items():
        if config.get(key)!=value:raise ValueError(f'P1 requires {key}={value!r}')
    for group,values in groups.items():
        for key,value in values.items():
            if config[group].get(key)!=value:raise ValueError(f'P1 requires {group}.{key}={value!r}')
    if (config['training']['microbatch_questions'],config['training']['gradient_accumulation']) not in ((8,4),(4,8)):
        raise ValueError('Only profiled micro8/acc4 or micro4/acc8 permitted')


def validate_profile(profile,config):
    expected={'status':'passed','train_only':True,'hostname':'alg26','physical_gpu':0,
              'zero_lora_pointer_native_bitexact':True,'lm_head_projection_calls':0,
              'head_and_lora_updated':True,'hidden_feature_cache':False,'lora_a_updated_after_nonzero_b':True,
              'selected_microbatch':config['training']['microbatch_questions'],
              'selected_accumulation':config['training']['gradient_accumulation']}
    for key,value in expected.items():
        if profile.get(key)!=value:raise ValueError(f'P1 preflight requires {key}={value!r}')
    counts={'lora_parameters':config['training']['expected_lora_parameters'],'head_parameters':config['head']['expected_parameters'],
            'trainable_parameters':config['training']['expected_trainable_parameters'],
            'frozen_parameters':config['training']['expected_frozen_parameters']}
    if any(profile.get('parameters',{}).get(k)!=v for k,v in counts.items()):raise ValueError('Profile parameter counts differ')
    frozen=profile.get('frozen_backbone_before',{})
    if (frozen!=profile.get('frozen_backbone_after') or len(frozen.get('state_sha256',''))!=64
            or frozen.get('unique_parameters')!=counts['frozen_parameters']):raise ValueError('Profile frozen Backbone changed')
    for name in ('head','lora'):
        gradient=profile.get('gradient_agreement',{}).get(name,{})
        if not (gradient.get('cosine',-1)>=.95 and 0<=gradient.get('relative_l2',999)<=.35):
            raise ValueError(f'Profile native/FLA {name} gradient disagreement')


def gpu_guard():
    if (socket.gethostname()!='alg26' or os.environ.get('CUDA_VISIBLE_DEVICES')!='0'
            or os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID'):
        raise ValueError('P1 requires alg26 physical GPU0: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0')


def device_provenance():
    result=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,name,pci.bus_id,driver_version',
                                    '--format=csv,noheader'],text=True).strip()
    torch_uuid=str(torch.cuda.get_device_properties(0).uuid)
    if torch_uuid.lower().removeprefix('gpu-')!=result.split(',')[0].strip().lower().removeprefix('gpu-'):
        raise ValueError('Torch logical GPU UUID differs from nvidia-smi physical GPU0 UUID')
    return {'hostname':socket.gethostname(),'physical_gpu':0,'nvidia_smi':result,'torch_gpu_uuid':torch_uuid,
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'cuda_device_order':os.environ.get('CUDA_DEVICE_ORDER')}


def prepare(config):
    validate_config(config);raw=ROOT/config['dataset']['raw_dir'];b0.validate_source(raw,config['dataset'])
    splits=prepare_splits(load_cases(raw),ROOT/config['dataset']['split_dir'],seed=42,source_manifest=raw/'download_manifest.json')
    baseline=json.loads((ROOT/config['baseline_run']/'metadata.json').read_text())
    baseline_config=yaml.safe_load((ROOT/config['baseline_run']/'config.yaml').read_text())
    split_hash=sha256_file(ROOT/config['dataset']['split_dir']/'manifest.json')
    if (baseline['status']!='complete' or baseline['dataset']!=config['dataset'] or baseline_config['model']!=config['model']
            or split_hash!=baseline['split_manifest_sha256']):raise ValueError('P1 requires completed original B0 model/data/splits')
    for source in ('src/jev/data/typed_decisions.py','src/jev/models/lm_readout.py','src/jev/evaluation/metrics.py'):
        if sha256_file(ROOT/source)!=baseline['source_sha256'][source]:raise ValueError(f'B0 scientific source changed: {source}')
    for name in ('g1_run','g2_run','p0_run'):
        reference=json.loads((ROOT/config[name]/'metadata.json').read_text())
        if (reference['status']!='complete' or reference['dataset']!=config['dataset']
                or reference['model_files_sha256']!=baseline['model_files_sha256'] or reference['split_manifest_sha256']!=split_hash):
            raise ValueError(f'P1 comparison {name} model/data/splits differ')
    if {k:len(v) for k,v in splits.items()}!={'train':960,'dev':120,'calibration':120,'test':400}:raise ValueError('Unexpected split sizes')
    return splits,baseline


def snapshot_sources(directory):
    paths=sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'tests').glob('test_*.py'))
    paths += [Path(__file__),ROOT/'experiments/01_baseline/run.py',ROOT/'experiments/02_supervised/configs/p1.yaml',
              ROOT/'experiments/02_supervised/profile_p1.py']
    hashes={}
    for path in paths:
        relative=path.relative_to(ROOT);target=directory/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target);hashes[str(relative)]=sha256_file(target)
    return hashes


def validate_resume(directory,config,metadata,fingerprint,profile,model_files,raw_hash,split_hash,smoke):
    if (yaml.safe_load((directory/'config.yaml').read_text())!=config
            or metadata.get('config_sha256')!=sha256_file(directory/'config.yaml')):raise ValueError('Resume config changed')
    if metadata['mode']!=('smoke_train' if smoke else 'train') or metadata['status']=='complete':raise ValueError('Invalid resume mode/status')
    for source,digest in metadata['source_sha256'].items():
        if sha256_file(ROOT/source)!=digest or sha256_file(directory/'source'/source)!=digest:raise ValueError(f'Resume source changed: {source}')
    if metadata['runtime_fingerprint']!=fingerprint:raise ValueError('Resume runtime changed')
    if metadata['profile_sha256']!=sha256_file(profile) or metadata['profile_sha256']!=sha256_file(directory/'preflight_train.json'):
        raise ValueError('Resume profile changed')
    if (metadata['model_files_sha256']!=model_files or metadata['raw_manifest_sha256']!=raw_hash
            or metadata['split_manifest_sha256']!=split_hash):raise ValueError('Resume source/model/splits changed')


def backend(model,config,kind):
    from jev.training.kernels import set_delta_backend
    return set_delta_backend(model,kind,ROOT/config['training']['fla_package_dir'])


def create_pair(config,seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    model,tokenizer,loading=load_local_model(config['model']);lora=config['lora']
    model=inject_lora(model,lora['rank'],lora['alpha'],lora['dropout'],lora['expected_target_modules'])
    # Independent RNG scope gives exactly P0's fresh Head initialization while
    # retaining G1's LoRA initialization and its subsequent RNG stream.
    devices=[torch.device(config['model']['device']).index or 0] if torch.device(config['model']['device']).type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        head=PointerHead(config['head']['hidden_size'],config['head']['projection_dim']).to(config['model']['device'])
    expected=json.loads((ROOT/config['p0_run']/f'seed{seed}/checkpoints/step000000/manifest.json').read_text())['head_sha256']
    if head_hash(head)!=expected:raise ValueError('Independent fresh Pointer initialization differs from corresponding P0 seed')
    model.eval();head.eval();return model,head,tokenizer,loading


def validate_counts(config,model,head):
    counts=assert_pointer_lora_trainable(model,head);training=config['training']
    expected={'lora_parameters':training['expected_lora_parameters'],'head_parameters':config['head']['expected_parameters'],
              'trainable_parameters':training['expected_trainable_parameters'],'frozen_parameters':training['expected_frozen_parameters']}
    if any(counts[k]!=v for k,v in expected.items()):raise ValueError('Joint trainable/frozen parameter counts differ')
    return counts


def frozen_check(model,expected):
    actual=frozen_backbone_state(model)
    if actual!=expected:raise RuntimeError('Original parameters or buffers changed')
    return actual


@contextmanager
def optimizer_offloaded(optimizer,device):
    for state in optimizer.state.values():
        for key,value in state.items():
            if isinstance(value,torch.Tensor):state[key]=value.cpu()
    try:yield
    finally:
        for state in optimizer.state.values():
            for key,value in state.items():
                if isinstance(value,torch.Tensor) and key!='step':state[key]=value.to(device)


def read_pointer(model,head,prepared):
    device=next(head.parameters()).device;synchronize(device);started=time.perf_counter()
    model.eval();head.eval()
    with torch.no_grad():scores=pointer_lora_logits(model,head,prepared)
    logits=[row[:n].float().cpu().tolist() for row,n in zip(scores,prepared['option_counts'])]
    probabilities=[torch.softmax(row[:n].float(),-1).cpu().tolist() for row,n in zip(scores,prepared['option_counts'])]
    synchronize(device)
    return {'label_logits':logits,'probabilities':probabilities,'inference_seconds':time.perf_counter()-started}


def records_for(case,result,prepared,elapsed):
    records=[]
    for decision,logits,probs,length in zip(case['decisions'],result['label_logits'],result['probabilities'],prepared['input_lengths']):
        row={k:v for k,v in decision.items() if k not in ('state','instructions')}
        row.update(label_logits=logits,probabilities=probs,input_tokens=length,case_latency_seconds=elapsed,
                   candidate_letters=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(decision['option_ids'])]))
        records.append(row)
    return records


def evaluate_native(name,cases,model,head,tokenizer,directory,config,seed,train,lock_path=None):
    hashes={'head_sha256':head_hash(head),'adapter_sha256':adapter_hash(model)}
    if name=='test':
        locked=require_test_lock(lock_path,config['seeds'])
        if any(locked[str(seed)].get(k)!=v for k,v in hashes.items()):raise ValueError('Test joint model differs from all-seed locked selection')
    elif name not in ('dev_epoch0','dev_epoch1','dev_epoch2','dev_epoch3'):raise ValueError('Only registered dev epochs/test permitted')
    target=directory/name
    if target.exists():
        if not (target/'summary.json').is_file():raise RuntimeError(f'Partial held-out output; refusing replay: {target}')
        summary=json.loads((target/'summary.json').read_text())
        if any(summary.get(k)!=v for k,v in hashes.items()):raise ValueError('Completed held-out joint state changed')
        return summary
    backend(model,config,'native');model.eval();head.eval();frozen=frozen_backbone_state(model)
    for case in b0.select_warmup(train,config['evaluation']['warmup_cases']):
        read_pointer(model,head,prepare_pointer_case(case['decisions'],tokenizer))
    device=config['model']['device'];synchronize(device)
    if torch.device(device).type=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
    target.mkdir();write_json(target/'started.json',{**hashes,'selections_sha256':sha256_file(lock_path) if lock_path else None})
    records=[];times=[]
    with (target/'predictions.jsonl').open('x') as stream:
        for index,case in enumerate(cases):
            synchronize(device);started=time.perf_counter()
            prepared=prepare_pointer_case(case['decisions'],tokenizer,config['evaluation']['max_input_tokens'],config['head']['max_candidates'])
            tokenization=time.perf_counter()-started;result=read_pointer(model,head,prepared);elapsed=time.perf_counter()-started
            rows=records_for(case,result,prepared,elapsed);records.extend(rows)
            for row in rows:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            times.append({'case_id':case['case_id'],'workflow':case['workflow'],'tokenization_seconds':tokenization,
                          'inference_seconds':result['inference_seconds'],'total_seconds':elapsed,
                          'input_lengths':prepared['input_lengths'],'padded_tokens':int(prepared['input_ids'].numel())})
            if (index+1)%40==0 or index+1==len(cases):stream.flush();logging.info('seed%d %s %d/%d cases',seed,name,index+1,len(cases))
    b0.save_logits(target/'logits.npz',records)
    metrics=evaluate_predictions(records,bootstrap_samples=config['evaluation']['bootstrap_samples'],seed=seed);write_json(target/'metrics.json',metrics)
    elapsed=np.asarray([row['total_seconds'] for row in times])
    timing={'n_cases':len(cases),'n_decisions':len(records),'total_seconds':float(elapsed.sum()),
            'mean_case_seconds':float(elapsed.mean()),'p50_case_seconds':float(np.percentile(elapsed,50)),
            'p95_case_seconds':float(np.percentile(elapsed,95)),'cases':times,
            'definition':'End-to-end uncached native original Backbone + unmerged LoRA + FP32 Pointer Head; five independent rows per case; '
                         'tokenization/offsets, transfers, Backbone, Head, normalization and CPU conversion included; warmup/load/metrics/file I/O excluded. '
                         'RTX 4090 measurements must not be interpreted as architecture speed differences versus prior RTX 5090 runs.'}
    write_json(target/'timing.json',timing)
    memory={k:int(f(device)) for k,f in [('peak_allocated_bytes',torch.cuda.max_memory_allocated),
             ('peak_reserved_bytes',torch.cuda.max_memory_reserved),('current_allocated_bytes',torch.cuda.memory_allocated)]} if torch.device(device).type=='cuda' else {}
    write_json(target/'memory.json',memory);frozen_check(model,frozen)
    value={'n_cases':len(cases),'n_decisions':len(records),'metrics':metrics['overall'],**hashes,
           'p50_case_seconds':timing['p50_case_seconds'],'p95_case_seconds':timing['p95_case_seconds'],'memory':memory,
           'memory_definition':'Original Backbone + unmerged LoRA + Head; joint optimizer states off GPU.',
           'step_zero':'Actual random Pointer Head + zero LoRA; no reused B0/P0 predictions' if name=='dev_epoch0' else None,
           'selections_sha256':sha256_file(lock_path) if lock_path else None}
    write_json(target/'summary.json',value);return value


def token_cache(cases,tokenizer,directory,config):
    decisions=list(iter_decisions(cases));cached=cache_pointer_decisions(decisions,tokenizer,config['evaluation']['max_input_tokens'],config['head']['max_candidates'])
    # A regenerated tokens-only digest also checks original ordering/identity on resume.
    evidence={'decisions':len(cached),'max_tokens':max(len(row['input_ids']) for row in cached),
              'cache_identity_sha256':digest_json(cached),'hidden_features_cached':False}
    path=directory/'tokenization.json'
    if path.exists() and json.loads(path.read_text())!=evidence:raise ValueError('Token cache identity changed')
    if not path.exists():write_json(path,evidence)
    return cached


def roundtrip(model,head,tokenizer,probe,checkpoint,config,directory):
    backend(model,config,'native');prepared=prepare_pointer_case(probe['decisions'],tokenizer)
    before=read_pointer(model,head,prepared)['label_logits'];load_joint_weights(model,head,checkpoint)
    after=read_pointer(model,head,prepared)['label_logits']
    if before!=after:raise RuntimeError('Joint native reload logits differ')
    write_json(directory/'reload_check_train.json',{'train_only':True,'case_id':probe['case_id'],
               'native_reload_bitexact':True,'checkpoint':str(checkpoint.relative_to(directory)),
               'head_sha256':head_hash(head),'adapter_sha256':adapter_hash(model)})


def finish_smoke(config,model,head,tokenizer,optimizer,scheduler,cached,pad,checkpoint,progress,directory,seed,frozen,initial):
    if progress['step']!=2:raise ValueError('Smoke requires checkpoint step2')
    roundtrip(model,head,tokenizer,config['_probe'],checkpoint,config,directory)
    restore_joint_checkpoint(model,head,optimizer,scheduler,checkpoint)
    groups=epoch_groups(len(cached),config['training']['effective_batch_questions'],seed,progress['epoch'])
    epoch=progress['epoch'] if progress['group']<len(groups) else progress['epoch']+1
    group=progress['group'] if progress['group']<len(groups) else 0
    indices=epoch_groups(len(cached),config['training']['effective_batch_questions'],seed,epoch)[group]
    before=(adapter_hash(model),head_hash(head));backend(model,config,'fla')
    stats=joint_train_update(model,head,cached,indices,pad,optimizer,scheduler,config['training']['microbatch_questions'],config['training']['max_grad_norm'])
    after=(adapter_hash(model),head_hash(head))
    if any(a==b for a,b in zip(before,after)) or before==initial:raise RuntimeError('Joint smoke parameters failed to update')
    frozen_check(model,frozen)
    write_json(directory/'smoke.json',{'train_only':True,'optimizer_updates':3,'checkpoint_step':2,'native_reload_bitexact':True,
               'resumed_third_update':True,'after_resume_loss':stats['loss'],'no_heldout_inference':True,
               'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)+sum(p.numel() for p in head.parameters())})
    return {'smoke':True}


def train_seed(config,seed,splits,directory,smoke):
    directory.mkdir(exist_ok=True);model,head,tokenizer,loading=create_pair(config,seed)
    frozen=frozen_backbone_state(model);counts=validate_counts(config,model,head)
    initial=(adapter_hash(model),head_hash(head));training=config['training']
    model_info={**(counts or {}),'loading_info':loading,'frozen_backbone':frozen,'initial_adapter_sha256':initial[0],
                'initial_head_sha256':initial[1],'head_initialization':'fresh independent per-seed RNG; identical to P0 step0',
                'target_modules':list(model._jev_lora_target_modules)}
    info_path=directory/'model.json'
    if info_path.exists() and json.loads(info_path.read_text())!=model_info:raise ValueError('Resume original model or initialization changed')
    if not info_path.exists():write_json(info_path,model_info)
    train=splits['train'][:8] if smoke else splits['train'];cached=token_cache(train,tokenizer,directory,config)
    pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    steps_per_epoch=math.ceil(len(cached)/training['effective_batch_questions'])
    optimizer=create_joint_optimizer(model,head,training['lora_learning_rate'],training['head_learning_rate'],training['weight_decay'],training['betas'],training['eps'])
    scheduler=make_scheduler(optimizer,training['epochs']*steps_per_epoch,training['warmup_fraction'])
    progress={'epoch':0,'group':0,'step':0};latest=directory/'latest.json'
    if latest.exists():
        point=json.loads(latest.read_text());checkpoint=directory/point['checkpoint']
        if checkpoint.resolve().parent!=(directory/'checkpoints').resolve():raise ValueError('Invalid joint recovery path')
        progress=restore_joint_checkpoint(model,head,optimizer,scheduler,checkpoint)
        if (progress['step']!=point['step'] or adapter_hash(model)!=point['adapter_sha256'] or head_hash(head)!=point['head_sha256']):
            raise ValueError('Joint latest generation mismatch')
    else:checkpoint,_=publish_joint_checkpoint(model,head,optimizer,scheduler,directory,progress,steps_per_epoch)
    reconcile_history(directory/'train.jsonl',progress['step']);frozen_check(model,frozen)
    smoke_config={**config,'_probe':train[0]}
    if smoke and progress['step']>=2:return finish_smoke(smoke_config,model,head,tokenizer,optimizer,scheduler,cached,pad,checkpoint,progress,directory,seed,frozen,initial)
    if not smoke:
        if progress['step']:
            zero=json.loads((directory/'dev_epoch0/summary.json').read_text())
            if zero['head_sha256']!=initial[1] or zero['adapter_sha256']!=initial[0]:raise ValueError('Original random dev0 evidence changed')
        else:
            with optimizer_offloaded(optimizer,config['model']['device']):evaluate_native('dev_epoch0',splits['dev'],model,head,tokenizer,directory,config,seed,train)
    peak=0
    if torch.device(config['model']['device']).type=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(config['model']['device'])
    for epoch in range(progress['epoch'],training['epochs']):
        groups=epoch_groups(len(cached),training['effective_batch_questions'],seed,epoch);start=progress['group'] if epoch==progress['epoch'] else 0
        backend(model,config,'fla')
        for group in range(start,len(groups)):
            started=time.perf_counter();stats=joint_train_update(model,head,cached,groups[group],pad,optimizer,scheduler,training['microbatch_questions'],training['max_grad_norm'])
            synchronize(config['model']['device']);progress={'epoch':epoch,'group':group+1,'step':epoch*steps_per_epoch+group+1}
            stats.update(progress,seconds=time.perf_counter()-started)
            with (directory/'train.jsonl').open('a') as stream:stream.write(json.dumps(stats,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
            if torch.device(config['model']['device']).type=='cuda':peak=max(peak,torch.cuda.max_memory_allocated(config['model']['device']))
            if progress['step']==2 or progress['step']%training['checkpoint_every_steps']==0 or progress['step']%steps_per_epoch==0:
                checkpoint,_=publish_joint_checkpoint(model,head,optimizer,scheduler,directory,progress,steps_per_epoch)
            if progress['step']<=2 or progress['step']%10==0:logging.info('seed%d epoch%d step%d loss=%.5f',seed,epoch+1,progress['step'],stats['loss'])
            if smoke and progress['step']==2:return finish_smoke(smoke_config,model,head,tokenizer,optimizer,scheduler,cached,pad,checkpoint,progress,directory,seed,frozen,initial)
        if not smoke:
            with optimizer_offloaded(optimizer,config['model']['device']):
                roundtrip(model,head,tokenizer,train[0],checkpoint,config,directory)
                evaluate_native(f'dev_epoch{epoch+1}',splits['dev'],model,head,tokenizer,directory,config,seed,train)
        frozen_check(model,frozen);progress={'epoch':epoch+1,'group':0,'step':(epoch+1)*steps_per_epoch}
        write_json(directory/'training_memory.json',{'peak_allocated_bytes':peak,'step':progress['step']})
        if torch.device(config['model']['device']).type=='cuda':torch.cuda.reset_peak_memory_stats(config['model']['device'])
    candidates=[]
    for epoch in range(training['epochs']+1):
        path=f'checkpoints/step{epoch*steps_per_epoch:06d}';manifest=validate_joint_checkpoint(directory/path)
        dev=json.loads((directory/f'dev_epoch{epoch}/summary.json').read_text())
        if any(manifest[k]!=dev[k] for k in ('head_sha256','adapter_sha256')):raise ValueError('Dev model/checkpoint joint hashes differ')
        candidates.append({'epoch':epoch,'step':epoch*steps_per_epoch,'hard_nll':dev['metrics']['hard_nll'],'checkpoint':path,
                           'head_sha256':manifest['head_sha256'],'adapter_sha256':manifest['adapter_sha256']})
    selected=dict(select_checkpoint(candidates));write_json(directory/'selection.json',selected)
    write_json(directory/'training_summary.json',{'optimizer_steps':training['epochs']*steps_per_epoch,'dev_candidates':candidates,
               'initial_head_sha256':initial[1],'initial_adapter_sha256':initial[0],'frozen_before':frozen,
               'frozen_after':frozen_check(model,frozen),'training_peak_allocated_bytes':peak})
    return selected


def validate_completed_selections(directory,selections):
    for seed,selected in selections.items():
        seed_dir=directory/f'seed{seed}';saved=json.loads((seed_dir/'selection.json').read_text())
        if saved!=selected:raise ValueError('Completed seed selection changed')
        checkpoint=seed_dir/selected['checkpoint']
        if checkpoint.resolve().parent!=(seed_dir/'checkpoints').resolve():raise ValueError('Invalid selected joint path')
        manifest=validate_joint_checkpoint(checkpoint)
        if any(manifest[k]!=selected[k] for k in ('head_sha256','adapter_sha256')):raise ValueError('Completed seed joint weights changed')


def execute(config,config_path,splits,baseline,smoke=False,resume=None):
    gpu_guard();stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    directory=Path(resume).resolve() if resume else ROOT/config['output_root']/f'{"smoke_train" if smoke else "train"}_{stamp}'
    profile=ROOT/config['output_root']/'preflight_train.json';validate_profile(json.loads(profile.read_text()),config)
    fingerprint=runtime_fingerprint(ROOT/config['training']['fla_package_dir']);fingerprint['device']=device_provenance()
    model_files=b0.model_file_hashes(config['model']['path']);raw_hash=sha256_file(ROOT/config['dataset']['raw_dir']/'download_manifest.json')
    if model_files!=baseline['model_files_sha256']:raise ValueError('Original model files changed from B0')
    if resume:
        metadata=json.loads((directory/'metadata.json').read_text())
        validate_resume(directory,config,metadata,fingerprint,profile,model_files,raw_hash,baseline['split_manifest_sha256'],smoke)
        if not smoke:validate_completed_selections(directory,metadata['completed_seeds'])
    else:
        directory.mkdir(parents=True,exist_ok=False);(directory/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        shutil.copyfile(profile,directory/profile.name);shutil.copytree(ROOT/config['dataset']['split_dir'],directory/'splits')
        metadata={'status':'running','mode':'smoke_train' if smoke else 'train','experiment_id':'p1','started_at':now(),
                  'completed_seeds':{},'test':{},'source_sha256':snapshot_sources(directory),'dataset':config['dataset'],
                  'model_files_sha256':model_files,'raw_manifest_sha256':raw_hash,'split_manifest_sha256':baseline['split_manifest_sha256'],
                  'runtime_fingerprint':fingerprint,'profile_sha256':sha256_file(profile),'config_sha256':sha256_file(directory/'config.yaml'),
                  'config_source':str(config_path.resolve()),'device':fingerprint['device'],
                  **{name:config[name] for name in ('baseline_run','g1_run','g2_run','p0_run')},
                  'protocol':{'backbone':'original frozen BF16 parameters + native FP32 rotary buffers; unmerged FP32 LoRA',
                              'training':'FP32 joint Head + LoRA parameters; BF16 AMP Backbone only; train-only token/position cache',
                              'head':'same P0 shared bias-free 1024->128 projections; independent per-seed fresh P0-matching initialization',
                              'selection':'actual random Head dev0 plus epochs1..3; min hard NLL; all three seeds locked before test',
                              'test':'native uncached end-to-end; optimizer absent; RTX 4090 timing not comparable causally to prior 5090',
                              'calibration_inference':False,'hidden_feature_cache':False,'candidate_order':'original; no order invariance guarantee'}}
    lease=(directory/'.run.lock').open('a')
    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:lease.close();raise RuntimeError('P1 already has a live worker') from None
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(directory/'run.log'),logging.StreamHandler()],force=True)
    (directory/'pid').write_text(str(os.getpid())+'\n');print(f'RUN_DIR={directory}',flush=True)
    try:
        torch.backends.cuda.matmul.fp32_precision='ieee';torch.backends.cudnn.conv.fp32_precision='ieee'
        metadata['status']='training';metadata.pop('error',None);write_json(directory/'metadata.json',metadata)
        for seed in config['seeds'][:1] if smoke else config['seeds']:
            if str(seed) not in metadata['completed_seeds']:
                metadata['completed_seeds'][str(seed)]=train_seed(config,seed,splits,directory/f'seed{seed}',smoke)
                write_json(directory/'metadata.json',metadata);gc.collect();torch.cuda.empty_cache()
        if not smoke:
            validate_completed_selections(directory,metadata['completed_seeds'])
            lock_path=directory/'selections.json';lock_selections(lock_path,metadata['completed_seeds'],config['seeds'])
            require_test_lock(lock_path,config['seeds'],metadata['completed_seeds'])
            metadata['status']='selections_locked';write_json(directory/'metadata.json',metadata)
            for seed in config['seeds']:
                if str(seed) in metadata['test']:
                    saved=json.loads((directory/f'seed{seed}/test/summary.json').read_text())
                    if saved!=metadata['test'][str(seed)]:raise ValueError('Completed test summary changed')
                    continue
                model,head,tokenizer,loading=create_pair(config,seed);validate_counts(config,model,head)
                selected=metadata['completed_seeds'][str(seed)];manifest=load_joint_weights(model,head,directory/f'seed{seed}'/selected['checkpoint'])
                if any(manifest[k]!=selected[k] for k in ('head_sha256','adapter_sha256')):raise ValueError('Locked joint selection changed')
                frozen=json.loads((directory/f'seed{seed}/model.json').read_text())['frozen_backbone'];frozen_check(model,frozen)
                metadata['status']='evaluating';write_json(directory/'metadata.json',metadata)
                metadata['test'][str(seed)]=evaluate_native('test',splits['test'],model,head,tokenizer,directory/f'seed{seed}',config,seed,splits['train'],lock_path)
                frozen_check(model,frozen);write_json(directory/'metadata.json',metadata)
                del model,head,tokenizer;gc.collect();torch.cuda.empty_cache()
            rows=[metadata['test'][str(seed)]['metrics'] for seed in config['seeds']]
            aggregate={key:{'mean':float(np.mean([row[key] for row in rows])),
                            'sample_std':float(np.std([row[key] for row in rows],ddof=1))}
                       for key in rows[0] if isinstance(rows[0][key],(int,float)) and not key.startswith('n_')}
            candidates=[read_rows(directory/f'seed{seed}/test/predictions.jsonl') for seed in config['seeds']]
            summary={'per_seed':metadata['test'],'aggregate':aggregate}
            for name,label in [('baseline_run','b0'),('g1_run','g1'),('g2_run','g2'),('p0_run','p0')]:
                paths=[ROOT/config[name]/'test/predictions.jsonl'] if label=='b0' else [ROOT/config[name]/f'seed{seed}/test/predictions.jsonl' for seed in config['seeds']]
                value=paired_seed_bootstrap([read_rows(path) for path in paths],candidates,config['evaluation']['bootstrap_samples'])
                value['definition']='Mean P1 seed correctness minus mean reference seed correctness; no probability ensemble; paired case percentile 95% CI'
                summary[f'paired_accuracy_delta_vs_{label}']=value
            write_json(directory/'summary.json',summary)
        metadata['status']='complete'
    except BaseException as error:
        metadata['status']='failed';metadata['error']={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
        logging.exception('P1 stopped; complete joint Head/adapter generations preserved');raise
    finally:
        metadata['updated_at']=now();write_json(directory/'metadata.json',metadata);fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
    return directory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'experiments/02_supervised/configs/p1.yaml')
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if args.prepare_only and (args.smoke or args.resume):parser.error('--prepare-only cannot combine with smoke/resume')
    config_path=args.resume/'config.yaml' if args.resume else args.config
    config=yaml.safe_load(config_path.read_text());splits,baseline=prepare(config)
    if args.prepare_only:
        print(json.dumps({'counts':{k:len(v) for k,v in splits.items()},'optimizer_steps_per_seed':450,
                          'trainable_parameters':11084800,'head_parameters':262144,'lora_parameters':10822656,
                          'training_backend':'fla','evaluation_backend':'native','host':'alg26','physical_gpu':0}));return
    execute(config,config_path,splits,baseline,args.smoke,args.resume)


if __name__=='__main__':main()

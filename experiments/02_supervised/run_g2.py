#!/usr/bin/env python3
"""G2: full text Backbone + tied LM Head, hard labels, three dev-selected seeds."""
import argparse
from datetime import datetime
import fcntl
import gc
import hashlib
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
from jev.data.typed_decisions import iter_decisions,load_cases,prepare_splits,sha256_file
from jev.models.lm_readout import load_local_model,prepare_case,read_case,synchronize
from jev.training.full import (enable_full_training,assert_full_trainable,parameter_hash,create_eval_copy,
    validate_full_checkpoint,load_full_weights,restore_full_checkpoint,offload_training_state,restore_training_state)
from jev.training.full_runner import validate_config,validate_profile,train_update,publish_checkpoint,reconcile_history,paired_seed_bootstrap
from jev.training.lora import cache_decisions
from jev.training.runner import (epoch_groups,make_scheduler,lock_selections,preserve_runtime_provenance,
    runtime_fingerprint,write_json,read_rows,select_checkpoint)
spec=importlib.util.spec_from_file_location('jev_g2_b0',ROOT/'experiments/01_baseline/run.py')
b0=importlib.util.module_from_spec(spec);spec.loader.exec_module(b0)


def now(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def prepare(config):
    validate_config(config)
    raw=ROOT/config['dataset']['raw_dir']; b0.validate_source(raw,config['dataset'])
    splits=prepare_splits(load_cases(raw),ROOT/config['dataset']['split_dir'],seed=42,source_manifest=raw/'download_manifest.json')
    baseline=ROOT/config['baseline_run']; previous=json.loads((baseline/'metadata.json').read_text())
    baseline_config=yaml.safe_load((baseline/'config.yaml').read_text())
    if previous['status']!='complete' or previous['dataset']!=config['dataset'] or baseline_config['model']!=config['model']:
        raise ValueError('G2 source/model must match completed pinned B0')
    if previous['split_manifest_sha256']!=sha256_file(ROOT/config['dataset']['split_dir']/'manifest.json'):
        raise ValueError('G2/B0 split provenance differs')
    for source in ['src/jev/data/typed_decisions.py','src/jev/models/lm_readout.py','src/jev/evaluation/metrics.py']:
        if previous['source_sha256'][source]!=sha256_file(ROOT/source): raise ValueError(f'B0 scientific source changed: {source}')
    if {key:len(value) for key,value in splits.items()}!={'train':960,'dev':120,'calibration':120,'test':400}:
        raise ValueError('Unexpected split sizes')
    g1=json.loads((ROOT/config['g1_run']/'metadata.json').read_text())
    if g1['status']!='complete' or g1['dataset']!=config['dataset'] or g1['split_manifest_sha256']!=previous['split_manifest_sha256']:
        raise ValueError('G2 comparison requires completed G1 with identical source/splits')
    return splits,previous


def sources(directory):
    paths=sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'tests').glob('test_*.py'))
    paths += [Path(__file__),ROOT/'experiments/01_baseline/run.py',ROOT/'experiments/02_supervised/configs/g2.yaml',ROOT/'experiments/02_supervised/profile_g2.py']
    hashes={}
    for path in paths:
        if not path.is_file(): raise ValueError(f'Missing scientific source: {path}')
        relative=path.relative_to(ROOT); target=directory/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target);hashes[str(relative)]=sha256_file(target)
    return hashes


def backend(model,config,kind):
    from jev.training.kernels import set_delta_backend
    return set_delta_backend(model,kind,(ROOT/config['training']['fla_package_dir']).resolve())


def memory(device):
    return {'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
            'current_allocated_bytes':torch.cuda.memory_allocated(device)}


def evaluate(name,cases,model,tokenizer,directory,config,seed,warmup):
    backend(model,config,'native');model.eval()
    for case in warmup: read_case(model,prepare_case(case['decisions'],tokenizer))
    synchronize(model.device);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(model.device)
    value=b0.run_split(name,cases,model,tokenizer,directory,{**config,'seed':seed})
    value['memory']=memory(model.device)
    value['memory_definition']='Native BF16 evaluation copy; FP32 training masters and optimizer reside on CPU.'
    write_json(directory/name/'summary.json',value)
    return value


def probe_logits(model,tokenizer,probe,config):
    backend(model,config,'native');model.eval()
    return read_case(model,prepare_case(probe['decisions'],tokenizer))['label_logits']


def isolated_probe(master,optimizer,tokenizer,probe,config):
    before=parameter_hash(master);ids=[id(p) for p in master.parameters()]
    token=offload_training_state(master,optimizer);torch.cuda.empty_cache()
    evaluation=None
    try:
        evaluation=create_eval_copy(master,device=config['model']['device'])
        value=probe_logits(evaluation,tokenizer,probe,config)
    finally:
        del evaluation;gc.collect();torch.cuda.empty_cache();restore_training_state(master,optimizer,token)
    if parameter_hash(master)!=before or ids!=[id(p) for p in master.parameters()]:
        raise RuntimeError('BF16 evaluation changed FP32 training masters or optimizer parameter identities')
    return value


def isolated_dev(master,optimizer,tokenizer,epoch,splits,directory,config,seed):
    name=f'dev_epoch{epoch}';dev=directory/name
    if dev.exists():
        if not (dev/'summary.json').is_file(): raise RuntimeError(f'Incomplete held-out evaluation, refusing replay: {dev}')
        return
    before=parameter_hash(master);ids=[id(p) for p in master.parameters()]
    token=offload_training_state(master,optimizer);torch.cuda.empty_cache();evaluation=None
    try:
        evaluation=create_eval_copy(master,device=config['model']['device'])
        evaluate(name,splits['dev'],evaluation,tokenizer,directory,config,seed,
                 b0.select_warmup(splits['train'],config['evaluation']['warmup_cases']))
    finally:
        del evaluation;gc.collect();torch.cuda.empty_cache();restore_training_state(master,optimizer,token)
    if parameter_hash(master)!=before or ids!=[id(p) for p in master.parameters()]:
        raise RuntimeError('Dev evaluation changed FP32 masters or parameter identities')
    write_json(directory/f'eval_integrity_epoch{epoch}.json',{'master_sha256_before':before,'master_sha256_after':parameter_hash(master),
                                                           'optimizer_parameter_identities_preserved':True})
    backend(master,config,config['training']['kernel_backend']);master.train()


def verify_smoke(model,optimizer,scheduler,tokenizer,probe,config,directory,cached,pad,next_indices,original_hash,peak):
    training=config['training']
    checkpoint=directory/json.loads((directory/'latest.json').read_text())['checkpoint']
    before=parameter_hash(model);logits=isolated_probe(model,optimizer,tokenizer,probe,config)
    restore_full_checkpoint(model,optimizer,scheduler,checkpoint)
    reloaded=isolated_probe(model,optimizer,tokenizer,probe,config)
    if before!=parameter_hash(model) or logits!=reloaded: raise RuntimeError('Lossless full checkpoint train reload failed')
    backend(model,config,training['kernel_backend'])
    # A real fused Adam update after restore verifies state-device continuation.
    following=train_update(model,cached,next_indices,pad,optimizer,scheduler,training['microbatch_questions'],training['max_grad_norm'])
    if parameter_hash(model)==before or before==original_hash: raise RuntimeError('Full weights failed to update')
    write_json(directory/'smoke.json',{'train_only':True,'optimizer_updates':3,'checkpoint_step':2,
               'fp32_checkpoint_bitexact':True,'bf16_reload_train_logits_equal':True,
               'fused_resume_update_passed':True,'original_weights_changed':True,'after_resume_loss':following['loss'],
               'training_peak_allocated_bytes':max(peak,torch.cuda.max_memory_allocated(model.device))})
    return {'smoke':True}


def train_seed(config,seed,splits,directory,baseline,smoke):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.fp32_precision='ieee';torch.backends.cudnn.conv.fp32_precision='ieee'
    directory.mkdir(exist_ok=True)
    # A partial held-out directory is never silently replayed, including on resume.
    for dev in directory.glob('dev_epoch*'):
        marker='reused_from_b0.json' if dev.name=='dev_epoch0' else 'summary.json'
        if not (dev/marker).is_file(): raise RuntimeError(f'Incomplete held-out evaluation: {dev}')
    model,tokenizer,loading=load_local_model(config['model']);probe=splits['train'][0]
    original_logits=probe_logits(model,tokenizer,probe,config)
    enable_full_training(model,config['training']['expected_parameters'])
    original_hash=parameter_hash(model);training=config['training']
    optimizer=torch.optim.AdamW(model.parameters(),lr=training['learning_rate'],weight_decay=training['weight_decay'],
                               betas=tuple(training['betas']),eps=training['eps'],fused=training['fused'])
    zero_logits=isolated_probe(model,optimizer,tokenizer,probe,config)
    if original_logits!=zero_logits: raise RuntimeError('Original/native BF16 evaluation-copy train parity failed')
    initial={'kind':'original_model_reference','parameter_sha256':original_hash,'train_case_id':probe['case_id'],
             'native_bf16_train_parity':True,'model_files_sha256':baseline['model_files_sha256']}
    reference=directory/'original_reference.json'
    if reference.exists() and json.loads(reference.read_text())!=initial: raise ValueError('Original reference changed')
    if not reference.exists(): write_json(reference,initial)
    model_info={**assert_full_trainable(model,training['expected_parameters']),'loading_info':loading,
                'master_dtype':'float32','autocast_dtype':'bfloat16','original_parameter_sha256':original_hash,
                'zero_step_train_parity':True,'head_and_embedding_tied':model.get_input_embeddings().weight is model.get_output_embeddings().weight}
    write_json(directory/'model.json',model_info)
    cached=cache_decisions(list(iter_decisions(splits['train'])),tokenizer,config['evaluation']['max_input_tokens'])
    if len(cached)!=4800: raise ValueError('Expected 4800 training decisions')
    write_json(directory/'tokenization.json',{'decisions':len(cached),'max_tokens':max(len(row['input_ids']) for row in cached),
               'cache_identity_sha256':hashlib.sha256(json.dumps(cached,sort_keys=True).encode()).hexdigest()})
    steps_per_epoch=math.ceil(len(cached)/training['effective_batch_questions']);total=training['epochs']*steps_per_epoch
    scheduler=make_scheduler(optimizer,total,training['warmup_fraction'])
    progress={'epoch':0,'group':0,'step':0};latest=directory/'latest.json'
    if latest.exists():
        point=json.loads(latest.read_text());checkpoint=directory/point['checkpoint']
        if checkpoint.resolve().parent!=(directory/'checkpoints').resolve(): raise ValueError('Invalid latest checkpoint path')
        progress=restore_full_checkpoint(model,optimizer,scheduler,checkpoint)
        if progress['step']!=point['step'] or parameter_hash(model)!=point['parameter_sha256']: raise ValueError('Latest pointer generation differs')
    reconcile_history(directory/'train.jsonl',progress['step'])
    if not smoke and not (directory/'dev_epoch0').exists():
        shutil.copytree(ROOT/config['baseline_run']/'dev',directory/'dev_epoch0')
        write_json(directory/'dev_epoch0/reused_from_b0.json',{'baseline_run':config['baseline_run'],
                   'native_bf16_train_parity':True,'metrics_sha256':sha256_file(ROOT/config['baseline_run']/'dev/metrics.json')})
    write_json(directory/'training_backend.json',backend(model,config,training['kernel_backend']))
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(model.device)
    peak_path=directory/'training_memory.json';peak=json.loads(peak_path.read_text())['peak_allocated_bytes'] if peak_path.exists() else 0
    pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if smoke and progress['step']>=2:
        if progress['step']!=2: raise ValueError('Unexpected smoke recovery checkpoint step')
        next_indices=epoch_groups(len(cached),training['effective_batch_questions'],seed,progress['epoch'])[progress['group']]
        result=verify_smoke(model,optimizer,scheduler,tokenizer,probe,config,directory,cached,pad,next_indices,original_hash,peak)
        del scheduler,optimizer,model;gc.collect();torch.cuda.empty_cache();return result
    for epoch in range(progress['epoch'],training['epochs']):
        groups=epoch_groups(len(cached),training['effective_batch_questions'],seed,epoch)
        start=progress['group'] if epoch==progress['epoch'] else 0
        for group in range(start,len(groups)):
            started=time.perf_counter()
            stats=train_update(model,cached,groups[group],pad,optimizer,scheduler,training['microbatch_questions'],training['max_grad_norm'])
            synchronize(model.device);progress={'epoch':epoch,'group':group+1,'step':epoch*steps_per_epoch+group+1}
            stats.update({**progress,'seconds':time.perf_counter()-started})
            with (directory/'train.jsonl').open('a') as stream:
                stream.write(json.dumps(stats,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
            peak=max(peak,torch.cuda.max_memory_allocated(model.device))
            write_json(peak_path,{'peak_allocated_bytes':peak,'step':progress['step']})
            if progress['step']==2 or progress['step']%training['checkpoint_every_steps']==0 or progress['step']%steps_per_epoch==0:
                checkpoint,manifest=publish_checkpoint(model,optimizer,scheduler,directory,progress,steps_per_epoch)
                logging.info('seed%d complete recovery checkpoint step%d %s',seed,progress['step'],checkpoint)
            if progress['step']%10==0 or progress['step']<=2 or smoke:
                logging.info('seed%d epoch%d step%d/%d loss=%.5f seconds=%.2f',seed,epoch+1,progress['step'],total,stats['loss'],stats['seconds'])
            if smoke and progress['step']>=2:
                result=verify_smoke(model,optimizer,scheduler,tokenizer,probe,config,directory,cached,pad,groups[group+1],original_hash,peak)
                del scheduler,optimizer,model;gc.collect();torch.cuda.empty_cache();return result
        isolated_dev(model,optimizer,tokenizer,epoch+1,splits,directory,config,seed)
        progress={'epoch':epoch+1,'group':0,'step':(epoch+1)*steps_per_epoch}
        torch.cuda.reset_peak_memory_stats(model.device)
    final_hash=parameter_hash(model)
    if final_hash==original_hash: raise RuntimeError('Full training did not modify original weights')
    candidates=[]
    for epoch in range(training['epochs']+1):
        metric=json.loads((directory/f'dev_epoch{epoch}/metrics.json').read_text())['overall']['hard_nll']
        checkpoint=f'checkpoints/step{epoch*steps_per_epoch:06d}' if epoch else 'original_reference.json'
        digest=validate_full_checkpoint(directory/checkpoint)['parameter_sha256'] if epoch else original_hash
        candidates.append({'epoch':epoch,'step':epoch*steps_per_epoch,'hard_nll':metric,'checkpoint':checkpoint,'parameter_sha256':digest})
    selected=dict(select_checkpoint(candidates));write_json(directory/'selection.json',selected)
    write_json(directory/'training_summary.json',{'original_parameter_sha256':original_hash,'final_parameter_sha256':final_hash,
               'optimizer_steps':total,'training_peak_allocated_bytes':peak,'dev_candidates':candidates})
    del scheduler,optimizer,model;gc.collect();torch.cuda.empty_cache();return selected


def selected_eval_model(config,selection,directory):
    model,tokenizer,_=load_local_model(config['model'])
    if selection['step']==0:
        reference=json.loads((directory/'original_reference.json').read_text())
        if reference['parameter_sha256']!=selection['parameter_sha256']: raise ValueError('Locked original reference changed')
        return model,tokenizer
    model.to('cpu');torch.cuda.empty_cache();enable_full_training(model,config['training']['expected_parameters'])
    manifest=load_full_weights(model,directory/selection['checkpoint'])
    if manifest['parameter_sha256']!=selection['parameter_sha256']: raise ValueError('Locked selected full weights changed')
    evaluation=create_eval_copy(model,device=config['model']['device']);del model;gc.collect()
    return evaluation,tokenizer


def execute(config,config_path,splits,baseline,smoke=False,resume=None):
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    directory=Path(resume).resolve() if resume else ROOT/config['output_root']/f'{"smoke_train" if smoke else "train"}_{stamp}'
    fingerprint=runtime_fingerprint(ROOT/config['training']['fla_package_dir']);profile=ROOT/config['output_root']/'kernel_profile_train.json'
    validate_profile(json.loads(profile.read_text()),config)
    files=b0.model_file_hashes(config['model']['path'])
    if files!=baseline['model_files_sha256']: raise ValueError('Original local model files differ from B0')
    raw_hash=sha256_file(ROOT/config['dataset']['raw_dir']/'download_manifest.json')
    if resume:
        metadata=json.loads((directory/'metadata.json').read_text())
        if yaml.safe_load((directory/'config.yaml').read_text())!=config: raise ValueError('Resume config differs')
        for source,expected in metadata['source_sha256'].items():
            if sha256_file(ROOT/source)!=expected or sha256_file(directory/'source'/source)!=expected:
                raise ValueError(f'Resume source changed: {source}')
        if metadata['mode']!=('smoke_train' if smoke else 'train') or metadata['status']=='complete': raise ValueError('Resume mode/status invalid')
        if metadata['model_files_sha256']!=files or metadata['raw_manifest_sha256']!=raw_hash: raise ValueError('Resume data/model provenance changed')
        if metadata['split_manifest_sha256']!=baseline['split_manifest_sha256']: raise ValueError('Resume splits changed')
        preserve_runtime_provenance(directory,profile,fingerprint,metadata,resume=True)
    else:
        directory.mkdir(parents=True,exist_ok=False);(directory/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        metadata={'status':'running','mode':'smoke_train' if smoke else 'train','started_at':now(),'source_sha256':sources(directory),
                  'completed_seeds':{},'test':{},'baseline_run':config['baseline_run'],'g1_run':config['g1_run'],'dataset':config['dataset'],
                  'model_files_sha256':files,'raw_manifest_sha256':raw_hash,'split_manifest_sha256':baseline['split_manifest_sha256'],
                  'config_sha256':sha256_file(directory/'config.yaml'),'config_source':str(config_path.resolve()),
                  'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
        shutil.copytree(ROOT/config['dataset']['split_dir'],directory/'splits')
        preserve_runtime_provenance(directory,profile,fingerprint,metadata)
    lease=(directory/'.run.lock').open('a')
    try: fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: lease.close();raise RuntimeError('This run already has a live worker') from None
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.FileHandler(directory/'run.log'),logging.StreamHandler()],force=True)
    (directory/'pid').write_text(str(os.getpid())+'\n');print(f'RUN_DIR={directory}',flush=True)
    try:
        metadata['device']={'name':torch.cuda.get_device_name(config['model']['device']),'logical':config['model']['device']}
        metadata['status']='training';metadata.pop('error',None);write_json(directory/'metadata.json',metadata)
        for seed in config['seeds'][:1] if smoke else config['seeds']:
            if str(seed) not in metadata['completed_seeds']:
                metadata['completed_seeds'][str(seed)]=train_seed(config,seed,splits,directory/f'seed{seed}',baseline,smoke)
                write_json(directory/'metadata.json',metadata)
        if not smoke:
            lock_selections(directory/'selections.json',metadata['completed_seeds'],config['seeds'])
            metadata['status']='selections_locked';write_json(directory/'metadata.json',metadata)
            for seed in config['seeds']:
                if str(seed) in metadata['test']: continue
                seed_dir=directory/f'seed{seed}'
                if (seed_dir/'test').exists():
                    if not (seed_dir/'test/summary.json').is_file(): raise RuntimeError('Partial test exists; refusing official inference replay')
                    metadata['test'][str(seed)]=json.loads((seed_dir/'test/summary.json').read_text());continue
                model,tokenizer=selected_eval_model(config,metadata['completed_seeds'][str(seed)],seed_dir)
                metadata['status']='evaluating';write_json(directory/'metadata.json',metadata)
                metadata['test'][str(seed)]=evaluate('test',splits['test'],model,tokenizer,seed_dir,config,seed,
                    b0.select_warmup(splits['train'],config['evaluation']['warmup_cases']))
                write_json(directory/'metadata.json',metadata);del model;gc.collect();torch.cuda.empty_cache()
            metrics=[metadata['test'][str(seed)]['metrics'] for seed in config['seeds']]
            aggregate={key:{'mean':float(np.mean([row[key] for row in metrics])),'sample_std':float(np.std([row[key] for row in metrics],ddof=1))}
                       for key in metrics[0] if isinstance(metrics[0][key],(int,float)) and not key.startswith('n_')}
            candidates=[read_rows(directory/f'seed{seed}/test/predictions.jsonl') for seed in config['seeds']]
            reference=[read_rows(ROOT/config['baseline_run']/'test/predictions.jsonl')]
            g1=[read_rows(ROOT/config['g1_run']/f'seed{seed}/test/predictions.jsonl') for seed in config['seeds']]
            write_json(directory/'summary.json',{'per_seed':metadata['test'],'aggregate':aggregate,
                       'paired_accuracy_delta_vs_b0':paired_seed_bootstrap(reference,candidates,config['evaluation']['bootstrap_samples']),
                       'paired_accuracy_delta_vs_g1':paired_seed_bootstrap(g1,candidates,config['evaluation']['bootstrap_samples'])})
        metadata['status']='complete'
    except BaseException as error:
        metadata['status']='failed';metadata['error']={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
        logging.exception('G2 stopped; complete recovery generations preserved');raise
    finally:
        metadata['updated_at']=now();write_json(directory/'metadata.json',metadata);fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
    return directory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'experiments/02_supervised/configs/g2.yaml')
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',type=Path);args=parser.parse_args()
    if args.prepare_only and (args.smoke or args.resume): parser.error('--prepare-only cannot combine --smoke/--resume')
    config_path=args.resume/'config.yaml' if args.resume else args.config
    config=yaml.safe_load(config_path.read_text());splits,baseline=prepare(config)
    if args.prepare_only:
        print(json.dumps({'counts':{key:len(value) for key,value in splits.items()},'train_decisions':4800,'optimizer_steps_per_seed':450,'baseline_run':config['baseline_run']}));return
    execute(config,config_path,splits,baseline,args.smoke,args.resume)

if __name__=='__main__': main()

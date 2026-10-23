"""G2 fixed protocol, atomic recovery pointers and seed-mean comparisons."""
from contextlib import nullcontext
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import numpy as np
import torch
from jev.data.typed_decisions import stable_argmax
from jev.training.full import save_full_checkpoint, validate_full_checkpoint, parameter_hash
from jev.training.lora import candidate_logits, collate_decisions, hard_candidate_loss
from jev.training.runner import write_json


def validate_config(config):
    if config.get('experiment_id') != 'g2' or config.get('seeds') != [42,43,44] or 'lora' in config:
        raise ValueError('G2 requires three fixed seeds and no LoRA')
    fixed = {'training': {'epochs':3,'effective_batch_questions':32,'learning_rate':2e-5,'weight_decay':.01,
             'max_grad_norm':1.,'warmup_fraction':.05,'scheduler':'cosine','gradient_checkpointing':False,
             'loss':'hard_candidate_cross_entropy','select_metric':'dev_hard_nll','eval_each_epoch':True,
             'include_step_zero':True,'kernel_backend':'fla','master_dtype':'float32','autocast_dtype':'bfloat16',
             'expected_parameters':752393024,'fused':True,'betas':[.9,.999],'eps':1e-8,'checkpoint_every_steps':50},
             'evaluation': {'splits':['dev','test'],'max_input_tokens':8192,'warmup_cases':3,'case_batch_size':1,
             'ece_bins':15,'temperature':1.,'use_cache':False,'prompt_version':'b0_v1','kernel_backend':'native',
             'merge_adapter':False}}
    for group, values in fixed.items():
        for key,value in values.items():
            if config.get(group,{}).get(key) != value: raise ValueError(f'G2 requires {group}.{key}={value!r}')
    if (config['training']['microbatch_questions'],config['training']['gradient_accumulation']) not in [(2,16),(4,8)]:
        raise ValueError('G2 requires profiled micro2/acc16 or micro4/acc8')


def validate_profile(profile,config):
    if profile.get('status') != 'passed' or profile.get('train_only') is not True:
        raise ValueError('G2 requires passed train-only profile')
    for key,configured in [('selected_microbatch','microbatch_questions'),('selected_accumulation','gradient_accumulation')]:
        if profile.get(key) != config['training'][configured]: raise ValueError(f'Profile {key} differs from configuration')


def train_update(model,cached,indices,pad,optimizer,scheduler,microbatch,max_grad_norm):
    model.train(); optimizer.zero_grad(set_to_none=True)
    context=(lambda:torch.autocast('cuda',dtype=torch.bfloat16)) if model.device.type=='cuda' else nullcontext
    total=0.
    for start in range(0,len(indices),microbatch):
        rows=[cached[i] for i in indices[start:start+microbatch]]
        batch=collate_decisions(rows,pad)
        with context(): loss=hard_candidate_loss(candidate_logits(model,batch),batch['option_counts'],batch['hard_targets'])
        weight=len(rows)/len(indices); (loss*weight).backward(); total+=float(loss.detach())*weight
    params=list(model.parameters())
    if any(p.grad is None for p in params): raise RuntimeError('Missing full-model gradient')
    norm=torch.nn.utils.clip_grad_norm_(params,max_grad_norm,error_if_nonfinite=True)
    lr=optimizer.param_groups[0]['lr']; optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    return {'loss':total,'grad_norm':float(norm),'learning_rate':lr,'questions':len(indices)}


def publish_checkpoint(model,optimizer,scheduler,directory,progress,steps_per_epoch):
    directory=Path(directory); checkpoint=directory/f'checkpoints/step{progress["step"]:06d}'
    if checkpoint.exists():
        manifest=validate_full_checkpoint(checkpoint)
        if manifest['progress']!=progress or manifest['parameter_sha256']!=parameter_hash(model):
            raise ValueError('Orphan checkpoint differs from replayed optimizer generation')
    else: manifest=save_full_checkpoint(model,optimizer,scheduler,checkpoint,progress)
    write_json(directory/'latest.json',{'checkpoint':str(checkpoint.relative_to(directory)),**progress,
                                       'parameter_sha256':manifest['parameter_sha256']})
    # Only validated full G2 generations under this run are eligible for pruning.
    for old in checkpoint.parent.glob('step[0-9][0-9][0-9][0-9][0-9][0-9]'):
        step=int(old.name[4:])
        if old!=checkpoint and step%steps_per_epoch and step<progress['step']:
            validate_full_checkpoint(old); shutil.rmtree(old)
    return checkpoint,manifest


def reconcile_history(path,committed_step):
    path=Path(path)
    if not path.exists():
        if committed_step: raise ValueError('Committed checkpoint is missing training history')
        return
    lines=path.read_text().splitlines(keepends=True); rows=[]; tail=[]
    for line in lines:
        try: row=json.loads(line)
        except json.JSONDecodeError:
            tail.append(line); continue
        if row.get('step',0)<=committed_step and not tail: rows.append((row,line))
        else: tail.append(line)
    if [row['step'] for row,_ in rows]!=list(range(1,committed_step+1)):
        raise ValueError('Training history does not cover committed checkpoint steps')
    if tail:
        archive=path.with_name(f'train.uncommitted.{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.jsonl')
        archive.write_text(''.join(tail))
        temporary=path.with_suffix('.jsonl.tmp'); temporary.write_text(''.join(line for _,line in rows)); temporary.replace(path)


def paired_seed_bootstrap(reference,candidates,samples=2000,seed=42):
    def keyed(rows):
        result={}
        for row in rows:
            key=(row['case_id'],row['question_id'])
            if key in result: raise ValueError('Duplicate prediction identity')
            result[key]=row
        return result
    if not reference or not candidates: raise ValueError('Both seed groups are required')
    base=[keyed(rows) for rows in reference]; other=[keyed(rows) for rows in candidates]; keys=set(base[0])
    if not keys or any(set(mapping)!=keys for mapping in base+other): raise ValueError('Prediction identities differ')
    cases={}
    for key in keys:
        rows=[mapping[key] for mapping in base+other]; canonical=rows[0]
        if any((row['option_ids'],row['hard_target'])!=(canonical['option_ids'],canonical['hard_target']) for row in rows):
            raise ValueError('Candidate/target identities differ')
        def correct(row): return float(stable_argmax(row['probabilities'],row['option_ids'])==row['hard_target'])
        delta=np.mean([correct(mapping[key]) for mapping in other])-np.mean([correct(mapping[key]) for mapping in base])
        cases.setdefault(key[0],[]).append(float(delta))
    values=np.array([sum(cases[key]) for key in sorted(cases)]); counts=np.array([len(cases[key]) for key in sorted(cases)])
    distribution=[]; rng=np.random.default_rng(seed)
    for start in range(0,samples,100):
        index=rng.integers(0,len(values),(min(100,samples-start),len(values)))
        distribution.extend((values[index].sum(1)/counts[index].sum(1)).tolist())
    low,high=np.quantile(distribution,[.025,.975])
    return {'mean_accuracy_delta':float(values.sum()/counts.sum()),'low':float(low),'high':float(high),
            'n_cases':len(values),'samples':samples,'seed':seed,'unit':'paired case',
            'definition':'Mean seed correctness G2 minus mean seed correctness reference; no probability ensemble; percentile 95% CI'}

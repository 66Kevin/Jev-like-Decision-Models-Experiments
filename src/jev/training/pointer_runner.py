"""P0 frozen-feature caches, Head-only optimization and immutable recovery."""
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from jev.data.typed_decisions import sha256_file
from jev.models.pointer import prepare_pointer_case, extract_pointer_features
from jev.models.lm_readout import label_probabilities, synchronize
from jev.training.lora import hard_candidate_loss
from jev.training.runner import rng_state, write_json


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def runtime_fingerprint():
    versions={}
    for name in ('torch','transformers','numpy','PyYAML','tokenizers','safetensors','huggingface-hub','accelerate'):
        versions[name]=importlib.metadata.version(name)
    return {'python':{'version':sys.version,'executable':str(Path(sys.executable).resolve()),
                      'implementation':platform.python_implementation()},'versions':versions,
            'torch_build':{'version':str(torch.__version__),'cuda':torch.version.cuda,
                           'git_version':torch.version.git_version,'config':torch.__config__.show()}}


def validate_config(config):
    fixed={'experiment_id':'p0','seeds':[42,43,44]}
    groups={'head':{'hidden_size':1024,'projection_dim':128,'max_candidates':5,'expected_parameters':262144},
            'training':{'epochs':3,'effective_batch_questions':32,'microbatch_questions':32,
                        'gradient_accumulation':1,'learning_rate':1e-3,'weight_decay':.01,
                        'betas':[.9,.999],'eps':1e-8,'max_grad_norm':1.,'warmup_fraction':.05,
                        'scheduler':'cosine','loss':'hard_candidate_cross_entropy',
                        'select_metric':'dev_hard_nll','kernel_backend':'native',
                        'checkpoint_every_steps':50,'expected_frozen_parameters':752393024},
            'evaluation':{'splits':['dev','test'],'max_input_tokens':8192,'warmup_cases':3,
                          'case_batch_size':1,'ece_bins':15,'temperature':1.,'use_cache':False,
                          'prompt_version':'b0_v1','kernel_backend':'native'}}
    for key,value in fixed.items():
        if config.get(key)!=value:raise ValueError(f'P0 requires {key}={value!r}')
    for group,mapping in groups.items():
        for key,value in mapping.items():
            if config.get(group,{}).get(key)!=value:raise ValueError(f'P0 requires {group}.{key}={value!r}')
    if 'lora' in config:raise ValueError('P0 does not use LoRA')


def build_cache_key(config,source_hashes,model_hashes,split_hash):
    return {'schema':'p0_feature_v1','model':config['model'],'model_files_sha256':model_hashes,
            'source_sha256':source_hashes,'split_manifest_sha256':split_hash,
            'runtime':runtime_fingerprint(),'backend':'native','original_weight_dtype':'bfloat16',
            'features_dtype':'float32 lossless from bfloat16','head_input_size':config['head']['hidden_size'],
            'max_candidates':config['head']['max_candidates'],
            'pooling':'last assistant prefix token; candidate final nonwhitespace character token',
            'batch_policy':'one case; five independent rows; left padding; mask positions',
            'max_input_tokens':config['evaluation']['max_input_tokens'],'prompt_version':'b0_v1'}


def feature_records(cases):
    return [{'case_id':d['case_id'],'question_id':d['question_id'],'workflow':d['workflow'],
             'option_ids':d['option_ids'],'option_texts':d['option_texts']} for case in cases for d in case['decisions']]


def assert_frozen(model,expected=None):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('Original Backbone must stay frozen in eval mode')
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Original Backbone acquired a gradient')
    digest=hashlib.sha256()
    for kind,items in [('parameter',model.named_parameters()),('buffer',model.named_buffers())]:
        for name,tensor in items:
            digest.update(f'{kind}:{name}:{tensor.dtype}:{list(tensor.shape)}'.encode())
            flat=tensor.detach().reshape(-1)
            for start in range(0,flat.numel(),1024*1024):
                digest.update(flat[start:start+1024*1024].contiguous().cpu().view(torch.uint8).numpy().tobytes())
    tied=None
    if hasattr(model,'get_input_embeddings'):
        tied=model.get_input_embeddings().weight is model.get_output_embeddings().weight
        if not tied:raise RuntimeError('Original embedding/LM Head tie changed')
    value={'state_sha256':digest.hexdigest(),'unique_parameters':sum(p.numel() for p in model.parameters()),
           'head_and_embedding_tied':tied,'parameter_dtypes':sorted({str(p.dtype) for p in model.parameters()}),
           'buffer_dtypes':sorted({str(b.dtype) for b in model.buffers()})}
    if expected is not None and value!=expected:raise RuntimeError('Frozen Backbone parameters/buffers changed')
    return value


def load_feature_cache(directory,key,cases):
    directory=Path(directory)
    manifest_path=directory/'manifest.json'
    if not manifest_path.is_file():raise RuntimeError(f'Incomplete feature cache: {directory}')
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('complete') is not True:raise ValueError('Incomplete feature cache manifest')
    if manifest.get('key')!=key:raise ValueError('Feature cache key differs')
    if set(manifest.get('files_sha256',{}))!={'features.npz','records.json'}:
        raise ValueError('Missing feature cache checksum entries')
    for name,digest in manifest['files_sha256'].items():
        if name not in ('features.npz','records.json') or sha256_file(directory/name)!=digest:
            raise ValueError('Feature cache checksum mismatch')
    records=json.loads((directory/'records.json').read_text())
    identities=feature_records(cases)
    if [{k:row[k] for k in identities[0]} for row in records]!=identities:
        raise ValueError('Feature cache identities differ')
    with np.load(directory/'features.npz',allow_pickle=False) as arrays:
        expected={'query_features','candidate_features','candidate_mask','option_counts',
                  'query_positions','candidate_positions','input_lengths'}
        if set(arrays.files)!=expected:raise ValueError('Feature cache tensor keys differ')
        result={name:torch.from_numpy(arrays[name].copy()) for name in arrays.files}
    n=len(records)
    if not n or n%5 or manifest['n_decisions']!=n or manifest['n_cases']!=len(cases):
        raise ValueError('Feature cache case/decision counts differ')
    hidden=key.get('head_input_size',result['query_features'].shape[-1])
    width=key.get('max_candidates',5)
    shapes={'query_features':(n,hidden),'candidate_features':(n,width,hidden),
            'candidate_mask':(n,width),'option_counts':(n,), 'query_positions':(n,),
            'candidate_positions':(n,width),'input_lengths':(n,)}
    for name,shape in shapes.items():
        if tuple(result[name].shape)!=shape:raise ValueError(f'Feature cache shape differs: {name}')
    if result['candidate_mask'].dtype!=torch.bool:raise ValueError('Feature cache mask dtype differs')
    for name in ('option_counts','query_positions','candidate_positions','input_lengths'):
        if result[name].dtype!=torch.int64:raise ValueError(f'Feature cache integer dtype differs: {name}')
    counts=torch.tensor([len(row['option_ids']) for row in records])
    mask=torch.arange(width)[None,:]<counts[:,None]
    if not torch.all((counts>=2)&(counts<=width)):raise ValueError('Invalid feature cache candidate count')
    if (not torch.all(result['input_lengths']>0)
            or not torch.all(result['input_lengths']<=result['query_positions']+1)
            or not torch.all(result['input_lengths']<=key.get('max_input_tokens',8192))
            or not torch.all(result['candidate_positions'][mask]>=0)
            or not torch.all(result['candidate_positions'][mask]<result['query_positions'][:,None].expand(n,width)[mask])):
        raise ValueError('Feature cache token positions/lengths differ')
    if torch.any(result['candidate_positions'][~mask]!=0) or torch.any(result['candidate_features'][~mask]!=0):
        raise ValueError('Invalid feature cache slots must be zero')
    for row,count in zip(records,counts):
        if len(row.get('prompt_sha256',''))!=64 or len(row.get('candidate_character_spans',[]))!=count:
            raise ValueError('Feature cache prompt/span provenance differs')
    if not torch.equal(result['option_counts'],counts) or not torch.equal(result['candidate_mask'],mask):
        raise ValueError('Feature cache candidate counts/mask differ')
    if result['query_features'].dtype!=torch.float32 or result['candidate_features'].dtype!=torch.float32:
        raise ValueError('Feature cache dtype differs')
    if result['query_features'].shape[0]!=n or result['candidate_mask'].shape[0]!=n:
        raise ValueError('Feature cache row count differs')
    if not torch.isfinite(result['query_features']).all() or not torch.isfinite(result['candidate_features']).all():
        raise ValueError('Nonfinite feature cache')
    targets=[d['hard_target'] for case in cases for d in case['decisions']]
    if manifest['hard_targets_sha256']!=digest_json(targets):raise ValueError('Hard target provenance changed')
    result.update(records=records,hard_targets=torch.tensor(targets,dtype=torch.long),
                  manifest_sha256=sha256_file(manifest_path),manifest=manifest)
    return result


def cache_features(directory,split,cases,model,tokenizer,key,max_input_tokens=8192,max_candidates=5):
    if split not in ('train','dev'):raise ValueError('Only train/dev feature caches are allowed')
    directory=Path(directory)
    if directory.exists():
        if (directory/'manifest.json').is_file():return load_feature_cache(directory,key,cases)
        if split=='dev':raise RuntimeError(f'Incomplete held-out feature cache, refusing replay: {directory}')
        # Preserve unfinished train evidence; only train may be extracted again.
        directory.rename(directory.with_name(f'{directory.name}.incomplete.{time.time_ns()}'))
    directory.mkdir(parents=True)
    write_json(directory/'started.json',{'split':split,'key_sha256':digest_json(key),'n_cases':len(cases)})
    arrays={name:[] for name in ('query_features','candidate_features','candidate_mask','option_counts',
                                 'query_positions','candidate_positions','input_lengths')}
    records=feature_records(cases);offset=0
    for index,case in enumerate(cases):
        prepared=prepare_pointer_case(case['decisions'],tokenizer,max_input_tokens,max_candidates)
        features=extract_pointer_features(model,prepared)
        for name in arrays:
            value=prepared['input_lengths'] if name=='input_lengths' else features[name]
            arrays[name].append(torch.as_tensor(value).cpu())
        for row,prompt,spans in zip(records[offset:],prepared['prompts'],prepared['candidate_character_spans']):
            row['prompt_sha256']=hashlib.sha256(prompt.encode()).hexdigest()
            row['candidate_character_spans']=spans
        offset+=len(case['decisions'])
        if (index+1)%40==0:
            import logging
            logging.info('%s feature cache: %d/%d cases',split,index+1,len(cases))
    packed={name:torch.cat(chunks).numpy() for name,chunks in arrays.items()}
    with (directory/'features.npz.tmp').open('wb') as stream:np.savez_compressed(stream,**packed)
    (directory/'features.npz.tmp').replace(directory/'features.npz')
    write_json(directory/'records.json',records)
    targets=[d['hard_target'] for case in cases for d in case['decisions']]
    manifest={'complete':True,'split':split,'key':key,'n_cases':len(cases),'n_decisions':len(records),
              'hard_targets_sha256':digest_json(targets),
              'files_sha256':{name:sha256_file(directory/name) for name in ('features.npz','records.json')}}
    write_json(directory/'manifest.json',manifest)
    return load_feature_cache(directory,key,cases)


def head_hash(head):
    digest=hashlib.sha256()
    for name,value in sorted(head.state_dict().items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _scores(head,cached,indices):
    device=next(head.parameters()).device
    return head(cached['query_features'][indices].to(device),cached['candidate_features'][indices].to(device),
                cached['candidate_mask'][indices].to(device))


def head_train_update(head,cached,indices,optimizer,scheduler,max_grad_norm):
    if {id(p) for group in optimizer.param_groups for p in group['params']}!={id(p) for p in head.parameters()}:
        raise ValueError('Optimizer must contain exactly Pointer Head parameters')
    head.train();optimizer.zero_grad(set_to_none=True)
    scores=_scores(head,cached,indices)
    loss=hard_candidate_loss(scores,cached['option_counts'][indices].tolist(),cached['hard_targets'][indices].tolist())
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
        raise RuntimeError('Missing or nonfinite Head gradient')
    norm=torch.nn.utils.clip_grad_norm_(head.parameters(),max_grad_norm,error_if_nonfinite=True)
    lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
    return {'loss':float(loss.detach()),'grad_norm':float(norm),'learning_rate':lr,'questions':len(indices)}


def score_cached(head,cached):
    """Score fixed five-question case groups, matching uncached Head arithmetic."""
    head.eval();result=[]
    with torch.no_grad():
        for start in range(0,len(cached['records']),5):
            indices=list(range(start,min(start+5,len(cached['records']))))
            if len({cached['records'][i]['case_id'] for i in indices})!=1:
                raise ValueError('Cached scoring requires five independent questions per case')
            scores=_scores(head,cached,indices)
            counts=cached['option_counts'][indices].tolist()
            probabilities=label_probabilities(scores,counts)
            for i,index in enumerate(indices):
                result.append({'probabilities':probabilities[i],'label_logits':scores[i,:counts[i]].cpu().tolist()})
    return result


def save_head_checkpoint(head,optimizer,scheduler,destination,progress):
    if any(value.dtype!=torch.float32 for value in head.state_dict().values()):
        raise ValueError('Head checkpoint requires FP32 parameters')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise ValueError('Head checkpoint generations are immutable')
    temporary=Path(tempfile.mkdtemp(prefix='.head_',dir=destination.parent))
    try:
        save_file({k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},str(temporary/'head.safetensors'))
        write_json(temporary/'config.json',{'hidden_size':head.hidden_size,'projection_dim':head.projection_dim,
                                         'parameters':sum(p.numel() for p in head.parameters()),'dtype':'float32'})
        torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'rng':rng_state(),'progress':progress},temporary/'trainer.pt')
        manifest={'complete':True,'progress':progress,'head_sha256':head_hash(head),
                  'files_sha256':{name:sha256_file(temporary/name) for name in ('head.safetensors','config.json','trainer.pt')}}
        write_json(temporary/'manifest.json',manifest);temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True);raise
    return manifest


def validate_head_checkpoint(directory):
    directory=Path(directory)
    if not (directory/'manifest.json').is_file():raise ValueError('Incomplete Head checkpoint')
    manifest=json.loads((directory/'manifest.json').read_text())
    if not manifest.get('complete'):raise ValueError('Incomplete Head checkpoint')
    for name,digest in manifest['files_sha256'].items():
        if name not in ('head.safetensors','config.json','trainer.pt') or sha256_file(directory/name)!=digest:
            raise ValueError('Head checkpoint checksum differs')
    if set(manifest['files_sha256'])!={'head.safetensors','config.json','trainer.pt'}:
        raise ValueError('Missing Head checkpoint files')
    return manifest


def load_head_weights(head,directory):
    manifest=validate_head_checkpoint(directory)
    config=json.loads((Path(directory)/'config.json').read_text())
    if (config['hidden_size'],config['projection_dim'])!=(head.hidden_size,head.projection_dim):
        raise ValueError('Head checkpoint architecture differs')
    tensors=load_file(str(Path(directory)/'head.safetensors'))
    expected=head.state_dict()
    if (config.get('dtype')!='float32' or config.get('parameters')!=sum(p.numel() for p in head.parameters())
            or set(tensors)!=set(expected) or any(value.dtype!=torch.float32 for value in expected.values())
            or any(value.dtype!=torch.float32 or value.shape!=expected[name].shape for name,value in tensors.items())):
        raise ValueError('Head checkpoint dtype/count/shapes differ')
    head.load_state_dict(tensors,strict=True)
    if head_hash(head)!=manifest['head_sha256']:raise ValueError('Head checkpoint parameter hash differs')
    return manifest


def restore_head_checkpoint(head,optimizer,scheduler,directory):
    manifest=load_head_weights(head,directory)
    state=torch.load(Path(directory)/'trainer.pt',map_location='cpu',weights_only=False)
    optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    if manifest['progress']!=state['progress']:raise ValueError('Head progress differs')
    random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy']);torch.set_rng_state(state['rng']['torch'])
    if state['rng']['cuda']:torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state['progress']


def publish_head_checkpoint(head,optimizer,scheduler,directory,progress,steps_per_epoch):
    directory=Path(directory);destination=directory/f'checkpoints/step{progress["step"]:06d}'
    if destination.exists():
        manifest=validate_head_checkpoint(destination)
        if manifest['progress']!=progress or manifest['head_sha256']!=head_hash(head):
            raise ValueError('Existing Head recovery generation differs')
    else:manifest=save_head_checkpoint(head,optimizer,scheduler,destination,progress)
    write_json(directory/'latest.json',{'checkpoint':str(destination.relative_to(directory)),**progress,
                                      'head_sha256':manifest['head_sha256']})
    for old in destination.parent.glob('step[0-9][0-9][0-9][0-9][0-9][0-9]'):
        step=int(old.name[4:])
        if old!=destination and step%steps_per_epoch and step<progress['step']:
            validate_head_checkpoint(old);shutil.rmtree(old)
    return destination,manifest


def require_test_lock(path,seeds,selections=None):
    path=Path(path)
    if not path.is_file():raise ValueError('All seeds must be locked before test')
    value=json.loads(path.read_text())
    if value.get('locked') is not True or set(value.get('seeds',{}))!={str(seed) for seed in seeds}:
        raise ValueError('All seeds must be locked before test')
    if selections is not None and value['seeds']!=selections:raise ValueError('Locked selections differ')
    return value['seeds']


def read_pointer_case(model,head,prepared):
    device=next(head.parameters()).device
    synchronize(device);started=time.perf_counter()
    features=extract_pointer_features(model,prepared)
    head.eval()
    with torch.no_grad():scores=head(features['query_features'].to(device),features['candidate_features'].to(device),
                                     features['candidate_mask'].to(device))
    probabilities=label_probabilities(scores,features['option_counts'])
    logits=[row[:count].cpu().tolist() for row,count in zip(scores,features['option_counts'])]
    synchronize(device)
    return {'probabilities':probabilities,'label_logits':logits,'inference_seconds':time.perf_counter()-started}

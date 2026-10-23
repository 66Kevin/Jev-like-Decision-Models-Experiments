"""P1 token-only caches, differentiable Pointer + LoRA, and joint recovery.

Only the CUDA training Backbone uses BF16 autocast. Native evaluation and the
FP32 decision Head explicitly disable autocast, including an outer context.
"""
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from jev.data.typed_decisions import sha256_file
from jev.models.pointer import prepare_pointer_case
from jev.training.lora import assert_adapter_only_trainable, hard_candidate_loss, _integer_vector
from jev.training.pointer_runner import head_hash
from jev.training.runner import rng_state, write_json


def cache_pointer_decisions(decisions, tokenizer, max_input_tokens=8192, max_candidates=5):
    """Cache unpadded tokens and offsets only, never frozen hidden features."""
    result = []
    for decision in decisions:
        target = _integer_vector([decision['hard_target']], 'hard target').item()
        count = len(decision['option_ids'])
        if not 0 <= target < count:
            raise ValueError('Hard target is outside valid candidates')
        prepared = prepare_pointer_case([decision], tokenizer, max_input_tokens, max_candidates)
        result.append({'input_ids': prepared['input_ids'][0].tolist(),
                       'query_position': int(prepared['query_positions'][0]),
                       'candidate_positions': prepared['candidate_positions'][0, :count].tolist(),
                       'candidate_character_spans': prepared['candidate_character_spans'][0],
                       'option_count': count, 'hard_target': target,
                       'prompt_sha256': hashlib.sha256(prepared['prompts'][0].encode()).hexdigest(),
                       'identity': {name: decision[name] for name in ('case_id','question_id','workflow')}})
    return result


def collate_pointer_decisions(cached, pad_token_id, max_candidates=5):
    """Left pad fresh microbatches and rebase every cached token position."""
    if not cached or pad_token_id is None or not isinstance(max_candidates,int) or not 2 <= max_candidates <= 26:
        raise ValueError('Nonempty token cache, padding ID and valid candidate width required')
    lengths = [len(row['input_ids']) for row in cached]
    if not min(lengths): raise ValueError('Empty cached prompt')
    targets = _integer_vector([row['hard_target'] for row in cached], 'hard targets')
    counts = [row['option_count'] for row in cached]
    width = max(lengths)
    ids = torch.full((len(cached),width),pad_token_id,dtype=torch.long)
    mask = torch.zeros_like(ids)
    candidate_positions = torch.zeros((len(cached),max_candidates),dtype=torch.long)
    candidate_mask = torch.zeros_like(candidate_positions,dtype=torch.bool)
    for i,(row,length,count,target) in enumerate(zip(cached,lengths,counts,targets.tolist())):
        if (not isinstance(count,int) or not 2 <= count <= max_candidates or not 0 <= target < count
                or row['query_position'] != length-1 or len(row['candidate_positions']) != count):
            raise ValueError('Invalid cached candidate counts, target or query position')
        positions = _integer_vector(row['candidate_positions'],'candidate positions')
        if ((positions < 0) | (positions >= length-1)).any():
            raise ValueError('Invalid cached candidate position')
        tokens = _integer_vector(row['input_ids'],'input IDs')
        if (tokens < 0).any(): raise ValueError('Invalid cached token IDs')
        ids[i,-length:] = tokens; mask[i,-length:] = 1
        candidate_positions[i,:count] = positions + width-length
        candidate_mask[i,:count] = True
    return {'input_ids':ids,'attention_mask':mask,'position_ids':(mask.cumsum(-1)-1).clamp(min=0),
            'query_positions':torch.full((len(cached),),width-1,dtype=torch.long),
            'candidate_positions':candidate_positions,'candidate_mask':candidate_mask,
            'input_lengths':lengths,'option_counts':counts,'hard_targets':targets,
            'identities':[dict(row['identity']) for row in cached]}


def assert_pointer_lora_trainable(model, head):
    report = assert_adapter_only_trainable(model)
    if any(not p.requires_grad or p.dtype != torch.float32 for p in head.parameters()):
        raise ValueError('Pointer Head parameters must all be trainable FP32')
    if not list(head.parameters()): raise ValueError('Pointer Head has no parameters')
    if {id(p) for p in model.parameters()} & {id(p) for p in head.parameters()}:
        raise ValueError('LoRA model and Head must have disjoint parameters')
    n = sum(p.numel() for p in head.parameters())
    return {'lora_parameters':report['trainable_parameters'],'head_parameters':n,
            'trainable_parameters':report['trainable_parameters']+n,
            'frozen_parameters':report['frozen_parameters'],
            'lora_trainable_tensors':report['trainable_tensors']}


def pointer_lora_logits(model, head, prepared):
    """Backpropagate query and candidate states directly through the text model.

    Mode is owned by the caller. Eval uses the native original-weight dtype;
    CUDA training enables BF16 AMP for Backbone only. No LM vocabulary output.
    """
    base = model.get_base_model()
    device = next(model.parameters()).device
    if next(head.parameters()).device != device:
        raise ValueError('Pointer Head and Backbone must share a device')
    ids = prepared['input_ids']
    if len(ids)==1 and (prepared['attention_mask']==0).any():
        raise ValueError('Native Qwen3.5 does not mask DeltaNet padding for a padded singleton')
    tensors = {name:prepared[name].to(device) for name in ('input_ids','attention_mask','position_ids')}
    # Explicitly override any caller autocast, including evaluation contexts.
    context = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type=='cuda' and model.training)
    with context:
        hidden = base.model(**tensors,use_cache=False,return_dict=True).last_hidden_state
    if hidden.ndim != 3 or hidden.shape[:2] != ids.shape:
        raise ValueError('Native Backbone returned incompatible hidden states')
    rows = torch.arange(len(ids),device=device)
    query_positions = prepared['query_positions'].to(device)
    positions = prepared['candidate_positions'].to(device)
    mask = prepared['candidate_mask'].to(device)
    if (query_positions.shape != (len(ids),) or positions.shape != mask.shape
            or positions.shape[0] != len(ids) or mask.dtype != torch.bool
            or ((query_positions < 0) | (query_positions >= hidden.shape[1])).any()
            or ((positions < 0) | (positions >= hidden.shape[1])).any()):
        raise ValueError('Invalid Pointer token positions or mask')
    with torch.autocast(device.type,enabled=False):
        query = hidden[rows,query_positions].float()
        candidates = hidden[rows[:,None],positions].float().masked_fill(~mask[...,None],0)
        return head(query,candidates,mask)


def frozen_backbone_state(model):
    """Hash every original parameter AND buffer while allowing LoRA train mode."""
    base = model.get_base_model()
    digest = hashlib.sha256(); n = 0; parameter_dtypes=set(); buffer_dtypes=set()
    for kind, items in (('parameter',base.named_parameters()),('buffer',base.named_buffers())):
        for name,tensor in items:
            if '.lora_' in name: continue
            if kind=='parameter':
                if tensor.requires_grad or tensor.grad is not None:
                    raise ValueError(f'Original parameter is trainable or has a gradient: {name}')
                n += tensor.numel(); parameter_dtypes.add(str(tensor.dtype))
            else: buffer_dtypes.add(str(tensor.dtype))
            name = name.replace('.base_layer.','.')
            digest.update(f'{kind}:{name}:{tensor.dtype}:{list(tensor.shape)}'.encode())
            flat=tensor.detach().reshape(-1)
            for start in range(0,flat.numel(),1024*1024):
                digest.update(flat[start:start+1024*1024].contiguous().cpu().view(torch.uint8).numpy().tobytes())
    tied=base.get_input_embeddings().weight is base.get_output_embeddings().weight
    if not tied: raise ValueError('Original embedding and LM Head tie changed')
    return {'state_sha256':digest.hexdigest(),'unique_parameters':n,'head_and_embedding_tied':tied,
            'parameter_dtypes':sorted(parameter_dtypes),'buffer_dtypes':sorted(buffer_dtypes)}


def create_joint_optimizer(model, head, lora_lr=2e-4, head_lr=1e-3, weight_decay=.01,
                           betas=(.9,.999), eps=1e-8):
    assert_pointer_lora_trainable(model,head)
    return torch.optim.AdamW([{'params':[p for p in model.parameters() if p.requires_grad],'lr':lora_lr},
                              {'params':list(head.parameters()),'lr':head_lr}],
                             weight_decay=weight_decay,betas=tuple(betas),eps=eps,foreach=False)


def _assert_optimizer(model,head,optimizer):
    assert_pointer_lora_trainable(model,head)
    groups = [[p for p in model.parameters() if p.requires_grad],list(head.parameters())]
    if (len(optimizer.param_groups)!=2 or any(len(g['params'])!=len(expected)
            or [id(p) for p in g['params']] != [id(p) for p in expected]
            for g,expected in zip(optimizer.param_groups,groups))):
        raise ValueError('Joint optimizer must contain exactly ordered LoRA and Head groups')
    return [p for group in groups for p in group]


def joint_train_update(model,head,cached,indices,pad,optimizer,scheduler,microbatch,max_grad_norm):
    if not indices or not isinstance(microbatch,int) or microbatch<1:
        raise ValueError('A nonempty update and positive microbatch are required')
    parameters = _assert_optimizer(model,head,optimizer)
    model.train(); head.train(); optimizer.zero_grad(set_to_none=True); total=0.
    for start in range(0,len(indices),microbatch):
        rows = [cached[i] for i in indices[start:start+microbatch]]
        batch = collate_pointer_decisions(rows,pad)
        scores = pointer_lora_logits(model,head,batch)
        loss = hard_candidate_loss(scores,batch['option_counts'],batch['hard_targets'])
        weight = len(rows)/len(indices)
        (loss*weight).backward(); total += float(loss.detach())*weight
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise RuntimeError('Missing or nonfinite joint LoRA/Head gradient')
    norm = torch.nn.utils.clip_grad_norm_(parameters,max_grad_norm,error_if_nonfinite=True)
    lora_lr,head_lr = [g['lr'] for g in optimizer.param_groups]
    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    return {'loss':total,'grad_norm':float(norm),'learning_rate':lora_lr,'lora_learning_rate':lora_lr,
            'head_learning_rate':head_lr,'questions':len(indices)}


def _adapter_tensors(model):
    from peft import get_peft_model_state_dict
    return get_peft_model_state_dict(model,save_embedding_layers=False)


def _tensor_hash(values):
    digest=hashlib.sha256()
    for name,tensor in sorted(values.items()):
        digest.update(f'{name}:{tensor.dtype}:{list(tensor.shape)}'.encode())
        digest.update(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def adapter_hash(model):
    return _tensor_hash(_adapter_tensors(model))


_JOINT_FILES = frozenset(('adapter/adapter_model.safetensors','adapter/adapter_config.json',
                          'head.safetensors','joint_config.json','trainer.pt'))


def save_joint_checkpoint(model,head,optimizer,scheduler,destination,progress):
    _assert_optimizer(model,head,optimizer)
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists(): raise ValueError('Joint checkpoint generations are immutable')
    temporary=Path(tempfile.mkdtemp(prefix='.joint_',dir=destination.parent))
    try:
        model.save_pretrained(temporary/'adapter',safe_serialization=True,save_embedding_layers=False)
        save_file({n:p.detach().cpu().contiguous() for n,p in head.state_dict().items()},str(temporary/'head.safetensors'))
        config={'hidden_size':head.hidden_size,'projection_dim':head.projection_dim,
                'head_parameters':sum(p.numel() for p in head.parameters()),
                'lora_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'dtype':'float32',
                'progress':progress,'adapter_sha256':adapter_hash(model),'head_sha256':head_hash(head)}
        write_json(temporary/'joint_config.json',config)
        torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'rng':rng_state(),'progress':progress,
                    'adapter_sha256':config['adapter_sha256'],'head_sha256':config['head_sha256']},temporary/'trainer.pt')
        manifest={'schema':'p1_joint_v1','complete':True,'progress':progress,
                  'adapter_sha256':config['adapter_sha256'],'head_sha256':config['head_sha256'],
                  'files_sha256':{name:sha256_file(temporary/name) for name in sorted(_JOINT_FILES)}}
        write_json(temporary/'manifest.json',manifest); temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True); raise
    return manifest


def validate_joint_checkpoint(directory):
    directory=Path(directory)
    if not (directory/'manifest.json').is_file(): raise ValueError('Incomplete joint checkpoint')
    manifest=json.loads((directory/'manifest.json').read_text())
    if (manifest.get('schema')!='p1_joint_v1' or manifest.get('complete') is not True
            or set(manifest.get('files_sha256',{}))!=_JOINT_FILES):
        raise ValueError('Incomplete joint checkpoint manifest')
    for name,digest in manifest['files_sha256'].items():
        if not (directory/name).is_file() or sha256_file(directory/name)!=digest:
            raise ValueError('Joint checkpoint checksum differs')
    config=json.loads((directory/'joint_config.json').read_text())
    state=torch.load(directory/'trainer.pt',map_location='cpu',weights_only=False)
    for key in ('progress','adapter_sha256','head_sha256'):
        if config.get(key)!=manifest[key] or state.get(key)!=manifest[key]:
            raise ValueError('Mixed joint checkpoint generation')
    adapter=load_file(str(directory/'adapter/adapter_model.safetensors'))
    if (not adapter or any(not n.endswith(('.lora_A.weight','.lora_B.weight')) for n in adapter)
            or any(p.dtype!=torch.float32 or p.ndim!=2 for p in adapter.values())
            or sum(p.numel() for p in adapter.values())!=config.get('lora_parameters')
            or _tensor_hash(adapter)!=manifest['adapter_sha256']):
        raise ValueError('Joint adapter dtype/count/hash differs')
    head_tensors=load_file(str(directory/'head.safetensors'))
    expected_keys={'query_projection.weight','candidate_projection.weight'}
    head_digest=hashlib.sha256()
    for name,tensor in sorted(head_tensors.items()):
        head_digest.update(name.encode());head_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    if (set(head_tensors)!=expected_keys or config.get('dtype')!='float32'
            or any(t.dtype!=torch.float32 or t.shape!=(config['projection_dim'],config['hidden_size']) for t in head_tensors.values())
            or sum(t.numel() for t in head_tensors.values())!=config.get('head_parameters')
            or head_digest.hexdigest()!=manifest['head_sha256']):
        raise ValueError('Joint Head dtype/count/shapes/hash differs')
    return manifest


def load_joint_weights(model,head,directory):
    from peft import set_peft_model_state_dict
    directory=Path(directory); manifest=validate_joint_checkpoint(directory)
    assert_pointer_lora_trainable(model,head)
    config=json.loads((directory/'joint_config.json').read_text())
    if (config.get('dtype')!='float32' or config.get('hidden_size')!=head.hidden_size
            or config.get('projection_dim')!=head.projection_dim
            or config.get('head_parameters')!=sum(p.numel() for p in head.parameters())):
        raise ValueError('Joint Head architecture or dtype differs')
    tensors=load_file(str(directory/'head.safetensors'))
    expected=head.state_dict()
    if (set(tensors)!=set(expected) or any(p.dtype!=torch.float32 or p.shape!=expected[n].shape for n,p in tensors.items())):
        raise ValueError('Joint Head tensor keys, dtype or shape differ')
    saved_config=json.loads((directory/'adapter/adapter_config.json').read_text())
    runtime_config=model.peft_config['default'].to_dict()
    for key in ('r','lora_alpha','lora_dropout','bias','target_modules','modules_to_save',
                'use_rslora','use_dora','fan_in_fan_out','rank_pattern','alpha_pattern','lora_bias'):
        saved,runtime=saved_config.get(key),runtime_config.get(key)
        if key=='target_modules':
            saved=sorted(saved) if isinstance(saved,(list,set,tuple)) else saved
            runtime=sorted(runtime) if isinstance(runtime,(list,set,tuple)) else runtime
        if saved!=runtime: raise ValueError(f'Joint adapter config differs: {key}')
    adapter=load_file(str(directory/'adapter/adapter_model.safetensors'))
    expected_adapter=_adapter_tensors(model)
    if (set(adapter)!=set(expected_adapter) or any(p.shape!=expected_adapter[n].shape for n,p in adapter.items())):
        raise ValueError('Joint adapter architecture differs')
    result=set_peft_model_state_dict(model,adapter)
    if result.unexpected_keys: raise ValueError('Unexpected joint adapter keys')
    head.load_state_dict(tensors,strict=True)
    if adapter_hash(model)!=manifest['adapter_sha256'] or head_hash(head)!=manifest['head_sha256']:
        raise ValueError('Loaded joint parameter hashes differ')
    return manifest


def restore_joint_checkpoint(model,head,optimizer,scheduler,directory):
    _assert_optimizer(model,head,optimizer)
    manifest=load_joint_weights(model,head,directory)
    state=torch.load(Path(directory)/'trainer.pt',map_location='cpu',weights_only=False)
    optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
    random.setstate(state['rng']['python']); np.random.set_state(state['rng']['numpy']); torch.set_rng_state(state['rng']['torch'])
    if state['rng']['cuda']: torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return manifest['progress']


def publish_joint_checkpoint(model,head,optimizer,scheduler,directory,progress,steps_per_epoch):
    if not isinstance(steps_per_epoch,int) or steps_per_epoch<1: raise ValueError('Invalid steps per epoch')
    directory=Path(directory); destination=directory/f'checkpoints/step{progress["step"]:06d}'
    if destination.exists():
        manifest=validate_joint_checkpoint(destination)
        if (manifest['progress']!=progress or manifest['adapter_sha256']!=adapter_hash(model)
                or manifest['head_sha256']!=head_hash(head)):
            raise ValueError('Existing joint recovery generation differs')
    else: manifest=save_joint_checkpoint(model,head,optimizer,scheduler,destination,progress)
    write_json(directory/'latest.json',{'checkpoint':str(destination.relative_to(directory)),**progress,
                                      'adapter_sha256':manifest['adapter_sha256'],'head_sha256':manifest['head_sha256']})
    for old in destination.parent.glob('step[0-9][0-9][0-9][0-9][0-9][0-9]'):
        step=int(old.name[4:])
        if old!=destination and step%steps_per_epoch and step<progress['step']:
            validate_joint_checkpoint(old); shutil.rmtree(old)
    return destination,manifest

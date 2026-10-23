#!/usr/bin/env python3
"""Train-only native frozen-backbone, Pointer span and Head-gradient preflight."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
import torch
import yaml
from jev.data.typed_decisions import load_cases, prepare_splits
from jev.models.lm_readout import build_messages, load_local_model
from jev.models.pointer import PointerHead, prepare_pointer_case, extract_pointer_features
from jev.training.full import parameter_hash
from jev.training.lora import hard_candidate_loss
from jev.training.runner import write_json


def main():
    config = yaml.safe_load((ROOT/'experiments/01_baseline/configs/b0.yaml').read_text())
    torch.manual_seed(42)
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.conv.fp32_precision = 'ieee'
    raw = ROOT/config['dataset']['raw_dir']
    splits = prepare_splits(load_cases(raw), ROOT/config['dataset']['split_dir'], seed=42,
                            source_manifest=raw/'download_manifest.json')
    model, tokenizer, loading = load_local_model(config['model'])
    original_hash = parameter_hash(model)
    buffer_before = {name:(buffer.dtype, buffer.detach().cpu().clone()) for name,buffer in model.named_buffers()}
    longest_case = None
    longest_length = 0
    for case in splits['train']:
        for decision in case['decisions']:
            prompt = tokenizer.apply_chat_template(build_messages(decision), tokenize=False,
                        add_generation_prompt=True, enable_thinking=False)
            length = len(tokenizer.encode(prompt, add_special_tokens=False))
            if length > longest_length:
                longest_length, longest_case = length, case
    cases = list(splits['train'][:8])
    workflows = {case['workflow'] for case in cases}
    for case in splits['train']:
        if case['workflow'] not in workflows:
            cases.append(case);workflows.add(case['workflow'])
    if longest_case not in cases:
        cases.append(longest_case)
    batches, features = [], []
    comparisons = []
    lm_calls = []
    def forbid_lm(*args):
        lm_calls.append(True)
        raise RuntimeError('P0 must not call LM vocabulary projection')
    for index,case in enumerate(cases):
        batch = prepare_pointer_case(case['decisions'], tokenizer)
        tensors = {key:batch[key].to(model.device) for key in ('input_ids','attention_mask','position_ids')}
        captured = []
        def capture(module, args, output):
            captured.append(output.last_hidden_state.detach().float().cpu().clone())
        hook = model.model.register_forward_hook(capture)
        with torch.no_grad():
            model(**tensors, use_cache=False, logits_to_keep=1)
        hook.remove()
        hidden = captured[0]
        q = hidden[torch.arange(5),batch['query_positions']]
        k = hidden[torch.arange(5)[:,None],batch['candidate_positions']]
        k = k.masked_fill(~batch['candidate_mask'][...,None],0)
        trap = model.lm_head.register_forward_pre_hook(forbid_lm)
        value = extract_pointer_features(model,batch)
        trap.remove()
        assert torch.equal(q,value['query_features']) and torch.equal(k,value['candidate_features'])
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())
        batches.append(batch);features.append(value)
        comparisons.append({'case_id':case['case_id'],'workflow':case['workflow'],
                            'input_lengths':batch['input_lengths'],'reference_features_bitexact':True,
                            'candidate_positions':batch['candidate_positions'].tolist()})
        print('native_case_passed',case['case_id'],flush=True)
    head = PointerHead().to(model.device)
    assert sum(parameter.numel() for parameter in head.parameters()) == 262144
    query = torch.cat([value['query_features'] for value in features])[:32].to(model.device)
    candidates = torch.cat([value['candidate_features'] for value in features])[:32].to(model.device)
    mask = torch.cat([value['candidate_mask'] for value in features])[:32].to(model.device)
    counts = mask.sum(-1).tolist()
    targets = [decision['hard_target'] for case in cases for decision in case['decisions']][:32]
    optimizer = torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=.01)
    losses = []
    before = {name:value.detach().cpu().clone() for name,value in head.named_parameters()}
    for step in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = hard_candidate_loss(head(query,candidates,mask),counts,targets)
        loss.backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   and parameter.grad.abs().sum()>0 for parameter in head.parameters())
        torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    assert all(not torch.equal(parameter.detach().cpu(),before[name]) for name,parameter in head.named_parameters())
    with torch.no_grad():
        cached_scores = head(features[0]['query_features'].to(model.device),
                             features[0]['candidate_features'].to(model.device),
                             features[0]['candidate_mask'].to(model.device)).cpu()
    fresh = extract_pointer_features(model,batches[0])
    with torch.no_grad():
        fresh_scores = head(fresh['query_features'].to(model.device),fresh['candidate_features'].to(model.device),
                            fresh['candidate_mask'].to(model.device)).cpu()
    assert torch.equal(cached_scores,fresh_scores)
    assert parameter_hash(model) == original_hash
    for name,buffer in model.named_buffers():
        dtype,value = buffer_before[name]
        assert buffer.dtype == dtype and torch.equal(buffer.detach().cpu(),value)
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    destination = ROOT/'outputs/02_supervised/p0/preflight_train.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    value = {'status':'passed','train_only':True,'physical_gpu':0,'native_backend':True,
             'frozen_parameters':752393024,'trainable_head_parameters':262144,'projection_dim':128,
             'backbone_sha256_before':original_hash,'backbone_sha256_after':original_hash,
             'original_buffers_unchanged':True,'head_embedding_tied':True,'lm_head_calls_during_pointer':len(lm_calls),
             'reference_features_bitexact':True,'cached_full_scores_bitexact':True,
             'max_train_tokens':longest_length,'tested_cases':comparisons,'head_updates':20,
             'finite_head_gradients':True,'head_weights_changed':True,'repeated_batch_initial_loss':losses[0],
             'repeated_batch_final_loss':losses[-1],
             'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
             'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'loading_info':loading}
    write_json(destination,value)
    print('P0_PREFLIGHT_PASSED',json.dumps({key:value[key] for key in ('max_train_tokens','repeated_batch_initial_loss',
           'repeated_batch_final_loss','peak_allocated_bytes')}),flush=True)


if __name__ == '__main__':
    main()

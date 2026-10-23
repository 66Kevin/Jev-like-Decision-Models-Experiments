#!/usr/bin/env python3
"""Train-only full-FT kernel, precision and worst-input memory preflight."""
import gc
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
import yaml
from jev.data.typed_decisions import iter_decisions, load_cases, prepare_splits
from jev.models.lm_readout import load_local_model
from jev.training.kernels import set_delta_backend
from jev.training.lora import cache_decisions, collate_decisions, candidate_logits, hard_candidate_loss
from jev.training.runner import write_json


def main():
    config = yaml.safe_load((ROOT / 'experiments/02_supervised/configs/g1.yaml').read_text())
    torch.manual_seed(42)
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.conv.fp32_precision = 'ieee'
    raw = ROOT / config['dataset']['raw_dir']
    splits = prepare_splits(load_cases(raw), ROOT / config['dataset']['split_dir'], seed=42,
                            source_manifest=raw / 'download_manifest.json')
    model, tokenizer, _ = load_local_model(config['model'])
    cached = cache_decisions(list(iter_decisions(splits['train'])), tokenizer)
    pad = tokenizer.pad_token_id
    # Gradient comparison uses two unequal, real training prompts.
    batch = collate_decisions([cached[0], cached[1]], pad)
    model.float().requires_grad_(True).train()
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert sum(p.numel() for p in model.parameters()) == 752393024
    gradients = {}
    profile = {'status': 'running', 'train_only': True, 'parameters': 752393024,
               'parameter_dtype': 'float32', 'autocast_dtype': 'bfloat16', 'optimizer': 'fused AdamW',
               'probe_case_ids': [r['identity'] for r in [cached[0], cached[1]]],
               'max_train_tokens': max(len(r['input_ids']) for r in cached)}
    destination = ROOT / 'outputs/02_supervised/g2/kernel_profile_train.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    for kind in ('native', 'fla'):
        set_delta_backend(model, kind, ROOT / '.runtime/fla')
        model.zero_grad(set_to_none=True)
        started = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = hard_candidate_loss(candidate_logits(model, batch), batch['option_counts'], batch['hard_targets'])
        loss.backward()
        sampled = []
        for name, p in model.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise RuntimeError(f'Missing/nonfinite {kind} gradient: {name}')
            flat = p.grad.detach().reshape(-1)
            count = min(1024, flat.numel())
            indices = torch.arange(count, dtype=torch.int64, device=flat.device) * (flat.numel()-1) // max(1, count-1)
            sampled.append(flat[indices].cpu())
        gradients[kind] = torch.cat(sampled).float()
        torch.cuda.synchronize()
        profile[kind] = {'loss': loss.item(), 'seconds': time.perf_counter()-started,
                         'all_parameter_gradients_finite': True, 'sample_count': len(gradients[kind])}
        print(kind, profile[kind], flush=True)
    a, b = gradients['native'], gradients['fla']
    cosine = torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=0).item()
    relative = ((a-b).norm()/a.norm()).item()
    profile['gradient_agreement'] = {'cosine': cosine, 'relative_l2': relative}
    if cosine < .95 or relative > .35:
        raise RuntimeError(f'Full gradient mismatch: {profile["gradient_agreement"]}')
    del gradients, a, b
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01, fused=True)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    longest = sorted(cached, key=lambda r: len(r['input_ids']), reverse=True)
    profile['memory_trials'] = []
    for micro in (2, 4):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        batch = collate_decisions(longest[:micro], pad)
        before = model.get_input_embeddings().weight.detach()[32:37].clone()
        started = time.perf_counter()
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = hard_candidate_loss(candidate_logits(model, batch), batch['option_counts'], batch['hard_targets'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            assert not torch.equal(before, model.get_input_embeddings().weight.detach()[32:37])
            assert all(s['exp_avg'].dtype == s['exp_avg_sq'].dtype == torch.float32 for s in optimizer.state.values())
            trial = {'microbatch': micro, 'status': 'passed', 'seconds': time.perf_counter()-started,
                     'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                     'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'head_updated': True}
        except torch.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
            trial = {'microbatch': micro, 'status': 'oom'}
        profile['memory_trials'].append(trial)
        print('memory', trial, flush=True)
        write_json(destination, profile)
    eligible = [r for r in profile['memory_trials'] if r['status']=='passed' and
                r['peak_allocated_bytes'] < .90*torch.cuda.get_device_properties(0).total_memory]
    if not eligible:
        raise RuntimeError('No full training microbatch passed with 10% memory headroom')
    chosen = max(eligible, key=lambda r: r['microbatch'])['microbatch']
    profile.update(status='passed', selected_microbatch=chosen, selected_accumulation=32//chosen,
                   optimizer_state_dtype='float32', tied_head_preserved=True)
    write_json(destination, profile)
    print('PROFILE_PASSED', chosen, flush=True)


if __name__ == '__main__':
    main()

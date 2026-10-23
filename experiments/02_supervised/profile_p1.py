#!/usr/bin/env python3
"""Server26 train-only P1 precision, gradient and full-update preflight."""
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
import torch
import yaml
from jev.data.typed_decisions import iter_decisions, load_cases, prepare_splits
from jev.models.lm_readout import load_local_model
from jev.models.pointer import PointerHead, prepare_pointer_case, extract_pointer_features
from jev.training.kernels import set_delta_backend
from jev.training.lora import inject_lora, hard_candidate_loss
from jev.training.pointer_lora import (
    adapter_hash, assert_pointer_lora_trainable, cache_pointer_decisions,
    collate_pointer_decisions, create_joint_optimizer, frozen_backbone_state,
    joint_train_update, pointer_lora_logits,
)
from jev.training.pointer_runner import head_hash
from jev.training.runner import make_scheduler, runtime_fingerprint, write_json


def sampled_gradients(model, head):
    samples = {'lora': [], 'head': []}
    parameters = ([('lora', n, p) for n, p in model.named_parameters() if p.requires_grad]
                  + [('head', n, p) for n, p in head.named_parameters()])
    for group, name, parameter in parameters:
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f'Missing/nonfinite {group} gradient: {name}')
        flat = parameter.grad.detach().reshape(-1)
        count = min(512, flat.numel())
        indices = torch.arange(count, device=flat.device) * (flat.numel()-1) // max(1, count-1)
        samples[group].append(flat[indices].float().cpu())
    return {group: torch.cat(values) for group, values in samples.items()}


def hash_lora_a(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if '.lora_A.' in name:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    if socket.gethostname() != 'alg26' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('P1 preflight requires server26 physical GPU0; no GPU work on20')
    directory = ROOT/'outputs/02_supervised/p1'
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory/'preflight_train.json'
    profile = {'status': 'running', 'train_only': True, 'hostname': socket.gethostname(),
               'physical_gpu': 0, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
               'python_executable': sys.executable, 'runtime_fingerprint': runtime_fingerprint(ROOT/'.runtime/fla')}
    write_json(destination, profile)
    try:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        torch.backends.cudnn.conv.fp32_precision = 'ieee'
        device = torch.device('cuda:0')
        profile['device_name'] = torch.cuda.get_device_name(device)
        if '4090' not in profile['device_name']:
            raise RuntimeError('Expected the designated RTX4090')
        profile['gpu_info'] = subprocess.check_output([
            'nvidia-smi', '-i', '0', '--query-gpu=index,uuid,name,driver_version,memory.total',
            '--format=csv,noheader'], text=True).strip()
        base = yaml.safe_load((ROOT/'experiments/02_supervised/configs/p0.yaml').read_text())
        raw = ROOT/base['dataset']['raw_dir']
        splits = prepare_splits(load_cases(raw), ROOT/base['dataset']['split_dir'], seed=42,
                                source_manifest=raw/'download_manifest.json')
        model, tokenizer, loading = load_local_model(base['model'])
        model.requires_grad_(False).eval()
        train = list(iter_decisions(splits['train']))
        cached = cache_pointer_decisions(train, tokenizer)
        if len(cached) != 4800:
            raise RuntimeError('Expected exactly4800 train questions')
        profile.update(train_decisions=len(cached), max_train_tokens=max(len(r['input_ids']) for r in cached),
                       loading_info=loading, probe_case_ids=[r['identity'] for r in cached[:8]])
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        head = PointerHead().to(device).eval()
        lm_calls = []
        def reject_lm_projection(module, args):
            lm_calls.append(1)
            raise RuntimeError('P1 must bypass the original LM vocabulary projection')
        model.lm_head.register_forward_pre_hook(reject_lm_projection)
        prepared = prepare_pointer_case(splits['train'][0]['decisions'], tokenizer)
        original = extract_pointer_features(model, prepared)
        with torch.no_grad():
            reference = head(original['query_features'].to(device), original['candidate_features'].to(device),
                             original['candidate_mask'].to(device)).cpu()
        model = inject_lora(model, rank=16, alpha=32, dropout=0., expected_count=186).eval()
        profile['native_backend'] = set_delta_backend(model, 'native')
        counts = assert_pointer_lora_trainable(model, head)
        if counts['trainable_parameters'] != 11084800 or counts['frozen_parameters'] != 752393024:
            raise RuntimeError('Unexpected P1 parameter partition')
        profile['parameters'] = counts
        frozen = frozen_backbone_state(model)
        profile['frozen_backbone_before'] = frozen
        with torch.no_grad():
            actual = pointer_lora_logits(model, head, collate_pointer_decisions(cached[:5], pad)).cpu()
        if not torch.equal(reference, actual):
            raise RuntimeError(f'Zero-LoRA native Pointer parity differs: {(reference-actual).abs().max().item()}')
        profile['zero_lora_pointer_native_bitexact'] = True
        profile['head_scores_dtype'] = str(actual.dtype)
        print('Zero-LoRA Pointer parity passed; checking native/FLA gradients', flush=True)
        gradients = {}
        batch = collate_pointer_decisions(cached[:2], pad)
        for kind in ('native', 'fla'):
            set_delta_backend(model, kind, ROOT/'.runtime/fla')
            model.train(); head.train(); model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
            started = time.perf_counter()
            scores = pointer_lora_logits(model, head, batch)
            if scores.dtype != torch.float32:
                raise RuntimeError('Head scores must remain FP32')
            loss = hard_candidate_loss(scores, batch['option_counts'], batch['hard_targets'])
            loss.backward()
            gradients[kind] = sampled_gradients(model, head)
            torch.cuda.synchronize()
            profile[kind+'_gradient_probe'] = {'loss': float(loss.detach()), 'seconds': time.perf_counter()-started,
                                               'all_trainable_gradients_finite': True}
        agreement = {}
        for group in ('lora', 'head'):
            a, b = gradients['native'][group].double(), gradients['fla'][group].double()
            cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
            relative = float((a-b).norm()/a.norm().clamp_min(1e-30))
            agreement[group] = {'cosine': cosine, 'relative_l2': relative}
            if cosine < .95 or relative > .35:
                raise RuntimeError(f'Native/FLA {group} gradient mismatch: {agreement[group]}')
        profile['gradient_agreement'] = agreement
        del gradients
        model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
        optimizer = create_joint_optimizer(model, head)
        scheduler = make_scheduler(optimizer, 450, .05)
        before_adapter, before_head, before_a = adapter_hash(model), head_hash(head), hash_lora_a(model)
        longest = sorted(range(len(cached)), key=lambda i: len(cached[i]['input_ids']), reverse=True)[:32]
        trials = []
        for micro in (8, 4):
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            try:
                updates = [joint_train_update(model, head, cached, longest, pad, optimizer, scheduler, micro, 1.)
                           for _ in range(3)]
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated(device)
                if peak > .9*torch.cuda.get_device_properties(device).total_memory:
                    trials.append({'microbatch': micro, 'status': 'insufficient_headroom',
                                   'peak_allocated_bytes': peak})
                    optimizer.zero_grad(set_to_none=True)
                    continue
                trial = {'microbatch': micro, 'accumulation': 32//micro, 'status': 'passed',
                         'full_32_question_updates': 3, 'peak_allocated_bytes': peak,
                         'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                         'seconds': time.perf_counter()-started, 'updates': updates}
                trials.append(trial)
                break
            except torch.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
                trials.append({'microbatch': micro, 'status': 'oom'})
        profile['memory_trials'] = trials
        if not trials or trials[-1]['status'] != 'passed':
            raise RuntimeError('No training microbatch passed')
        if adapter_hash(model) == before_adapter or head_hash(head) == before_head or hash_lora_a(model) == before_a:
            raise RuntimeError('Joint updates did not change Head, LoRA B and subsequent LoRA A')
        if frozen_backbone_state(model) != frozen:
            raise RuntimeError('Original parameters or buffers changed')
        set_delta_backend(model, 'native'); model.eval(); head.eval()
        with torch.no_grad():
            after = pointer_lora_logits(model, head, collate_pointer_decisions(cached[:5], pad))
        if not torch.isfinite(after).all():
            raise RuntimeError('Nonfinite native inference after joint training')
        profile.update(status='passed', selected_microbatch=trials[-1]['microbatch'],
                       selected_accumulation=trials[-1]['accumulation'], lm_head_projection_calls=len(lm_calls),
                       head_and_lora_updated=True, lora_a_updated_after_nonzero_b=True,
                       frozen_backbone_after=frozen_backbone_state(model), hidden_feature_cache=False)
        print('P1_PREFLIGHT_PASSED', profile['selected_microbatch'], agreement, flush=True)
    except BaseException as error:
        profile.update(status='failed', error={'type': type(error).__name__, 'message': str(error),
                                             'traceback': traceback.format_exc()})
        raise
    finally:
        write_json(destination, profile)


if __name__ == '__main__':
    main()

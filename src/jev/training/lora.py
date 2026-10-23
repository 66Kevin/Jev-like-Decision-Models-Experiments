"""G1: frozen original weights, LoRA projections and hard candidate-only loss.

The data formatter and exact letter tokenization are delegated to B0. Targets
are never forwarded to the language model, whose original LM Head stays frozen.
"""
import re

import torch
from torch import nn
from torch.nn import functional as F

from jev.models.lm_readout import prepare_case


TARGET_SUFFIXES = frozenset(('q_proj', 'k_proj', 'v_proj', 'o_proj',
                             'in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b',
                             'out_proj', 'gate_proj', 'up_proj', 'down_proj'))
_EXCLUDED_PATHS = frozenset(('lm_head', 'embed_tokens', 'embedding', 'embeddings', 'word_embeddings'))
_ADAPTER_NAME = re.compile(r'\.lora_[AB]\.[^.]+\.weight$')
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def discover_lora_targets(model, expected_count=None):
    """Return complete Linear names rather than permissive PEFT suffix matching."""
    targets = []
    for name, module in model.named_modules():
        parts = name.split('.')
        if _EXCLUDED_PATHS.intersection(parts) or parts[-1] not in TARGET_SUFFIXES:
            continue
        if not isinstance(module, nn.Linear):
            raise ValueError(f'LoRA target must be Linear: {name} ({type(module).__name__})')
        targets.append(name)
    targets.sort()
    if not targets:
        raise ValueError('No supported Linear LoRA targets found')
    if expected_count is not None and len(targets) != expected_count:
        raise ValueError(f'LoRA target count {len(targets)} differs from expected count {expected_count}')
    return targets


def assert_adapter_only_trainable(model):
    """Fail closed if any base parameter or non-FP32 adapter can be optimized."""
    trainable, frozen, tensors = 0, 0, 0
    for name, parameter in model.named_parameters():
        adapter = bool(_ADAPTER_NAME.search(name))
        if adapter:
            prefix = name.split('.lora_', 1)[0].split('.')
            if (prefix[-1] not in TARGET_SUFFIXES or _EXCLUDED_PATHS.intersection(prefix)
                    or not parameter.requires_grad or parameter.dtype != torch.float32):
                raise ValueError(f'Invalid trainable LoRA parameter: {name}, {parameter.dtype}')
        elif parameter.requires_grad:
            raise ValueError(f'Original parameter must not be trainable: {name}')
        if parameter.requires_grad:
            trainable += parameter.numel()
            tensors += 1
        else:
            frozen += parameter.numel()
    if not trainable:
        raise ValueError('No trainable LoRA adapter parameters')
    return {'trainable_parameters': trainable, 'frozen_parameters': frozen,
            'total_parameters': trainable + frozen, 'trainable_tensors': tensors}


def inject_lora(model, rank=16, alpha=32, dropout=0.0, expected_count=None):
    """Attach unmerged PEFT adapters, preserving original/tied weights in place."""
    from peft import LoraConfig, get_peft_model

    if rank <= 0 or alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError('Invalid LoRA rank, alpha or dropout')
    if any('lora_' in name for name, _ in model.named_parameters()):
        raise ValueError('Model already contains LoRA adapters')
    targets = discover_lora_targets(model, expected_count)
    model.requires_grad_(False)
    # No task_type wrapper: forward arguments (notably logits_to_keep) must be
    # passed through unchanged to the native Qwen3.5 implementation.
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                        bias='none', target_modules=targets, inference_mode=False,
                        init_lora_weights=True)
    wrapped = get_peft_model(model, config, autocast_adapter_dtype=True)
    for name, parameter in wrapped.named_parameters():
        adapter = bool(_ADAPTER_NAME.search(name))
        parameter.requires_grad_(adapter)
        if adapter and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
    assert_adapter_only_trainable(wrapped)
    wrapped._jev_lora_target_modules = tuple(targets)
    return wrapped


def _integer_vector(values, name, device=None):
    try:
        result = torch.as_tensor(values, device=device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f'{name} must be an integer vector') from error
    if result.ndim != 1 or result.dtype not in _INTEGER_DTYPES:
        raise ValueError(f'{name} must be an integer vector')
    return result.long()


def cache_decisions(decisions, tokenizer, max_input_tokens=8192):
    """Cache B0 prompt tokens once; retain only hard targets and row identities."""
    cached = []
    for decision in decisions:
        target = _integer_vector([decision['hard_target']], 'hard target')[0].item()
        count = len(decision['option_ids'])
        if not 0 <= target < count:
            raise ValueError('Hard target is outside valid candidates')
        prepared = prepare_case([decision], tokenizer, max_input_tokens=max_input_tokens)
        cached.append({'input_ids': prepared['input_ids'][0].tolist(),
                       'hard_target': target, 'option_count': count,
                       'label_token_ids': list(prepared['label_token_ids']),
                       'identity': {key: decision[key] for key in
                                    ('case_id', 'question_id', 'workflow')}})
    return cached


def collate_decisions(cached, pad_token_id):
    """Left-pad arbitrary questions; keep target tensors separate from inputs."""
    if not cached:
        raise ValueError('No cached decisions supplied')
    if pad_token_id is None:
        raise ValueError('A padding token ID is required')
    lengths = [len(row['input_ids']) for row in cached]
    if not all(lengths):
        raise ValueError('Empty cached prompt')
    counts = [row['option_count'] for row in cached]
    max_count = max(counts)
    labels = list(max(cached, key=lambda row: row['option_count'])['label_token_ids'])
    if not 2 <= max_count <= 26 or len(labels) != max_count or len(set(labels)) != len(labels):
        raise ValueError('Invalid cached candidate letter mapping')
    targets = _integer_vector([row['hard_target'] for row in cached], 'hard targets')
    for row, target in zip(cached, targets.tolist()):
        count = row['option_count']
        if (not isinstance(count, int) or not 2 <= count <= max_count
                or row['label_token_ids'] != labels[:count]
                or not 0 <= target < count):
            raise ValueError('Invalid cached candidate count, mapping or target')
    ids = torch.full((len(cached), max(lengths)), pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(cached):
        ids[index, -lengths[index]:] = torch.tensor(row['input_ids'], dtype=torch.long)
        mask[index, -lengths[index]:] = 1
    return {'input_ids': ids, 'attention_mask': mask,
            'position_ids': (mask.cumsum(-1) - 1).clamp(min=0),
            'input_lengths': lengths, 'option_counts': counts, 'label_token_ids': labels,
            'hard_targets': targets, 'identities': [dict(row['identity']) for row in cached]}


def candidate_logits(model, prepared):
    """Differentiable native B0 readout; never compute full-vocabulary label CE."""
    if len(prepared['input_ids']) == 1 and (prepared['attention_mask'] == 0).any():
        raise ValueError('Native Qwen3.5 does not mask DeltaNet padding for a padded singleton')
    device = model.device
    tensors = {key: prepared[key].to(device) for key in
               ('input_ids', 'attention_mask', 'position_ids')}
    output = model(**tensors, use_cache=False, logits_to_keep=1)
    return output.logits[:, -1, prepared['label_token_ids']].float()


def hard_candidate_loss(logits, option_counts, targets):
    """FP32 hard CE over each row's valid letters, including a strict mask."""
    if logits.ndim != 2 or not logits.is_floating_point() or logits.shape[0] == 0:
        raise ValueError('Expected a nonempty two-dimensional floating logits tensor')
    counts = _integer_vector(option_counts, 'option counts', logits.device)
    targets = _integer_vector(targets, 'hard targets', logits.device)
    if len(counts) != len(logits) or len(targets) != len(logits):
        raise ValueError('Expected one option count and hard target per logits row')
    if ((counts < 2) | (counts > logits.shape[1])).any():
        raise ValueError('Invalid candidate count')
    if ((targets < 0) | (targets >= counts)).any():
        raise ValueError('Hard target is outside valid candidates')
    valid = torch.arange(logits.shape[1], device=logits.device)[None, :] < counts[:, None]
    if not torch.isfinite(logits[valid]).all():
        raise ValueError('Nonfinite valid candidate logits')
    masked = logits.float().masked_fill(~valid, -torch.inf)
    return F.cross_entropy(masked, targets)

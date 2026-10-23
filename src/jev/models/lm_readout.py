"""B0: frozen Qwen3.5, exact next-token letter readout, no generation."""
import json
import string
import time

import torch

from jev.data.typed_decisions import model_input


SYSTEM_PROMPT = (
    'You are a decision classifier. Read the state, instructions, and candidates. '
    'Choose the single best candidate. Reply with only its uppercase letter '
    '(A, B, C, etc.), without explanation or punctuation.'
)


def build_messages(decision):
    task = model_input(decision)
    count = len(task['option_ids'])
    if not 2 <= count <= 26 or len(task['option_texts']) != count:
        raise ValueError('B0 requires between 2 and 26 aligned candidates')
    candidates = '\n'.join(f'{letter}. {identity}: {text}' for letter, identity, text in
                           zip(string.ascii_uppercase, task['option_ids'], task['option_texts']))
    state = json.dumps(task['state'], ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False)
    content = (f'State:\n{state}\n\nDecision type: {task["type"]}\n\n'
               f'Instructions:\n{task["instructions"]}\n\nCandidates:\n{candidates}\n\n'
               'Answer with one candidate letter:')
    return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': content}]


def prepare_case(decisions, tokenizer, max_input_tokens=8192):
    """Each question has its own context; pad on the left and never truncate."""
    if not decisions:
        raise ValueError('No decisions supplied')
    sequences, prompts, counts = [], [], []
    max_count = max(len(d['option_ids']) for d in decisions)
    if max_count > 26:
        raise ValueError('B0 supports at most 26 candidate letters')
    label_ids = []
    for letter in string.ascii_uppercase[:max_count]:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f'Label {letter} must be a single token')
        label_ids.append(ids[0])
    if len(set(label_ids)) != len(label_ids):
        raise ValueError('Candidate letters must have distinct token IDs')
    for decision in decisions:
        prompt = tokenizer.apply_chat_template(build_messages(decision), tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > max_input_tokens:
            raise ValueError(f'{decision["case_id"]}/{decision["question_id"]}: '
                             f'{len(ids)} tokens exceeds {max_input_tokens}; truncation forbidden')
        if not ids:
            raise ValueError('Empty tokenized prompt')
        count = len(decision['option_ids'])
        for letter, token_id in zip(string.ascii_uppercase[:count], label_ids[:count]):
            continuation = tokenizer.encode(prompt + letter, add_special_tokens=False)
            if continuation != ids + [token_id]:
                raise ValueError(f'Label {letter} is not an exact single-token continuation')
        sequences.append(ids); prompts.append(prompt); counts.append(count)
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if pad is None:
        raise ValueError('Tokenizer has no padding or EOS token')
    width = max(map(len, sequences))
    input_ids = torch.full((len(sequences), width), pad, dtype=torch.long)
    mask = torch.zeros_like(input_ids)
    for i, ids in enumerate(sequences):
        input_ids[i, -len(ids):] = torch.tensor(ids)
        mask[i, -len(ids):] = 1
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    return {'input_ids': input_ids, 'attention_mask': mask, 'position_ids': positions,
            'input_lengths': list(map(len, sequences)), 'option_counts': counts,
            'label_token_ids': label_ids, 'prompts': prompts}


def label_probabilities(label_logits, option_counts):
    if label_logits.ndim != 2 or len(label_logits) != len(option_counts):
        raise ValueError('Expected one logits row for each decision')
    probabilities = []
    for row, count in zip(label_logits, option_counts):
        if not 2 <= count <= len(row) or not torch.isfinite(row[:count]).all():
            raise ValueError('Invalid candidate count or nonfinite label logits')
        probabilities.append(torch.softmax(row[:count].float(), dim=-1).cpu().tolist())
    return probabilities


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def read_case(model, prepared):
    """Native forward projects only the last position; normalize valid label logits."""
    if len(prepared['input_ids']) == 1 and (prepared['attention_mask'] == 0).any():
        raise ValueError('Native Qwen3.5 does not mask DeltaNet padding for a padded singleton')
    device = model.device
    synchronize(device)
    started = time.perf_counter()
    tensors = {key: prepared[key].to(device) for key in
               ('input_ids', 'attention_mask', 'position_ids')}
    output = model(**tensors, use_cache=False, logits_to_keep=1)
    selected = output.logits[:, -1, prepared['label_token_ids']].float()
    probabilities = label_probabilities(selected, prepared['option_counts'])
    logits = [row[:count].cpu().tolist() for row, count in
              zip(selected, prepared['option_counts'])]
    synchronize(device)
    return {'probabilities': probabilities, 'label_logits': logits,
            'inference_seconds': time.perf_counter() - started}


def load_local_model(config):
    """Load the text tower directly and fail closed on any weight mismatch."""
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

    if config['dtype'] != 'bfloat16':
        raise ValueError('B0 protocol uses bfloat16')
    if not config.get('local_files_only', True):
        raise ValueError('B0 model must load from local files only')
    tokenizer = AutoTokenizer.from_pretrained(config['path'], local_files_only=True)
    text_config = Qwen3_5TextConfig.from_pretrained(config['path'], local_files_only=True)
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        config['path'], config=text_config, key_mapping={r'^model.language_model\.': 'model.'},
        dtype=torch.bfloat16, attn_implementation=config['attention_implementation'],
        device_map={'': config['device']}, local_files_only=True, output_loading_info=True)
    if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError(f'Checkpoint weight loading mismatch: {loading}')
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer, loading

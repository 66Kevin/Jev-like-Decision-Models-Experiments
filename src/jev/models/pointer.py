"""P0 native frozen text features and shared candidate-matching Head.

The formatter and token IDs remain identical to B0. Candidate endpoints are
located through the complete anchored candidate block, never word searches.
"""
import math
import string

import torch

from jev.data.typed_decisions import model_input
from jev.models.lm_readout import build_messages, prepare_case


def prepare_pointer_case(decisions, tokenizer, max_input_tokens=8192, max_candidates=5):
    """Extend B0 inputs with left-pad-adjusted query/candidate token positions.

    Candidate character spans include the original full text (trailing spaces
    included); positions select the last token covering its final nonspace char.
    Invalid candidate slots have position zero and a false mask.
    """
    if not getattr(tokenizer, 'is_fast', False):
        raise ValueError('Pointer positions require a fast tokenizer with offsets')
    if not isinstance(max_candidates, int) or not 2 <= max_candidates <= 26:
        raise ValueError('max_candidates must be between 2 and 26')
    prepared = prepare_case(decisions, tokenizer, max_input_tokens)
    if max(prepared['option_counts']) > max_candidates:
        raise ValueError('Candidate count exceeds max_candidates')
    batch, width = prepared['input_ids'].shape
    candidate_positions = torch.zeros((batch, max_candidates), dtype=torch.long)
    candidate_mask = torch.zeros((batch, max_candidates), dtype=torch.bool)
    all_spans = []
    for row, decision in enumerate(decisions):
        task = model_input(decision)
        content = build_messages(decision)[1]['content']
        lines = [f'{letter}. {identity}: {text}' for letter, identity, text in
                 zip(string.ascii_uppercase, task['option_ids'], task['option_texts'])]
        suffix = '\n\nCandidates:\n' + '\n'.join(lines) + '\n\nAnswer with one candidate letter:'
        if not content.endswith(suffix):
            raise ValueError('Candidate block must be the exact anchored user-content suffix')
        prompt = prepared['prompts'][row]
        content_start = prompt.find(content)
        if content_start < 0 or prompt.find(content, content_start + 1) >= 0:
            raise ValueError('Rendered prompt must contain unique complete user content')
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded.get('input_ids')
        offsets = encoded.get('offset_mapping')
        length = prepared['input_lengths'][row]
        baseline_ids = prepared['input_ids'][row, width - length:].tolist()
        if ids != baseline_ids:
            raise ValueError('Full-prompt offset tokenization differs from B0 encoding')
        if offsets is None or len(offsets) != length:
            raise ValueError('Missing or misaligned full-prompt token offsets')
        for span in offsets:
            if (len(span) != 2 or any(not isinstance(value, int) for value in span)
                    or not 0 <= span[0] <= span[1] <= len(prompt)):
                raise ValueError('Invalid tokenizer offset mapping')
        cursor = content_start + len(content) - len(suffix) + len('\n\nCandidates:\n')
        spans = []
        for index, (letter, identity, text) in enumerate(zip(
                string.ascii_uppercase, task['option_ids'], task['option_texts'])):
            if not isinstance(text, str) or not text.rstrip():
                raise ValueError('Candidate text must not be empty or whitespace only')
            prefix = f'{letter}. {identity}: '
            start, stop = cursor + len(prefix), cursor + len(prefix) + len(text)
            if prompt[start:stop] != text:
                raise ValueError('Candidate text span differs from anchored candidate block')
            final_character = start + len(text.rstrip()) - 1
            covering = [i for i, (begin, end) in enumerate(offsets)
                        if begin <= final_character < end]
            if not covering:
                raise ValueError('No token offset covers final candidate character')
            candidate_positions[row, index] = width - length + covering[-1]
            candidate_mask[row, index] = True
            spans.append((start, stop))
            cursor = stop + 1  # One newline separates complete candidate lines.
        all_spans.append(spans)
    prepared.update(query_positions=torch.full((batch,), width - 1, dtype=torch.long),
                    candidate_positions=candidate_positions, candidate_mask=candidate_mask,
                    candidate_character_spans=all_spans)
    return prepared


def _validate_mask(mask, batch, candidates):
    if mask.dtype != torch.bool or mask.shape != (batch, candidates):
        raise ValueError('Candidate mask must be a correctly shaped bool tensor')
    counts = mask.sum(-1)
    if not torch.all((counts >= 2) & (counts <= 26)):
        raise ValueError('Each decision needs between 2 and 26 valid candidates')
    prefix = torch.arange(candidates, device=mask.device)[None, :] < counts[:, None]
    if not torch.equal(mask, prefix):
        raise ValueError('Valid candidate mask must be a contiguous prefix')


def extract_pointer_features(model, prepared):
    """Run only the frozen text Backbone; return ordinary FP32 CPU features.

    Call outside inference_mode: returned tensors must remain usable as inputs
    saved for backward by the trainable Head. No parameter casts or mode changes
    occur here, preserving native BF16 weights and native FP32 rotary buffers.
    """
    if torch.is_inference_mode_enabled():
        raise ValueError('Feature extraction must run outside inference_mode')
    if model.training or model.model.training:
        raise ValueError('Frozen Backbone must be in eval mode')
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('All original model parameters must remain frozen')
    input_ids = prepared['input_ids']
    if len(input_ids) == 1 and (prepared['attention_mask'] == 0).any():
        raise ValueError('Native Qwen3.5 does not mask DeltaNet padding for a padded singleton')
    mask = prepared['candidate_mask']
    positions = prepared['candidate_positions']
    if positions.ndim != 2:
        raise ValueError('Expected a candidate-position matrix')
    _validate_mask(mask, len(input_ids), positions.shape[1])
    device = next(model.parameters()).device
    tensors = {key: prepared[key].to(device) for key in
               ('input_ids', 'attention_mask', 'position_ids')}
    with torch.no_grad():
        output = model.model(**tensors, use_cache=False, return_dict=True)
        hidden = output.last_hidden_state
        if hidden.ndim != 3 or hidden.shape[:2] != input_ids.shape:
            raise ValueError('Native Backbone returned incompatible hidden states')
        query_positions = prepared['query_positions'].to(device)
        candidate_positions = positions.to(device)
        batch_indices = torch.arange(len(input_ids), device=device)
        query = hidden[batch_indices, query_positions]
        candidates = hidden[batch_indices[:, None], candidate_positions]
        candidates = candidates.masked_fill(~mask.to(device)[..., None], 0)
        if not torch.isfinite(query).all() or not torch.isfinite(candidates).all():
            raise ValueError('Native Backbone features must be finite')
        return {'query_features': query.detach().float().cpu().clone(),
                'candidate_features': candidates.detach().float().cpu().clone(),
                'candidate_mask': mask.detach().cpu().clone(),
                'option_counts': list(prepared['option_counts']),
                'query_positions': prepared['query_positions'].detach().cpu().clone(),
                'candidate_positions': positions.detach().cpu().clone()}


class PointerHead(torch.nn.Module):
    """Shared bias-free matching projections, no position-specific parameters.

    Raw FP32 scores are zero at invalid slots. Call candidate-only hard CE or
    label_probabilities with the original counts; never softmax padded slots.
    Fixed-feature permutation equivariance does not imply prompt invariance.
    """
    def __init__(self, hidden_size=1024, projection_dim=128):
        super().__init__()
        if not isinstance(hidden_size, int) or not isinstance(projection_dim, int) or min(hidden_size, projection_dim) < 1:
            raise ValueError('Hidden and projection dimensions must be positive integers')
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.query_projection = torch.nn.Linear(hidden_size, projection_dim, bias=False, dtype=torch.float32)
        self.candidate_projection = torch.nn.Linear(hidden_size, projection_dim, bias=False, dtype=torch.float32)
        torch.nn.init.xavier_uniform_(self.query_projection.weight)
        torch.nn.init.xavier_uniform_(self.candidate_projection.weight)

    def forward(self, query_features, candidate_features, candidate_mask):
        if (query_features.ndim != 2 or candidate_features.ndim != 3
                or query_features.shape[0] != candidate_features.shape[0]
                or query_features.shape[-1] != self.hidden_size
                or candidate_features.shape[-1] != self.hidden_size):
            raise ValueError('Incompatible query/candidate feature shapes')
        _validate_mask(candidate_mask, len(query_features), candidate_features.shape[1])
        if any(value.dtype != torch.float32 for value in (query_features, candidate_features)):
            raise ValueError('Pointer Head requires FP32 query and candidate features')
        if any(value.device != self.query_projection.weight.device for value in
               (query_features, candidate_features, candidate_mask)):
            raise ValueError('Pointer features, mask and Head must share a device')
        candidates = candidate_features.masked_fill(~candidate_mask[..., None], 0)
        if not torch.isfinite(query_features).all() or not torch.isfinite(candidates).all():
            raise ValueError('Valid Pointer features must be finite')
        query = self.query_projection(query_features)
        keys = self.candidate_projection(candidates)
        scores = (query[:, None, :] * keys).sum(-1) / math.sqrt(self.projection_dim)
        scores = scores.masked_fill(~candidate_mask, 0)
        if not torch.isfinite(scores).all():
            raise ValueError('Pointer Head scores must be finite')
        return scores

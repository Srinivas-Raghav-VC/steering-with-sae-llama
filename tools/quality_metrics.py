#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quality metrics utilities:
- Perplexity computation using the current LM
- Sentence-level cosine similarity using model hidden states (pooled)

Designed to avoid extra dependencies and work offline with the LLM already loaded.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch


@torch.no_grad()
def compute_perplexity(model, tokenizer, text: str, max_length: int = 512) -> float:
    """Compute perplexity of the given text under the LM.

    Uses teacher forcing: shift labels and average token NLL. Limited to max_length tokens.
    """
    if not text:
        return float("nan")
    device = next(model.parameters()).device
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)

    # labels = input_ids; model will compute cross-entropy loss
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = out.loss
    try:
        ppl = float(math.exp(loss.item()))
    except OverflowError:
        ppl = float("inf")
    return ppl


@torch.no_grad()
def cosine_similarity_model(model, tokenizer, a: str, b: str, max_length: int = 256) -> float:
    """Compute cosine similarity between two texts using pooled last hidden states.

    Pooling: 0.7 * masked mean + 0.3 * last valid token.
    """
    if not a or not b:
        return float("nan")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    def _encode(text: str) -> torch.Tensor:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, use_cache=False, output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[-1]  # [1, T, D]
        mask = enc["attention_mask"].unsqueeze(-1)  # [1, T, 1]
        mean_pool = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        # Find last valid position
        lengths = enc["attention_mask"].sum(dim=1)  # [1]
        last_idx = (lengths - 1).clamp(min=0)
        last_vec = hidden[torch.arange(hidden.size(0), device=device), last_idx, :]
        pooled = 0.7 * mean_pool + 0.3 * last_vec
        return pooled.squeeze(0).to(dtype=torch.float32)

    va = _encode(a)
    vb = _encode(b)
    # Cosine similarity
    if va.numel() == 0 or vb.numel() == 0:
        return float("nan")
    va = torch.nn.functional.normalize(va, dim=-1)
    vb = torch.nn.functional.normalize(vb, dim=-1)
    sim = torch.dot(va, vb).item()
    return float(sim)

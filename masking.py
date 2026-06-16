# ============================================================
# smeft_rl/masking.py
# Logit masking (FIX-C), KL schedule (FIX-D),
# epsilon schedule (FIX-NEW), and random sequence generation.
# ============================================================

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    TOP_K, MAX_NEW_TOKENS,
    KL_COEF_START, KL_COEF_END, KL_WARMUP,
    EPSILON_START, EPSILON_END, EPSILON_DECAY,
    DEVICE,WC_NAME_MAP,
)
from .vocab import stoi, PAD_ID, START_ID, END_ID, UNK_ID
#from .config import OP_SCORES

# pre-compute valid operator token IDs for random generation
VALID_OP_IDS = [stoi[op] for op in WC_NAME_MAP.keys() if op in stoi]


def mask_logits(logits, step, emitted_ids=None):
    """
    FIX-C: Apply logit masking during generation (sampling only).
    Never call this inside recompute_logp — see generate.py for explanation.

    Rules:
      - Always block PAD, UNK, START
      - Suppress already-emitted operators (no repeats)
      - Force END at step >= MAX_NEW_TOKENS
      - Top-k filter at step >= 1
    """
    logits = logits.clone()
    logits[:, PAD_ID]   = -1e9
    logits[:, UNK_ID]   = -1e9
    logits[:, START_ID] = -1e9

    if emitted_ids is not None and emitted_ids.size(1) > 0:
        V        = logits.size(1)
        suppress = torch.zeros(logits.size(0), V, dtype=torch.bool,
                               device=logits.device)
        suppress.scatter_(1, emitted_ids.clamp(0, V - 1), True)
        suppress[:, PAD_ID]   = False
        suppress[:, START_ID] = False
        suppress[:, END_ID]   = False
        logits[suppress] = -1e9

    if step >= MAX_NEW_TOKENS:
        forced            = torch.full_like(logits, -1e9)
        forced[:, END_ID] = 0.0
        return forced

    # FIX-C: top-k filter — concentrate probability mass
    if step >= 1:
        topk_vals, _ = torch.topk(logits, TOP_K, dim=-1)
        threshold     = topk_vals[:, -1].unsqueeze(1)
        logits        = logits.masked_fill(logits < threshold, -1e9)

    return logits


def safe_softmax(logits):
    logits = torch.nan_to_num(logits, nan=-1e4, posinf=-1e4, neginf=-1e4)
    return F.softmax(logits, dim=-1)


def get_kl_coef(batch_idx):
    """FIX-D: KL coefficient — near-zero early, ramps up later."""
    if batch_idx >= KL_WARMUP:
        return KL_COEF_END
    t = batch_idx / KL_WARMUP
    return KL_COEF_START + t * (KL_COEF_END - KL_COEF_START)


def get_epsilon(batch_idx):
    """FIX-NEW: epsilon-greedy decay schedule."""
    return max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** batch_idx))


def generate_random_seqs(n, min_len=2, max_len=4):
    """
    FIX-NEW: Sample short random sequences from valid operator IDs.
    No domain knowledge — just uses the operator vocab.
    """
    seqs = []
    for _ in range(n):
        length = np.random.randint(min_len, max_len + 1)
        chosen = np.random.choice(
            VALID_OP_IDS,
            size=min(length, len(VALID_OP_IDS)),
            replace=False,
        )
        seq = [START_ID] + list(chosen) + [END_ID]
        seqs.append(torch.tensor(seq, dtype=torch.long, device=DEVICE))

    max_l  = max(s.size(0) for s in seqs)
    padded = torch.stack([
        F.pad(s, (0, max_l - s.size(0)), value=PAD_ID) for s in seqs
    ])
    return padded

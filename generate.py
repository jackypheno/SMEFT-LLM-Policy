# ============================================================
# smeft_rl/generate.py
# Autoregressive generation and log-prob recomputation.
#
# KEY BUG FIXES (do not revert):
#
#   generate(): mask_logits is applied to POLICY logits only.
#     Applying it to ref_policy logits causes -1e9 sentinel values
#     when the policy samples a token masked in the reference
#     distribution (e.g. END_ID). This corrupts pol_ref_lp and
#     causes KL explosion.
#
#   recompute_logp(): mask_logits is NOT applied at all.
#     We are evaluating log P(token | context) for tokens that
#     were already sampled. The true unmasked distribution must
#     be used. Applying mask_logits here causes -1e9 for any
#     target token the mask disfavors (especially END_ID).
# ============================================================

import torch
import torch.nn.functional as F

from .config  import MAX_NEW_TOKENS, ADV_STD_MIN, DEVICE
from .masking import mask_logits, safe_softmax

from .vocab import stoi, PAD_ID, START_ID, END_ID, UNK_ID

def generate(policy, prompts, ref_policy=None):
    """
    Autoregressively generate sequences from policy.
    Returns (sequences, policy_logp_sum, ref_logp_sum, ended).

    ref_policy logits are intentionally NOT masked — see module docstring.
    """
    B_   = prompts.size(0)
    out  = prompts.clone()
    logp_sum     = torch.zeros(B_, device=DEVICE)
    ref_logp_sum = torch.zeros(B_, device=DEVICE)
    ended        = torch.zeros(B_, dtype=torch.bool, device=DEVICE)

    for step in range(MAX_NEW_TOKENS + 1):

        # ── policy: mask then sample ─────────────────────────
        logits = policy(out)[:, -1, :]
        logits = mask_logits(logits, step, out)   # mask applies to policy only
        probs  = safe_softmax(logits)
        nxt    = torch.distributions.Categorical(probs).sample()

        lp       = F.log_softmax(logits, dim=-1)
        lp_taken = lp.gather(1, nxt.unsqueeze(1)).squeeze(1)
        logp_sum = logp_sum + lp_taken * (~ended).float()

        # ── reference: unmasked true distribution ────────────
        if ref_policy is not None:
            with torch.no_grad():
                ref_logits = ref_policy(out)[:, -1, :]
                # NO mask_logits here — see module docstring
                ref_lp       = F.log_softmax(ref_logits, dim=-1)
                ref_lp_taken = ref_lp.gather(1, nxt.unsqueeze(1)).squeeze(1)
            ref_logp_sum = ref_logp_sum + ref_lp_taken * (~ended).float()

        # ── append token ─────────────────────────────────────
        nxt_append        = nxt.clone()
        nxt_append[ended] = PAD_ID
        out   = torch.cat([out, nxt_append.unsqueeze(1)], dim=1)
        ended |= (nxt == END_ID)
        if ended.all():
            break

    return out, logp_sum, ref_logp_sum, ended


def recompute_logp(policy, seqs):
    """
    Recompute log P(seq | policy) for a batch of already-generated sequences.
    Used during the PPO update step.

    mask_logits is deliberately NOT applied — see module docstring.
    """
    B_, T        = seqs.shape
    logp_sum     = torch.zeros(B_, device=DEVICE)
    entropy_sum  = torch.zeros(B_, device=DEVICE)
    step_count   = torch.zeros(B_, device=DEVICE)
    ended        = torch.zeros(B_, dtype=torch.bool, device=DEVICE)

    for step in range(1, T):
        context = seqs[:, :step]
        target  = seqs[:, step]

        logits  = policy(context)[:, -1, :]
        # NO mask_logits — must use true unmasked distribution

        lp       = F.log_softmax(logits, dim=-1)
        lp_taken = lp.gather(1, target.clamp(0).unsqueeze(1)).squeeze(1)

        probs   = lp.exp()
        entropy = -(probs * lp.clamp(min=-50)).sum(dim=-1)

        is_real     = (target != PAD_ID) & (~ended)
        logp_sum    = logp_sum    + lp_taken * is_real.float()
        entropy_sum = entropy_sum + entropy  * is_real.float()
        step_count  = step_count  + is_real.float()
        ended      |= (target == END_ID)

    entropy_mean = (entropy_sum / step_count.clamp(min=1)).mean()
    return logp_sum, entropy_mean

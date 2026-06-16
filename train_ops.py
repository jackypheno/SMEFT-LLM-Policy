# ============================================================
# smeft_rl/train_ops.py
# PPO loss, advantage normalization (FIX-B), replay buffer
# update (FIX-NEW3), and optimizer/scheduler construction.
# ============================================================

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .config  import (
    ADV_STD_MIN, EPS, ENTROPY_COEF, REPLAY_FRAC,
    REPLAY_MIN_REWARD, LR_PEAK, N_BATCHES, DEVICE,
)
from .generate import recompute_logp

from .vocab import stoi, PAD_ID, START_ID, END_ID, UNK_ID

def build_optimizer_and_scheduler(model, n_batches=N_BATCHES):
    """FIX-F: AdamW + OneCycleLR warmup."""
    opt = torch.optim.AdamW(model.parameters(), lr=LR_PEAK)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr           = LR_PEAK,
        total_steps      = n_batches,
        pct_start        = 5 / n_batches,
        anneal_strategy  = "cos",
        div_factor       = 10,
        final_div_factor = 100,
    )
    return opt, scheduler


def compute_advantages(all_rewards, all_policy_mask):
    """
    FIX-B: Group-relative advantage normalization (GRPO-style).
    Each group normalized independently using policy-arm rewards only.
    Returns list of G advantage tensors, same shape as all_rewards[i].
    """
    advantages = []
    for i in range(len(all_rewards)):
        r_i   = all_rewards[i]
        mask  = all_policy_mask[i]
        r_pol = r_i[mask]

        if r_pol.numel() > 1:
            r_mean = r_pol.mean()
            r_std  = r_pol.std().clamp(min=ADV_STD_MIN)
        else:
            r_mean = r_i.mean()
            r_std  = torch.tensor(ADV_STD_MIN, device=DEVICE)

        advantages.append((r_i - r_mean) / r_std)
    return advantages


def ppo_update(
    model, opt,
    all_seqs, all_ref_logp_sum, all_rewards,
    all_policy_mask, advantages,
    replay_buffer, kl_coef,
    n_update_steps, batch_idx,
):
    """
    Run N_UPDATE_STEPS PPO gradient updates for one batch.
    Returns final (loss, grad_norm, log_ratio) for diagnostics.
    """
    last_loss      = None
    last_grad_norm = None
    last_log_ratio = None

    for update_step in range(n_update_steps):
        opt.zero_grad()
        losses = []

        for i in range(len(all_seqs)):
            mask = all_policy_mask[i]
            if mask.sum() == 0:
                continue

            pol_seqs = all_seqs[i][mask]
            pol_ref  = all_ref_logp_sum[i][mask]
            pol_adv  = advantages[i][mask]

            new_logp_sum, entropy_mean = recompute_logp(model, pol_seqs)

            raw_log_ratio = new_logp_sum - pol_ref
            log_ratio     = torch.clamp(raw_log_ratio, -3, 3)

            # diagnostic print — first update step, first group only
            if update_step == 0 and i == 0:
                print(f"\n  [diag] batch={batch_idx} update={update_step} group={i}")
                print(f"    new_logp_sum  mean={new_logp_sum.mean():.2f}  "
                      f"min={new_logp_sum.min():.2f}  max={new_logp_sum.max():.2f}")
                print(f"    pol_ref       mean={pol_ref.mean():.2f}  "
                      f"min={pol_ref.min():.2f}  max={pol_ref.max():.2f}")
                print(f"    log_ratio RAW mean={raw_log_ratio.mean():.2f}  "
                      f"min={raw_log_ratio.min():.2f}  max={raw_log_ratio.max():.2f}")
                print(f"    log_ratio CLAMPED mean={log_ratio.mean():.2f}  "
                      f"hitting_clamp={(raw_log_ratio.abs()>3).float().mean():.1%}")
                print(f"    sentinel check: {(pol_ref.abs()>100).sum().item()} remaining")

            ratio     = torch.exp(log_ratio)
            unclipped = ratio * pol_adv
            clipped   = torch.clamp(ratio, 1 - EPS, 1 + EPS) * pol_adv
            ppo_loss  = -torch.min(unclipped, clipped)
            kl_pen    = kl_coef * log_ratio.abs()

            losses.append((ppo_loss + kl_pen).mean() - ENTROPY_COEF * entropy_mean)
            last_log_ratio = log_ratio

        # FIX-NEW3: replay buffer loss
        if len(replay_buffer) >= 8:
            from .config import B
            n_replay  = max(1, int(REPLAY_FRAC * B))
            rb_idxs   = np.random.choice(len(replay_buffer), n_replay, replace=False)
            rb_seqs   = pad_sequence(
                [replay_buffer[j]["seq"] for j in rb_idxs],
                batch_first=True, padding_value=PAD_ID
            ).to(DEVICE)
            rb_rewards = torch.tensor(
                [replay_buffer[j]["reward"] for j in rb_idxs],
                device=DEVICE, dtype=torch.float,
            )
            rb_logp, _ = recompute_logp(model, rb_seqs)
            rb_adv     = ((rb_rewards - rb_rewards.mean()) /
                          rb_rewards.std().clamp(min=ADV_STD_MIN))
            replay_loss = -(rb_logp * rb_adv).mean() * REPLAY_FRAC
            losses.append(replay_loss)

        loss = torch.stack(losses).mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        last_loss      = loss
        last_grad_norm = float(grad_norm)

    return last_loss, last_grad_norm, last_log_ratio

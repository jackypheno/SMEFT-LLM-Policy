import sys
from pathlib import Path
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib

sys.path.insert(
    0,
    "/home/kumarj/links/projects/def-london/kumarj/gitrepos/smeft-anomaly"
)

from smeft_new.pulls_linear import (
    constraints_ewp,
    CHI2_SM,
    flavio_pull_fn,
    BASELINE_PULLS ,
)


OBS_NAMES = constraints_ewp

from smeft_new.state import SMEFTState, step
from smeft_new.helpers import print_policy_diagnostics
from smeft_new.sensitivity_matrix import M_ij


# ------------------------------------------------------------
# LOAD OPERATOR VOCABULARY
# ------------------------------------------------------------

import argparse
import warnings
import flavio
import wilson
from wilson import wcxf
        
warnings.filterwarnings("ignore")

def _build_catalogue_from_wcxf() -> dict[str, list[str]]:
    """
    Query the wcxf package for every WC in the SMEFT Warsaw basis and
    organise them by their wcxf 'sector' label.

    Returns
    -------
    dict  {sector_name: [wc_name, ...]}  — ordered by sector, then WC name
    """
    try:
        basis_obj = wcxf.Basis["SMEFT", "Warsaw"]
        catalogue: dict[str, list[str]] = {}
        for sector_name, sector_data in basis_obj.sectors.items():
            if sector_name == "dB=de=dmu=dtau=0":
                wcs = sorted(sector_data.keys())
                if wcs:
                    catalogue[sector_name] = wcs
        return catalogue
    except Exception as exc:
        warnings.warn(
            f"Could not load Warsaw basis from wcxf ({exc}). "
            "Falling back to a minimal built-in catalogue."
        )
        return _fallback_catalogue()
OPERATOR_CATALOGUE: dict[str, list[str]] = _build_catalogue_from_wcxf()
operator_vocab: list[str] = [wc for ops in OPERATOR_CATALOGUE.values() for wc in ops]
print(f"Loaded {len(operator_vocab)} operators")
print(operator_vocab[:100])

# Adding a copy method to SMEFTState class
# This assumes SMEFTState is a dataclass or similar simple class
def _smeftstate_copy(self):
    # Need to also copy active_ops and step for a complete state copy
    new_state = SMEFTState(pulls=np.copy(self.pulls), chi2=self.chi2, obs_names=self.obs_names)
    new_state.active_ops = list(self.active_ops) if hasattr(self, 'active_ops') else []
    new_state.step = self.step if hasattr(self, 'step') else 0
    return new_state

SMEFTState.copy = _smeftstate_copy

def _smeftstate_to_dict(self):
    return {
        "pulls": self.pulls.tolist(),
        "chi2": self.chi2,
        "obs_names": self.obs_names,
        "active_ops": getattr(self, 'active_ops', []),
        "step": getattr(self, 'step', 0),
    }

SMEFTState.to_dict = _smeftstate_to_dict


SEED = 42

print(f"Using seed = {SEED}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_smeft_policy_diagnostics(history: dict, save_path: str = None):
    """
    Generates a 6-panel diagnostic dashboard for SMEFT operator sampling RL agent.

    The function automatically handles histories with unequal lengths.
    """

    sns.set_theme(style="whitegrid", palette="muted")

    fig, axes = plt.subplots(
        3, 2,
        figsize=(16, 14),
        sharex=True
    )

    # ------------------------------------------------------------
    # Convert steps to a 1D NumPy array
    # ------------------------------------------------------------
    steps = np.asarray(history["step"]).ravel()

    # ------------------------------------------------------------
    # Helper: exponential moving average
    # ------------------------------------------------------------
    def ema(y, weight=0.85):
        y = np.asarray(y, dtype=float).ravel()

        if len(y) == 0:
            return y

        smoothed = np.empty_like(y)

        smoothed[0] = y[0]

        for i in range(1, len(y)):
            smoothed[i] = (
                weight * smoothed[i - 1]
                + (1.0 - weight) * y[i]
            )

        return smoothed

    # ------------------------------------------------------------
    # Helper: align a history array with training steps
    #
    # If the history is shorter than steps, use the most recent
    # matching steps.
    # ------------------------------------------------------------
    def align_with_steps(values, name="unknown"):
        values = np.asarray(values, dtype=float).ravel()

        n = min(len(steps), len(values))

        if n == 0:
            print(f"Warning: '{name}' has no data.")
            return np.array([]), np.array([])

        if len(steps) != len(values):
            print(
                f"Warning: length mismatch for '{name}': "
                f"steps={len(steps)}, values={len(values)}. "
                f"Plotting the last {n} entries."
            )

        return steps[-n:], values[-n:]

    # ============================================================
    # 1. Mean Reward vs Time
    # ============================================================

    ax = axes[0, 0]

    x, reward = align_with_steps(
        history["reward"],
        "reward"
    )

    ax.plot(
        x,
        reward,
        alpha=0.3,
        color="steelblue",
        label="Raw"
    )

    ax.plot(
        x,
        ema(reward),
        color="steelblue",
        lw=2,
        label="EMA Smoothed"
    )

    ax.set_title(
        r"1. Trajectory Mean Reward ($-\Delta\chi^2$)",
        fontweight="bold"
    )

    ax.set_ylabel("Reward")
    ax.legend(loc="upper left")

    # ============================================================
    # 2. Loss Breakdown
    # ============================================================

    ax = axes[0, 1]

    x, policy_loss = align_with_steps(
        history["policy_loss"],
        "policy_loss"
    )

    ax.plot(
        x,
        ema(policy_loss),
        label="Policy Loss",
        color="crimson"
    )

    x, value_loss = align_with_steps(
        history["value_loss"],
        "value_loss"
    )

    ax.plot(
        x,
        ema(value_loss),
        label="Value Loss",
        color="darkorange"
    )

    x, entropy = align_with_steps(
        history["entropy"],
        "entropy"
    )

    ax.plot(
        x,
        ema(entropy),
        label="Entropy Bonus",
        color="purple"
    )

    ax.set_title(
        "2. Loss Components",
        fontweight="bold"
    )

    ax.set_ylabel("Loss Value")
    ax.set_yscale("symlog")
    ax.legend(loc="upper right")

    # ============================================================
    # 3. Gradient Norms
    # ============================================================

    ax = axes[1, 0]

    x, policy_grad = align_with_steps(
        history["policy_grad_norm"],
        "policy_grad_norm"
    )

    ax.plot(
        x,
        policy_grad,
        label="Policy Net Grad Norm",
        color="teal",
        alpha=0.7
    )

    x, value_grad = align_with_steps(
        history["value_grad_norm"],
        "value_grad_norm"
    )

    ax.plot(
        x,
        value_grad,
        label="Value Net Grad Norm",
        color="coral",
        alpha=0.7
    )

    ax.axhline(
        1.0,
        linestyle="--",
        color="gray",
        label="Gradient Clip Threshold"
    )

    ax.set_title(
        r"3. Gradient Norms ($L_2$)",
        fontweight="bold"
    )

    ax.set_ylabel(r"$||g||_2$")
    ax.set_yscale("log")
    ax.legend(loc="upper right")

    # ============================================================
    # 4. Operator Sampling Probabilities
    # ============================================================

    ax = axes[1, 1]

    for op_name, probs in history["operator_probs"].items():

        x, probs = align_with_steps(
            probs,
            f"operator_probs[{op_name}]"
        )

        if len(probs) == 0:
            continue

        ax.plot(
            x,
            ema(probs, weight=0.8),
            label=fr"${op_name}$",
            lw=1.8
        )

    ax.set_title(
        r"4. Operator Softmax Probabilities $\pi(a|s)$",
        fontweight="bold"
    )

    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.0)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1)
    )

    # ============================================================
    # 5. Policy Entropy
    # ============================================================

    ax = axes[2, 0]

    x, entropy = align_with_steps(
        history["entropy"],
        "entropy"
    )

    ax.plot(
        x,
        entropy,
        color="mediumpurple",
        lw=2,
        label=r"Policy Entropy $H(\pi)$"
    )

    ax.set_title(
        "5. Exploration / Policy Entropy",
        fontweight="bold"
    )

    ax.set_ylabel("Entropy (nits)")
    ax.set_xlabel("Training Steps / Episodes")
    ax.legend(loc="upper right")

    # ============================================================
    # 6. Best-Fit Chi^2
    # ============================================================

    ax = axes[2, 1]

    x, best_chi2 = align_with_steps(
        history["best_chi2"],
        "best_chi2"
    )

    ax.plot(
        x,
        best_chi2,
        color="firebrick",
        lw=2,
        label=r"Current Best $\chi^2$"
    )

    ax.set_title(
        r"6. Global Fit Optimization ($\chi^2$ Drop)",
        fontweight="bold"
    )

    ax.set_ylabel(r"$\chi^2$")
    ax.set_xlabel("Training Steps / Episodes")
    ax.legend(loc="upper right")

    # ------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------

    plt.tight_layout(pad=2.0)

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.close()


n_ops = len(operator_vocab)
n_obs = len(constraints_ewp)

M_ij = np.zeros(
    (n_ops,n_obs),
    dtype=np.float32
)

print("M_ij shape =", M_ij.shape)
print("operator_vocab =", len(operator_vocab))
print("constraints =", len(constraints_ewp))


"""
SMEFT policy training with STEP-LEVEL reward shaping.

WHAT CHANGED FROM THE ORIGINAL SCRIPT AND WHY
------------------------------------------------------------------
1. compute_rollout_reward() had a real syntax bug:
       if abs(delta_chi2) <= min_delta_chi2:
       reward -= 30.0          # <- not indented, IndentationError
   Fixed below.

2. THE CORE FIX (credit assignment):
   The original loop did:
       batch_log_probs.append(result["log_probs"].sum())
       ... actor_loss = -(advantage * log_probs_sum).mean()
   This applies ONE scalar (the trajectory reward) to every operator's
   log-prob equally. If a rollout picks [good, bad, bad, bad], the good
   operator gets the same negative gradient as the three bad ones.

   Fix: score each operator individually from its own chi2 contribution
   (compute_step_rewards), turn per-step rewards into a return-to-go
   G_t per step, and use the critic's V(s_t) as a baseline:
       A_t = G_t - V(s_t)
       actor_loss  = -(A_t.detach() * log_prob_t).mean()   summed over ALL steps
       critic_loss =  MSE(V(s_t), G_t.detach())
   Now a good operator gets a positive advantage even inside a trajectory
   that ends up bad overall, and vice versa.

3. The critic head was computed every step (state_val) but discarded.
   rollout() now stores it; the training loop now actually uses it as
   the baseline instead of ignoring it.

4. Top-20%-only trajectory filtering is removed. With step-level credit,
   you don't need to throw away 80% of rollouts to get a clean signal —
   every step from every rollout now contributes correctly-signed
   gradient. This also improves sample efficiency substantially.
"""

import os
import json
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. PHYSICS-GATED SMEFT POLICY (unchanged from original)
# ============================================================
class SMEFTTransformerPolicy(nn.Module):
    def __init__(
        self,
        op_vocab,
        obs_names,
        sensitivity_matrix,
        embed_dim=32,
        n_heads=4,
        alpha_init=1.0,
        pull_gain=2.5,
    ):
        super().__init__()
        self.op_vocab = op_vocab
        self.obs_names = obs_names
        self.num_ops = len(op_vocab)
        self.num_obs = len(obs_names)
        self.embed_dim = embed_dim

        sm_shape = tuple(sensitivity_matrix.shape)
        expected = (self.num_obs, self.num_ops)
        expected_T = (self.num_ops, self.num_obs)

        if sm_shape == expected:
            M = sensitivity_matrix.T
        elif sm_shape == expected_T:
            M = sensitivity_matrix
        else:
            raise ValueError(
                f"sensitivity_matrix shape {sm_shape} doesn't match either "
                f"(num_obs={self.num_obs}, num_ops={self.num_ops}) or transpose."
            )

        self.register_buffer("M", M.float())

        self.op_embed = nn.Embedding(self.num_ops, embed_dim)
        self.pull_proj = nn.Linear(1, embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, batch_first=True
        )

        self.actor_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1),
        )

        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(pull_gain, dtype=torch.float32))

    def forward(self, state, fired_ops=None):
        device = self.M.device

        pulls_data = state.pulls if hasattr(state, "pulls") else state["pulls"]
        if not isinstance(pulls_data, torch.Tensor):
            pulls_tensor = torch.tensor(pulls_data, dtype=torch.float32, device=device)
        else:
            pulls_tensor = pulls_data.to(device)

        if pulls_tensor.shape[-1] != self.num_obs:
            raise ValueError(
                f"pulls vector has length {pulls_tensor.shape[-1]} but expected "
                f"num_obs={self.num_obs}."
            )

        pulls = pulls_tensor.unsqueeze(-1)

        direct_physics_logits = torch.matmul(self.M, pulls).squeeze(-1) * self.beta

        op_ids = torch.arange(self.num_ops, device=device)
        op_k = self.op_embed(op_ids).unsqueeze(0)
        pull_v = self.pull_proj(pulls).unsqueeze(0)

        attn_out, attn_weights = self.cross_attn(query=op_k, key=pull_v, value=pull_v)

        neural_logits = self.actor_head(attn_out).squeeze(0).squeeze(-1) * self.alpha

        final_logits = neural_logits #+ direct_physics_logits

        if fired_ops is not None and len(fired_ops) > 0:
            mask = torch.zeros(self.num_ops, dtype=torch.bool, device=device)
            mask[list(fired_ops)] = True
            final_logits = final_logits.masked_fill(mask, -1e9)

        state_summary = torch.cat([op_k.mean(dim=1), pull_v.mean(dim=1)], dim=-1)
        state_value = self.critic_head(state_summary).squeeze()

        diagnostics = {
            "base_logits": neural_logits.detach(),
            "direct_logits": direct_physics_logits.detach(),
            "pulls": pulls_tensor.detach().cpu().numpy(),
        }

        return final_logits, state_value, attn_weights.squeeze(0), diagnostics


# ============================================================
# 2. STEP-LEVEL REWARD (the actual fix)
# ============================================================
def compute_step_rewards(
    history,
    alpha_step=0.1,
    dead_step_penalty=0.5,
    min_delta_chi2=1e-3,
    solved_bonus=10.0,
    chi2_threshold=1e-6,
):
    """
    One reward per proposed operator, in trajectory order.

    history[i]["delta_chi2"] = trial_chi2 - old_chi2 (negative = improvement)
    so step_chi2_drop = -delta_chi2 is positive when the operator helped.

    This is what actually solves "3 bad operators drag down the 1 good one":
    each operator's reward now comes from ITS OWN chi2 contribution, not
    from the trajectory's final outcome.
    """
    step_rewards = []
    for step in history:
        delta = step["delta_chi2"]
        step_chi2_drop = -delta

        r = step_chi2_drop - alpha_step

        if step_chi2_drop < min_delta_chi2:
            r -= dead_step_penalty

        if step["trial_chi2"] < chi2_threshold:
            r += solved_bonus

        step_rewards.append(r)
    return step_rewards


def compute_returns_to_go(step_rewards, gamma=0.99):
    """G_t = sum_{k=t}^{T} gamma^(k-t) * r_k, computed backward."""
    returns = [0.0] * len(step_rewards)
    running = 0.0
    for t in reversed(range(len(step_rewards))):
        running = step_rewards[t] + gamma * running
        returns[t] = running
    return returns


def compute_rollout_reward(
    final_state,
    baseline_chi2,
    trajectory_length=None,
    history=None,
    alpha_step=0.1,
    dead_step_penalty=0.5,
    min_delta_chi2=1e-3,
    solved_bonus=10.0,
    chi2_threshold=1e-6,
    clip_reward=False,
    clip_range=(-20.0, 20.0),
):
    """
    Trajectory-level summary reward. Still useful for logging / picking the
    best trajectory found, but NOT used for the policy gradient anymore —
    see compute_step_rewards() for that.
    """
    if isinstance(final_state, dict):
        final_chi2 = final_state["chi2_total"]
    elif hasattr(final_state, "chi2"):
        final_chi2 = final_state.chi2
    else:
        raise ValueError("final_state must contain chi2_total")

    delta_chi2 = baseline_chi2 - final_chi2
    reward = delta_chi2

    if abs(delta_chi2) <= min_delta_chi2:
        reward -= 30.0

    if final_chi2 < chi2_threshold:
        reward += solved_bonus

    if trajectory_length is not None:
        reward -= alpha_step * trajectory_length

    if history is not None:
        for step in history:
            step_delta = step.get("delta_chi2", 0.0) if isinstance(step, dict) else step
            step_chi2_drop = -step_delta
            if step_chi2_drop < min_delta_chi2:
                reward -= dead_step_penalty

    if clip_reward:
        reward = np.clip(reward, clip_range[0], clip_range[1])

    return float(reward), float(delta_chi2)


# ============================================================
# 3. ROLLOUT — now also stores per-step state values for the critic
# ============================================================
def rollout(
    policy,
    pull_fn,
    obs_names,
    baseline_pulls,
    baseline_chi2,
    max_steps=10,
    deterministic=False,
    verbose=True,
    trajectory_idx=1,
    total_trajectories=30,
):
    device = policy.M.device

    current_state = SMEFTState(
        pulls=np.array(baseline_pulls, dtype=np.float32),
        chi2=baseline_chi2,
        obs_names=obs_names,
    )
    current_state.active_ops = []
    current_state.step = 0

    all_evaluated_op_indices = set()
    accepted_op_indices = []

    log_probs = []
    entropies = []
    attentions = []
    diagnostics = []
    state_values = []  # <-- NEW: critic's V(s_t) at each step, previously discarded
    state_trajectory = [current_state.copy()]
    history = []

    if verbose:
        print(f"\n--- Trajectory [{trajectory_idx:02d} / {total_trajectories:02d}] ---")

    for step in range(max_steps):
        logits, state_val, attn_weights, diag = policy(
            current_state, fired_ops=all_evaluated_op_indices
        )

        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits) if deterministic else dist.sample()
        action_idx = action.item()

        all_evaluated_op_indices.add(action_idx)

        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        attentions.append(attn_weights.detach())
        diagnostics.append(diag)
        state_values.append(state_val)  # <-- NEW

        proposed_op = policy.op_vocab[action_idx]

        trial_ops = current_state.active_ops + [proposed_op]
        _, trial_pulls, trial_chi2, _, _ = pull_fn(trial_ops)

        trial_chi2 = float(trial_chi2)
        old_chi2 = float(current_state.chi2)

        chi2_reduction = old_chi2 - trial_chi2
        delta_raw = trial_chi2 - old_chi2

        accepted = False
        if chi2_reduction > deltachi2_min:#1e-2:
            accepted = True
            accepted_op_indices.append(action_idx)

            current_state = SMEFTState(
                pulls=np.array(trial_pulls, dtype=np.float32),
                chi2=trial_chi2,
                obs_names=obs_names,
            )
            current_state.active_ops = trial_ops
            current_state.step = step + 1
        else:
            current_state.step = step + 1

        state_trajectory.append(current_state.copy())

        if verbose:
            status = "ACCEPT" if accepted else "REJECT"
            print(
                f"STEP {step:02d}: ADD {proposed_op:20s} "
                f"Δχ²={delta_raw:+8.3f} (Drop: {chi2_reduction:8.3f}) {status}"
            )

        history.append({
            "step": step,
            "proposed_operator": proposed_op,
            "accepted": accepted,
            "state_before": state_trajectory[-2].to_dict(),
            "state_after": current_state.to_dict(),
            "trial_chi2": float(trial_chi2),
            "delta_chi2": float(delta_raw),
        })

    if verbose:
        final_ops_str = " -> ".join(current_state.active_ops) if current_state.active_ops else "None (Pure SM)"
        total_delta_chi2 = float(baseline_chi2) - current_state.chi2
        print(f"FINAL SEQUENCE [{len(current_state.active_ops)} ops]: {final_ops_str}")
        print(f"FINAL RESULTS   : Total Δχ² = -{total_delta_chi2:.3f} | Final χ² = {current_state.chi2:.3f}")

    return {
        "final_state": current_state,
        "state_trajectory": state_trajectory,
        "ops": accepted_op_indices,
        "op_names": current_state.active_ops,
        "log_probs": torch.stack(log_probs),
        "entropies": torch.stack(entropies),
        "state_values": torch.stack(state_values),  # <-- NEW
        "attention": attentions,
        "diagnostics": diagnostics,
        "history": history,
    }


# ============================================================
# 4. SETUP & INITIALIZATION
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "SMEFTState" not in globals():
    class SMEFTState:
        """Minimal fallback — replace with your real SMEFTState import."""
        def __init__(self, pulls, chi2, obs_names):
            self.pulls = pulls
            self.chi2 = chi2
            self.obs_names = obs_names
            self.active_ops = []
            self.step = 0

        def copy(self):
            s = SMEFTState(np.array(self.pulls), self.chi2, self.obs_names)
            s.active_ops = list(self.active_ops)
            s.step = self.step
            return s

        def to_dict(self):
            return {
                "pulls": np.asarray(self.pulls).tolist(),
                "chi2": self.chi2,
                "active_ops": list(self.active_ops),
                "step": self.step,
            }

if "operator_vocab" not in globals():
    operator_vocab = [f"op_{i}" for i in range(20)]
if "OBS_NAMES" not in globals():
    OBS_NAMES = [f"obs_{i}" for i in range(10)]
if "BASELINE_PULLS" not in globals():
    BASELINE_PULLS = np.random.randn(len(OBS_NAMES)).astype(np.float32)
if "CHI2_SM" not in globals():
    CHI2_SM = 100.0
if "M_ij" not in globals():
    M_ij = np.random.randn(len(OBS_NAMES), len(operator_vocab)).astype(np.float32)

if "flavio_pull_fn" not in globals():
    def flavio_pull_fn(op_list):
        fake_chi2 = CHI2_SM - len(op_list) * 2.5 + np.random.uniform(-0.5, 0.5)
        fake_pulls = np.random.randn(len(OBS_NAMES)).astype(np.float32)
        return None, fake_pulls, fake_chi2, True, None

M_tensor = torch.tensor(M_ij, dtype=torch.float32)
M_mean = M_tensor.mean()
M_std = M_tensor.std() + 1e-6
M_normalized = (M_tensor - M_mean) / M_std

model = SMEFTTransformerPolicy(
    op_vocab=operator_vocab,
    obs_names=OBS_NAMES,
    sensitivity_matrix=M_normalized,
    embed_dim=32,
    n_heads=4,
    alpha_init=1.0,
    pull_gain=2.5,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

#MAX_STEPS = 10
#ROLLOUTS_PER_BATCH = 25
#N_BATCHES = 40

# run 1 , got partial results
#MAX_STEPS = 10
#ROLLOUTS_PER_BATCH = 25
#N_BATCHES = 10

MAX_STEPS = 10
ROLLOUTS_PER_BATCH = 25
N_BATCHES = 150

deltachi2_min=1e-2 # condn to accept an operator

GRAD_CLIP = 0.5
GAMMA = 0.99             # discount for return-to-go within a trajectory
CRITIC_LOSS_COEF = 0.5
ADV_NORMALIZE = True     # normalize advantages across the whole batch of steps

OUTPUT_DIR = "/home/kumarj/links/scratch/tmp_real/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS_LOG_PATH = os.path.join(OUTPUT_DIR, "metrics_log.txt")
THEORY_ARCHIVE_PATH = os.path.join(OUTPUT_DIR, "theory_archive.json")
REWARD_DISTS_PATH = os.path.join(OUTPUT_DIR, "reward_distributions.json")

with open(METRICS_LOG_PATH, "w") as f:
    f.write(
        "batch\tmean_traj_reward\tmax_traj_reward\tmean_delta_chi2\t"
        "actor_loss\tcritic_loss\ttotal_loss\talpha\tbeta\t"
        "mean_advantage\tpolicy_grad_norm\n"
    )

if not os.path.exists(THEORY_ARCHIVE_PATH):
    with open(THEORY_ARCHIVE_PATH, "w") as f:
        json.dump([], f, indent=2)

if not os.path.exists(REWARD_DISTS_PATH):
    with open(REWARD_DISTS_PATH, "w") as f:
        json.dump({}, f, indent=2)

theory_archive = []
reward_distributions = {}

best_reward = -float("inf")
best_ops = None

# ============================================================
# 5. TRAINING LOOP — step-level credit assignment
# ============================================================
for batch_idx in range(N_BATCHES):
    model.train()

    progress = batch_idx / float(N_BATCHES)
    entropy_coef = 0.01 * (1.0 - progress) + 0.0001 * progress

    # trajectory-level, for logging / archive only
    batch_traj_rewards = []
    batch_delta_chi2 = []
    batch_ops = []

    # step-level, this is what the gradient is actually computed from
    all_log_probs = []
    all_entropies = []
    all_state_values = []
    all_returns = []

    batch_attn_correlations = []
    batch_base_norms = []
    batch_direct_norms = []

    print(f"\n==================== BATCH {batch_idx:03d} ====================")

    for rollout_idx in range(ROLLOUTS_PER_BATCH):
        result = rollout(
            policy=model,
            pull_fn=flavio_pull_fn,
            obs_names=OBS_NAMES,
            baseline_pulls=BASELINE_PULLS,
            baseline_chi2=CHI2_SM,
            max_steps=MAX_STEPS,
            deterministic=False,
            verbose=True,
            trajectory_idx=rollout_idx + 1,
            total_trajectories=ROLLOUTS_PER_BATCH,
        )
        final_state = result["final_state"]
        hist = result["history"]
        traj_ops = result["ops"]
        traj_len = len(traj_ops)

        # trajectory-level summary (logging / best-trajectory tracking only)
        traj_reward, delta_chi2 = compute_rollout_reward(
            final_state=final_state,
            baseline_chi2=CHI2_SM,
            trajectory_length=traj_len,
            history=hist,
            alpha_step=0.1,
            dead_step_penalty=0.5,
        )

        # --- STEP-LEVEL REWARD: this is the actual fix ---
        step_rewards = compute_step_rewards(
            hist, alpha_step=0.1, dead_step_penalty=0.5,
        )
        returns_to_go = compute_returns_to_go(step_rewards, gamma=GAMMA)
        returns_t = torch.tensor(returns_to_go, dtype=torch.float32, device=DEVICE)

        all_log_probs.append(result["log_probs"])
        all_entropies.append(result["entropies"])
        all_state_values.append(result["state_values"])
        all_returns.append(returns_t)

        theory_archive.append({
            "batch": batch_idx,
            "rollout": rollout_idx,
            "traj_reward": float(traj_reward),
            "step_rewards": [float(r) for r in step_rewards],
            "delta_chi2": float(delta_chi2),
            "chi2_min": float(final_state.chi2),
            "operators": result["op_names"],
            "pulls": np.asarray(final_state.pulls).tolist(),
            "history": hist,
        })

        with open(THEORY_ARCHIVE_PATH, "w") as f:
            json.dump(theory_archive, f, indent=2)

        batch_traj_rewards.append(float(traj_reward))
        batch_delta_chi2.append(float(delta_chi2))
        batch_ops.append(traj_ops)

        for attn_matrix, diag in zip(result["attention"], result["diagnostics"]):
            abs_pulls = np.abs(diag["pulls"])
            if np.sum(abs_pulls) > 0:
                step_corrs = []
                attn_np = attn_matrix.cpu().numpy()
                for op_idx in range(attn_np.shape[0]):
                    row_attn = attn_np[op_idx]
                    corr, _ = scipy.stats.spearmanr(row_attn, abs_pulls)
                    if not np.isnan(corr):
                        step_corrs.append(corr)
                if step_corrs:
                    batch_attn_correlations.append(np.mean(step_corrs))
            batch_base_norms.append(torch.norm(diag["base_logits"]).detach().cpu().item())
            batch_direct_norms.append(torch.norm(diag["direct_logits"]).detach().cpu().item())

        if traj_reward > best_reward:
            best_reward = traj_reward
            best_ops = list(traj_ops)

    reward_distributions[f"batch_{batch_idx}"] = batch_traj_rewards

    mean_attn_corr = float(np.mean(batch_attn_correlations)) if batch_attn_correlations else 0.0
    avg_base_norm = float(np.mean(batch_base_norms)) if batch_base_norms else 1e-8
    avg_direct_norm = float(np.mean(batch_direct_norms)) if batch_direct_norms else 0.0
    logit_component_ratio = avg_direct_norm / (avg_base_norm + 1e-8)

    # ------------------------------------------------------------
    # STEP-LEVEL ACTOR-CRITIC LOSS
    # Flatten ALL steps from ALL rollouts in the batch into one long
    # sequence of (log_prob, return, value) triples. No trajectory-level
    # top-k filtering: every step contributes, correctly signed.
    # ------------------------------------------------------------
    flat_log_probs = torch.cat(all_log_probs)
    flat_entropies = torch.cat(all_entropies)
    flat_values = torch.cat(all_state_values)
    flat_returns = torch.cat(all_returns)

    advantages = flat_returns - flat_values.detach()
    if ADV_NORMALIZE and advantages.numel() > 1 and advantages.std() > 1e-6:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    actor_loss = -(advantages.detach() * flat_log_probs).mean()
    critic_loss = F.mse_loss(flat_values, flat_returns.detach())
    entropy_loss = -entropy_coef * flat_entropies.mean()

    total_loss = actor_loss + CRITIC_LOSS_COEF * critic_loss + entropy_loss

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()

    raw_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP).item()
    optimizer.step()

    batch_mean_traj_reward = float(np.mean(batch_traj_rewards))
    batch_max_traj_reward = float(np.max(batch_traj_rewards))
    batch_mean_delta_chi2 = float(np.mean(batch_delta_chi2))
    mean_advantage = float(advantages.mean().item())

    with open(METRICS_LOG_PATH, "a") as f:
        f.write(
            f"{batch_idx}\t{batch_mean_traj_reward:.6f}\t{batch_max_traj_reward:.6f}\t"
            f"{batch_mean_delta_chi2:.6f}\t{actor_loss.item():.6f}\t{critic_loss.item():.6f}\t"
            f"{total_loss.item():.6f}\t{model.alpha.item():.6f}\t{model.beta.item():.6f}\t"
            f"{mean_advantage:.6f}\t{raw_grad_norm:.6f}\n"
        )

    print(f"\n>>> SUMMARY BATCH {batch_idx:03d} <<<")
    print(
        f"Actor: {actor_loss.item():.4f} | Critic: {critic_loss.item():.4f} | "
        f"Mean Δχ²: {batch_mean_delta_chi2:.2f} | Max Traj Reward: {batch_max_traj_reward:.1f} | "
        f"Logit Ratio: {logit_component_ratio:.3f}\n"
    )




with open("/home/kumarj/links/scratch/tmp_real/theory_archive.json", "r") as f:
    results = json.load(f)

# Keep only the best occurrence of each unique operator set
unique_theories = {}

for theory in results:

    # Operator ordering should not matter
    key = tuple(sorted(theory["operators"]))

    # Keep the one with the highest reward
    if (
        key not in unique_theories
        or theory["traj_reward"] > unique_theories[key]["traj_reward"]
    ):
        unique_theories[key] = theory

# Sort unique theories by reward
top_theories = sorted(
    unique_theories.values(),
    key=lambda theory: theory["traj_reward"],
    reverse=True
)[:20]

print("\n========== TOP 15 UNIQUE THEORIES ==========\n")

for rank, theory in enumerate(top_theories, start=1):

    print(
        f"{rank:2d}. "
        f"Reward = {theory['traj_reward']:.6f}, "
        f"Operators = {sorted(theory['operators'])}, "
        f"chi2_min = {theory['chi2_min']:.4f}, "
        f"Δχ² = {theory['delta_chi2']:.4f}"
    )    


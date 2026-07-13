"""
Order-invariant GRPO for SMEFT subset selection.

    candidate operators -> self-attention encoder -> one binary logit/operator
    -> sample the whole subset simultaneously.

The action is a binary vector a in {0,1}^N.  Its log-probability is the sum of
Bernoulli log-probabilities.  GRPO is otherwise unchanged: sample groups of
candidate subsets, compute group-relative advantages, and apply a clipped
policy-ratio update.

Important:
- The reward is evaluated ONLY on the complete subset.  It does not need
  to decompose into individual operator rewards.
- A shared latent context z is sampled once per candidate subset and fed to all
  operator heads.  Conditional on z the bits are Bernoulli, but marginalizing
  over z gives a mixture of Bernoulli policies, which can represent correlated
  operator choices better than a single fully factorized Bernoulli policy.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ============================================================
# SETUP
# ============================================================

repo = Path(__file__).parent
os.chdir(repo)
if repo not in sys.path:
    sys.path.insert(0, repo)

save_dir = repo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# OPERATOR VOCABULARY
# ============================================================

vocab = Path("smeft_rl/vocab_0f2f.txt").read_text().split()
SPECIAL = {"PAD", "START", "END", "UNK"}

# The policy acts on operators only.
OPS = [tok for tok in vocab if tok not in SPECIAL]
N_OPS = len(OPS)
op_to_idx = {op: i for i, op in enumerate(OPS)}

print(f"Device: {device}")
print(f"Number of candidate operators: {N_OPS}")

# ============================================================
# ORDER-INVARIANT SET POLICY
# ============================================================

class SMEFTSetPolicy(nn.Module):
    """
    Self-attention over the fixed set of candidate operators.

    Each operator gets a learned identity embedding.  The Transformer has no
    positional encoding, so permuting the operator table simply permutes the
    outputs (permutation equivariance).

    A per-rollout latent vector z is added as shared context to every operator.
    This makes the overall subset distribution a mixture of factorized
    Bernoulli distributions and allows correlated modes such as:
        mode 1 -> select A and B together
        mode 2 -> select C and D together

    """
    def __init__(self, n_ops,
        d_model=256, nhead=4, nlayers=4, dim_feedforward=512, latent_dim=64):
        super().__init__()
        self.n_ops = n_ops
        self.d_model = d_model
        self.latent_dim = latent_dim

        self.op_embedding = nn.Embedding(n_ops, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.final_ln = nn.LayerNorm(d_model)

        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # One binary logit for each context-aware operator embedding.
        self.selection_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def encode_operators(self):
        ids = torch.arange(self.n_ops, device=self.op_embedding.weight.device)
        h = self.op_embedding(ids)  # [N, D]

        h = self.encoder(h.unsqueeze(0)).squeeze(0)  # [N, D]
        return self.final_ln(h)

    def forward(self, z):
        """
        z: [B, latent_dim]
        returns logits: [B, N_OPS]
        """
        B = z.shape[0]
        op_h = self.encode_operators().unsqueeze(0).expand(B, -1, -1)
        context = self.latent_proj(z).unsqueeze(1).expand(-1, self.n_ops, -1)
        h = torch.cat([op_h, context], dim=-1)
        return self.selection_head(h).squeeze(-1)


model = SMEFTSetPolicy(N_OPS, d_model = 32, nhead = 1, nlayers = 1, dim_feedforward=32, latent_dim = 8).to(device)

# Initialization to avoid all the probabilities to be close to 0.5
target_initial_size = 3
p0 = target_initial_size / N_OPS
initial_bias = np.log(p0 / (1.0 - p0))
nn.init.constant_(model.selection_head[-1].bias, initial_bias)

# ============================================================
# HYPERPARAMETERS
# ============================================================

B = 50                  # number of GRPO groups
G = 10                   # candidate subsets per group
N_BATCHES = 50
N_UPDATE_STEPS = 20

EPS = 0.2
ADV_STD_MIN = 0.1
LR_PEAK = 1e-4

LATENT_DIM = model.latent_dim

# Exploration: mix policy samples with uniformly random sparse subsets.
EPSILON_START = 0.8
EPSILON_END = 0.05
EPSILON_DECAY = 0.88
RANDOM_MAX_OPS = 4

opt = torch.optim.AdamW(model.parameters(), lr=LR_PEAK)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    opt,
    max_lr=LR_PEAK,
    total_steps=N_BATCHES,
    pct_start=min(0.99, 5 / N_BATCHES),
    anneal_strategy="cos",
    div_factor=10,
    final_div_factor=100
)

OP_SCORES = {
    "G": 0.400,
    "phi": 0.200,
    "W": 0.300,
    "phiWB": 0.800,
    "phiD": 1.000,
    "phiW": 0.01,
    "phiB": 0.04,
    "phiWtilde": 0.08,
    "phiBox": 0.09
}

_sorted_scores = sorted(OP_SCORES.values(), reverse=True)
MAX_RAW_SCORE = sum(_sorted_scores[:3])
JUNK_PENALTY = 0.10
LENGTH_PENALTY = 0.15
FREE_OPS = 3


def _compute_reward(ops):
    if len(ops) == 0:
        return -0.5

    known_ops = set(OP_SCORES)
    good_ops = [op for op in ops if op in known_ops]
    junk_ops = [op for op in ops if op not in known_ops]

    # max score = 1
    score = sum(OP_SCORES[op] for op in good_ops) / MAX_RAW_SCORE
    # penalty for subsets with non contributing operators
    score -= JUNK_PENALTY * len(junk_ops)
    # penalty for subsets with more than 3 operators
    score -= LENGTH_PENALTY * max(0, len(ops) - FREE_OPS)

    # Kept only to preserve the original toy experiment.
    if good_ops:
        best_single = max(OP_SCORES[op] for op in good_ops) / MAX_RAW_SCORE
        score += 0.1 * best_single

    return float(np.clip(score, -1.0, 1.0))


# ============================================================
# REWARD CACHE / TRACKING
# ============================================================

reward_cache = {}
total_calls = 0
best_reward_so_far = -999.0
first_best_found = None
best_ops_so_far = None

def get_reward(ops):
    global total_calls, best_reward_so_far, first_best_found, best_ops_so_far

    key = tuple(sorted(ops))
    total_calls += 1

    if key not in reward_cache:
        reward_cache[key] = _compute_reward(ops)

    r = reward_cache[key]

    if r > best_reward_so_far:
        best_reward_so_far = r
        first_best_found = len(reward_cache)
        best_ops_so_far = list(ops)
        print(
            f"\n*** NEW BEST: reward={r:.4f} ops={ops} "
            f"at unique eval #{first_best_found} ***"
        )

    return r


# ============================================================
# BINARY SUBSET DISTRIBUTION
# ============================================================

@torch.no_grad()
def sample_policy(policy, n):
    """
    Sample n unordered subsets.

    z is stored with the rollout.  During GRPO updates, the same z is reused so
    the action log-probability can be recomputed exactly conditional on z.
    """
    z = torch.randn(n, LATENT_DIM, device=device)
    logits = policy(z)
    dist = torch.distributions.Bernoulli(logits=logits)
    actions = dist.sample()

    # Do NOT top-k/project actions here unless you also model that transformed
    # distribution. Prefer a size penalty in the reward.
    old_logp = dist.log_prob(actions).sum(dim=-1)

    return actions, z, old_logp


# ============================================================
# DIAGNOSTICS
# ============================================================

@torch.no_grad()
def run_pre_diagnostics():
    print("\n" + "=" * 70)
    print("PRE-TRAINING DIAGNOSTICS — ORDER-INVARIANT SET POLICY")
    print("=" * 70)

    z = torch.zeros(1, LATENT_DIM, device=device)
    probs = torch.sigmoid(model(z))[0]

    for op in OP_SCORES:
        if op in op_to_idx:
            print(f"{op:15s} p(select)={probs[op_to_idx[op]].item():.4f}")

    print(f"Expected subset size at z=0: {probs.sum().item():.2f}/{N_OPS}")
    print("=" * 70 + "\n")


def print_search_efficiency(batch_idx):
    unique_evals = len(reward_cache)
    duplicates = total_calls - unique_evals
    hit_rate = duplicates / max(total_calls, 1)

    print(f"\n{'=' * 55}")
    print(f"SEARCH EFFICIENCY (batch {batch_idx})")
    print(f"Unique subsets evaluated : {unique_evals:,}")
    print(f"Total reward calls       : {total_calls:,}")
    print(f"Cache hit rate           : {hit_rate:.1%}")
    print(f"Best reward              : {best_reward_so_far:.4f}")
    print(f"Best subset              : {best_ops_so_far}")
    print(f"Best found at unique eval: {first_best_found}")
    print(f"{'=' * 55}\n")


# ============================================================
# TRAINING
# ============================================================

run_pre_diagnostics()

theory_archive = []
HIGH_REWARD_THRESHOLD = 0.95
coverage_rl = set()
coverage_random = set()

model.train()

for batch_idx in range(N_BATCHES):

    epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** batch_idx))

    # --------------------------------------------------------
    # STEP 1: collect B groups, each with G candidate subsets
    # --------------------------------------------------------
    groups = []

    with torch.no_grad():
        for group_idx in range(B):
            # GRPO candidates from the current policy.
            actions, z, old_logp = sample_policy(model, G)

            rewards = []
            for j in range(G):
                # Action to operators
                ops = [OPS[i] for i in torch.nonzero(actions[j] > 0.5, as_tuple=False).flatten()]
                r = get_reward(ops)
                rewards.append(r)
                theory_archive.append({"ops": ops, "reward": r, "source": "policy"})

                if r >= HIGH_REWARD_THRESHOLD and ops:
                    coverage_rl.add(frozenset(ops))

            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

            # Group-relative advantage: this is the defining GRPO step.
            adv = (rewards - rewards.mean()) / rewards.std().clamp(min=ADV_STD_MIN)

            groups.append({"actions": actions, "z": z, "old_logp": old_logp, "rewards": rewards, "advantages": adv})

            # Separate random exploration for comparaison
            if np.random.rand() < epsilon:
                #    Uniformly sample sparse subsets from ALL operators
                rand_actions = torch.zeros(1, N_OPS, device=device)
                for b in range(1):
                    k = np.random.randint(1, RANDOM_MAX_OPS + 1)
                    ids = np.random.choice(N_OPS, size=min(k, N_OPS), replace=False)
                    rand_actions[b, torch.as_tensor(ids, dtype=torch.long, device=device)] = 1.0

                # Action to operators
                rand_ops = [OPS[i] for i in torch.nonzero(rand_actions[0] > 0.5, as_tuple=False).flatten()]
                rand_r = get_reward(rand_ops)
                theory_archive.append({
                    "ops": rand_ops,
                    "reward": rand_r,
                    "source": "random",
                })
                if rand_r >= HIGH_REWARD_THRESHOLD and rand_ops:
                    coverage_random.add(frozenset(rand_ops))

    # --------------------------------------------------------
    # STEP 2: GRPO clipped policy update
    # --------------------------------------------------------
    for update_step in range(N_UPDATE_STEPS):
        opt.zero_grad()
        losses = []

        for group in groups:
            actions = group["actions"]
            z = group["z"]
            old_logp = group["old_logp"]
            adv = group["advantages"]

            logits = model(z)
            # Log P(a | z)
            dist = torch.distributions.Bernoulli(logits=logits)
            new_logp = dist.log_prob(actions).sum(dim=-1)


            # GRPO importance ratio: NEW policy / rollout policy.
            log_ratio = torch.clamp(new_logp - old_logp, -3.0, 3.0)
            ratio = torch.exp(log_ratio)

            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - EPS, 1 + EPS) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()

            losses.append(policy_loss)

        loss = torch.stack(losses).mean()

        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    scheduler.step()

    # --------------------------------------------------------
    # STEP 3: diagnostics
    # --------------------------------------------------------
    if batch_idx % 5 == 0:
        all_batch_rewards = torch.cat([g["rewards"] for g in groups])
        all_batch_adv = torch.cat([g["advantages"] for g in groups])

        print_search_efficiency(batch_idx)
        print(
            f"lr={scheduler.get_last_lr()[0]:.2e} "
            f"reward mean={all_batch_rewards.mean().item():.3f} "
            f"std={all_batch_rewards.std().item():.3f} "
            f"max={all_batch_rewards.max().item():.3f}"
        )
        print(
            f"adv mean={all_batch_adv.mean().item():.3f} "
            f"std={all_batch_adv.std().item():.3f}"
        )
        print(f"grad norm={float(grad_norm):.3f}")
        print(f"coverage RL={len(coverage_rl)} random={len(coverage_random)}")

        # Estimate marginal selection frequencies from fresh policy samples.
        with torch.no_grad():
            diag_actions, _, _ = sample_policy(model, 256)
            freqs = diag_actions.mean(dim=0)
            sizes = diag_actions.sum(dim=1)

        print(
            f"sampled subset size: mean={sizes.mean().item():.2f}, "
            f"min={sizes.min().item():.0f}, max={sizes.max().item():.0f}"
        )
        for op in OP_SCORES:
            if op in op_to_idx:
                print(f"freq({op:15s})={freqs[op_to_idx[op]].item():.1%}")


# ============================================================
# FINAL OUTPUT
# ============================================================

best = {}
for item in theory_archive:
    key = tuple(sorted(item["ops"]))
    best[key] = max(best.get(key, -float("inf")), item["reward"])

print("\nTOP-15 UNIQUE SUBSETS")
for rank, (ops, reward) in enumerate(
    sorted(best.items(), key=lambda x: x[1], reverse=True)[:15], start=1
):
    print(f"{rank:>2}. reward={reward:.4f}  ops={list(ops)}")


print_search_efficiency(N_BATCHES - 1)
print("Training complete.")
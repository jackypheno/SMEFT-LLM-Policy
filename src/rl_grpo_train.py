# ============================================================
# rl_grpo_train.py
#
# Stage-2 RL fine-tuning for your SMEFT generator
# Simplified GRPO (Albert-style)
#
# Uses:
#   - your pretrained SMEFTPolicy
#   - masked generation
#   - group-relative rewards
#   - KL regularization to reference model
#   - Jaccard diversity bonus
#
# ============================================================

import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from pathlib import Path

from src.policy import SMEFTPolicy

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "/home/kumarj/links/scratch/anomaly/models/policy2/smeft_policy.pt"

LR = 1e-5
STEPS = 5000
GROUP_SIZE = 8
MAX_LEN = 128

TEMPERATURE = 0.9

BETA_KL = 0.01
ETA_JACCARD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# VOCAB
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
vocab = (BASE_DIR / "vocab.txt").read_text().split()

stoi = {t:i for i,t in enumerate(vocab)}
itos = {i:t for t,i in stoi.items()}
VOCAB_SIZE = len(vocab)

# ============================================================
# MODEL
# ============================================================

policy = SMEFTPolicy(VOCAB_SIZE).to(device)
policy.load_state_dict(torch.load(MODEL_PATH, map_location=device))
policy.train()

# frozen reference model
ref_model = deepcopy(policy).to(device)
ref_model.eval()

opt = torch.optim.AdamW(policy.parameters(), lr=LR)

# ============================================================
# GRAMMAR MASK
# ============================================================

def grammar_mask(seq_tokens):

    mask = torch.full((VOCAB_SIZE,), float("-inf"), device=device)

    last = seq_tokens[-1]

    def allow(tok):
        if tok in stoi:
            mask[stoi[tok]] = 0.0

    def allow_prefix(prefix):
        for t in vocab:
            if t.startswith(prefix):
                mask[stoi[t]] = 0.0

    # parse NUM_OPS
    target_ops = 1
    for tok in seq_tokens:
        if tok.startswith("NUM_OPS_"):
            target_ops = int(tok.split("_")[-1])

    completed = seq_tokens.count("OP_END")

    # ----------------------------------------------------

    if last == "THEORY_START":
        allow_prefix("NUM_OPS_")
        return mask

    if last.startswith("NUM_OPS_"):
        allow_prefix("SCALE_")
        return mask

    if last.startswith("SCALE_"):
        allow("OP_START")
        return mask

    if last == "OP_START":
        allow("CLASS_4F")
        return mask

    if last == "CLASS_4F":
        allow_prefix("BASE_")
        return mask

    if last.startswith("BASE_"):
        allow("p1"); allow("p2"); allow("p3")
        return mask

    if last.startswith("p"):
        allow("r1"); allow("r2"); allow("r3")
        return mask

    if last.startswith("r"):
        allow("s1"); allow("s2"); allow("s3")
        return mask

    if last.startswith("s"):
        allow("t1"); allow("t2"); allow("t3")
        return mask

    if last.startswith("t"):
        allow_prefix("COEF_")
        return mask

    if last.startswith("COEF_"):
        allow("OP_END")
        return mask

    if last == "OP_END":
        if completed < target_ops:
            allow("OP_START")
        else:
            allow("THEORY_END")
        return mask

    allow("THEORY_END")
    return mask


# ============================================================
# GENERATE + LOGPROB
# ============================================================

@torch.no_grad()
def generate_one(model, prompt=["THEORY_START"]):

    seq_tokens = prompt[:]
    seq_ids = [stoi[t] for t in seq_tokens]

    total_logprob = 0.0

    for _ in range(MAX_LEN):

        # converts the ids list to tensor 
        x = torch.tensor([seq_ids], dtype=torch.long, device=device)

        # model output = logits for last token position
        logits = model(x)[:, -1, :] / TEMPERATURE
        logits = logits.squeeze(0)

        # returns penalities =  [0, -inf, 0, -inf]
        mask = grammar_mask(seq_tokens)

        masked_logits = logits + mask

        probs = F.softmax(masked_logits, dim=-1)

        nxt = torch.multinomial(probs, 1).item()

        logp = torch.log(probs[nxt] + 1e-12)
        total_logprob += logp

        tok = itos[nxt]

        seq_ids.append(nxt)
        seq_tokens.append(tok)

        if tok == "THEORY_END":
            break

    return seq_tokens, total_logprob
#   returns state and total probs

# ============================================================
# REWARD
# ============================================================

def parse_bases(tokens):
    return [t for t in tokens if t.startswith("BASE_")]

def parse_coeffs(tokens):
    vals = []
    for t in tokens:
        if t.startswith("COEF_"):
            try:
                vals.append(float(t.replace("COEF_", "")))
            except:
                pass
    return vals


def reward_fn(tokens):
    # computes rewards for one trajectory= one theory
    reward = 0.0

    # -----------------------------
    # unique operators reward
    # -----------------------------
    bases = parse_bases(tokens)
    reward += len(set(bases))

    # -----------------------------
    # duplicate penalty
    # -----------------------------
    reward -= (len(bases) - len(set(bases))) * 1.5

    # -----------------------------
    # coefficient diversity
    # -----------------------------
    coeffs = parse_coeffs(tokens)

    if len(coeffs) > 1:
        reward += np.std(coeffs)

    # -----------------------------
    # valid ending bonus
    # -----------------------------
    if tokens[-1] == "THEORY_END":
        reward += 2.0

    return reward


# ============================================================
# JACCARD BONUS
# ============================================================

def jaccard(a, b):
    sa = set(parse_bases(a))
    sb = set(parse_bases(b))

    if len(sa | sb) == 0:
        return 0.0

    return len(sa & sb) / len(sa | sb)


def diversity_bonus(group):

    bonuses = []

    for i in range(len(group)):

        sims = []

        for j in range(len(group)):
            if i != j:
                sims.append(jaccard(group[i], group[j]))

        avg_sim = np.mean(sims) if sims else 0.0

        bonuses.append(-avg_sim)   # lower similarity = better

    return bonuses


# ============================================================
# RL TRAIN LOOP
# ============================================================

for step in range(STEPS):

    samples = []
    logps = []
    rewards = []

    # ----------------------------------------
    # sample group
    # ----------------------------------------
    for _ in range(GROUP_SIZE):
        # it loads state and total prob of trajectory
        toks, logp = generate_one(policy)
        # computes rewards for current state
        r = reward_fn(toks)

#   At each training step, the policy generates multiple trajectories, 
#   evaluates them with a reward function, and stores them for later optimization.
        samples.append(toks)
        logps.append(logp)
        rewards.append(r)

    # ----------------------------------------
    # diversity bonus
    # ----------------------------------------
    div_bonus = diversity_bonus(samples)

    rewards = np.array(rewards) + ETA_JACCARD * np.array(div_bonus)

    # ----------------------------------------
    # group-relative advantages
    # ----------------------------------------
    mean_r = rewards.mean()
    std_r = rewards.std() + 1e-8

    adv = (rewards - mean_r) / std_r

    # ----------------------------------------
    # policy loss
    # ----------------------------------------
    loss = 0.0

    for i in range(GROUP_SIZE):
        loss += -adv[i] * logps[i]

    loss = loss / GROUP_SIZE

    # ----------------------------------------
    # KL regularization
    # crude token-level KL using same prompt
    # ----------------------------------------
    prompt = torch.tensor([[stoi["THEORY_START"]]], device=device)

    p_logits = policy(prompt)[:, -1, :]
    r_logits = ref_model(prompt)[:, -1, :]

    p = F.log_softmax(p_logits, dim=-1)
    q = F.softmax(r_logits, dim=-1)

    kl = F.kl_div(p, q, reduction="batchmean")

    loss = loss + BETA_KL * kl

    # ----------------------------------------
    # optimize # update weights
    # ----------------------------------------
    opt.zero_grad()
    loss.backward()
    opt.step()

    # ----------------------------------------
    # print
    # ----------------------------------------
    if step % 20 == 0:

        best = int(np.argmax(rewards))

        print("="*60)
        print("step:", step)
        print("loss:", float(loss))
        print("reward mean:", rewards.mean())
        print("best sample:")
        print(" ".join(samples[best]))
        print("="*60)


# ============================================================
# SAVE
# ============================================================

torch.save(policy.state_dict(),
           "/home/kumarj/links/scratch/anomaly/models/policy2/smeft_policy_rl.pt")

print("RL fine-tuned model saved.")

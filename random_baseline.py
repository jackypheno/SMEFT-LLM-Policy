# random_baseline.py
"""
Random search baseline for SMEFT RL evaluation.
Run from Jupyter: %run random_baseline.py
Produces: random_baseline_results.pkl
"""
import pickle
import random
from itertools import combinations
from pathlib import Path
from .reward import _compute_reward, _physics_reward

# op_vocab, _compute_reward, all_seqs, high_seqs
# call with: %run random_baseline.py

def generate_random_baseline(
    vocab:        list,
    n_rollouts:   int   = 10_000,
    max_len:      int   = 5,
    min_reward:   float = 0.9,
    save_path:    str   = "random_baseline_results.pkl",
    seed:         int   = 42,
) -> dict:
    """
    Simulates a pure random search agent:
    - samples sequence length uniformly from 1..max_len
    - samples operators without replacement
    - records reward, length, ops, from_policy=False for every rollout
    """
    random.seed(seed)
    archive  = []
    high     = []

    print(f"Running random baseline: {n_rollouts} rollouts, max_len={max_len}")
    for i in range(n_rollouts):
        length = random.randint(1, max_len)
        seq    = tuple(sorted(random.sample(vocab, min(length, len(vocab)))))
        r      = _physics_reward(list(seq), None, True)

        entry = {
            "ops":         seq,
            "reward":      r,
            "from_policy": False,
            "rollout_idx": i,
        }
        archive.append(entry)
        if r >= min_reward:
            high.append(seq)

    # deduplicate high reward
    high_deduped = list(set(high))

    results = {
        "archive":      archive,
        "high_seqs":    high_deduped,
        "n_rollouts":   n_rollouts,
        "max_len":      max_len,
        "min_reward":   min_reward,
    }

    with open(save_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Done. {len(high_deduped)} unique high-reward sequences found.")
    print(f"Saved → {save_path}")
    return results

"""
special_tokens = {"START", "END", "UNK", "PAD"}
op_vocab = [v for v in vocab if v not in special_tokens]

if __name__ == "__main__":
    # uses op_vocab already in Jupyter namespace via %run
    results = generate_random_baseline(
        vocab      = op_vocab,
        n_rollouts = 10_000,
        max_len    = 5,
        min_reward = 0.9,
    )
"""    

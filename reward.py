# ============================================================
# smeft_rl/reward.py
# Reward computation, caching, and efficiency tracking.
# Edit _compute_reward() to change the reward function.
# ============================================================

from .config import (
    OP_SCORES, FREE_OPS, JUNK_PENALTY, LENGTH_PENALTY,
    TERMINATION_BONUS, MAX_LEN, WC_NAME_MAP,
)

from .shared import SEARCH_SPACE_SIZE
from .ground import all_seqs

from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter


# derived constant
_sorted_scores = sorted(OP_SCORES.values(), reverse=True)
MAX_RAW_SCORE  = sum(_sorted_scores[:3])

reward_cache       = {}
total_calls        = 0
best_reward_so_far = -999.0
first_best_found   = None
best_ops_so_far    = None
efficiency_history = []


def reset_reward_state():
    """Call this at the start of each experiment run."""
    global reward_cache, total_calls, best_reward_so_far
    global first_best_found, best_ops_so_far, efficiency_history
    reward_cache       = {}
    total_calls        = 0
    best_reward_so_far = -999.0
    first_best_found   = None
    best_ops_so_far    = None
    efficiency_history = []


def _compute_reward(ops, raw_ops, terminated):
    """
    Reward with partial credit shaping (FIX-NEW2).
    Edit this function to change the reward structure.
    """
    if len(ops) == 0:
        return -0.5

    known_ops = set(OP_SCORES.keys())
    good_ops  = [op for op in ops if op in known_ops]
    junk_ops  = [op for op in ops if op not in known_ops]

    score  = sum(OP_SCORES[op] for op in good_ops) / MAX_RAW_SCORE
    score -= JUNK_PENALTY * len(junk_ops)
#    score -= LENGTH_PENALTY * max(0, len(ops) - FREE_OPS)

    if len(ops) > MAX_LEN:
        score -= 1.0 + 0.5 * (len(ops) - MAX_LEN)   # hard penalty: 1.0 base + 0.5 per extra token
    else:
        score -= LENGTH_PENALTY * max(0, len(ops) - FREE_OPS)

    if terminated:
        score += TERMINATION_BONUS
    else:
        score -= 0.10

    # FIX-NEW2: partial credit for best single operator
    if good_ops:
        best_single = max(OP_SCORES[op] for op in good_ops) / MAX_RAW_SCORE
        score      += 0.1 * best_single

    return float(min(max(score, -1.0), 1.0))

"""
def _physics_reward(ops, raw_ops=None, terminated=None):
    if len(ops) == 0:
        return 0.0

    known_ops = set(OP_SCORES.keys())
    good_ops  = [op for op in ops if op in known_ops]

    if not good_ops:
        return 0.0

    score = sum(OP_SCORES[op] for op in good_ops) / MAX_RAW_SCORE
    return float(min(max(score, 0.0), 1.0))
"""

def _physics_reward(ops, raw_ops=None, terminated=None):
    if len(ops) == 0:
        return 0.0

    known_ops = set(OP_SCORES.keys())
    good_ops  = [op for op in ops if op in known_ops]

    if not good_ops:
        return 0.0

    # mean over ALL ops in sequence — zero-score ops dilute reward
    all_scores = [OP_SCORES.get(op, 0.0) for op in ops]
    score      = sum(all_scores) / (MAX_RAW_SCORE * len(ops))

    return float(min(max(score, 0.0), 1.0))


def get_reward(ops, raw_ops, terminated):
    """
    Cached reward lookup. Updates global tracking state.
    Prints a message when a new best reward is found.
    """
    global total_calls, best_reward_so_far, first_best_found, best_ops_so_far

    key = tuple(sorted(ops))
    total_calls += 1

    if key not in reward_cache:
        reward_cache[key] = _compute_reward(ops, raw_ops, terminated)

    r = reward_cache[key]

    if r > best_reward_so_far:
        best_reward_so_far = r
        first_best_found   = len(reward_cache)
        best_ops_so_far    = list(ops)
        print(f"\n  *** NEW BEST: reward={r:.4f}  ops={list(ops)}"
              f"  found at unique eval #{first_best_found}"
              f"  ({100*first_best_found/SEARCH_SPACE_SIZE:.1f}% coverage) ***")

    efficiency_history.append((len(reward_cache), best_reward_so_far))
    return r

def high_seqs(length: int, min_reward: float, vocab: list) -> tuple[int, list[tuple]]:
    """
    Returns all operator combinations whose reward meets the threshold.

    Args:
        length:     maximum sequence length (inclusive), starting from 1
        min_reward: minimum reward value (inclusive) to keep a sequence
        vocab:      list of operator tokens

    Returns:
        (count, sequences)  where sequences is a list of high-reward tuples
    """
    _, sequences = all_seqs(length, vocab)

    high_reward_seqs = []
    for seq in sequences:
        reward = _physics_reward(list(seq),None, True)
        if reward >= min_reward:
            high_reward_seqs.append(seq)

    return len(high_reward_seqs), high_reward_seqs


def plot_high_seqs(length: int, min_reward: float, vocab: list, save_dir) -> None:
    """
    Plots a histogram of reward values for high-reward sequences and
    annotates the figure with their count and fraction of the total.

    Args:
        length:     maximum sequence length (inclusive), starting from 1
        min_reward: minimum reward threshold
        vocab:      list of operator tokens
    """
    total_count, all_sequences = all_seqs(length, vocab)
    high_count, high_sequences = high_seqs(length, min_reward, vocab)

    # Compute rewards for high-reward sequences
    rewards = [_physics_reward(list(seq),None, True) for seq in high_sequences]

    bg_color      = "#0f1117"
    panel_color   = "#1a1d27"
    accent_color  = "#7c6af7"   # violet
    accent2_color = "#f7a26a"   # amber
    text_color    = "#e8e8f0"
    sub_color     = "#8888aa"

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(panel_color)

    if rewards:
        n_bins = min(40, max(10, high_count // 20))
        counts, bin_edges, patches = ax.hist(
            rewards,
            bins=n_bins,
            color=accent_color,
            edgecolor=bg_color,
            linewidth=0.6,
            alpha=0.9,
            zorder=3,
        )

        # Colour bars by height for depth
        max_count = counts.max()
        for patch, c in zip(patches, counts):
            patch.set_facecolor(
                plt.cm.cool(0.3 + 0.7 * c / max_count) if max_count > 0 else accent_color
            )

        # Threshold line
        ax.axvline(
            min_reward,
            color=accent2_color,
            linewidth=1.8,
            linestyle="--",
            zorder=4,
            label=f"Min reward = {min_reward}",
        )

        # Mean line
        mean_r = np.mean(rewards)
        ax.axvline(
            mean_r,
            color="#6af7c0",
            linewidth=1.4,
            linestyle=":",
            zorder=4,
            label=f"Mean = {mean_r:.3f}",
        )
    else:
        ax.text(
            0.5, 0.5,
            "No sequences meet the reward threshold",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=13, color=sub_color,
        )

    fraction = high_count / total_count if total_count else 0
    info_text = (
        f"High-reward sequences : {high_count:,}\n"
        f"Total sequences        : {total_count:,}\n"
        f"Fraction               : {fraction:.4%}\n"
        f"Max seq length         : {length}"
    )
    ax.text(
        0.98, 0.97,
        info_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10,
        color=text_color,
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#22253a",
            edgecolor=accent_color,
            linewidth=1.2,
            alpha=0.9,
        ),
        zorder=5,
    )

    ax.set_title(
        f"High-Reward Models  (reward ≥ {min_reward})",
        color=text_color, fontsize=14, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Reward", color=sub_color, fontsize=11, labelpad=8)
    ax.set_ylabel("Count", color=sub_color, fontsize=11, labelpad=8)

    ax.tick_params(colors=sub_color, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    ax.grid(axis="y", color="#333355", linewidth=0.6, linestyle="--", zorder=0)
    ax.grid(axis="x", color="#333355", linewidth=0.3, linestyle=":", zorder=0)

    if rewards:
        ax.legend(
            frameon=True,
            facecolor="#22253a",
            edgecolor=accent_color,
            labelcolor=text_color,
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_dir/"high_reward_sequences.png", dpi=150, bbox_inches="tight")
    print("Plot saved in {save_dir} /high_reward_sequences.png")
    plt.show()
"""
def evaluate_policy(
    policy_seqs: list[tuple],
    length: int,
    min_reward: float,
    vocab: list,
    top_n_operators: int = 20,
) -> None:

    print("Computing ground truth high-reward sequences...")
    gt_count, gt_seqs = high_seqs(length, min_reward, vocab)

    gt_seqs_norm = [tuple(sorted(s)) for s in gt_seqs]
    gt_set        = set(gt_seqs_norm)
    gt_rewards    = [_compute_reward(list(s), None, True) for s in gt_seqs_norm]

    print("Computing policy rewards...")
    policy_seqs_norm    = [tuple(sorted(s)) for s in policy_seqs]
    policy_rewards      = [_compute_reward(list(s), None, True) for s in policy_seqs_norm]
    policy_high_seqs    = [s for s, r in zip(policy_seqs_norm, policy_rewards) if r >= min_reward]
    policy_high_rewards = [r for r in policy_rewards if r >= min_reward]

    policy_set      = set(policy_high_seqs)
    true_positives  = policy_set & gt_set
    recall          = len(true_positives) / gt_count if gt_count else 0
    precision       = len(true_positives) / len(policy_set) if policy_set else 0
    false_discovery = 1 - precision
    false_negative  = 1 - recall

    gt_op_counts     = Counter(op for seq in gt_seqs_norm    for op in seq)
    policy_op_counts = Counter(op for seq in policy_high_seqs for op in seq)

    all_top_ops    = list({op for op, _ in
                           (gt_op_counts + policy_op_counts).most_common(top_n_operators)})
    gt_op_vals     = [gt_op_counts.get(op, 0)     for op in all_top_ops]
    policy_op_vals = [policy_op_counts.get(op, 0) for op in all_top_ops]

    gt_lengths     = [len(s) for s in gt_seqs_norm]
    policy_lengths = [len(s) for s in policy_high_seqs]
    max_len        = max(length, max(policy_lengths, default=1))
    len_bins       = list(range(1, max_len + 2))

    gt_len_hist,     _ = np.histogram(gt_lengths,     bins=len_bins)
    policy_len_hist, _ = np.histogram(policy_lengths, bins=len_bins)

    bg_color    = "#0f1117"
    panel_color = "#1a1d27"
    gt_color    = "#7c6af7"
    pol_color   = "#f7a26a"
    text_color  = "#e8e8f0"
    sub_color   = "#8888aa"
    grid_color  = "#333355"
    good_color  = "#6af7c0"
    warn_color  = "#f76a7c"

    fig = plt.figure(figsize=(16, 12), facecolor=bg_color)
    fig.suptitle(
        f"Policy vs Ground Truth Evaluation   (reward ≥ {min_reward}, max length {length})",
        color=text_color, fontsize=15, fontweight="bold", y=0.98,
    )

    gs   = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32,
                            left=0.07, right=0.97, top=0.93, bottom=0.07)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    for ax in axes:
        ax.set_facecolor(panel_color)
        ax.tick_params(colors=sub_color, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_color)
        ax.grid(axis="y", color=grid_color, linewidth=0.5, linestyle="--", zorder=0)

    ax0   = axes[0]
    all_r = gt_rewards + policy_high_rewards
    r_min = min(all_r) if all_r else 0
    r_max = max(all_r) if all_r else 1
    bins  = np.linspace(r_min, r_max, 30)

    gt_hist,     bin_edges = np.histogram(gt_rewards,          bins=bins)
    policy_hist, _         = np.histogram(policy_high_rewards, bins=bins)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bar_width   = (bin_edges[1] - bin_edges[0]) * 0.42

    ax0.bar(bin_centers - bar_width/2, gt_hist,     bar_width,
            color=gt_color,  alpha=0.9, label=f"Ground truth ({gt_count:,})",
            zorder=3, edgecolor=bg_color, lw=0.4)
    ax0.bar(bin_centers + bar_width/2, policy_hist, bar_width,
            color=pol_color, alpha=0.9, label=f"Policy high-reward ({len(policy_high_seqs):,})",
            zorder=3, edgecolor=bg_color, lw=0.4)
#    ax0.axvline(min_reward, color="white", linewidth=1.4, linestyle="--",
#                label=f"Threshold = {min_reward}", zorder=4)
    ax0.set_title("Reward Distribution (Side-by-Side)", color=text_color,
                  fontsize=11, fontweight="bold")
    ax0.set_xlabel("Reward", color=sub_color, fontsize=9)
    ax0.set_ylabel("Count",  color=sub_color, fontsize=9)
    ax0.legend(frameon=True, facecolor="#22253a", edgecolor=gt_color,
               labelcolor=text_color, fontsize=8)
    ax0.grid(axis="x", color=grid_color, linewidth=0.3, linestyle=":", zorder=0)

    ax1    = axes[1]
    x_lens = np.arange(1, max_len + 1)
    width  = 0.38
    ax1.bar(x_lens - width/2, gt_len_hist,     width, color=gt_color,  alpha=0.85,
            label="Ground truth", zorder=3, edgecolor=bg_color, lw=0.4)
    ax1.bar(x_lens + width/2, policy_len_hist, width, color=pol_color, alpha=0.85,
            label="Policy",       zorder=3, edgecolor=bg_color, lw=0.4)
    ax1.set_title("Sequence Length Distribution", color=text_color,
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("Length", color=sub_color, fontsize=9)
    ax1.set_ylabel("Count",  color=sub_color, fontsize=9)
    ax1.set_xticks(x_lens)
    ax1.xaxis.set_tick_params(colors=sub_color)
    ax1.legend(frameon=True, facecolor="#22253a", edgecolor=gt_color,
               labelcolor=text_color, fontsize=8)

    ax2   = axes[2]
    x_ops = np.arange(len(all_top_ops))
    ax2.bar(x_ops - width/2, gt_op_vals,     width, color=gt_color,  alpha=0.85,
            label="Ground truth", zorder=3, edgecolor=bg_color, lw=0.4)
    ax2.bar(x_ops + width/2, policy_op_vals, width, color=pol_color, alpha=0.85,
            label="Policy",       zorder=3, edgecolor=bg_color, lw=0.4)
    ax2.set_title(f"Top-{top_n_operators} Operator Frequencies", color=text_color,
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("Operator", color=sub_color, fontsize=9)
    ax2.set_ylabel("Count",    color=sub_color, fontsize=9)
    ax2.set_xticks(x_ops)
    ax2.set_xticklabels(all_top_ops, rotation=45, ha="right", fontsize=7, color=sub_color)
    ax2.legend(frameon=True, facecolor="#22253a", edgecolor=gt_color,
               labelcolor=text_color, fontsize=8)
    ax2.grid(axis="x", color=grid_color, linewidth=0.3, linestyle=":", zorder=0)

    ax3 = axes[3]
    ax3.axis("off")
    ax3.set_title("Evaluation Summary", color=text_color, fontsize=11, fontweight="bold")

    metrics = [
        ("Ground truth high-reward",  f"{gt_count:,}",                   gt_color),
        ("Policy sequences (total)",  f"{len(policy_seqs):,}",           pol_color),
        ("Policy high-reward",        f"{len(policy_high_seqs):,}",      pol_color),
        ("True positives (overlap)",  f"{len(true_positives):,}",        good_color),
        ("Recall  (GT coverage)",     f"{recall:.2%}",
         good_color if recall >= 0.5 else warn_color),
        ("Precision",                 f"{precision:.2%}",
         good_color if precision >= 0.5 else warn_color),
        ("False discovery rate",      f"{false_discovery:.2%}",
         warn_color if false_discovery >= 0.3 else good_color),
        ("Missed GT sequences",       f"{gt_count - len(true_positives):,}",
         warn_color if (gt_count - len(true_positives)) > 0 else good_color),
        ("GT mean reward",            f"{np.mean(gt_rewards):.4f}"           if gt_rewards           else "N/A", text_color),
        ("Policy mean reward",        f"{np.mean(policy_high_rewards):.4f}"  if policy_high_rewards  else "N/A", text_color),
    ]

    row_h = 0.085
    for i, (label, value, color) in enumerate(metrics):
        y = 0.92 - i * row_h
        if i % 2 == 0:
            ax3.add_patch(mpatches.FancyBboxPatch(
                (0.01, y - 0.03), 0.98, row_h - 0.005,
                boxstyle="round,pad=0.01", linewidth=0,
                facecolor="#22253a", transform=ax3.transAxes, zorder=1,
            ))
        ax3.text(0.05, y, label, transform=ax3.transAxes,
                 color=sub_color, fontsize=9, va="center", fontfamily="monospace", zorder=2)
        ax3.text(0.95, y, value, transform=ax3.transAxes,
                 color=color, fontsize=9, va="center", ha="right",
                 fontfamily="monospace", fontweight="bold", zorder=2)

    plt.savefig("policy_vs_truth_eval.png", dpi=150, bbox_inches="tight", facecolor=bg_color)
    plt.show()
    print(f"\nPlot saved → policy_vs_truth_eval.png")
    print(f"Recall: {recall:.2%} | Precision: {precision:.2%} | True positives: {len(true_positives):,}")

# policy_seqs = [...]   # list of tuples from your RL policy rollouts
# evaluate_policy(policy_seqs, length=5, min_reward=0.5, vocab=op_vocab)

n, seqs = all_seqs(5, op_vocab)
print(f"Total sequences: {n}")          # → 15,082,602

n_high, high = high_seqs(5, 0.5, op_vocab)
print(f"High-reward sequences: {n_high}")
print("First 5:", high[:5])

plot_high_seqs(5, 0.5, op_vocab)
"""



def run_pre_diagnostics(model, stoi, device):
    """Print initial model probabilities and reward sanity checks."""
    import torch
    import torch.nn.functional as F
    from .vocab import START_ID

    print("\n" + "="*60)
    print("PRE-TRAINING DIAGNOSTICS")
    print("="*60)
    print(f"MAX_RAW_SCORE = {MAX_RAW_SCORE:.2f}  (top-3 tier-1 sum)")
    print(f"  perfect seq reward  = "
          f"{_compute_reward(['phi','W','phiWB'], ['phi','W','phiWB'], True):.4f}")
    print(f"  single phi reward   = "
          f"{_compute_reward(['phi'], ['phi'], True):.4f}")
    print(f"  empty seq reward    = "
          f"{_compute_reward([], [], False):.4f}")
    with torch.no_grad():
        test_prompt = torch.tensor([[START_ID]] * 8, device=device)
        logits      = model(test_prompt)[:, -1, :]
        probs       = F.softmax(logits, dim=-1).mean(0)
        print("\nInitial operator probabilities:")
        for op in WC_NAME_MAP:
            if op in stoi:
                print(f"  {op:15s}  {probs[stoi[op]].item():.4f}")
    print("="*60 + "\n")


def print_search_efficiency(batch_idx, n_batches):
    """Print search efficiency stats."""
    if total_calls == 0:
        return
    unique_evals       = len(reward_cache)
    duplicates         = total_calls - unique_evals
    hit_rate           = duplicates / total_calls
    coverage           = unique_evals / SEARCH_SPACE_SIZE
    batches_so_far     = batch_idx + 1
    evals_per_batch    = unique_evals / batches_so_far
    batches_to_exhaust = (SEARCH_SPACE_SIZE - unique_evals) / max(evals_per_batch, 1)

    print(f"\n{'='*48}")
    print(f"  SEARCH EFFICIENCY  (batch {batch_idx})")
    print(f"{'='*48}")
    print(f"  Unique evals           : {unique_evals:>8,}  ({coverage:.1%} of space)")
    print(f"  Total calls (w/ dupes) : {total_calls:>8,}")
    print(f"  Cache hit rate         : {hit_rate:>8.1%}")
    print(f"  Best reward            : {best_reward_so_far:>8.4f}  ops={best_ops_so_far}")
    if first_best_found is not None:
        print(f"  Best found at eval #   : {first_best_found:>8,}  "
              f"({100*first_best_found/SEARCH_SPACE_SIZE:.1f}% coverage)")
    print(f"  Unique evals/batch     : {evals_per_batch:>8.1f}")
    print(f"  Batches to exhaust     : {batches_to_exhaust:>8.1f}  "
          f"(you have {n_batches - batches_so_far} remaining)")
    print(f"{'='*48}\n")

"""
def evaluate_policy(
    policy_seqs:    list[tuple],
    length:         int,
    min_reward:     float,
    vocab:          list,
    random_seqs:    list[tuple] = None,
    random_archive: list[dict]  = None,
    top_n_operators: int        = 20,
) -> None:

    print("Computing ground truth...")
    gt_count, gt_seqs   = high_seqs(length, min_reward, vocab)
    gt_seqs_norm        = [tuple(sorted(s)) for s in gt_seqs]
    gt_set              = set(gt_seqs_norm)
    gt_rewards          = [_physics_reward(list(s), None, True) for s in gt_seqs_norm]

    print("Computing policy rewards...")
    policy_seqs_norm    = [tuple(sorted(s)) for s in policy_seqs]
    policy_rewards      = [_physics_reward(list(s), None, True) for s in policy_seqs_norm]
    policy_high_seqs    = [s for s, r in zip(policy_seqs_norm, policy_rewards) if r >= min_reward]
    policy_high_rewards = [r for r in policy_rewards if r >= min_reward]
    policy_set          = set(policy_high_seqs)
    tp_policy           = gt_set & policy_set
    recall              = len(tp_policy) / gt_count if gt_count else 0
    precision           = len(tp_policy) / len(policy_set) if policy_set else 0
    false_discovery     = 1 - precision

    has_random = random_seqs is not None
    if has_random:
        print("Computing random baseline...")
        rand_seqs_norm    = [tuple(sorted(s)) for s in random_seqs]
        rand_set          = set(rand_seqs_norm)
        rand_rewards      = [_physics_reward(list(s), None, True) for s in rand_seqs_norm]
        tp_random         = gt_set & rand_set
        rand_recall       = len(tp_random) / gt_count if gt_count else 0
        rand_precision    = len(tp_random) / len(rand_set) if rand_set else 0
        rand_lengths      = [len(e["ops"]) for e in random_archive
                             if _physics_reward(list(e["ops"]), None, True) >= min_reward]
        rand_op_counts    = Counter(op for s in rand_seqs_norm for op in s)
    else:
        rand_rewards = rand_lengths = []

    gt_op_counts     = Counter(op for s in gt_seqs_norm     for op in s)
    policy_op_counts = Counter(op for s in policy_high_seqs for op in s)
    combined         = gt_op_counts + policy_op_counts
    if has_random:
        combined += rand_op_counts
    all_top_ops    = [op for op, _ in combined.most_common(top_n_operators)]
    gt_op_vals     = [gt_op_counts.get(op, 0)     for op in all_top_ops]
    policy_op_vals = [policy_op_counts.get(op, 0) for op in all_top_ops]
    rand_op_vals   = [rand_op_counts.get(op, 0)   for op in all_top_ops] if has_random else []

    gt_lengths     = [len(s) for s in gt_seqs_norm]
    policy_lengths = [len(s) for s in policy_high_seqs]
    max_len        = max(length, max(policy_lengths, default=1))
    len_bins       = list(range(1, max_len + 2))
    gt_len_hist,     _ = np.histogram(gt_lengths,     bins=len_bins)
    policy_len_hist, _ = np.histogram(policy_lengths, bins=len_bins)
    rand_len_hist,   _ = np.histogram(rand_lengths,   bins=len_bins) if has_random else (np.zeros(max_len), None)

    bg_color    = "#0f1117"
    panel_color = "#1a1d27"
    gt_color    = "#7c6af7"
    pol_color   = "#f7a26a"
#    rand_color  = "#4a4a6a"    # muted — background reference
    rand_color  = "#8888cc"    # lighter muted blue-purple, visible but clearly background
    text_color  = "#e8e8f0"
    sub_color   = "#8888aa"
    grid_color  = "#333355"
    good_color  = "#6af7c0"
    warn_color  = "#f76a7c"

    fig = plt.figure(figsize=(16, 12), facecolor=bg_color)
    fig.suptitle(
        f"Policy vs Ground Truth Evaluation   (reward ≥ {min_reward}, max length {length})",
        color=text_color, fontsize=15, fontweight="bold", y=0.98,
    )
    gs   = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32,
                            left=0.07, right=0.97, top=0.93, bottom=0.07)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    for ax in axes:
        ax.set_facecolor(panel_color)
        ax.tick_params(colors=sub_color, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_color)
        ax.grid(axis="y", color=grid_color, linewidth=0.5, linestyle="--", zorder=0)

    ax0   = axes[0]
    all_r = gt_rewards + policy_high_rewards + rand_rewards
    r_min = min(all_r) if all_r else 0
    r_max = max(all_r) if all_r else 1
    bins  = np.linspace(r_min, r_max, 30)

    gt_hist,     bin_edges = np.histogram(gt_rewards,          bins=bins)
    policy_hist, _         = np.histogram(policy_high_rewards, bins=bins)
    rand_hist,   _         = np.histogram(rand_rewards,        bins=bins) if has_random else (np.zeros(len(bins)-1), None)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    if has_random:
        bar_w = (bin_edges[1] - bin_edges[0]) * 0.3
        # random in background, widest
        ax0.bar(bin_centers,           rand_hist,   bar_w * 2.8,
                color=rand_color, alpha=0.35, label=f"Random ({len(rand_seqs_norm):,})",
                zorder=1, edgecolor="none")
        ax0.bar(bin_centers - bar_w/2, gt_hist,     bar_w,
                color=gt_color,   alpha=0.9,  label=f"Ground truth ({gt_count:,})",
                zorder=3, edgecolor=bg_color, lw=0.4)
        ax0.bar(bin_centers + bar_w/2, policy_hist, bar_w,
                color=pol_color,  alpha=0.9,  label=f"Policy ({len(policy_high_seqs):,})",
                zorder=3, edgecolor=bg_color, lw=0.4)
    else:
        bar_w = (bin_edges[1] - bin_edges[0]) * 0.42
        ax0.bar(bin_centers - bar_w/2, gt_hist,     bar_w, color=gt_color,  alpha=0.9,
                label=f"Ground truth ({gt_count:,})", zorder=3, edgecolor=bg_color, lw=0.4)
        ax0.bar(bin_centers + bar_w/2, policy_hist, bar_w, color=pol_color, alpha=0.9,
                label=f"Policy ({len(policy_high_seqs):,})", zorder=3, edgecolor=bg_color, lw=0.4)

    ax0.axvline(min_reward, color="white", linewidth=1.4, linestyle="--",
                label=f"Threshold = {min_reward}", zorder=4)
    ax0.set_title("Reward Distribution", color=text_color, fontsize=11, fontweight="bold")
    ax0.set_xlabel("Reward", color=sub_color, fontsize=9)
    ax0.set_ylabel("Count",  color=sub_color, fontsize=9)
    ax0.legend(frameon=True, facecolor="#22253a", edgecolor=gt_color,
               labelcolor=text_color, fontsize=8)
    ax0.grid(axis="x", color=grid_color, linewidth=0.3, linestyle=":", zorder=0)

    ax1    = axes[1]
    x_lens = np.arange(1, max_len + 1)
    if has_random:
        w = 0.26
        ax1.bar(x_lens - w,   rand_len_hist,   w, color=rand_color, alpha=0.5,
                label="Random",       zorder=2, edgecolor=bg_color, lw=0.4)
        ax1.bar(x_lens,       gt_len_hist,     w, color=gt_color,   alpha=0.85,
                label="Ground truth", zorder=3, edgecolor=bg_color, lw=0.4)
        ax1.bar(x_lens + w,   policy_len_hist, w, color=pol_color,  alpha=0.85,
                label="Policy",       zorder=3, edgecolor=bg_color, lw=0.4)
    else:
        w = 0.38
        ax1.bar(x_lens - w/2, gt_len_hist,     w, color=gt_color,  alpha=0.85,
                label="Ground truth", zorder=3, edgecolor=bg_color, lw=0.4)
        ax1.bar(x_lens + w/2, policy_len_hist, w, color=pol_color, alpha=0.85,
                label="Policy",       zorder=3, edgecolor=bg_color, lw=0.4)

    ax1.set_title("Sequence Length Distribution", color=text_color, fontsize=11, fontweight="bold")
    ax1.set_xlabel("Length", color=sub_color, fontsize=9)
    ax1.set_ylabel("Count",  color=sub_color, fontsize=9)
    ax1.set_xticks(x_lens)
    ax1.xaxis.set_tick_params(colors=sub_color)
    ax1.legend(frameon=True, facecolor="#22253a", edgecolor=gt_color,
               labelcolor=text_color, fontsize=8)

    ax2   = axes[2]
    x_ops = np.arange(len(all_top_ops))
    if has_random:
        w = 0.26
        ax2.bar(x_ops - w,   rand_op_vals,   w, color=rand_color, alpha=0.5,
                label="Random",       zorder=2, edgecolor=bg_color, lw=0.4)
        ax2.bar(x_ops,       gt_op_vals,     w, color=gt_color,   alpha=0.85,
                label="Ground truth", zorder=3, edgecolor=bg_color, lw=0.4)
        ax2.bar(x_ops + w,   policy_op_vals, w, color=pol_color,  alpha=0.85,
                label="Policy",       zorder=3, edgecolor=bg_color, lw=0.4)
    else:
        w = 0.38
        ax2.bar(x_ops - w/2, gt_op_vals,     w, color=gt_color,  alpha=0.85,
                label="Ground truth", zorder=3, edgecolor=bg_color, lw=0.4)
        ax2.bar(x_ops + w/2, policy_op_vals, w, color=pol_color, alpha=0.85,
                label="Policy",       zorder=3, edgecolor=bg_color, lw=0.4)

    ax2.set_title(f"Top-{top_n_operators} Operator Frequencies", color=text_color,
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("Operator", color=sub_color, fontsize=9)
    ax2.set_ylabel("Count",    color=sub_color, fontsize=9)
    ax2.set_xticks(x_ops)
    ax2.set_xticklabels(all_top_ops, rotation=45, ha="right", fontsize=7, color=sub_color)
    ax2.legend(frameon=True, facecolor="#22253a", edgecolor=gt_color,
               labelcolor=text_color, fontsize=8)
    ax2.grid(axis="x", color=grid_color, linewidth=0.3, linestyle=":", zorder=0)

    ax3 = axes[3]
    ax3.axis("off")
    ax3.set_title("Evaluation Summary", color=text_color, fontsize=11, fontweight="bold")

#    rand_n_rollouts = rand_results["n_rollouts"] if has_random else 0

    rand_n_rollouts = len(random_archive) if has_random else 0
    speedup = (rand_n_rollouts / max(len(tp_random), 1)) / \
              (len(policy_seqs) / max(len(tp_policy), 1)) if has_random and tp_random else float("inf")

    metrics = [
        ("Ground truth high-reward",   f"{gt_count:,}",                  gt_color),
        ("Policy rollouts (total)",     f"{len(policy_seqs):,}",          pol_color),
        ("Policy high-reward",          f"{len(policy_high_seqs):,}",     pol_color),
        ("Policy true positives",       f"{len(tp_policy):,}",            good_color),
        ("Policy recall",               f"{recall:.2%}",
         good_color if recall >= 0.5 else warn_color),
        ("Policy precision",            f"{precision:.2%}",
         good_color if precision >= 0.5 else warn_color),
        ("Policy false discovery",      f"{false_discovery:.2%}",
         warn_color if false_discovery >= 0.3 else good_color),
        ("── Random baseline ──",       "──────────",                     sub_color),
        ("Random rollouts (total)",     f"{rand_n_rollouts:,}" if has_random else "N/A", rand_color),
        ("Random true positives",       f"{len(tp_random):,}"  if has_random else "N/A",
         good_color if has_random else sub_color),
        ("Random recall",               f"{rand_recall:.2%}"   if has_random else "N/A",
         good_color if has_random and rand_recall >= 0.5 else warn_color),
        ("Speedup vs random",           f"{speedup:.1f}×"      if has_random else "N/A",
         good_color if has_random and speedup > 1 else warn_color),
    ]

    row_h = 0.075
    for i, (label, value, color) in enumerate(metrics):
        y = 0.94 - i * row_h
        if i % 2 == 0:
            ax3.add_patch(mpatches.FancyBboxPatch(
                (0.01, y - 0.025), 0.98, row_h - 0.005,
                boxstyle="round,pad=0.01", linewidth=0,
                facecolor="#22253a", transform=ax3.transAxes, zorder=1,
            ))
        ax3.text(0.05, y, label, transform=ax3.transAxes,
                 color=sub_color, fontsize=8.5, va="center", fontfamily="monospace", zorder=2)
        ax3.text(0.95, y, value, transform=ax3.transAxes,
                 color=color, fontsize=8.5, va="center", ha="right",
                 fontfamily="monospace", fontweight="bold", zorder=2)

    plt.savefig("policy_vs_truth_eval.png", dpi=150, bbox_inches="tight", facecolor=bg_color)
    plt.show()
    print(f"\nSaved → policy_vs_truth_eval.png")
    print(f"Policy  — Recall: {recall:.2%} | Precision: {precision:.2%} | TP: {len(tp_policy):,}")
    if has_random:
        print(f"Random  — Recall: {rand_recall:.2%} | Precision: {rand_precision:.2%} | TP: {len(tp_random):,}")
        print(f"Speedup : {speedup:.1f}× fewer rollouts per true positive vs random")
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def evaluate_policy(
    policy_seqs,
    length,
    min_reward,
    vocab,
    random_seqs=None,
    random_archive=None,
):
    gt_count, gt_seqs   = high_seqs(length, min_reward, vocab)
    gt_seqs_norm        = [tuple(sorted(s)) for s in gt_seqs]
    gt_rewards          = [_physics_reward(list(s), None, True) for s in gt_seqs_norm]
    gt_lengths          = [len(s) for s in gt_seqs_norm]

    policy_seqs_norm    = [tuple(sorted(s)) for s in policy_seqs]
    policy_rewards      = [_physics_reward(list(s), None, True) for s in policy_seqs_norm]
    policy_high_seqs    = [s for s, r in zip(policy_seqs_norm, policy_rewards)
                           if r >= min_reward]
    policy_high_rewards = [r for r in policy_rewards if r >= min_reward]
    policy_lengths      = [len(s) for s in policy_high_seqs]

    has_random = random_seqs is not None and random_archive is not None
    if has_random:
        rand_seen = {}
        for e in random_archive:
            key = tuple(sorted(e["ops"]))
            r   = _physics_reward(list(e["ops"]), None, True)
            if key not in rand_seen or r > rand_seen[key]:
                rand_seen[key] = r
        rand_high_rewards = [r for r in rand_seen.values() if r >= min_reward]
        rand_lengths      = [len(k) for k, r in rand_seen.items() if r >= min_reward]
    else:
        rand_high_rewards = []
        rand_lengths      = []

    gt_col   = "#5B4FCF"
    pol_col  = "#E07B39"
    rand_col = "#9999BB"

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")

    for ax in (ax0, ax1):
        ax.set_facecolor("white")
        ax.tick_params(colors="black", labelsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#333333")
            ax.spines[spine].set_linewidth(1.2)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.8,
                linestyle="--", zorder=0)

    def round_r(rewards, decimals=2):
        return [round(r, decimals) for r in rewards]

    gt_r_rounded   = round_r(gt_rewards)
    pol_r_rounded  = round_r(policy_high_rewards)
    rand_r_rounded = round_r(rand_high_rewards) if has_random else []

    all_unique_r = sorted(set(gt_r_rounded + pol_r_rounded + rand_r_rounded))

    gt_counts   = Counter(gt_r_rounded)
    pol_counts  = Counter(pol_r_rounded)
    rand_counts = Counter(rand_r_rounded) if has_random else {}

    gt_vals   = [gt_counts.get(r, 0)   for r in all_unique_r]
    pol_vals  = [pol_counts.get(r, 0)  for r in all_unique_r]
    rand_vals = [rand_counts.get(r, 0) for r in all_unique_r] if has_random else []

    x_pos = np.arange(len(all_unique_r))

    if has_random:
        w = 0.25
        ax0.bar(x_pos - w,     rand_vals, w,
                color=rand_col, alpha=0.50,
                label=f"Random ({len(rand_high_rewards):,})",
                zorder=2, edgecolor="white", lw=0.5)
        ax0.bar(x_pos,         gt_vals,   w,
                color=gt_col,   alpha=0.90,
                label=f"Ground truth ({gt_count:,})",
                zorder=3, edgecolor="white", lw=0.5)
        ax0.bar(x_pos + w,     pol_vals,  w,
                color=pol_col,  alpha=0.90,
                label=f"Policy ({len(policy_high_seqs):,})",
                zorder=3, edgecolor="white", lw=0.5)
    else:
        w = 0.35
        ax0.bar(x_pos - w / 2, gt_vals,  w,
                color=gt_col,  alpha=0.90,
                label=f"Ground truth ({gt_count:,})",
                zorder=3, edgecolor="white", lw=0.5)
        ax0.bar(x_pos + w / 2, pol_vals, w,
                color=pol_col, alpha=0.90,
                label=f"Policy ({len(policy_high_seqs):,})",
                zorder=3, edgecolor="white", lw=0.5)

    # threshold vertical line — placed just before first bar
    ax0.axvline(-0.5, color="black", linewidth=1.8,
                linestyle="--", zorder=5,
                label=f"Threshold = {min_reward}")

    ax0.set_xticks(x_pos)
    ax0.set_xticklabels([f"{r:.2f}" for r in all_unique_r],
                         rotation=45, ha="right", fontsize=11)
    ax0.set_xlim(-0.6, len(all_unique_r) - 0.4)

    ax0.set_title("Reward Distribution of Models",
                  fontsize=15, fontweight="bold", color="black", pad=12)
    ax0.set_xlabel("Reward",           fontsize=14, color="black", labelpad=8)
    ax0.set_ylabel("Number of models", fontsize=14, color="black", labelpad=8)
    ax0.tick_params(axis="both", labelsize=13, colors="black")
    ax0.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC",
               labelcolor="black", fontsize=11, loc="upper right")

    max_len  = max(length,
                   max(policy_lengths, default=1),
                   max(gt_lengths,     default=1))
    x_lens   = np.arange(1, max_len + 1)
    len_bins = np.arange(0.5, max_len + 1.5, 1)

    gt_lh,   _ = np.histogram(gt_lengths,     bins=len_bins)
    pol_lh,  _ = np.histogram(policy_lengths, bins=len_bins)
    rand_lh, _ = np.histogram(rand_lengths,   bins=len_bins) \
                 if has_random else (np.zeros(max_len, dtype=int), None)

    # remove empty length bins
    nonzero_len = (gt_lh > 0) | (pol_lh > 0) | \
                  (rand_lh > 0 if has_random
                   else np.zeros(max_len, dtype=bool))
    x_nz      = x_lens[nonzero_len]
    gt_lh_nz  = gt_lh[nonzero_len]
    pol_lh_nz = pol_lh[nonzero_len]
    if has_random:
        rand_lh_nz = rand_lh[nonzero_len]

    if has_random:
        w = 0.26
        ax1.bar(x_nz - w,   rand_lh_nz,  w,
                color=rand_col, alpha=0.50, label="Random",
                zorder=2, edgecolor="white", lw=0.5)
        ax1.bar(x_nz,        gt_lh_nz,   w,
                color=gt_col,   alpha=0.90, label="Ground truth",
                zorder=3, edgecolor="white", lw=0.5)
        ax1.bar(x_nz + w,    pol_lh_nz,  w,
                color=pol_col,  alpha=0.90, label="Policy",
                zorder=3, edgecolor="white", lw=0.5)
    else:
        w = 0.38
        ax1.bar(x_nz - w / 2, gt_lh_nz,  w,
                color=gt_col,  alpha=0.90, label="Ground truth",
                zorder=3, edgecolor="white", lw=0.5)
        ax1.bar(x_nz + w / 2, pol_lh_nz, w,
                color=pol_col, alpha=0.90, label="Policy",
                zorder=3, edgecolor="white", lw=0.5)

    ax1.set_xticks(x_nz)
    ax1.set_xticklabels([str(x) for x in x_nz], fontsize=13)
    ax1.set_xlim(x_nz[0] - 0.6, x_nz[-1] + 0.6)

    ax1.set_title("Operator Count Distribution",
                  fontsize=15, fontweight="bold", color="black", pad=12)
    ax1.set_xlabel("Number of operators", fontsize=14, color="black", labelpad=8)
    ax1.set_ylabel("Number of models",    fontsize=14, color="black", labelpad=8)
    ax1.tick_params(axis="both", labelsize=13, colors="black")
    ax1.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC",
               labelcolor="black", fontsize=11, loc="upper right")

    plt.tight_layout(pad=2.5)
    plt.savefig("two_panels.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.savefig("two_panels.pdf", bbox_inches="tight",
                facecolor="white")
    plt.show()
    print("Saved → two_panels.png")
    print("Saved → two_panels.pdf") 

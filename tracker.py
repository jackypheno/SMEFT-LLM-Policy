# ============================================================
# smeft_rl/tracker.py
# RewardTracker: records rollouts and produces reward plots.
# print_theories: prints and saves theory archive.
# ============================================================

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

from .config    import OP_SCORES, FREE_OPS, MAX_NEW_TOKENS, WC_NAME_MAP
from .reward    import efficiency_history, MAX_RAW_SCORE
from .shared import SEARCH_SPACE_SIZE

OP_COLORS = {
    "G":          "#FF0000",
    "phi":        "#FFD700",
    "W":          "#00FF00",
    "phiWB":      "#0080FF",
    "phiD":       "#FF6600",
    "phiW":       "#FF00FF",
    "phiB":       "#00FFFF",
    "phiWtilde":  "#FFFFFF",
    "phiBox":     "#7FFF00",
}
_FALLBACK_COLORS = ["#FF0000","#00FF00","#0080FF","#FFD700",
                    "#FF00FF","#00FFFF","#FF6600","#FFFFFF",
                    "#FF69B4","#7FFF00","#9400D3","#FF4500"]


class RewardTracker:
    def __init__(self):
        self.tracked_ops = list(OP_SCORES.keys())
        self.rollout_idx = []
        self.rewards     = []
        self.seq_lengths = []
        self.op_present  = {op: [] for op in self.tracked_ops}
        self._counter    = 0

    def record(self, ops, r):
        self._counter += 1
        self.rollout_idx.append(self._counter)
        self.rewards.append(r)
        self.seq_lengths.append(len(ops))
        op_set = set(ops)
        for op in self.tracked_ops:
            self.op_present[op].append(1 if op in op_set else 0)

    def plot(self, save_path, window=20):
        idx = np.array(self.rollout_idx, dtype=float)
        R   = np.array(self.rewards,    dtype=float)
        if len(R) < 2:
            return

        def rolling(arr, w):
            w    = min(w, len(arr))
            mean = np.convolve(arr, np.ones(w)/w, mode="same")
            std  = np.array([arr[max(0,i-w//2):i+w//2+1].std()
                             for i in range(len(arr))])
            return mean, std

        roll_mean, roll_std = rolling(R, window)
        op_rolls = {op: rolling(np.array(self.op_present[op], dtype=float),
                                window)[0]
                    for op in self.tracked_ops}
        lengths     = np.array(self.seq_lengths, dtype=float)
        roll_len, _ = rolling(lengths, window)

        fig = plt.figure(figsize=(12, 12), facecolor="#0e1117")
        gs  = GridSpec(4, 1, figure=fig,
                       height_ratios=[3, 2, 2, 2], hspace=0.5)

        ax_r, ax_op, ax_len, ax_eff = [fig.add_subplot(gs[i]) for i in range(4)]

        for ax in (ax_r, ax_op, ax_len, ax_eff):
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e", labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

        # reward panel
        ax_r.scatter(idx, R, s=4, alpha=0.25, color="#58a6ff",
                     linewidths=0, label="per-rollout", zorder=2)
        ax_r.fill_between(idx, roll_mean-roll_std, roll_mean+roll_std,
                          alpha=0.18, color="#58a6ff")
        ax_r.plot(idx, roll_mean, color="#58a6ff", linewidth=1.8,
                  label=f"rolling mean (w={window})", zorder=3)
        ax_r.axhline(1.0, color="#3fb950", linewidth=0.8,
                     linestyle="--", alpha=0.6, label="max reward")
        ax_r.axhline(0.0, color="#8b949e", linewidth=0.6,
                     linestyle=":", alpha=0.5)
        ax_r.set_xlim(1, max(idx))
        ax_r.set_ylim(-1.05, 1.1)
        ax_r.set_ylabel("Reward", color="#c9d1d9", fontsize=10)
        ax_r.set_title("SMEFT RL — Reward vs Rollout",
                        color="#c9d1d9", fontsize=12, pad=10, fontweight="bold")
        ax_r.legend(fontsize=8, framealpha=0.3, facecolor="#161b22",
                    edgecolor="#30363d", labelcolor="#c9d1d9")
        best_i = int(np.argmax(R))
        ax_r.annotate(f"best: {R[best_i]:.3f} @ rollout {int(idx[best_i])}",
                      xy=(idx[best_i], R[best_i]),
                      xytext=(idx[best_i]+max(len(idx)*0.03, 2), R[best_i]-0.12),
                      fontsize=7.5, color="#3fb950",
                      arrowprops=dict(arrowstyle="->", color="#3fb950", lw=0.8))

        # operator frequency panel
        for i, op in enumerate(self.tracked_ops):
            color = OP_COLORS.get(op, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])
            ax_op.plot(idx, op_rolls[op], color=color, linewidth=1.6,
                       label=op, alpha=0.9)
        ax_op.set_xlim(1, max(idx))
        ax_op.set_ylim(-0.05, 1.05)
        ax_op.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax_op.set_ylabel("Frequency", color="#c9d1d9", fontsize=10)
        ax_op.set_title(f"Operator Frequency (rolling w={window})",
                        color="#c9d1d9", fontsize=10, pad=8)
        ax_op.legend(fontsize=7, ncol=4, framealpha=0.3, facecolor="#161b22",
                     edgecolor="#30363d", labelcolor="#c9d1d9")

        # sequence length panel
        ax_len.scatter(idx, lengths, s=4, alpha=0.20, color="#e3b341",
                       linewidths=0, zorder=2)
        ax_len.plot(idx, roll_len, color="#e3b341", linewidth=1.8, zorder=3)
        ax_len.axhline(FREE_OPS, color="#3fb950", linewidth=0.9,
                       linestyle="--", alpha=0.7,
                       label=f"free threshold ({FREE_OPS})")
        ax_len.set_xlim(1, max(idx))
        ax_len.set_ylim(0, MAX_NEW_TOKENS+1)
        ax_len.set_ylabel("Seq length", color="#c9d1d9", fontsize=10)
        ax_len.set_title("Sequence Length vs Rollout",
                         color="#c9d1d9", fontsize=10, pad=8)
        ax_len.legend(fontsize=8, framealpha=0.3, facecolor="#161b22",
                      edgecolor="#30363d", labelcolor="#c9d1d9")

        # search efficiency panel
        if len(efficiency_history) > 1:
            eff_x      = np.array([e[0] for e in efficiency_history], dtype=float)
            eff_y      = np.array([e[1] for e in efficiency_history], dtype=float)
            coverage_x = 100 * eff_x / SEARCH_SPACE_SIZE

            ax_eff.plot(coverage_x, eff_y, color="#3fb950", linewidth=1.8,
                        label="best reward found", zorder=3)
            ax_eff.axhline(1.0, color="#3fb950", linewidth=0.8,
                           linestyle="--", alpha=0.5, label="max reward")
            random_x = np.linspace(0, 100, 200)
            random_y = random_x / 100
            ax_eff.plot(random_x, random_y, color="#f85149", linewidth=1.2,
                        linestyle="--", alpha=0.7, label="random search baseline")
            ax_eff.set_xlim(0, min(max(coverage_x)+5, 110))
            ax_eff.set_ylim(-0.05, 1.1)
            ax_eff.set_xlabel("Search space coverage (%)",
                              color="#c9d1d9", fontsize=10)
            ax_eff.set_ylabel("Best reward found",
                              color="#c9d1d9", fontsize=10)
            ax_eff.set_title("RL Search Efficiency vs Random Baseline",
                             color="#c9d1d9", fontsize=10, pad=8)
            ax_eff.legend(fontsize=8, framealpha=0.3, facecolor="#161b22",
                          edgecolor="#30363d", labelcolor="#c9d1d9")

        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[plot] saved → {save_path}")


def print_theories(theory_archive, save_dir):
    best       = {}
    first_seen = {}
    best_seq   = {}

    for i, item in enumerate(theory_archive):
        key = tuple(sorted(item["ops"]))
        r   = item["reward"]
        if key not in first_seen:
            first_seen[key] = i + 1
        if key not in best or r > best[key]:
            best[key]     = r
            best_seq[key] = item["tokens"]

    sorted_theories = sorted(best.items(), key=lambda x: x[1], reverse=True)
    print(f"\nUnique theories explored: {len(best)}")
    print("\n" + "="*70)
    print("TOP-15 THEORIES")
    print("="*70)
    for i, (ops, r) in enumerate(sorted_theories[:15]):
        tokens     = best_seq[ops]
        if "END" in tokens:
            tokens = tokens[:tokens.index("END")+1]
        discovered = first_seen[ops]
        pct        = 100.0 * discovered / max(len(theory_archive), 1)
        print(f"Rank {i+1:>3}  reward={r:.4f}  "
              f"first @ rollout {discovered} ({pct:.1f}%)\n"
              f"         {' → '.join(tokens)}\n")

    export = []
    seen   = set()
    for i, item in enumerate(theory_archive):
        key    = tuple(sorted(item["ops"]))
        is_new = key not in seen
        seen.add(key)
        tokens = item["tokens"]
        if "END" in tokens:
            tokens = tokens[:tokens.index("END")+1]
        export.append({
            "index":      i + 1,
            "is_new":     is_new,
            "reward":     item["reward"],
            "seq":        tokens,
            "first_seen": first_seen[tuple(sorted(item["ops"]))],
        })
    path = str(save_dir / "theory_archive.json")
    with open(path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Archive saved → {path}")


import numpy as np
import matplotlib.pyplot as plt

def plot_length_distribution(theory_archive, reward_threshold=0.7,
                              save_path=None):
    seen     = {}
    for item in theory_archive:
        key = tuple(sorted(item["ops"]))
        if key not in seen:
            seen[key] = item
        # keep highest reward if same set appears multiple times
        elif item["reward"] > seen[key]["reward"]:
            seen[key] = item

    unique_archive = list(seen.values())
    print(f"Total entries    : {len(theory_archive)}")
    print(f"Unique sets      : {len(unique_archive)}")
    print(f"High reward sets : {sum(1 for i in unique_archive if i['reward'] >= reward_threshold)}")

    all_lengths  = [len(item["ops"]) for item in unique_archive]
    high_lengths = [len(item["ops"]) for item in unique_archive
                    if item["reward"] >= reward_threshold]
    low_lengths  = [len(item["ops"]) for item in unique_archive
                    if item["reward"] < reward_threshold]
    # ... rest of plot code unchanged, just uses unique_archive

    max_len = max(all_lengths) if all_lengths else 1
    bins    = np.arange(0, max_len + 2) - 0.5  # centred bins

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="#0e1117")

    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # --- Panel 1: all sequences ---
    axes[0].hist(all_lengths, bins=bins, color="#58a6ff",
                 alpha=0.8, density=True, edgecolor="#0e1117")
    axes[0].axvline(np.mean(all_lengths), color="#e3b341", linewidth=1.5,
                    linestyle="--", label=f"mean={np.mean(all_lengths):.2f}")
    axes[0].set_xlabel("Sequence length (# operators)", color="#c9d1d9", fontsize=10)
    axes[0].set_ylabel("Density",                        color="#c9d1d9", fontsize=10)
    axes[0].set_title("All sequences",                   color="#c9d1d9", fontsize=11)
    axes[0].legend(fontsize=8, framealpha=0.3, facecolor="#161b22",
                   edgecolor="#30363d", labelcolor="#c9d1d9")

    # --- Panel 2: high reward vs low reward overlaid ---
    axes[1].hist(low_lengths,  bins=bins, color="#f85149", alpha=0.6,
                 density=True, label=f"reward < {reward_threshold}",
                 edgecolor="#0e1117")
    axes[1].hist(high_lengths, bins=bins, color="#3fb950", alpha=0.7,
                 density=True, label=f"reward ≥ {reward_threshold}",
                 edgecolor="#0e1117")
    axes[1].axvline(np.mean(high_lengths), color="#3fb950", linewidth=1.5,
                    linestyle="--",
                    label=f"high mean={np.mean(high_lengths):.2f}")
    axes[1].axvline(np.mean(low_lengths),  color="#f85149", linewidth=1.5,
                    linestyle="--",
                    label=f"low mean={np.mean(low_lengths):.2f}")
    axes[1].set_xlabel("Sequence length (# operators)", color="#c9d1d9", fontsize=10)
    axes[1].set_title("High vs Low Reward",              color="#c9d1d9", fontsize=11)
    axes[1].legend(fontsize=8, framealpha=0.3, facecolor="#161b22",
                   edgecolor="#30363d", labelcolor="#c9d1d9")

    # --- Panel 3: length vs mean reward (scatter + binned mean) ---
    from collections import defaultdict
    length_rewards = defaultdict(list)
    for item in theory_archive:
        length_rewards[len(item["ops"])].append(item["reward"])

    lengths   = sorted(length_rewards.keys())
    mean_r    = [np.mean(length_rewards[l]) for l in lengths]
    std_r     = [np.std(length_rewards[l])  for l in lengths]
    counts    = [len(length_rewards[l])     for l in lengths]

    axes[2].bar(lengths, mean_r, color="#58a6ff", alpha=0.7,
                yerr=std_r, capsize=4,
                error_kw=dict(ecolor="#e3b341", capsize=4))
    axes[2].axhline(reward_threshold, color="#3fb950", linewidth=1.0,
                    linestyle="--", alpha=0.7,
                    label=f"threshold={reward_threshold}")

    # annotate count per length
    for l, r, c in zip(lengths, mean_r, counts):
        axes[2].text(l, r + max(std_r)*0.1, f"n={c}",
                     ha="center", fontsize=8, color="#8b949e")

    axes[2].set_xlabel("Sequence length (# operators)", color="#c9d1d9", fontsize=10)
    axes[2].set_ylabel("Mean reward ± std",              color="#c9d1d9", fontsize=10)
    axes[2].set_title("Mean Reward by Length",           color="#c9d1d9", fontsize=11)
    axes[2].legend(fontsize=8, framealpha=0.3, facecolor="#161b22",
                   edgecolor="#30363d", labelcolor="#c9d1d9")

    # summary stats
    fig.text(0.5, 0.01,
        f"Total sequences: {len(all_lengths)}  |  "
        f"High reward (≥{reward_threshold}): {len(high_lengths)} "
        f"({100*len(high_lengths)/len(all_lengths):.1f}%)  |  "
        f"Unique lengths: {sorted(set(all_lengths))}",
        ha="center", fontsize=8, color="#8b949e"
    )

    plt.suptitle("Sequence Length Distribution",
                 color="#c9d1d9", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"saved → {save_path}")
    else:
        plt.show()



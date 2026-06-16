# ============================================================
# smeft_rl/coverage.py
# Tracks unique high-reward operator sets per arm (RL vs random)
# and produces the coverage comparison plots for the paper.
# ============================================================

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from .config import HIGH_REWARD_THRESHOLD
from .shared import SEARCH_SPACE_SIZE

class CoverageTracker:
    """
    Tracks unique high-reward operator sets discovered by each arm.

    Usage:
        tracker = CoverageTracker()
        # inside rollout loop:
        tracker.record(ops, reward, from_policy=True/False)
        # after training:
        tracker.save(save_dir)
        tracker.plot(save_dir)
    """

    def __init__(self, threshold=HIGH_REWARD_THRESHOLD):
        self.threshold        = threshold
        self.coverage_rl      = set()    # frozensets found by policy arm
        self.coverage_random  = set()    # frozensets found by random arm
        self.rl_curve         = []       # (eval_number, cumulative_count)
        self.random_curve     = []
        self.eval_counter     = 0

    def record(self, ops, reward, from_policy: bool):
        """Call once per evaluated sequence."""
        self.eval_counter += 1
        if reward >= self.threshold and ops:
            key = frozenset(ops)
            if from_policy:
                self.coverage_rl.add(key)
            else:
                self.coverage_random.add(key)
        self.rl_curve.append((self.eval_counter, len(self.coverage_rl)))
        self.random_curve.append((self.eval_counter, len(self.coverage_random)))

    def summary(self):
        print(f"  coverage RL     : {len(self.coverage_rl)} unique sets "
              f"(reward >= {self.threshold})")
        print(f"  coverage random : {len(self.coverage_random)} unique sets")

    def save(self, save_dir):
        save_dir = Path(save_dir)
        with open(save_dir / "coverage_rl.json",     "w") as f:
            json.dump(self.rl_curve, f)
        with open(save_dir / "coverage_random.json", "w") as f:
            json.dump(self.random_curve, f)
        print(f"Coverage curves saved to {save_dir}")
        print(f"  RL unique sets:     {[sorted(s) for s in self.coverage_rl]}")
        print(f"  Random unique sets: {[sorted(s) for s in self.coverage_random]}")

    def plot(self, save_dir):
        save_dir = Path(save_dir)

        rl_evals,     rl_counts     = zip(*self.rl_curve)
        random_evals, random_counts = zip(*self.random_curve)
        rl_evals,     rl_counts     = np.array(rl_evals),     np.array(rl_counts)
        random_evals, random_counts = np.array(random_evals), np.array(random_counts)

        max_eval  = min(rl_evals.max(), random_evals.max())
        eval_grid = np.linspace(0, max_eval, 500).astype(int)

        def interp(evals, counts, grid):
            out = np.zeros(len(grid), dtype=int)
            for j, e in enumerate(grid):
                idx    = np.searchsorted(evals, e, side="right") - 1
                out[j] = counts[idx] if idx >= 0 else 0
            return out

        rl_interp     = interp(rl_evals,     rl_counts,     eval_grid)
        random_interp = interp(random_evals, random_counts, eval_grid)

        window      = max(1, len(eval_grid) // 20)
        rl_rate     = np.convolve(
            np.diff(np.concatenate([[0], rl_interp])),
            np.ones(window)/window, mode="same") * 100
        random_rate = np.convolve(
            np.diff(np.concatenate([[0], random_interp])),
            np.ones(window)/window, mode="same") * 100

        # ── Figure 1: cumulative coverage + discovery rate ────
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.suptitle("RL Policy vs. Random Search: Solution Space Coverage",
                     fontsize=13, y=1.01)

        ax = axes[0]
        ax.step(eval_grid, rl_interp,     where="post",
                color="steelblue", lw=2.0, label="RL policy (PPO-GRPO)")
        ax.step(eval_grid, random_interp, where="post",
                color="tomato",    lw=2.0, ls="--", label="Random search")
        ax.set_xlabel("Number of evaluations", fontsize=11)
        ax.set_ylabel(f"Unique sets with reward ≥ {self.threshold}", fontsize=11)
        ax.set_title("Cumulative coverage of solution space", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        if rl_interp[-1] > 0:
            ax.annotate(f"RL: {rl_interp[-1]} sets",
                        xy=(eval_grid[-1], rl_interp[-1]),
                        xytext=(-60, 8), textcoords="offset points",
                        fontsize=9, color="steelblue",
                        arrowprops=dict(arrowstyle="->", color="steelblue", lw=1))
        if random_interp[-1] > 0:
            ax.annotate(f"Random: {random_interp[-1]} sets",
                        xy=(eval_grid[-1], random_interp[-1]),
                        xytext=(-60, -18), textcoords="offset points",
                        fontsize=9, color="tomato",
                        arrowprops=dict(arrowstyle="->", color="tomato", lw=1))

        ax = axes[1]
        ax.plot(eval_grid, rl_rate,     color="steelblue", lw=1.8, label="RL policy")
        ax.plot(eval_grid, random_rate, color="tomato",    lw=1.8,
                ls="--", label="Random search")
        ax.set_xlabel("Number of evaluations", fontsize=11)
        ax.set_ylabel("New unique sets per 100 evaluations", fontsize=11)
        ax.set_title("Discovery rate over training", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        out = save_dir / "coverage_comparison.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

        targets   = [1, 2, 5, 10, 20]
        rl_to_n, random_to_n, valid = [], [], []
        for n in targets:
            ri = np.searchsorted(rl_counts,     n, side="left")
            rr = np.searchsorted(random_counts, n, side="left")
            if ri < len(rl_evals) and rr < len(random_evals):
                rl_to_n.append(rl_evals[ri])
                random_to_n.append(random_evals[rr])
                valid.append(n)

        if valid:
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            x = np.arange(len(valid))
            w = 0.35
            ax2.bar(x - w/2, rl_to_n,     width=w,
                    color="steelblue", label="RL policy")
            ax2.bar(x + w/2, random_to_n, width=w,
                    color="tomato",    label="Random search")
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"{n} sets" for n in valid])
            ax2.set_ylabel("Evaluations required", fontsize=11)
            ax2.set_title("Evaluations to discover N unique high-reward sets",
                          fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(axis="y", alpha=0.3)
            ax2.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            out2 = save_dir / "coverage_speed.png"
            fig2.savefig(out2, dpi=150, bbox_inches="tight")
            plt.close(fig2)
            print(f"Saved {out2}")

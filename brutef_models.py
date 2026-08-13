"""
brutef_models.py

Brute-force scan over all combinations of 1, 2, 3, and 4 operators drawn
from operator_vocab. For each combination, runs flavio_pull_fn to minimize
chi2 and record the best-fit Wilson coefficients, predictions, and pulls.

Designed to run as a single SLURM task with many CPUs (ProcessPoolExecutor
fans out across cores within one process) -- NOT via srun --ntasks=N or MPI.

Example SLURM launch:
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=192
    python -u brutef_models.py
"""

import itertools
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# --- adjust these two imports to match your actual module names ---
from pulls_linear import flavio_pull_fn
# --------------------------------------------------------------------

# ------------------------------------------------------------
# LOAD OPERATOR VOCABULARY
# ------------------------------------------------------------

VOCAB_FILE = Path(
    "/home/kumarj/links/projects/def-london/kumarj/gitrepos/"
    "smeft-anomaly/smeft_new/vocab_0f2f.txt"
)

with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    operator_vocab = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

# Remove old sequence special tokens if they are present
SPECIAL_TOKENS = {"START", "END", "PAD", "UNK"}

operator_vocab = [
    op for op in operator_vocab
    if op not in SPECIAL_TOKENS
]

print(f"Loaded {len(operator_vocab)} operators")
print(operator_vocab[:10])

#operator_vocab=operator_vocab[:2]



# Output / checkpoint location. Project dirs are often read-only on compute
# nodes -- scratch is writable.
OUTPUT_DIR = Path("/home/kumarj/links/scratch")
CHECKPOINT_PATH = OUTPUT_DIR / "scan_results_checkpoint.pkl"
FINAL_RAW_PATH = OUTPUT_DIR / "scan_results_raw.pkl"
FINAL_DF_PATH = OUTPUT_DIR / "scan_results.pkl"

MAX_COMBO_SIZE = 4
CHECKPOINT_EVERY = 500  # write partial results to disk every N completed combos


def get_available_cpus() -> int:
    """
    Return the number of CPUs actually allocated to this job.
    Prefers SLURM's own accounting, falls back to sched_getaffinity
    (respects cgroup limits), falls back to os.cpu_count().
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        return int(slurm_cpus)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def generate_all_combos(vocab: list, max_size: int = MAX_COMBO_SIZE) -> list:
    """Generate every combination of size 1..max_size from vocab."""
    combos = []
    for r in range(1, max_size + 1):
        combos.extend(itertools.combinations(vocab, r))
    return combos


def _run_single_combo(combo: tuple) -> dict:
    """
    Worker function: runs flavio_pull_fn on one operator combination.
    Must be top-level (picklable) for ProcessPoolExecutor.
    """
    fired_ops = list(combo)
    try:
        obs_names, pulls, chi2_min, predictions, best_wc = flavio_pull_fn(fired_ops)
        return {
            "combo": combo,
            "n_ops": len(combo),
            "chi2_min": chi2_min,
            "best_wc": best_wc,
            "predictions": predictions,
            "pulls": pulls,
            "obs_names": obs_names,
            "error": None,
        }
    except Exception as e:
        # Don't let one bad combination kill the whole run
        return {
            "combo": combo,
            "n_ops": len(combo),
            "chi2_min": None,
            "best_wc": None,
            "predictions": None,
            "pulls": None,
            "obs_names": None,
            "error": repr(e),
        }


def run_parallel_scan(
    vocab: list,
    max_size: int = MAX_COMBO_SIZE,
    max_workers: int = None,
    checkpoint_path: Path = CHECKPOINT_PATH,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> pd.DataFrame:
    """
    Run flavio_pull_fn over every combination of size 1..max_size from
    vocab, in parallel across max_workers processes. Periodically
    checkpoints partial results to disk so a timeout/crash doesn't lose
    everything computed so far.
    """
    combos = generate_all_combos(vocab, max_size)
    n_total = len(combos)
    print(f"Total combinations to fit (sizes 1-{max_size} from {len(vocab)} ops): {n_total}",
          flush=True)
    print(f"Using max_workers={max_workers}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_single_combo, combo): combo for combo in combos}

        n_done = 0
        for future in as_completed(futures):
            results.append(future.result())
            n_done += 1

            if n_done % checkpoint_every == 0 or n_done == n_total:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0.0
                eta = (n_total - n_done) / rate if rate > 0 else float("inf")
                print(
                    f"  {n_done}/{n_total} done "
                    f"({elapsed:.1f}s elapsed, {rate:.2f} combos/s, "
                    f"ETA {eta:.1f}s)",
                    flush=True,
                )
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(results, f)

    total_elapsed = time.time() - t0
    print(f"Scan finished: {n_total} combos in {total_elapsed:.1f}s", flush=True)

    # Save final raw results (list of dicts) as a backup before building the DataFrame
    with open(FINAL_RAW_PATH, "wb") as f:
        pickle.dump(results, f)

    df = pd.DataFrame(results)
    return df


def main():
    n_workers = get_available_cpus()
    print(f"Detected {n_workers} available CPUs", flush=True)

    df = run_parallel_scan(operator_vocab, max_size=MAX_COMBO_SIZE, max_workers=n_workers)

    failed = df[df["error"].notna()]
    if not failed.empty:
        print(f"\n{len(failed)} combinations failed to fit:", flush=True)
        print(failed[["combo", "error"]].to_string(), flush=True)

    ok = df[df["error"].isna()].copy()
    ok = ok.sort_values("chi2_min", ascending=True).reset_index(drop=True)

    print("\nTop 20 best-fitting combinations overall:", flush=True)
    print(ok[["combo", "n_ops", "chi2_min"]].head(20).to_string(), flush=True)

    print("\nBest combination per size:", flush=True)
    for n in range(1, MAX_COMBO_SIZE + 1):
        subset = ok[ok["n_ops"] == n]
        if not subset.empty:
            best = subset.iloc[0]
            print(f"  size {n}: {best['combo']}  chi2_min={best['chi2_min']:.4f}", flush=True)

    ok.to_pickle(FINAL_DF_PATH)
    print(f"\nFull results saved to {FINAL_DF_PATH}", flush=True)

    # Clean up the intermediate checkpoint now that the final result is safely written
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()


if __name__ == "__main__":
    main()

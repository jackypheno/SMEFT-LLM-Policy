"""
print_best_theories.py

Load scan_results.pkl (or scan_results_raw.pkl / the checkpoint file) and
print the best-fitting operator combinations.

Usage:
    python print_best_theories.py
    python print_best_theories.py --path /home/kumarj/links/scratch/scan_results.pkl --top 30
    python print_best_theories.py --size 3          # only 3-operator combos
    python print_best_theories.py --op C_ll          # only combos containing this operator
    python print_best_theories.py --show-wc          # also print best-fit Wilson coefficients
    python print_best_theories.py --show-pulls        # also print per-observable pulls
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("/home/kumarj/links/scratch/scan_results.pkl")


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    with open(path, "rb") as f:
        obj = pickle.load(f)

    # scan_results.pkl was saved with df.to_pickle -> it's already a DataFrame
    if isinstance(obj, pd.DataFrame):
        return obj

    # scan_results_raw.pkl / checkpoint files are a list of dicts
    if isinstance(obj, list):
        return pd.DataFrame(obj)

    raise TypeError(f"Unrecognized pickle contents: {type(obj)}")


def main():
    parser = argparse.ArgumentParser(description="Print best-fitting operator combinations")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH,
                         help="Path to results pickle (DataFrame or raw list)")
    parser.add_argument("--top", type=int, default=5000,
                         help="Number of top combinations to print")
    parser.add_argument("--size", type=int, default=None, choices=[1, 2, 3, 4],
                         help="Only show combinations of this size")
    parser.add_argument("--op", type=str, default=None,
                         help="Only show combinations containing this operator")
    parser.add_argument("--show-wc", action="store_true",
                         help="Also print best-fit Wilson coefficients for each combo")
    parser.add_argument("--show-pulls", action="store_true",
                         help="Also print per-observable pulls for each combo")
    args = parser.parse_args()

    df = load_results(args.path)
    print(f"Loaded {len(df)} results from {args.path}\n")

    # separate failed fits
    if "error" in df.columns:
        failed = df[df["error"].notna()]
        ok = df[df["error"].isna()].copy()
        if not failed.empty:
            print(f"({len(failed)} combinations failed to fit -- excluded below)\n")
    else:
        ok = df.copy()

    # optional filters
    if args.size is not None:
        ok = ok[ok["n_ops"] == args.size]

    if args.op is not None:
        ok = ok[ok["combo"].apply(lambda c: args.op in c)]

    if ok.empty:
        print("No matching results found.")
        return

    ok = ok.sort_values("chi2_min", ascending=True).reset_index(drop=True)

    print(f"Top {min(args.top, len(ok))} best-fitting combinations"
          + (f" (size={args.size})" if args.size else "")
          + (f" (containing {args.op})" if args.op else "")
          + ":\n")

    for i, row in ok.head(args.top).iterrows():
        print(f"[{i+1}] combo={row['combo']}  n_ops={row['n_ops']}  chi2_min={row['chi2_min']:.4f}")
        if args.show_wc:
            print(f"     best_wc: {row['best_wc']}")
        if args.show_pulls:
            pulls_str = ", ".join(
                f"{name}={pull:.2f}" for name, pull in zip(row["obs_names"], row["pulls"])
            )
            print(f"     pulls: {pulls_str}")
        print()

    print("Best combination per size:")
    for n in sorted(ok["n_ops"].unique()):
        subset = ok[ok["n_ops"] == n]
        if not subset.empty:
            best = subset.iloc[0]
            print(f"  size {n}: {best['combo']}  chi2_min={best['chi2_min']:.4f}")


if __name__ == "__main__":
    main()

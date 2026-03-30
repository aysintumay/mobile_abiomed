"""
Compute mean ± std of final_normalized_reward across seeds from results CSVs.
Usage:
    python compute_results.py [results_*.csv ...]
    python compute_results.py  # auto-discovers all results_*.csv in cwd
"""
import sys
import glob
import pandas as pd
import numpy as np


def load_csv(path):
    """Load a results CSV, skipping malformed rows."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            parts = line.split(",")
            if len(parts) != 5:
                continue
            task, dataset, seed, reward, reward_std = parts
            # Skip header or non-numeric values
            try:
                rows.append({
                    "task": task,
                    "dataset": dataset,
                    "seed": int(seed),
                    "final_normalized_reward": float(reward),
                    "final_reward_std": float(reward_std),
                    "source_file": path,
                })
            except ValueError:
                continue
    return pd.DataFrame(rows)


def summarize(df):
    """For each (task, dataset), keep last entry per seed, then compute mean±std."""
    results = []
    for (task, dataset), group in df.groupby(["task", "dataset"]):
        # Keep the last logged entry per seed
        last_per_seed = group.sort_index().groupby("seed").last()
        rewards = last_per_seed["final_normalized_reward"].values
        mean = np.mean(rewards)
        std = np.std(rewards, ddof=1) if len(rewards) > 1 else 0.0
        results.append({
            "task": task,
            "dataset": dataset,
            "n_seeds": len(rewards),
            "seeds": sorted(last_per_seed.index.tolist()),
            "mean": mean,
            "std": std,
            "per_seed_rewards": rewards.tolist(),
        })
    return results


def infer_model(path):
    """Infer model variant from filename suffix."""
    name = path.replace("\\", "/").split("/")[-1]  # basename
    name = name.replace("results_", "").replace(".csv", "")
    for variant in ("diffusion", "realnvp", "vae", "kde"):
        if name.endswith(f"_{variant}"):
            return variant
    return "mobile"


def main():
    files = sys.argv[1:] or sorted(glob.glob("results_*.csv"))
    if not files:
        print("No results CSV files found.")
        return

    # Group files by model variant
    from collections import defaultdict
    by_model = defaultdict(list)
    for path in files:
        model = infer_model(path)
        df = load_csv(path)
        if df.empty:
            print(f"  [skip] {path} — no valid rows")
            continue
        df["model"] = model
        by_model[model].append(df)

    if not by_model:
        print("No valid data loaded.")
        return

    for model in sorted(by_model.keys()):
        combined = pd.concat(by_model[model], ignore_index=True)
        # Deduplicate: keep last entry per (task, dataset, seed)
        combined = combined.groupby(["task", "dataset", "seed"]).last().reset_index()

        summaries = summarize(combined)
        if not summaries:
            continue

        print(f"\n{'='*90}")
        print(f"Model: {model.upper()}")
        print(f"{'='*90}")
        print(f"{'Dataset':<45} {'Seeds':<20} {'Mean':>10} {'Std':>10}")
        print("-" * 90)
        for r in sorted(summaries, key=lambda x: x["dataset"]):
            seeds_str = str(r["seeds"])
            print(f"{r['dataset']:<45} {seeds_str:<20} {r['mean']:>10.2f} {r['std']:>10.2f}")

        print()
        for r in sorted(summaries, key=lambda x: x["dataset"]):
            seed_vals = ", ".join(
                f"seed {s}: {v:.2f}"
                for s, v in zip(r["seeds"], r["per_seed_rewards"])
            )
            print(f"  {r['dataset']}: {seed_vals}")
            print(f"    => mean={r['mean']:.4f} ± {r['std']:.4f} (n={r['n_seeds']})")


if __name__ == "__main__":
    main()

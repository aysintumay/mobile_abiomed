#!/usr/bin/env python3
"""
Script to aggregate evaluation results from multiple seed runs into a single CSV.
Collects results from the most recent runs for each seed.
"""

import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
import glob


def find_latest_runs(task_name, log_dir="log", num_seeds=3):
    """Find the latest run directory for each seed."""
    task_dir = os.path.join(log_dir, task_name)

    if not os.path.exists(task_dir):
        print(f"Error: Task directory not found: {task_dir}")
        return []

    seed_runs = {}

    # Iterate through all algorithm directories
    for algo_dir in glob.glob(os.path.join(task_dir, "*")):
        if not os.path.isdir(algo_dir):
            continue

        # Find all seed runs
        for run_dir in glob.glob(os.path.join(algo_dir, "seed_*")):
            # Extract seed number
            seed_str = os.path.basename(run_dir).split("&")[0].replace("seed_", "")
            try:
                seed = int(seed_str)
            except ValueError:
                continue

            # Extract timestamp
            timestamp_str = os.path.basename(run_dir).split("&timestamp_")[1] if "&timestamp_" in run_dir else ""

            # Keep track of the latest run for each seed
            if seed not in seed_runs or timestamp_str > seed_runs[seed][1]:
                csv_path = os.path.join(run_dir, "record", "policy_training_progress.csv")
                if os.path.exists(csv_path):
                    seed_runs[seed] = (csv_path, timestamp_str, run_dir)

    return seed_runs


def aggregate_results(task_name, output_file="results.csv", log_dir="log"):
    """Aggregate evaluation results from multiple seeds."""

    seed_runs = find_latest_runs(task_name, log_dir)

    if not seed_runs:
        print(f"No runs found for task: {task_name}")
        return

    print(f"Found {len(seed_runs)} seed runs:")
    for seed, (csv_path, timestamp, run_dir) in sorted(seed_runs.items()):
        print(f"  Seed {seed}: {run_dir}")

    # Read all CSVs
    all_data = []
    for seed, (csv_path, _, _) in sorted(seed_runs.items()):
        try:
            df = pd.read_csv(csv_path)

            # Filter to only evaluation rows (those with eval metrics)
            eval_cols = [col for col in df.columns if col.startswith('eval/')]
            if not eval_cols:
                print(f"Warning: No evaluation columns found in {csv_path}")
                continue

            # Add seed column
            df['seed'] = seed

            # Keep only timestep and eval columns
            keep_cols = ['timestep', 'seed'] + eval_cols
            df_filtered = df[keep_cols].dropna(subset=eval_cols, how='all')

            all_data.append(df_filtered)

        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue

    if not all_data:
        print("No data to aggregate!")
        return

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Sort by timestep and seed
    combined_df = combined_df.sort_values(['timestep', 'seed'])

    # Save combined results
    combined_df.to_csv(output_file, index=False)
    print(f"\nSaved combined results to {output_file}")

    # Calculate and print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    # Group by timestep and calculate mean/std across seeds
    eval_cols = [col for col in combined_df.columns if col.startswith('eval/') and not col.endswith('_std')]

    if eval_cols:
        summary_data = []
        for timestep in sorted(combined_df['timestep'].unique()):
            timestep_data = combined_df[combined_df['timestep'] == timestep]

            summary_row = {'timestep': timestep}
            for col in eval_cols:
                if col in timestep_data.columns:
                    values = timestep_data[col].dropna()
                    if len(values) > 0:
                        summary_row[f'{col}_mean'] = values.mean()
                        summary_row[f'{col}_std'] = values.std()

            summary_data.append(summary_row)

        summary_df = pd.DataFrame(summary_data)
        summary_file = output_file.replace('.csv', '_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved summary statistics to {summary_file}")

        # Print final performance
        if len(summary_df) > 0:
            print("\nFinal Performance (last evaluation):")
            last_row = summary_df.iloc[-1]
            for col in summary_df.columns:
                if col != 'timestep' and pd.notna(last_row[col]):
                    print(f"  {col}: {last_row[col]:.4f}")

    print("\n" + "="*60)
    print(f"Total evaluations: {len(combined_df)}")
    print(f"Seeds: {sorted(combined_df['seed'].unique())}")
    print(f"Timesteps: {sorted(combined_df['timestep'].unique())}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate evaluation results from multiple seed runs')
    parser.add_argument('--task', type=str, required=True, help='Task name (e.g., halfcheetah-medium-expert-v2)')
    parser.add_argument('--output', type=str, default='results.csv', help='Output CSV file path')
    parser.add_argument('--log-dir', type=str, default='log', help='Base log directory')

    args = parser.parse_args()

    aggregate_results(args.task, args.output, args.log_dir)


if __name__ == "__main__":
    main()

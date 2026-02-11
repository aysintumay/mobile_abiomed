#!/usr/bin/env python
"""
Evaluate OOD Detection for Sparse Diffusion Models

This script evaluates OOD detection performance for the 4 sparse diffusion models:
1. halfcheetah-medium-expert-sparse-57.5
2. halfcheetah-medium-expert-sparse-72.5
3. hopper-medium-expert-sparse-78
4. walker2d-medium-expert-sparse-73

Uses the original OOD test data from /public/d4rl/ood_test/{env}-medium-expert-v2/

Results are saved to JSON files for plotting.
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_ood import DiffusionOOD
from monte_carlo_sampling_unconditional import build_model_from_ckpt
from ddim_training_unconditional import NPZTargetDataset
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


# Configuration for sparse models
SPARSE_MODELS = {
    "halfcheetah_sparse_57.5": {
        "model_dir": "/public/gormpo/models/halfcheetah_medium_expert_sparse_2/diffusion",
        "ood_base_dir": "/public/d4rl/ood_test/halfcheetah-medium-expert-v2",
        "train_data": "/public/d4rl/sparse_datasets/diffusion_processed/halfcheetah_medium_expert_sparse_57.5_train.npz",
        "env_name": "halfcheetah-medium-expert-v2",
        "sparse_level": "57.5%",
    },
    "halfcheetah_sparse_72.5": {
        "model_dir": "/public/gormpo/models/halfcheetah_medium_expert_sparse_3/diffusion",
        "ood_base_dir": "/public/d4rl/ood_test/halfcheetah-medium-expert-v2",
        "train_data": "/public/d4rl/sparse_datasets/diffusion_processed/halfcheetah_medium_expert_sparse_72.5_train.npz",
        "env_name": "halfcheetah-medium-expert-v2",
        "sparse_level": "72.5%",
    },
    "hopper_sparse_78": {
        "model_dir": "/public/gormpo/models/hopper_medium_expert_sparse_3/diffusion",
        "ood_base_dir": "/public/d4rl/ood_test/hopper-medium-expert-v2",
        "train_data": "/public/d4rl/sparse_datasets/diffusion_processed/hopper_medium_expert_sparse_78_train.npz",
        "env_name": "hopper-medium-expert-v2",
        "sparse_level": "78%",
    },
    "walker2d_sparse_73": {
        "model_dir": "/public/gormpo/models/walker2d_medium_expert_sparse_3/diffusion",
        "ood_base_dir": "/public/d4rl/ood_test/walker2d-medium-expert-v2",
        "train_data": "/public/d4rl/sparse_datasets/diffusion_processed/walker2d_medium_expert_sparse_73_train.npz",
        "env_name": "walker2d-medium-expert-v2",
        "sparse_level": "73%",
    },
}

# OOD distance levels to test
OOD_DISTANCES = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0]


def load_ood_data(ood_path: str, device: str = "cuda") -> torch.Tensor:
    """Load OOD test data from pickle file."""
    with open(ood_path, "rb") as f:
        data = pickle.load(f)

    # Handle different formats
    if isinstance(data, dict):
        if "target" in data:
            arr = data["target"]
        elif "observations" in data and "actions" in data:
            obs = data["observations"]
            actions = data["actions"]
            arr = np.concatenate([obs, actions], axis=1)
        elif "next_observations" in data and "actions" in data:
            obs = data["next_observations"]
            actions = data["actions"]
            arr = np.concatenate([obs, actions], axis=1)
        else:
            # Try first array-like value
            for v in data.values():
                if isinstance(v, np.ndarray):
                    arr = v
                    break
            else:
                raise ValueError(f"Cannot find data array in {ood_path}")
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise ValueError(f"Unknown data format in {ood_path}")

    return torch.FloatTensor(arr).to(device)


def load_train_data(npz_path: str, device: str = "cuda", max_samples: int = 5000) -> torch.Tensor:
    """Load training data from NPZ file."""
    dataset = NPZTargetDataset(npz_path)
    data = dataset.target[:max_samples]
    return data.to(device)


def load_diffusion_model(model_dir: str, device: str = "cuda") -> Tuple[DiffusionOOD, dict]:
    """Load trained diffusion model."""
    ckpt_path = os.path.join(model_dir, "checkpoint.pt")
    sched_dir = os.path.join(model_dir, "scheduler")

    # Also try model.pt if checkpoint.pt doesn't exist
    if not os.path.exists(ckpt_path):
        alt_ckpt = os.path.join(model_dir, "model.pt")
        if os.path.exists(alt_ckpt):
            ckpt_path = alt_ckpt
        else:
            raise FileNotFoundError(f"No checkpoint found in {model_dir}")

    model, train_cfg = build_model_from_ckpt(ckpt_path, device)

    # Load scheduler
    try:
        scheduler = DDIMScheduler.from_pretrained(sched_dir)
    except Exception:
        try:
            scheduler = DDPMScheduler.from_pretrained(sched_dir)
        except Exception:
            scheduler = DDIMScheduler(
                num_train_timesteps=50000,
                beta_schedule="linear",
                prediction_type="epsilon",
            )

    model.eval()
    ood_model = DiffusionOOD(model, scheduler, device=device, num_inference_steps=100)

    return ood_model, train_cfg


def evaluate_single_ood(
    ood_model: DiffusionOOD,
    train_data: torch.Tensor,
    ood_data: torch.Tensor,
    batch_size: int = 256,
) -> Dict:
    """Evaluate OOD detection for a single OOD dataset."""
    # Score both datasets
    train_scores = ood_model.score_samples(train_data, batch_size=batch_size).cpu().numpy()
    ood_scores = ood_model.score_samples(ood_data, batch_size=batch_size).cpu().numpy()

    # Create labels (0 = in-distribution, 1 = OOD)
    y_true = np.concatenate([
        np.zeros(len(train_scores)),
        np.ones(len(ood_scores))
    ])

    # Use negative log prob as anomaly score (lower log prob = more anomalous)
    y_scores = np.concatenate([-train_scores, -ood_scores])

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Compute precision-recall curve and average precision
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    # Compute separation statistics
    results = {
        "roc_auc": float(roc_auc),
        "avg_precision": float(avg_precision),
        "train_log_prob_mean": float(train_scores.mean()),
        "train_log_prob_std": float(train_scores.std()),
        "ood_log_prob_mean": float(ood_scores.mean()),
        "ood_log_prob_std": float(ood_scores.std()),
        "log_prob_gap": float(train_scores.mean() - ood_scores.mean()),
        "n_train": len(train_scores),
        "n_ood": len(ood_scores),
        # Store raw scores for later analysis
        "train_scores": train_scores.tolist(),
        "ood_scores": ood_scores.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }

    return results


def evaluate_model(
    model_name: str,
    config: Dict,
    ood_distances: List[float],
    device: str = "cuda",
    batch_size: int = 256,
    max_train_samples: int = 5000,
) -> Dict:
    """Evaluate a single sparse diffusion model across all OOD distances."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Model dir: {config['model_dir']}")
    print(f"{'='*60}")

    # Check if model exists
    if not os.path.exists(config["model_dir"]):
        print(f"  [SKIP] Model directory not found: {config['model_dir']}")
        return {"error": "model_not_found", "model_dir": config["model_dir"]}

    # Load model
    try:
        ood_model, _ = load_diffusion_model(config["model_dir"], device)
        print(f"  Loaded diffusion model")
    except Exception as e:
        print(f"  [ERROR] Failed to load model: {e}")
        return {"error": str(e)}

    # Load training data
    try:
        train_data = load_train_data(config["train_data"], device, max_train_samples)
        print(f"  Loaded {len(train_data)} training samples")
    except Exception as e:
        print(f"  [ERROR] Failed to load training data: {e}")
        return {"error": str(e)}

    # Set threshold based on training data
    ood_model.set_threshold(train_data, anomaly_fraction=0.01, batch_size=batch_size)

    results = {
        "model_name": model_name,
        "env_name": config["env_name"],
        "sparse_level": config["sparse_level"],
        "model_dir": config["model_dir"],
        "train_data_path": config["train_data"],
        "threshold": ood_model.threshold,
        "n_train_samples": len(train_data),
        "ood_results": {},
    }

    # Evaluate each OOD distance
    for dist in ood_distances:
        # Try different filename formats
        ood_files = [
            f"ood-distance-{dist}.pkl",
            f"ood-distance-{int(dist)}.pkl" if dist == int(dist) else None,
            f"ood-distance-{dist}-uniform.pkl",
        ]
        ood_files = [f for f in ood_files if f is not None]

        ood_path = None
        for fname in ood_files:
            candidate = os.path.join(config["ood_base_dir"], fname)
            if os.path.exists(candidate):
                ood_path = candidate
                break

        if ood_path is None:
            print(f"  [SKIP] OOD distance {dist}: file not found")
            results["ood_results"][str(dist)] = {"error": "file_not_found"}
            continue

        try:
            ood_data = load_ood_data(ood_path, device)
            print(f"  Evaluating OOD distance {dist}: {len(ood_data)} samples")

            ood_result = evaluate_single_ood(
                ood_model, train_data, ood_data, batch_size
            )
            ood_result["ood_path"] = ood_path
            results["ood_results"][str(dist)] = ood_result

            print(f"    ROC AUC: {ood_result['roc_auc']:.4f}, "
                  f"Log prob gap: {ood_result['log_prob_gap']:.2f}")

        except Exception as e:
            print(f"  [ERROR] OOD distance {dist}: {e}")
            results["ood_results"][str(dist)] = {"error": str(e)}

    return results


def save_results(results: Dict, output_dir: str):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save full results
    full_path = os.path.join(output_dir, f"ood_results_full_{timestamp}.json")
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {full_path}")

    # Save summary (without raw scores for smaller file)
    summary = {}
    for model_name, model_results in results["models"].items():
        if "error" in model_results:
            summary[model_name] = model_results
            continue

        summary[model_name] = {
            "env_name": model_results.get("env_name"),
            "sparse_level": model_results.get("sparse_level"),
            "threshold": model_results.get("threshold"),
            "ood_results": {},
        }

        for dist, ood_res in model_results.get("ood_results", {}).items():
            if "error" in ood_res:
                summary[model_name]["ood_results"][dist] = ood_res
            else:
                summary[model_name]["ood_results"][dist] = {
                    "roc_auc": ood_res["roc_auc"],
                    "avg_precision": ood_res["avg_precision"],
                    "train_log_prob_mean": ood_res["train_log_prob_mean"],
                    "ood_log_prob_mean": ood_res["ood_log_prob_mean"],
                    "log_prob_gap": ood_res["log_prob_gap"],
                }

    summary_path = os.path.join(output_dir, f"ood_results_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump({"timestamp": timestamp, "models": summary}, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    # Also save a latest symlink-style copy
    latest_full = os.path.join(output_dir, "ood_results_full_latest.json")
    latest_summary = os.path.join(output_dir, "ood_results_summary_latest.json")
    with open(latest_full, "w") as f:
        json.dump(results, f, indent=2)
    with open(latest_summary, "w") as f:
        json.dump({"timestamp": timestamp, "models": summary}, f, indent=2)

    return full_path, summary_path


def plot_results(results: Dict, output_dir: str):
    """Generate plots from results."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: ROC AUC vs OOD distance for all models
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D']

    for idx, (model_name, model_results) in enumerate(results["models"].items()):
        if "error" in model_results:
            continue

        distances = []
        aucs = []
        for dist_str, ood_res in model_results.get("ood_results", {}).items():
            if "error" not in ood_res:
                distances.append(float(dist_str))
                aucs.append(ood_res["roc_auc"])

        if distances:
            sorted_pairs = sorted(zip(distances, aucs))
            distances, aucs = zip(*sorted_pairs)

            label = f"{model_results['env_name']} ({model_results['sparse_level']})"
            ax.plot(distances, aucs, marker=markers[idx % len(markers)],
                   color=colors[idx % len(colors)], label=label, linewidth=2, markersize=8)

    ax.set_xlabel("OOD Distance", fontsize=12)
    ax.set_ylabel("ROC AUC", fontsize=12)
    ax.set_title("OOD Detection Performance: Sparse Diffusion Models", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])

    plot_path = os.path.join(output_dir, "roc_auc_vs_distance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")

    # Plot 2: Log probability distributions for each model (at distance 1.0)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (model_name, model_results) in enumerate(results["models"].items()):
        if "error" in model_results or idx >= 4:
            continue

        ax = axes[idx]

        # Use distance 1.0 for comparison
        dist_key = "1.0"
        if dist_key not in model_results.get("ood_results", {}):
            dist_key = "1"

        if dist_key in model_results.get("ood_results", {}) and "error" not in model_results["ood_results"][dist_key]:
            ood_res = model_results["ood_results"][dist_key]

            train_scores = np.array(ood_res["train_scores"])
            ood_scores = np.array(ood_res["ood_scores"])

            # Plot histograms
            bins = 50
            ax.hist(train_scores, bins=bins, alpha=0.6, label="In-distribution", color="blue", density=True)
            ax.hist(ood_scores, bins=bins, alpha=0.6, label="OOD (dist=1.0)", color="red", density=True)

            # Add threshold line
            if model_results.get("threshold") is not None:
                ax.axvline(x=model_results["threshold"], color='green', linestyle='--',
                          linewidth=2, label=f'Threshold')

            ax.set_xlabel("Log Probability (ELBO)", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.set_title(f"{model_results['env_name']}\n({model_results['sparse_level']} sparse)", fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "log_prob_distributions.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")

    # Plot 3: ROC curves at distance 1.0
    fig, ax = plt.subplots(figsize=(8, 8))

    for idx, (model_name, model_results) in enumerate(results["models"].items()):
        if "error" in model_results:
            continue

        dist_key = "1.0"
        if dist_key not in model_results.get("ood_results", {}):
            dist_key = "1"

        if dist_key in model_results.get("ood_results", {}) and "error" not in model_results["ood_results"][dist_key]:
            ood_res = model_results["ood_results"][dist_key]

            fpr = ood_res["fpr"]
            tpr = ood_res["tpr"]
            roc_auc = ood_res["roc_auc"]

            label = f"{model_results['env_name']} ({model_results['sparse_level']}) - AUC={roc_auc:.3f}"
            ax.plot(fpr, tpr, label=label, linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves at OOD Distance 1.0", fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, "roc_curves_dist1.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate OOD detection for sparse diffusion models")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Specific models to evaluate (default: all)")
    parser.add_argument("--distances", nargs="+", type=float, default=None,
                       help="OOD distances to test (default: 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0)")
    parser.add_argument("--output-dir", type=str,
                       default="/home/ubuntu/Projects/GORMPO/results/sparse_diffusion_ood",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")

    args = parser.parse_args()

    # Select models
    models_to_eval = SPARSE_MODELS
    if args.models:
        models_to_eval = {k: v for k, v in SPARSE_MODELS.items() if k in args.models}

    # Select distances
    distances = args.distances if args.distances else OOD_DISTANCES

    print("="*60)
    print("Sparse Diffusion Model OOD Evaluation")
    print("="*60)
    print(f"Models: {list(models_to_eval.keys())}")
    print(f"OOD distances: {distances}")
    print(f"Device: {args.device}")
    print(f"Output dir: {args.output_dir}")

    # Run evaluation
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "ood_distances": distances,
            "device": args.device,
            "batch_size": args.batch_size,
            "max_train_samples": args.max_train_samples,
        },
        "models": {},
    }

    for model_name, config in models_to_eval.items():
        model_results = evaluate_model(
            model_name, config, distances,
            device=args.device,
            batch_size=args.batch_size,
            max_train_samples=args.max_train_samples,
        )
        all_results["models"][model_name] = model_results

    # Save results
    full_path, summary_path = save_results(all_results, args.output_dir)

    # Generate plots
    if not args.no_plot:
        plot_results(all_results, args.output_dir)

    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()

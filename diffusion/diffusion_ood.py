"""
Diffusion Model Out-of-Distribution (OOD) Detection

This script provides OOD detection for unconditional diffusion models using ELBO-based
log-likelihood estimation. It follows the same approach as Neural ODE and RealNVP OOD detection.
"""

import argparse
import json
import os
import sys
from typing import Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
from sklearn.metrics import roc_curve, auc
from scipy.stats import gaussian_kde

try:
    import yaml
except Exception:
    yaml = None

from monte_carlo_sampling_unconditional import (
    build_model_from_ckpt,
)
from ddim_training_unconditional import (
    NPZTargetDataset,
    log_prob_elbo,
)
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


class DiffusionOOD:
    """
    Diffusion Model-based Out-of-Distribution (OOD) Detection wrapper.

    Uses ELBO (Evidence Lower Bound) to compute log-likelihood for OOD detection.
    """

    def __init__(
        self,
        model: nn.Module,
        scheduler,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        num_inference_steps: int = None
    ):
        """
        Initialize the Diffusion OOD detector.

        Args:
            model: Trained diffusion model (epsilon prediction)
            scheduler: Diffusion scheduler (DDPM or DDIM)
            device: Device to run computations on
            num_inference_steps: Number of timesteps for ELBO (None = use all 50K, e.g. 20 for fast)
        """
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.threshold = None
        self.model.to(device)
        self.model.eval()

    def score_samples(self, x: torch.Tensor, batch_size: int = 512, num_inference_steps: int = None) -> torch.Tensor:
        """
        Compute log probability of data points using ELBO.

        Args:
            x: Input tensor of shape (num_samples, dim)
            batch_size: Batch size for processing (default: 512)
            num_inference_steps: Number of timesteps to use for ELBO (default: use self.num_inference_steps)
                If specified, uniformly subsample timesteps for faster approximation

        Returns:
            Log probabilities (ELBO) for each sample
        """
        self.model.eval()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)

        # Use instance default if not specified
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps

        # Process in batches to avoid OOM
        all_log_probs = []
        for i in range(0, len(x), batch_size):
            batch = x[i:min(i+batch_size, len(x))]
            with torch.no_grad():
                log_probs = log_prob_elbo(
                    model=self.model,
                    scheduler=self.scheduler,
                    x0=batch,
                    device=self.device,
                    num_inference_steps=num_inference_steps
                )
            all_log_probs.append(log_probs.cpu())

        return torch.cat(all_log_probs)

    def set_threshold(
        self,
        val_data: torch.Tensor,
        anomaly_fraction: float = 0.01,
        batch_size: int = 512
    ) -> float:
        """
        Set threshold for anomaly detection based on validation data.

        Args:
            val_data: Validation dataset (assumed to be normal data)
            anomaly_fraction: Fraction of validation data to classify as anomalies
            batch_size: Batch size for processing validation data

        Returns:
            The computed threshold value
        """
        self.model.eval()
        all_log_probs = []

        # Process in batches
        num_samples = len(val_data)
        for i in range(0, num_samples, batch_size):
            batch = val_data[i:min(i+batch_size, num_samples)]
            batch = batch.to(self.device)
            log_probs = self.score_samples(batch, batch_size=batch_size)
            all_log_probs.append(log_probs)

        log_probs = torch.cat(all_log_probs)

        # Set threshold as percentile of validation log probabilities
        self.threshold = torch.quantile(log_probs, anomaly_fraction).item()

        print(f'Threshold set to {self.threshold:.4f} '
              f'(marking {anomaly_fraction*100:.1f}% of validation data as anomalies)')

        return self.threshold

    def predict_anomaly(self, x: torch.Tensor, batch_size: int = 512) -> np.ndarray:
        """
        Predict anomalies based on log probability threshold.

        Args:
            x: Input data
            batch_size: Batch size for processing

        Returns:
            Boolean array indicating anomalies (True = anomaly)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")

        log_probs = self.score_samples(x, batch_size)
        return log_probs.numpy() < self.threshold

    def predict(self, x: torch.Tensor, batch_size: int = 512) -> np.ndarray:
        """
        Predict anomalies based on threshold.

        Args:
            x: Test data
            batch_size: Batch size for processing

        Returns:
            predictions: 1 for normal, -1 for anomaly
        """
        anomalies = self.predict_anomaly(x, batch_size)
        return np.where(anomalies, -1, 1)

    def evaluate_anomaly_detection(
        self,
        normal_data: torch.Tensor,
        anomaly_data: torch.Tensor,
        plot: bool = True,
        save_path: Optional[str] = None,
        batch_size: int = 512
    ) -> dict:
        """
        Evaluate anomaly detection performance.

        Args:
            normal_data: Normal test data
            anomaly_data: Anomalous test data
            plot: Whether to plot ROC curve
            save_path: Path to save the ROC curve plot
            batch_size: Batch size for processing

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        # Compute log probabilities
        normal_log_probs = self.score_samples(normal_data, batch_size).numpy()
        anomaly_log_probs = self.score_samples(anomaly_data, batch_size).numpy()

        # Create labels (0 = normal, 1 = anomaly)
        y_true = np.concatenate([
            np.zeros(len(normal_log_probs)),
            np.ones(len(anomaly_log_probs))
        ])

        # Use negative log prob as anomaly score
        y_scores = np.concatenate([-normal_log_probs, -anomaly_log_probs])

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Compute accuracy with current threshold
        if self.threshold is not None:
            predictions = np.concatenate([
                normal_log_probs < self.threshold,
                anomaly_log_probs < self.threshold
            ])
            accuracy = (predictions == y_true).mean()
        else:
            accuracy = None

        results = {
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'normal_log_prob_mean': normal_log_probs.mean(),
            'normal_log_prob_std': normal_log_probs.std(),
            'anomaly_log_prob_mean': anomaly_log_probs.mean(),
            'anomaly_log_prob_std': anomaly_log_probs.std()
        }

        if plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2)
            ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('ROC Curve for OOD Detection (Diffusion)', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"ROC curve saved to {save_path}")
            else:
                os.makedirs('figures/diffusion_ood', exist_ok=True)
                plt.savefig('figures/diffusion_ood/diffusion_roc_curve.png', dpi=300, bbox_inches='tight')
                print("ROC curve saved to figures/diffusion_ood/diffusion_roc_curve.png")
            plt.close(fig)

        return results

    def save_model(self, save_path: str, train_data: Optional[torch.Tensor] = None):
        """
        Save the Diffusion OOD model metadata.

        Args:
            save_path: Base path for saving (without extension)
            train_data: Optional training data to compute statistics
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        # Compute training log probabilities if provided
        metadata = {
            'threshold': self.threshold,
            'device': self.device,
        }

        if train_data is not None:
            self.model.eval()
            train_log_probs = self.score_samples(train_data.to(self.device))
            metadata['mean'] = train_log_probs.mean().item()
            metadata['std'] = train_log_probs.std().item()

        import pickle
        with open(f"{save_path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Metadata saved to: {save_path}_metadata.pkl")


def plot_likelihood_distributions(
    model: DiffusionOOD,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    ood_data: Optional[torch.Tensor] = None,
    threshold: Optional[float] = None,
    title: str = "Likelihood Distribution (Diffusion ELBO)",
    save_dir: str = "figures/diffusion_ood",
    bins: int = 50,
    batch_size: int = 512
):
    """
    Visualize log-likelihood distributions for train, val, and OOD data.

    Args:
        model: Diffusion OOD model
        train_data: In-distribution training set
        val_data: Held-out validation set
        ood_data: Optional OOD dataset
        threshold: Threshold value for anomaly detection
        title: Title for the plot
        save_dir: Directory to save figures
        bins: Number of histogram bins
        batch_size: Batch size for processing
    """
    os.makedirs(save_dir, exist_ok=True)

    # Compute log-likelihoods
    print("Computing log-likelihoods for train data...")
    logp_train = model.score_samples(train_data, batch_size).numpy()

    print("Computing log-likelihoods for validation data...")
    logp_val = model.score_samples(val_data, batch_size).numpy()

    logp_ood = None
    if ood_data is not None:
        print("Computing log-likelihoods for OOD data...")
        logp_ood = model.score_samples(ood_data, batch_size).numpy()

    if threshold is None:
        threshold = model.threshold

    # Plot train and validation
    plt.figure(figsize=(10, 6))
    sns.histplot(logp_train, bins=bins, color="blue", alpha=0.4, label="Train", kde=True)
    sns.histplot(logp_val, bins=bins, color="green", alpha=0.4, label="Validation", kde=True)
    if threshold is not None:
        plt.axvline(x=threshold, color='tab:red', linestyle='--', linewidth=2, label='Threshold')
    plt.xlabel("Log-likelihood (ELBO)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"{title} - Train/Val", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    save_path = os.path.join(save_dir, "diffusion_train_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure at {save_path}")
    plt.close()

    # Plot OOD if available
    if logp_ood is not None:
        plt.figure(figsize=(10, 6))
        sns.histplot(logp_ood, bins=bins, color="red", alpha=0.4, label="OOD", kde=True)
        sns.histplot(logp_val, bins=bins, color="green", alpha=0.3, label="Validation", kde=True)
        if threshold is not None:
            plt.axvline(x=threshold, color='tab:red', linestyle='--', linewidth=2, label='Threshold')
        plt.xlabel("Log-likelihood (ELBO)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(f"{title} - OOD vs Validation", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)

        save_path = os.path.join(save_dir, "diffusion_ood_distribution.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure at {save_path}")
        plt.close()


def load_rl_data_for_ood(data_path: str, val_ratio: float = 0.2, test_ratio: float = 0.2, max_samples: int = 10000):
    """
    Load RL dataset for OOD detection (similar to neuralODE approach).

    Args:
        data_path: Path to dataset
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        max_samples: Maximum samples to load (for speed)

    Returns:
        Tuple of (train_data, val_data, test_data, input_dim)
    """
    import sys
    import os
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common.util import load_dataset_with_validation_split

    # Create args object for load_dataset_with_validation_split
    class Args:
        pass

    args = Args()
    args.data_path = data_path
    args.task = os.path.basename(data_path).replace('.pkl', '').replace('.npz', '')

    # Load the dataset using common utility
    dataset_result = load_dataset_with_validation_split(
        args=args,
        env=None,
        val_split_ratio=val_ratio,
        test_split_ratio=test_ratio
    )

    train_dataset = dataset_result['train_data']
    val_dataset = dataset_result['val_data']
    test_dataset = dataset_result['test_data']

    # Extract next_observations + actions (like Neural ODE)
    train_next_obs = torch.FloatTensor(train_dataset['next_observations'])
    train_actions = torch.FloatTensor(train_dataset['actions'])
    train_data = torch.cat([train_next_obs, train_actions], dim=1)

    val_next_obs = torch.FloatTensor(val_dataset['next_observations'])
    val_actions = torch.FloatTensor(val_dataset['actions'])
    val_data = torch.cat([val_next_obs, val_actions], dim=1)

    test_next_obs = torch.FloatTensor(test_dataset['next_observations'])
    test_actions = torch.FloatTensor(test_dataset['actions'])
    test_data = torch.cat([test_next_obs, test_actions], dim=1)

    # Limit samples if needed
    if len(train_data) > max_samples:
        ratio = max_samples / len(train_data)
        train_data = train_data[:max_samples]
        val_data = val_data[:int(len(val_data) * ratio)]
        test_data = test_data[:int(len(test_data) * ratio)]
        print(f"Limited data to {max_samples} train samples")

    input_dim = train_data.shape[1]

    print(f"Loaded data from {data_path}")
    print(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
    print(f"Input dimension: {input_dim}")

    return train_data, val_data, test_data, input_dim


def parse_ood_args():
    """Parse command line arguments for OOD detection."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="")
    config_args, remaining_argv = config_parser.parse_known_args()

    yaml_defaults = {}
    if config_args.config:
        try:
            with open(config_args.config, "r") as f:
                config = yaml.safe_load(f)
                yaml_defaults = {k.replace("-", "_"): v for k, v in config.items()}
        except Exception as e:
            print(f"Failed to load YAML config: {e}")

    def dget(key, default):
        return yaml_defaults.get(key, default)

    parser = argparse.ArgumentParser(
        description="Diffusion Model OOD Detection",
        parents=[config_parser]
    )

    parser.add_argument('--model-dir', type=str,
                        required=('model_dir' not in yaml_defaults and 'out' not in yaml_defaults),
                        default=dget('model_dir', dget('out', None)),
                        help='Directory with checkpoint.pt and scheduler/')
    parser.add_argument('--data-path', type=str, default=dget('data_path', dget('npz', '')),
                        help='Path to dataset (NPZ or pickle)')
    parser.add_argument('--val-ratio', type=float, default=dget('val_ratio', 0.2),
                        help='Validation split ratio')
    parser.add_argument('--test-ratio', type=float, default=dget('test_ratio', 0.2),
                        help='Test split ratio')
    parser.add_argument('--anomaly-fraction', type=float, default=dget('anomaly_fraction', 0.01),
                        help='Fraction for anomaly threshold')
    parser.add_argument('--batch-size', type=int, default=dget('batch_size', 256),
                        help='Batch size')
    parser.add_argument('--max-samples', type=int, default=dget('max_samples', 10000),
                        help='Maximum samples to load')
    parser.add_argument('--num-inference-steps', type=int, default=dget('num_inference_steps', None),
                        help='Number of timesteps for ELBO (None=all 50K, e.g. 20 for fast approximation)')
    parser.add_argument('--device', type=str, default=dget('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--save-model-path', type=str, default=dget('save_model_path', ''),
                        help='Path to save OOD model')
    parser.add_argument('--plot-results', action='store_true', default=dget('plot_results', True))
    parser.add_argument('--save-dir', type=str, default=dget('save_dir', 'figures/diffusion_ood'))

    args = parser.parse_args(remaining_argv)
    return args


if __name__ == "__main__":
    args = parse_ood_args()

    print(f"Loading Diffusion model from: {args.model_dir}")

    # Load data
    print(f"\nLoading data from: {args.data_path}")
    train_data, val_data, test_data, input_dim = load_rl_data_for_ood(
        args.data_path,
        args.val_ratio,
        args.test_ratio,
        args.max_samples
    )

    # Subsample to 6000 samples like Neural ODE
    n_samples = 6000
    if len(train_data) > n_samples:
        print(f"\nSubsampling data to {n_samples} train samples for faster computation...")
        ratio = n_samples / len(train_data)
        n_val = int(len(val_data) * ratio)
        n_test = int(len(test_data) * ratio)

        train_data = train_data[:n_samples]
        val_data = val_data[:n_val]
        test_data = test_data[:n_test]
        print(f"Subsampled - Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")

    # Move data to device
    train_data = train_data.to(args.device)
    val_data = val_data.to(args.device)
    test_data = test_data.to(args.device)

    # Load Diffusion model
    ckpt_path = os.path.join(args.model_dir, "checkpoint.pt")
    sched_dir = os.path.join(args.model_dir, "scheduler")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint at {ckpt_path}")
    if not os.path.exists(sched_dir):
        raise FileNotFoundError(f"Missing scheduler directory at {sched_dir}")

    model, train_cfg = build_model_from_ckpt(ckpt_path, args.device)

    # Load scheduler (try DDIM first, fallback to DDPM)
    try:
        scheduler = DDIMScheduler.from_pretrained(sched_dir)
        print("Loaded DDIMScheduler")
    except Exception:
        try:
            scheduler = DDPMScheduler.from_pretrained(sched_dir)
            print("Loaded DDPMScheduler")
        except Exception as e:
            print(f"Warning: Could not load scheduler: {e}")
            print("Creating default DDIMScheduler")
            scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_schedule="linear",
                prediction_type="epsilon",
            )

    model.eval()
    print("Model loaded successfully")

    # Create OOD wrapper
    print(f"Using num_inference_steps: {args.num_inference_steps if args.num_inference_steps else 'all (50K)'}")
    ood_model = DiffusionOOD(model, scheduler, device=args.device, num_inference_steps=args.num_inference_steps)

    # Set threshold
    print(f"\nSetting threshold with {args.anomaly_fraction*100}% anomaly fraction...")
    ood_model.set_threshold(val_data, args.anomaly_fraction, args.batch_size)

    # Get predictions on training data
    print("\nComputing predictions on training data...")
    predictions_tr = ood_model.predict(train_data, args.batch_size)
    scores_tr = ood_model.score_samples(train_data, args.batch_size)
    scores_test_in_dist = ood_model.score_samples(test_data, args.batch_size)

    # Create OOD test data like RealNVP/Neural ODE: 10% of train data + noisy versions
    print("\nCreating OOD test data (10% train + noisy versions)...")
    small_train = train_data[predictions_tr == 1][: int(0.1 * len(train_data))].cpu().numpy()
    noisy_train = small_train + np.random.normal(0, 0.1, small_train.shape)
    ood_test_data = torch.FloatTensor(np.concatenate([small_train, noisy_train], axis=0)).to(args.device)

    print(f"OOD test data created: {len(ood_test_data)} samples (target: ~600)")

    # Evaluate on OOD test data
    print("\nTesting on OOD data...")
    predictions_test = ood_model.predict(ood_test_data, args.batch_size)
    scores_test = ood_model.score_samples(ood_test_data, args.batch_size)

    print(f"Scores test OOD: {scores_test.mean().item():.3f}")
    print(f"Scores test ID: {scores_test_in_dist.mean().item():.3f}")
    anomaly_count = (np.array(predictions_test) == -1).sum()
    print(f"Max density score: {scores_test.max().item():.3f}")
    print(f"Min density score: {scores_test.min().item():.3f}")
    print(f"OOD data anomalies detected: {anomaly_count}/{len(ood_test_data)} ({(anomaly_count/len(ood_test_data)):.1%})")

    # Evaluate OOD detection with ROC curve
    print("\nEvaluating OOD detection performance (ROC curve)...")
    n_normal = min(len(test_data), len(ood_test_data))
    results = ood_model.evaluate_anomaly_detection(
        normal_data=test_data[:n_normal],
        anomaly_data=ood_test_data[:n_normal],
        plot=args.plot_results,
        save_path=os.path.join(args.save_dir, "diffusion_roc_curve.png"),
        batch_size=args.batch_size
    )

    print(f"\nROC Evaluation Results:")
    print(f"  ROC AUC: {results['roc_auc']:.3f}")
    if results['accuracy'] is not None:
        print(f"  Accuracy: {results['accuracy']:.3f}")
    print(f"  Normal log prob: {results['normal_log_prob_mean']:.3f} ± {results['normal_log_prob_std']:.3f}")
    print(f"  Anomaly log prob: {results['anomaly_log_prob_mean']:.3f} ± {results['anomaly_log_prob_std']:.3f}")

    # Plot likelihood distributions
    if args.plot_results:
        print("\nPlotting likelihood distributions...")
        plot_likelihood_distributions(
            model=ood_model,
            train_data=train_data,
            val_data=val_data,
            ood_data=ood_test_data,
            save_dir=args.save_dir,
            batch_size=args.batch_size
        )

    # Save model if requested
    if args.save_model_path:
        print(f"\nSaving OOD model to: {args.save_model_path}")
        ood_model.save_model(args.save_model_path, train_data)

    print("\nDiffusion OOD detection completed!")

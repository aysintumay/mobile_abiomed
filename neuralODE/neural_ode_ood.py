import argparse
import json
import math
import os
import sys
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, Subset

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

from neuralODE.neural_ode_density import (
    ContinuousNormalizingFlow,
    NPZTargetDataset,
    ODEFunc,
)
from common.buffer import ReplayBuffer
from common.util import load_dataset_with_validation_split


class NeuralODEOOD:
    """
    Neural ODE-based Out-of-Distribution (OOD) Detection wrapper.

    This class wraps a ContinuousNormalizingFlow model and provides
    OOD detection functionality similar to RealNVP.
    """

    def __init__(
        self,
        flow: ContinuousNormalizingFlow,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the Neural ODE OOD detector.

        Args:
            flow: Trained ContinuousNormalizingFlow model
            device: Device to run computations on
        """
        self.flow = flow
        self.device = device
        self.threshold = None
        self.flow.to(device)

    def score_samples(self, x: torch.Tensor, device: str = 'cuda') -> np.ndarray:
        """
        Compute log probability of data points (matches RealNVP interface).

        Args:
            x: Input tensor of shape (batch_size, dim)
            device: Device to use (ignored, uses model's device)

        Returns:
            Log probabilities as numpy array
        """
        self.flow.eval()
        # Note: Neural ODE requires gradients for divergence computation
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)

        # Enable gradients even inside no_grad context (needed for divergence computation)
        with torch.enable_grad():
            log_probs = self.flow.log_prob(x)
        return log_probs.detach().cpu().numpy()

    def set_threshold(
        self,
        val_data: torch.Tensor,
        anomaly_fraction: float = 0.01
    ):
        """
        Set threshold for anomaly detection based on validation data.

        Args:
            val_data: Validation dataset (assumed to be normal data)
            anomaly_fraction: Fraction of validation data to classify as anomalies
        """
        self.flow.eval()
        val_data = val_data.to(self.device)

        # Use score_samples which returns numpy array
        log_probs = self.score_samples(val_data)

        # Set threshold as percentile of validation log probabilities
        self.threshold = float(np.percentile(log_probs, anomaly_fraction * 100))

        print(f'Threshold set to {self.threshold:.4f} '
              f'(marking {anomaly_fraction*100:.1f}% of validation data as anomalies)')

        return self.threshold

    def predict_anomaly(self, x: torch.Tensor) -> np.ndarray:
        """
        Predict anomalies based on log probability threshold.

        Args:
            x: Input data

        Returns:
            Boolean array indicating anomalies (True = anomaly)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")

        log_probs = self.score_samples(x)
        return log_probs < self.threshold

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Predict anomalies based on threshold (matches RealNVP interface).

        Args:
            x: Test data

        Returns:
            predictions: 1 for normal, -1 for anomaly
        """
        anomalies = self.predict_anomaly(x)
        return np.where(anomalies, -1, 1)

    def evaluate_anomaly_detection(
        self,
        normal_data: torch.Tensor,
        anomaly_data: torch.Tensor,
        plot: bool = True,
        save_path: Optional[str] = None
    ) -> dict:
        """
        Evaluate anomaly detection performance.

        Args:
            normal_data: Normal test data
            anomaly_data: Anomalous test data
            plot: Whether to plot ROC curve
            save_path: Path to save the ROC curve plot

        Returns:
            Dictionary with evaluation metrics
        """
        self.flow.eval()

        # Move data to device
        normal_data = normal_data.to(self.device)
        anomaly_data = anomaly_data.to(self.device)

        # Compute log probabilities for normal data
        normal_log_probs = self.score_samples(normal_data)

        # Compute log probabilities for anomaly data
        anomaly_log_probs = self.score_samples(anomaly_data)

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
            ax.set_title('ROC Curve for OOD Detection (Neural ODE)', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"ROC curve saved to {save_path}")
            else:
                plt.savefig('figures/neural_ode_roc_curve.png', dpi=300, bbox_inches='tight')
                print("ROC curve saved to figures/neural_ode_roc_curve.png")
            plt.close(fig)

        return results

    def save_model(self, save_path: str, train_data: Optional[torch.Tensor] = None):
        """
        Save the Neural ODE OOD model and metadata (matches RealNVP interface).

        Args:
            save_path: Base path for saving (without extension)
            train_data: Optional training data to compute statistics
        """
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Save model state dict
        torch.save(self.flow.state_dict(), f"{save_path}_model.pt")

        # Compute training log probabilities if provided
        metadata = {
            'threshold': self.threshold,
            'device': str(self.device),
        }

        if train_data is not None:
            self.flow.eval()
            train_data = train_data.to(self.device)
            train_log_probs = self.score_samples(train_data)
            metadata['mean'] = float(np.mean(train_log_probs))
            metadata['std'] = float(np.std(train_log_probs))

        with open(f"{save_path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Model saved to: {save_path}_model.pt")
        print(f"Metadata saved to: {save_path}_metadata.pkl")

    @classmethod
    def load_model(
        cls,
        save_path: str,
        target_dim: Optional[int] = None,
        hidden_dims: Tuple[int, ...] = (512, 512),
        activation: str = "silu",
        time_dependent: bool = True,
        solver: str = "dopri5",
        t0: float = 0.0,
        t1: float = 1.0,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Load a saved Neural ODE OOD model.

        Args:
            save_path: Base path for loading (without extension)
            target_dim: Dimension of target data (optional, will be read from metadata if not provided)
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            time_dependent: Whether to use time-dependent ODE
            solver: ODE solver
            t0, t1: Integration time bounds
            rtol, atol: ODE solver tolerances
            device: Device to load model on

        Returns:
            Dictionary with loaded model, threshold, and statistics
        """
        import glob

        # Support multiple formats:
        # 1. New format: {save_path}/metadata.pkl + {save_path}/model.pt
        # 2. Old format: {save_path}_metadata.pkl + {save_path}_model.pt
        # 3. Standalone format: {save_path}.pt with embedded metadata (no separate metadata file)
        metadata_path_new = os.path.join(save_path, "metadata.pkl")
        metadata_path_old = f"{save_path}_metadata.pkl"
        model_path_new = os.path.join(save_path, "model.pt")
        model_path_old = f"{save_path}_model.pt"
        model_path_standalone = f"{save_path}.pt" if not save_path.endswith('.pt') else save_path

        metadata = {}
        metadata_path = None
        model_path = None

        if os.path.exists(metadata_path_new):
            metadata_path = metadata_path_new
            # Check for model.pt, fallback to latest checkpoint
            if os.path.exists(model_path_new):
                model_path = model_path_new
            else:
                # Find latest checkpoint
                ckpt_pattern = os.path.join(save_path, "checkpoint_epoch_*.pt")
                ckpts = glob.glob(ckpt_pattern)
                if ckpts:
                    # Sort by epoch number and get latest
                    ckpts.sort(key=lambda x: int(x.split('_epoch_')[-1].replace('.pt', '')))
                    model_path = ckpts[-1]
                    print(f"model.pt not found, using latest checkpoint: {model_path}")
                else:
                    raise FileNotFoundError(
                        f"Could not find model.pt or any checkpoint in {save_path}"
                    )
        elif os.path.exists(metadata_path_old):
            metadata_path = metadata_path_old
            model_path = model_path_old
        elif os.path.exists(model_path_standalone):
            # Standalone format: model.pt exists, check for metadata in same directory
            model_path = model_path_standalone
            # Check for metadata.pkl in the same directory as model.pt
            model_dir = os.path.dirname(model_path)
            metadata_in_dir = os.path.join(model_dir, "metadata.pkl")
            if os.path.exists(metadata_in_dir):
                metadata_path = metadata_in_dir
                print(f"Loading model with metadata from same directory: {model_path}")
            else:
                metadata_path = None
                print(f"Loading standalone model (metadata embedded in checkpoint): {model_path}")
        else:
            raise FileNotFoundError(
                f"Could not find metadata file at {metadata_path_new} or {metadata_path_old}, "
                f"nor standalone model at {model_path_standalone}"
            )

        # Load metadata from file if available
        if metadata_path is not None:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

        # Load checkpoint to check for embedded metadata
        ckpt = torch.load(model_path, map_location=device)

        # Handle standalone format with embedded metadata (toy experiments format)
        # This format stores: ode_func_state_dict, hidden_dims, time_dependent, solver, input_dim
        if isinstance(ckpt, dict) and 'ode_func_state_dict' in ckpt:
            # Toy experiments format - extract metadata from checkpoint
            if target_dim is None:
                target_dim = ckpt.get('input_dim', ckpt.get('target_dim'))
            hidden_dims = tuple(ckpt.get('hidden_dims', hidden_dims))
            time_dependent = ckpt.get('time_dependent', time_dependent)
            solver = ckpt.get('solver', solver)
            # activation defaults to 'silu' if not in checkpoint
            activation = ckpt.get('activation', activation)

        # Use metadata values if available, otherwise use provided arguments or checkpoint values
        hidden_dims = metadata.get('hidden_dims', hidden_dims)
        activation = metadata.get('activation', activation)
        time_dependent = metadata.get('time_dependent', time_dependent)
        solver = metadata.get('solver', solver)
        t0 = metadata.get('t0', t0)
        t1 = metadata.get('t1', t1)
        rtol = metadata.get('rtol', rtol)
        atol = metadata.get('atol', atol)
        target_dim = metadata.get('target_dim', target_dim)

        # If target_dim still not found, try to infer from environment name in path
        if target_dim is None:
            # Standard D4RL environment dimensions (obs_dim + act_dim)
            env_dims = {
                'halfcheetah': 23,  # 17 obs + 6 act
                'hopper': 14,       # 11 obs + 3 act
                'walker2d': 23,     # 17 obs + 6 act
                'walker': 23,       # alias for walker2d
            }
            save_path_lower = save_path.lower()
            for env_name, dim in env_dims.items():
                if env_name in save_path_lower:
                    target_dim = dim
                    print(f"Inferred target_dim={target_dim} from environment '{env_name}' in path")
                    break

        if target_dim is None:
            raise ValueError(
                f"target_dim not found in metadata or checkpoint, not provided as argument, "
                f"and could not be inferred from path '{save_path}'. "
                "Please provide target_dim explicitly or use a path containing the environment name "
                "(halfcheetah, hopper, walker2d)."
            )

        # Create ODE function and flow
        odefunc = ODEFunc(
            dim=target_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            time_dependent=time_dependent,
        ).to(device)

        flow = ContinuousNormalizingFlow(
            func=odefunc,
            t0=t0,
            t1=t1,
            solver=solver,
            rtol=rtol,
            atol=atol,
        ).to(device)

        # Load model state dict (handle multiple checkpoint formats)
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                flow.load_state_dict(ckpt['model_state_dict'])
            elif 'ode_func_state_dict' in ckpt:
                # Toy experiments format - load just the ODE function state
                odefunc.load_state_dict(ckpt['ode_func_state_dict'])
            else:
                # Assume the dict is the state dict itself
                flow.load_state_dict(ckpt)
        else:
            flow.load_state_dict(ckpt)
        flow.eval()

        # Create OOD wrapper
        ood_model = cls(flow, device=device)

        # Extract threshold, mean, std from checkpoint if not in metadata
        # Check checkpoint first (for embedded metadata), then fall back to metadata dict
        if isinstance(ckpt, dict):
            threshold = ckpt.get('threshold', metadata.get('threshold'))
            mean_val = ckpt.get('mean', metadata.get('mean'))
            std_val = ckpt.get('std', metadata.get('std'))
        else:
            threshold = metadata.get('threshold')
            mean_val = metadata.get('mean')
            std_val = metadata.get('std')

        ood_model.threshold = threshold

        print(f"Model loaded from: {model_path}")
        if metadata_path:
            print(f"Metadata loaded from: {metadata_path}")
        else:
            print("Metadata extracted from checkpoint (standalone format)")
        print(f"Threshold: {ood_model.threshold}")

        model_dict = {
            'model': ood_model,
            'threshold': threshold,
            'mean': mean_val,
            'std': std_val
        }

        return model_dict



def plot_likelihood_distributions(
    model: NeuralODEOOD,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    ood_data: Optional[torch.Tensor] = None,
    threshold: Optional[float] = None,
    title: str = "Likelihood Distribution (Neural ODE)",
    save_dir: str = "figures",
    bins: int = 50
):
    """
    Visualize log-likelihood distributions for train, val, and OOD data.

    Args:
        model: Neural ODE OOD model
        train_data: In-distribution training set
        val_data: Held-out validation set
        ood_data: Optional OOD dataset
        threshold: Threshold value for anomaly detection
        title: Title for the plot
        save_dir: Directory to save figures
        bins: Number of histogram bins
    """
    os.makedirs(save_dir, exist_ok=True)

    # Compute log-likelihoods
    print("Computing log-likelihoods for train data...")
    train_data = train_data.to(model.device)
    logp_train = model.score_samples(train_data)

    print("Computing log-likelihoods for validation data...")
    val_data = val_data.to(model.device)
    logp_val = model.score_samples(val_data)

    logp_ood = None
    if ood_data is not None:
        print("Computing log-likelihoods for OOD data...")
        ood_data = ood_data.to(model.device)
        logp_ood = model.score_samples(ood_data)

    if threshold is None:
        threshold = model.threshold

    # Plot train and validation
    plt.figure(figsize=(10, 6))
    sns.histplot(logp_train, bins=bins, color="blue", alpha=0.4, label="Train", kde=True)
    sns.histplot(logp_val, bins=bins, color="green", alpha=0.4, label="Validation", kde=True)
    if threshold is not None:
        plt.axvline(x=threshold, color='tab:red', linestyle='--', linewidth=2, label='Threshold')
    plt.xlabel("Log-likelihood", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"{title} - Train/Val", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    save_path = os.path.join(save_dir, "neural_ode_train_distribution.png")
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
        plt.xlabel("Log-likelihood", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(f"{title} - OOD vs Validation", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)

        save_path = os.path.join(save_dir, "neural_ode_ood_distribution.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure at {save_path}")
        plt.close()


def plot_tsne(tsne_data, preds, title, save_dir="figures"):
    """
    Plot t-SNE visualization of OOD predictions.

    Args:
        tsne_data: 2D t-SNE embeddings
        preds: Predictions (1 for ID, -1 for OOD)
        title: Plot title
        save_dir: Directory to save figure
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_data[preds == 1, 0], tsne_data[preds == 1, 1],
                color='blue', label='ID', alpha=0.5)
    plt.scatter(tsne_data[preds == -1, 0], tsne_data[preds == -1, 1],
                color='red', label='OOD', alpha=0.5)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved t-SNE plot to {save_path}")
    plt.close()


def load_rl_data_for_ood(args, env=None, val_split_ratio=0.2, test_split_ratio=0.2):
    """
    Load RL dataset and prepare next_observations + actions for OOD detection.

    Args:
        args: Arguments containing data_path, obs_shape, action_dim
        env: Environment object (for datasets that need it)
        val_split_ratio: Fraction of data for validation
        test_split_ratio: Fraction of data for testing

    Returns:
        Tuple of (train_data, val_data, test_data, input_dim)
    """
    dataset_result = load_dataset_with_validation_split(
        args=args,
        env=env,
        val_split_ratio=val_split_ratio,
        test_split_ratio=test_split_ratio
    )

    train_dataset = dataset_result['train_data']
    val_dataset = dataset_result['val_data']
    test_dataset = dataset_result['test_data']

    # Extract concatenated next_observations + actions
    train_next_obs = torch.FloatTensor(train_dataset['next_observations'])
    train_actions = torch.FloatTensor(train_dataset['actions'])
    train_data = torch.cat([train_next_obs, train_actions], dim=1)

    val_next_obs = torch.FloatTensor(val_dataset['next_observations'])
    val_actions = torch.FloatTensor(val_dataset['actions'])
    val_data = torch.cat([val_next_obs, val_actions], dim=1)

    test_next_obs = torch.FloatTensor(test_dataset['next_observations'])
    test_actions = torch.FloatTensor(test_dataset['actions'])
    test_data = torch.cat([test_next_obs, test_actions], dim=1)

    input_dim = train_data.shape[1]

    print(f"OOD input shape: {train_data.shape}")
    print(f"OOD input dimension: {input_dim}")
    print(f"Next observations dim: {train_next_obs.shape[1]}, Actions dim: {train_actions.shape[1]}")

    return train_data, val_data, test_data, input_dim


@dataclass
class OODConfig:
    """Configuration for Neural ODE OOD detection."""
    model_path: str
    npz_path: str = ""
    data_path: str = ""
    task: str = "hopper-medium-v2"
    obs_dim: Tuple[int, ...] = (11,)
    action_dim: int = 3
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    anomaly_fraction: float = 0.01
    batch_size: int = 512
    hidden_dims: Tuple[int, ...] = (512, 512)
    activation: str = "silu"
    time_dependent: bool = True
    solver: str = "dopri5"
    t0: float = 0.0
    t1: float = 1.0
    rtol: float = 1e-5
    atol: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_model_path: str = ""
    plot_results: bool = True
    save_dir: str = "figures"


def parse_ood_args() -> OODConfig:
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
        description="Neural ODE OOD Detection",
        parents=[config_parser]
    )

    parser.add_argument('--model-path', type=str,
                        required=('model_path' not in yaml_defaults),
                        default=dget('model_path', None),
                        help='Path to trained Neural ODE model')
    parser.add_argument('--npz-path', type=str, default=dget('npz_path', ''),
                        help='Path to NPZ dataset')
    parser.add_argument('--data-path', type=str, default=dget('data_path', ''),
                        help='Path to RL dataset (pickle or npz)')
    parser.add_argument('--task', type=str, default=dget('task', 'hopper-medium-v2'),
                        help='Task name')
    parser.add_argument('--obs-dim', type=int, nargs='+', default=dget('obs_dim', [11]),
                        help='Observation dimension')
    parser.add_argument('--action-dim', type=int, default=dget('action_dim', 3),
                        help='Action dimension')
    parser.add_argument('--val-ratio', type=float, default=dget('val_ratio', 0.2),
                        help='Validation split ratio')
    parser.add_argument('--test-ratio', type=float, default=dget('test_ratio', 0.2),
                        help='Test split ratio')
    parser.add_argument('--anomaly-fraction', type=float, default=dget('anomaly_fraction', 0.01),
                        help='Fraction for anomaly threshold')
    parser.add_argument('--batch-size', type=int, default=dget('batch_size', 512),
                        help='Batch size')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=dget('hidden_dims', [512, 512]),
                        help='Hidden dimensions')
    parser.add_argument('--activation', type=str, default=dget('activation', 'silu'),
                        choices=['silu', 'tanh'])
    parser.add_argument('--solver', type=str, default=dget('solver', 'dopri5'))
    parser.add_argument('--device', type=str, default=dget('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--save-model-path', type=str, default=dget('save_model_path', ''),
                        help='Path to save OOD model')
    parser.add_argument('--plot-results', action='store_true', default=dget('plot_results', True))
    parser.add_argument('--save-dir', type=str, default=dget('save_dir', 'figures'))

    args = parser.parse_args(remaining_argv)

    hidden_dims = tuple(args.hidden_dims) if isinstance(args.hidden_dims, list) else tuple([args.hidden_dims])
    obs_dim = tuple(args.obs_dim) if isinstance(args.obs_dim, list) else tuple([args.obs_dim])

    return OODConfig(
        model_path=args.model_path,
        npz_path=args.npz_path,
        data_path=args.data_path,
        task=args.task,
        obs_dim=obs_dim,
        action_dim=args.action_dim,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        anomaly_fraction=args.anomaly_fraction,
        batch_size=args.batch_size,
        hidden_dims=hidden_dims,
        activation=args.activation,
        solver=args.solver,
        device=args.device,
        save_model_path=args.save_model_path,
        plot_results=args.plot_results,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    cfg = parse_ood_args()

    print(f"Loading Neural ODE model from: {cfg.model_path}")

    # Load data
    if cfg.npz_path:
        # Load from NPZ file
        dataset = NPZTargetDataset(cfg.npz_path)
        # Split into train/val/test
        n = len(dataset)
        n_train = int(n * (1 - cfg.val_ratio - cfg.test_ratio))
        n_val = int(n * cfg.val_ratio)

        train_data = dataset.target[:n_train]
        val_data = dataset.target[n_train:n_train+n_val]
        test_data = dataset.target[n_train+n_val:]
        target_dim = dataset.target_dim
    else:
        # Load RL data
        import gym

        class Args:
            pass

        args = Args()
        args.data_path = cfg.data_path
        args.task = cfg.task
        args.obs_dim = cfg.obs_dim
        args.obs_shape = cfg.obs_dim
        args.action_dim = cfg.action_dim

        env = None
        if not cfg.data_path:
            try:
                env = gym.make(cfg.task)
            except Exception as e:
                print(f"Warning: Could not create environment: {e}")

        train_data, val_data, test_data, target_dim = load_rl_data_for_ood(
            args, env, cfg.val_ratio, cfg.test_ratio
        )

    print(f"Data loaded - Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")

    # Subsample to speed up computation (like RealNVP)
    # Use first 6000 samples for train, and proportionally reduce val/test
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

    # Move data to device early
    train_data = train_data.to(cfg.device)
    val_data = val_data.to(cfg.device)
    test_data = test_data.to(cfg.device)

    # Load Neural ODE model
    odefunc = ODEFunc(
        dim=target_dim,
        hidden_dims=cfg.hidden_dims,
        activation=cfg.activation,
        time_dependent=cfg.time_dependent
    ).to(cfg.device)

    flow = ContinuousNormalizingFlow(
        func=odefunc,
        t0=cfg.t0,
        t1=cfg.t1,
        solver=cfg.solver,
        rtol=cfg.rtol,
        atol=cfg.atol
    ).to(cfg.device)

    # Load checkpoint
    checkpoint = torch.load(cfg.model_path, map_location=cfg.device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        flow.load_state_dict(checkpoint['model_state_dict'])
    else:
        flow.load_state_dict(checkpoint)

    flow.eval()
    print("Model loaded successfully")

    # Create OOD wrapper
    ood_model = NeuralODEOOD(flow, device=cfg.device)

    # Set threshold (now on subsampled data)
    print(f"\nSetting threshold with {cfg.anomaly_fraction*100}% anomaly fraction...")
    ood_model.set_threshold(val_data, cfg.anomaly_fraction)

    # Get predictions on training data
    print("\nComputing predictions on training data...")
    predictions_tr = ood_model.predict(train_data)
    scores_tr = ood_model.score_samples(train_data)
    scores_test_in_dist = ood_model.score_samples(test_data)

    # Create OOD test data like RealNVP: 10% of train data + noisy versions
    print("\nCreating OOD test data (10% train + noisy versions)...")
    small_train = train_data[predictions_tr == 1][: int(0.1 * len(train_data))].cpu().numpy()
    noisy_train = small_train + np.random.normal(0, 0.1, small_train.shape)
    ood_test_data = torch.FloatTensor(np.concatenate([small_train, noisy_train], axis=0)).to(cfg.device)

    print(f"OOD test data created: {len(ood_test_data)} samples (target: ~6000)")

    # Evaluate on OOD test data
    print("\nTesting on OOD data...")
    predictions_test = ood_model.predict(ood_test_data)
    scores_test = ood_model.score_samples(ood_test_data)

    print(f"Scores test OOD: {scores_test.mean():.3f}")
    print(f"Scores test ID: {scores_test_in_dist.mean():.3f}")
    anomaly_count = (np.array(predictions_test) == -1).sum()
    print(f"Max density score: {scores_test.max():.3f}")
    print(f"Min density score: {scores_test.min():.3f}")
    print(f"OOD data anomalies detected: {anomaly_count}/{len(ood_test_data)} ({(anomaly_count/len(ood_test_data)):.1%})")

    # Evaluate OOD detection with ROC curve using the real OOD test data
    # Use half of test_data as "normal" and ood_test_data as "anomaly"
    print("\nEvaluating OOD detection performance (ROC curve)...")
    n_normal = min(len(test_data), len(ood_test_data))
    results = ood_model.evaluate_anomaly_detection(
        normal_data=test_data[:n_normal],
        anomaly_data=ood_test_data[:n_normal],
        plot=cfg.plot_results,
        save_path=os.path.join(cfg.save_dir, "neural_ode_roc_curve.png")
    )

    print(f"\nROC Evaluation Results:")
    print(f"  ROC AUC: {results['roc_auc']:.3f}")
    if results['accuracy'] is not None:
        print(f"  Accuracy: {results['accuracy']:.3f}")
    print(f"  Normal log prob: {results['normal_log_prob_mean']:.3f} ± {results['normal_log_prob_std']:.3f}")
    print(f"  Anomaly log prob: {results['anomaly_log_prob_mean']:.3f} ± {results['anomaly_log_prob_std']:.3f}")

    # Plot likelihood distributions
    if cfg.plot_results:
        print("\nPlotting likelihood distributions...")
        plot_likelihood_distributions(
            model=ood_model,
            train_data=train_data,
            val_data=val_data,
            ood_data=ood_test_data,
            save_dir=cfg.save_dir
        )

    # Save model if requested
    if cfg.save_model_path:
        print(f"\nSaving OOD model to: {cfg.save_model_path}")
        ood_model.save_model(cfg.save_model_path, train_data)

    print("\nNeural ODE OOD detection completed!")

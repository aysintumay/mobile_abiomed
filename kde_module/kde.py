import os
import sys
import time
import pickle
import argparse

import numpy as np
import torch
import faiss
import yaml
from sklearn.preprocessing import StandardScaler
import gym
import d4rl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.buffer import ReplayBuffer as ReplayBufferAbiomed
from common.util import create_synthetic_data
from helpers.plotter import (
    plot_likelihood_distributions,
    plot_tsne,
    evaluate_anomaly_detection,
    evaluate_and_plot_density
)


def get_d4rl_data(args, val=None):
    """
    Load environment and dataset for D4RL tasks.

    Args:
        args: Arguments containing task name and optional data_path
        val: Whether to return validation data

    Returns:
        env, dataset, [dataset_val]: Environment and datasets
    """
    env = gym.make(args.task)

    # Check if a custom data_path is provided (e.g., for sparse datasets)
    if hasattr(args, 'data_path') and args.data_path is not None:
        print(f"Loading D4RL dataset from custom path: {args.data_path}")
        try:
            with open(args.data_path, "rb") as f:
                dataset = pickle.load(f)
            print(f"Successfully loaded pickle file: {args.data_path}")
            print(f"  Dataset keys: {dataset.keys()}")
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            print("Falling back to standard D4RL dataset...")
            dataset = d4rl.qlearning_dataset(env)
    else:
        # Standard D4RL dataset loading
        dataset = d4rl.qlearning_dataset(env)

    if val:
        return env, dataset, None
    else:
        return env, dataset


def get_env_data(args, val=None):
    """
    Load environment and dataset based on environment type.

    Args:
        args: Arguments containing environment configuration
        val: Whether to return validation data

    Returns:
        env, dataset, [dataset_val]: Environment and datasets
    """
    if args.env == "abiomed":
        return get_abiomed_data(args, val)
    else:
        return get_d4rl_data(args, val)


def get_abiomed_data(args, val=None):
    """
    Load environment and dataset for Abiomed.

    Args:
        args: Arguments containing environment configuration
        val: Whether to return validation data

    Returns:
        env, dataset, [dataset_val]: Environment and datasets
    """
    from abiomed_env.rl_env import AbiomedRLEnvFactory

    env = AbiomedRLEnvFactory.create_env(
        model_name=args.model_name,
        model_path=args.model_path,
        data_path=args.data_path_wm,
        max_steps=args.max_steps,
        gamma1=args.gamma1,
        gamma2=args.gamma2,
        gamma3=args.gamma3,
        action_space_type=args.action_space_type,
        reward_type="smooth",
        normalize_rewards=True,
        noise_rate=args.noise_rate,
        noise_scale=args.noise_scale,
        seed=args.seed,
        device=f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu",
    )

    if args.data_path is None:
        dataset1 = env.world_model.data_train
        dataset2 = env.world_model.data_val
        dataset3 = env.world_model.data_test
        dataset = [dataset1, dataset2, dataset3]
    else:
        try:
            with open(args.data_path, "rb") as f:
                dataset = pickle.load(f)
            print("Opened pickle file for synthetic dataset")
        except Exception:
            dataset = np.load(args.data_path)
            dataset = {k: dataset[k] for k in dataset.files}
            print("Opened npz file for synthetic dataset")

    if val:
        print("Validation data is supported for Abiomed dataset")
        dataset_val = env.world_model.data_test
        return env, dataset, dataset_val
    else:
        print("No validation data for Abiomed dataset")
        return env, dataset


class PercentileThresholdKDE:
    """
    KDE-based anomaly detection using percentile thresholds
    """

    def __init__(
        self,
        bandwidth=1.0,
        n_neighbors=100,
        use_gpu=True,
        normalize=True,
        percentile=5.0,
        pca=None,
        devid=0,
    ):
        self.bandwidth = bandwidth
        self.n_neighbors = n_neighbors
        # Check if GPU is available and faiss has GPU support
        self.use_gpu = use_gpu and hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0
        self.normalize = normalize
        self.percentile = percentile 

        self.index = None
        self.training_data = None
        self.scaler = None
        self.threshold = None
        self.is_fitted = False
        self.pca = pca
        self.devid = devid

    def fit(self, X, X_val, verbose=True):
        """
        Fit the model and compute percentile-based threshold.

        Args:
            X: Training data (n_samples, n_features)
            X_val: Validation data for threshold computation
            verbose: Print fitting information
        """
        start_time = time.time()

        if torch.is_tensor(X):
            X = X.cpu().numpy()

        X = X.astype(np.float32)

        # Normalize data if requested
        if self.normalize:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        # Fit PCA and transform data if enabled
        if (X.shape[1] > 30) and (self.pca):
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=7)
            self.pca.fit(X)
            X = self.pca.transform(X)
            if verbose:
                print(f"Reduced to {X.shape[1]} features using PCA")

        self.training_data = X.copy()
        n_samples, n_features = X.shape

        if verbose:
            print(f"Training on {n_samples} samples with {n_features} features")
            print(f"Percentile threshold: {self.percentile}%")
            print(f"Bandwidth: {self.bandwidth}, K-neighbors: {self.n_neighbors}")

        # Build FAISS index
        # Use IVF for GPU with large datasets to avoid CUBLAS matrix size issues
        use_ivf = (n_samples >= 1000000) or (self.use_gpu and n_samples > 50000)

        if use_ivf:
            nlist = min(int(np.sqrt(n_samples)), 4096)
            quantizer = faiss.IndexFlatL2(n_features)
            self.index = faiss.IndexIVFFlat(quantizer, n_features, nlist)
            if hasattr(self.index, "train"):
                self.index.train(X)
            if verbose:
                print(f"Using IndexIVFFlat with {nlist} clusters for efficient GPU search")
        else:
            self.index = faiss.IndexFlatL2(n_features)
            if verbose:
                print(f"Using IndexFlatL2 for exact search")

        # Move to GPU if available and GPU functions exist
        if self.use_gpu and hasattr(faiss, 'index_cpu_to_gpu'):
            try:
                gpu_resources = faiss.StandardGpuResources()
                # Set temp memory to avoid CUBLAS issues
                gpu_resources.setTempMemory(256 * 1024 * 1024)  # 256MB temp memory
                self.index = faiss.index_cpu_to_gpu(
                    gpu_resources, self.devid, self.index
                )
                if verbose:
                    print("Using GPU acceleration")
            except Exception as e:
                if verbose:
                    print(f"GPU failed, using CPU: {e}")
                self.use_gpu = False
        else:
            if self.use_gpu and verbose:
                print("GPU functions not available (using faiss-cpu), training on CPU")
            self.use_gpu = False

        self.index.add(X)

        # Configure IVF search parameters for better recall
        if use_ivf:
            # nprobe controls how many clusters to search (higher = more accurate but slower)
            nprobe = min(max(int(nlist * 0.1), 10), nlist)  # Search 10% of clusters, min 10
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = nprobe
                if verbose:
                    print(f"Set nprobe={nprobe} for IVF search quality")

        # Compute density scores on validation data to find threshold
        if verbose:
            print("Computing density scores for threshold...")

        density_scores = self._score_samples_internal(X_val)

        # Find percentile threshold (lower density = higher anomaly score)
        self.threshold = np.percentile(density_scores, self.percentile)

        self.is_fitted = True
        fit_time = time.time() - start_time

        if verbose:
            print(f"Fitting completed in {fit_time:.2f} seconds")
            print(f"Threshold (log-density): {self.threshold:.4f}")
            print(f"Expected anomaly rate: {self.percentile}%")

        return self

    def _score_samples_internal(self, X, batch_size=5000):
        """Internal method to compute density scores with batching for large queries."""
        k = min(self.n_neighbors, self.training_data.shape[0])

        # Process in batches if dataset is large or if using GPU with large training data
        n_samples = X.shape[0]
        # Use smaller batches for GPU when training data is large to avoid CUBLAS errors
        if self.use_gpu:
            if self.training_data.shape[0] > 100000:
                # Very large datasets need tiny batches to avoid CUBLAS matrix size limits
                batch_size = min(batch_size, 100)
            elif self.training_data.shape[0] > 50000:
                batch_size = min(batch_size, 200)

        if n_samples > batch_size:
            log_densities = []
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                X_batch = X[i:batch_end]
                distances, _ = self.index.search(X_batch, k)

                # Gaussian kernel
                kernel_values = np.exp(-0.5 * distances / (self.bandwidth**2))
                density = np.mean(kernel_values, axis=1)
                log_density = np.log(density + 1e-10)
                log_densities.append(log_density)

            return np.concatenate(log_densities)
        else:
            distances, _ = self.index.search(X, k)

            # Gaussian kernel
            kernel_values = np.exp(-0.5 * distances / (self.bandwidth**2))
            density = np.mean(kernel_values, axis=1)
            log_density = np.log(density + 1e-10)

            return log_density

    def score_samples(self, X, device='cuda'):
        """
        Compute log-density estimates for new data

        Args:
            X: Test data (n_samples, n_features)

        Returns:
            log_density: Log-density estimates
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if torch.is_tensor(X):
            X = X.cpu().numpy()

        X = X.astype(np.float32)

        if self.normalize and self.scaler is not None:
            X = self.scaler.transform(X)
        if self.pca:
            X = self.pca.transform(X)

        return self._score_samples_internal(X)

    def predict(self, X):
        """
        Predict anomalies based on threshold

        Args:
            X: Test data

        Returns:
            predictions: 1 for normal, -1 for anomaly
        """
        scores = self.score_samples(X)
        return np.where(scores >= self.threshold, 1, -1)

    def decision_function(self, X):
        """
        Anomaly score (higher = more anomalous)

        Args:
            X: Test data

        Returns:
            anomaly_scores: Higher values indicate more anomalous
        """
        density_scores = self.score_samples(X)
        # Convert to anomaly scores (negative log-density relative to threshold)
        return self.threshold - density_scores

    def get_threshold_stats(self):
        """Get statistics about the threshold"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return {
            "threshold": self.threshold,
            "percentile": self.percentile,
            "bandwidth": self.bandwidth,
            "n_neighbors": self.n_neighbors,
            "n_training_samples": self.training_data.shape[0],
        }

    def save_model(self, base_path):
        """Save FAISS index and metadata separately."""
        # Transfer from GPU to CPU if needed
        if self.use_gpu and hasattr(faiss, 'index_gpu_to_cpu'):
            print("Transferring from GPU to CPU for saving...")
            index_cpu = faiss.index_gpu_to_cpu(self.index)
        else:
            # Already on CPU or using CPU-only faiss
            index_cpu = self.index

        faiss.write_index(index_cpu, f"{base_path}.faiss")

        # Save metadata
        metadata = {
            "threshold": self.threshold,
            "bandwidth": self.bandwidth,
            "n_neighbors": self.n_neighbors,
            "percentile": self.percentile,
            "normalize": self.normalize,
            "scaler": self.scaler,
            "training_data": self.training_data,
            "is_fitted": self.is_fitted,
            "model_params": self.get_threshold_stats(),
            "pca": self.pca if hasattr(self, "pca") else None,
        }

        with open(f"{base_path}_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        print(f"Model saved to {base_path}.faiss and {base_path}_metadata.pkl")

    @classmethod
    def load_model(cls, load_path, use_gpu=True, devid=0):
        """
        Load a saved model.

        Args:
            load_path: Base path for loading (without extension)
            use_gpu: Whether to move index to GPU after loading
            devid: GPU device ID

        Returns:
            dict: Dictionary containing model, index, threshold, and scaler
        """
        # Load FAISS index
        index = faiss.read_index(f"{load_path}.faiss")

        # Load metadata
        with open(f"{load_path}_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        # Create model instance
        model = cls(
            bandwidth=metadata["bandwidth"],
            n_neighbors=metadata["n_neighbors"],
            use_gpu=use_gpu,
            normalize=metadata["normalize"],
            percentile=metadata["percentile"],
            devid=devid,
        )

        # Restore state
        model.index = index
        model.threshold = metadata["threshold"]
        model.scaler = metadata["scaler"]
        model.training_data = metadata["training_data"]
        model.is_fitted = metadata["is_fitted"]
        model.pca = metadata.get("pca", None)

        # Move to GPU if requested
        if use_gpu and hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
            try:
                gpu_resources = faiss.StandardGpuResources()
                # Set temp memory to avoid CUBLAS issues
                gpu_resources.setTempMemory(256 * 1024 * 1024)  # 256MB temp memory
                model.index = faiss.index_cpu_to_gpu(gpu_resources, devid, model.index)
                model.use_gpu = True
                print(f"Model loaded and moved to GPU {devid}")
            except Exception as e:
                print(f"Could not move to GPU: {e}, using CPU")
                model.use_gpu = False
        else:
            model.use_gpu = False
            if use_gpu:
                print("GPU not available, using CPU")

        model_dict = {
            "model": model,
            "model_index": model.index,
            "thr": model.threshold,
            "scaler": metadata["scaler"],
        }
        print(f"Model loaded from {load_path}")
        return model_dict


def load_data(data_path, test_size, validation_size, args=None):
    """
    Load and split data for training/validation/testing.

    Args:
        data_path: Path to data file
        test_size: Fraction of data to use for testing
        validation_size: Fraction of data to use for validation
        args: Additional arguments for environment setup

    Returns:
        dict: Dictionary containing train/val/test splits and metadata
    """
    print(f"Loading data from {data_path}")
    if args.env == "abiomed":
        env, data = get_env_data(args)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        replay_buffer = ReplayBufferAbiomed(state_dim, action_dim)
        replay_buffer.convert_abiomed(data, env)
        X = np.concatenate([replay_buffer.next_state, replay_buffer.action], axis=1)
    elif hasattr(args, 'task') and args.task:
        # D4RL dataset loading
        env, data = get_d4rl_data(args)
        X = np.concatenate([data["next_observations"], data["actions"]], axis=1)
        print(f"Loaded D4RL dataset: {args.task}")
        print(f"  next_observations shape: {data['next_observations'].shape}")
        print(f"  Actions shape: {data['actions'].shape}")
    else:
        # Load from file path
        try:
            with open(data_path, "rb") as f:
                data = pickle.load(f)
            print("Loaded pickle file")
        except Exception:
            data = np.load(data_path)
            data = {k: data[k] for k in data.files}
            print("Loaded npz file")

        X = np.concatenate([data["next_observations"], data["actions"]], axis=1)

    n_samples = len(X)

    if args.temporal_split:
        # Temporal split (no shuffle)
        test_end = int(n_samples * (1 - test_size))
        val_end = int(test_end * (1 - validation_size))

        train_idx = np.arange(0, val_end)
        val_idx = np.arange(val_end, test_end)
        test_idx = np.arange(test_end, n_samples)
        X_train = X[train_idx]
        X_val = X[val_idx] if len(val_idx) > 0 else None
        X_test = X[test_idx]
    else:
        # Random train/test split
        np.random.seed(42)
        total_size = n_samples
        val_test_size = int(total_size * (validation_size + test_size))
        val_size = int(total_size * validation_size)
        train_size = total_size - val_test_size

        # Create random indices for splitting
        indices = np.random.permutation(total_size)
        train_indices = indices[:train_size]
        test_indices = indices[train_size + val_size:]
        val_indices = indices[train_size:val_size + train_size]

        X_train = X[train_indices]
        X_val = X[val_indices] if len(val_indices) > 0 else None
        X_test = X[test_indices]

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "split_info": {
            "total_samples": n_samples,
            "train_samples": len(X_train),
            "val_samples": len(X_val) if X_val is not None else 0,
            "test_samples": len(X_test),
        },
    }


def find_optimal_percentile(
    X_train,
    X_val=None,
    percentiles=None,
    bandwidth=1.0,
    n_neighbors=100,
    use_gpu=True,
    metric="density_range",
):
    """
    Find optimal percentile threshold using unsupervised criteria (no labels needed)

    Args:
        X_train: Training data
        X_val: Validation data (if None, uses part of training data)
        percentiles: List of percentiles to test
        bandwidth: KDE bandwidth
        n_neighbors: Number of neighbors for FAISS
        use_gpu: Use GPU acceleration
        metric: 'density_range', 'stability', or 'separation'
    """
    if percentiles is None:
        percentiles = [1, 2, 3, 5, 7, 10, 15, 20]

    print(f"\n=== Finding Optimal Percentile (Unsupervised) ===")
    print(f"Testing percentiles: {percentiles}")
    print(f"Optimization metric: {metric}")

    if X_val is None:
        # Use 20% of training data for validation
        n_val = int(0.2 * len(X_train))
        indices = np.random.permutation(len(X_train))
        X_train_subset = X_train[indices[n_val:]]
        X_val = X_train[indices[:n_val]]
        print(f"Created validation set: {len(X_val)} samples")
    else:
        X_train_subset = X_train

    best_score = -np.inf
    best_percentile = percentiles[0]
    results = []

    for p in percentiles:
        print(f"Testing percentile: {p}%")

        model = PercentileThresholdKDE(
            bandwidth=bandwidth, n_neighbors=n_neighbors, use_gpu=use_gpu, percentile=p
        )

        model.fit(X_train_subset, X_val, verbose=False)

        # Get density scores for validation data
        val_scores = model.score_samples(X_val)
        val_predictions = model.predict(X_val)

        # Compute unsupervised quality metrics
        if metric == "density_range":
            # Maximize the range of density scores (better separation)
            score = np.max(val_scores) - np.min(val_scores)

        elif metric == "stability":
            # Minimize variance in "normal" region (more stable threshold)
            normal_scores = val_scores[val_predictions == 1]
            if len(normal_scores) > 0:
                score = -np.var(normal_scores)  # Negative because we want low variance
            else:
                score = -np.inf

        elif metric == "separation":
            # Maximize separation between normal and anomalous regions
            normal_scores = val_scores[val_predictions == 1]
            anomaly_scores = val_scores[val_predictions == -1]

            if len(normal_scores) > 0 and len(anomaly_scores) > 0:
                normal_mean = np.mean(normal_scores)
                anomaly_mean = np.mean(anomaly_scores)
                pooled_std = np.sqrt(
                    (np.var(normal_scores) + np.var(anomaly_scores)) / 2
                )

                # Cohen's d (effect size)
                score = abs(normal_mean - anomaly_mean) / (pooled_std + 1e-10)
            else:
                score = -np.inf

        anomaly_rate = (val_predictions == -1).mean()

        results.append(
            {
                "percentile": p,
                "score": score,
                "anomaly_rate": anomaly_rate,
                "threshold": model.threshold,
                "val_score_mean": np.mean(val_scores),
                "val_score_std": np.std(val_scores),
            }
        )

        print(
            f"  Score: {score:.4f}, Anomaly rate: {anomaly_rate:.1%}, "
            f"Threshold: {model.threshold:.4f}"
        )

        if score > best_score:
            best_score = score
            best_percentile = p
    #compute negative log likelihood
    loss = -val_scores.mean()
    print(f"  Negative log likelihood: {loss:.4f}")
    print(f"\nBest percentile: {best_percentile}% (Score: {best_score:.4f})")
    return best_percentile, results


def main():
    print("Running", __file__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kde/walker2d_medium_expert.yaml")
    args, remaining_argv = parser.parse_known_args()

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    # Data loading arguments
    parser.add_argument('--data_path', type=str, default=None, help='Path to data file')
    parser.add_argument('--task', type=str, default='', help='D4RL task name (e.g., hopper-medium-v2)')

    # KDE model hyperparameters
    parser.add_argument('--bandwidth', type=float, default=1.0, help='KDE bandwidth')
    parser.add_argument('--k_neighbors', type=int, default=100, help='Number of neighbors')
    parser.add_argument('--percentile', type=float, default=5.0, help='Percentile threshold')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')

    # Data splitting arguments
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Test set fraction"
    )
    parser.add_argument(
        "--val_size", type=float, default=0.2, help="Validation set fraction"
    )
    parser.add_argument(
        "--temporal_split",
        action="store_true",
        help="Use temporal split (no shuffle) for time series",
    )
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")

    # Model arguments
    # parser.add_argument(
    #     "--percentile", type=float, default=5.0, help="Percentile threshold"
    # )
    # parser.add_argument('--bandwidth', type=float, default=1.0, help='KDE bandwidth')
    # parser.add_argument('--k_neighbors', type=int, default=100, help='Number of neighbors')
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU")
    parser.add_argument(
        "--no_normalize", action="store_true", help="Disable data normalization"
    )

    # Optimization arguments
    parser.add_argument(
        "--optimize_percentile", action="store_true", help="Find optimal percentile"
    )
    parser.add_argument(
        "--optimization_metric",
        choices=["density_range", "stability", "separation"],
        default="density_range",
        help="Metric for percentile optimization",
    )
    parser.add_argument("--pca", action="store_true")

    # Output arguments
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    # parser.add_argument('--save_model', type=str, default="trained_kde", help='Save model path')
    # parser.add_argument('--save_results', type=str, help='Save results to file')
    parser.add_argument(
        "--devid", type=int, default=0, help="GPU device ID (if using GPU)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/abiomed/models/kde",
        help="Path to save model and results",
    )

    # Synthetic data fallback (if no data_path provided)
    parser.add_argument(
        "--n_samples", type=int, default=10000, help="Number of synthetic samples"
    )
    parser.add_argument(
        "--n_features", type=int, default=2, help="Number of synthetic features"
    )
    parser.add_argument(
        "--anomaly_rate", type=float, default=0.05, help="Synthetic anomaly rate"
    )

    # noisy dataset parameters
    parser.add_argument(
        "--action", action="store_true", help="Create dataset with noisy actions"
    )
    parser.add_argument(
        "--transition",
        action="store_true",
        help="Create dataset with noisy transitions",
    )
    parser.add_argument("--env", type=str, default="")

    parser.add_argument("--fig_save_path", type=str, default="figures", help="Path to save figures")

    parser.set_defaults(**config)
    args = parser.parse_args(remaining_argv)

    print("=== Percentile-Based KDE Anomaly Detection ===")

    # Load data
    print(f"\nLoading data from: {args.data_path}")

    print(f"\nSplitting data...")
    data_splits = load_data(
        args.data_path,
        test_size=args.test_size,
        validation_size=args.val_size,
        args=args,
    )

    X_train = data_splits["X_train"]
    X_val = data_splits["X_val"]
    X_test = data_splits["X_test"]

    # No true labels available
    y_test = None

    print(f"\nFinal data splits:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape if X_val is not None else 'None'}")
    print(f"  Test: {X_test.shape}")

    # Find optimal percentile if requested
    if args.optimize_percentile:
        print(f"\nOptimizing percentile using '{args.optimization_metric}' metric...")
        optimal_percentile, search_results = find_optimal_percentile(
            X_train,
            X_val,
            bandwidth=args.bandwidth,
            n_neighbors=args.k_neighbors,
            use_gpu=not args.no_gpu,
            metric=args.optimization_metric,
        )
        percentile_to_use = optimal_percentile
    else:
        percentile_to_use = args.percentile
    #calculate negative log likelihood on validation set

    # Train final model
    print(f"\nTraining final model with {percentile_to_use}% threshold...")
    model = PercentileThresholdKDE(
        bandwidth=args.bandwidth,
        n_neighbors=args.k_neighbors,
        use_gpu=not args.no_gpu,
        normalize=not args.no_normalize,
        percentile=percentile_to_use,
        pca=args.pca,
        devid=args.devid,
    )

    start_time = time.time()
    model.fit(X_train, X_val)
    fit_time = time.time() - start_time
    # model.load_model(args.model_save_path, use_gpu=not args.no_gpu, devid=args.devid)
    # print(f"Training completed in {fit_time:.2f} seconds")
    print("Creating synthetic data...")
    # normal_data, anomaly_data = create_synthetic_data(
    #     n_samples=X_test.shape[0],
    #     dim=X_test.shape[1],
    #     anomaly_type='outlier'
    # )

    # evaluate_and_plot_density(model, X_val, X_train, X_test)

    # Print train and test scores
    # print(f"\n=== Density Scores ===")
    # train_scores = model.score_samples(X_train)
    # val_scores = model.score_samples(X_val)
    # test_scores = model.score_samples(X_test)

    # print(f"Train scores - Mean: {train_scores.mean():.4f}, Std: {train_scores.std():.4f}, Min: {train_scores.min():.4f}, Max: {train_scores.max():.4f}")
    # print(f"Validation scores - Mean: {val_scores.mean():.4f}, Std: {val_scores.std():.4f}, Min: {val_scores.min():.4f}, Max: {val_scores.max():.4f}")
    # print(f"Test scores - Mean: {test_scores.mean():.4f}, Std: {test_scores.std():.4f}, Min: {test_scores.min():.4f}, Max: {test_scores.max():.4f}")

    # # Print anomaly test scores (synthetic OOD data)
    # anomaly_scores = model.score_samples(anomaly_data)
    # print(f"\n=== Anomaly Test Scores (Synthetic OOD) ===")
    # print(f"Anomaly scores - Mean: {anomaly_scores.mean():.4f}, Std: {anomaly_scores.std():.4f}, Min: {anomaly_scores.min():.4f}, Max: {anomaly_scores.max():.4f}")

    # # Print detection rates
    # train_preds = model.predict(X_train)
    # val_preds = model.predict(X_val)
    # test_preds = model.predict(X_test)
    # anomaly_preds = model.predict(anomaly_data)

    # print(f"\n=== Detection Rates ===")
    # print(f"Train anomaly rate: {(train_preds == -1).mean():.2%}")
    # print(f"Validation anomaly rate: {(val_preds == -1).mean():.2%}")
    # print(f"Test anomaly rate: {(test_preds == -1).mean():.2%}")
    # print(f"Synthetic OOD anomaly rate: {(anomaly_preds == -1).mean():.2%}")

    # plot_likelihood_distributions(
    #                     model,
    #                     X_train,
    #                     X_val,
    #                     ood_data=X_test,
    #                     thr = model.threshold,
    #                     title="Likelihood Distribution",
    #                     savepath=args.fig_save_path,
    #                     bins=50
    #                 )

    # Use command-line save_path if provided, otherwise fall back to config model_save_path
    save_path = None
    if hasattr(args, 'save_path') and args.save_path != '/abiomed/models/kde':
        # Command line argument was explicitly provided (not default)
        save_path = args.save_path
    elif config.get('model_save_path', False):
        # Use config file value
        save_path = config.get('model_save_path', 'saved_models/vae')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save_model(save_path)
        print(f"Model saved to: {save_path}.faiss and {save_path}_metadata.pkl")


    # Print threshold statistics
    stats = model.get_threshold_stats()
    print(f"\n=== Model Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print(f"\n=== Usage Example ===")
    print("# To use the trained model on new data:")
    print("scores = model.score_samples(new_data)")
    print("predictions = model.predict(new_data)  # 1=normal, -1=anomaly")
    print("anomaly_scores = model.decision_function(new_data)  # higher=more anomalous")
    print(f"# Threshold: {model.threshold:.4f}")


if __name__ == "__main__":
    main()

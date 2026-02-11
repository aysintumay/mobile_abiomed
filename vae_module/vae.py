import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import argparse
import yaml
import os
import sys
import pickle
import gym
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.buffer import ReplayBuffer
from common.util import load_dataset_with_validation_split, create_synthetic_data
from helpers.plotter import plot_training_curves
from helpers.plotter import plot_likelihood_distributions
from helpers.plotter import plot_likelihood_distributions


class Encoder(nn.Module):
    """Encoder network for VAE."""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Mean and log variance layers
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """Decoder network for VAE."""

    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()

        layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class VAE(nn.Module):
    """Variational Autoencoder for density estimation and anomaly detection."""

    def __init__(
        self,
        input_dim: int = 2,
        latent_dim: int = 16,
        hidden_dims: List[int] = [256, 256],
        device: str = 'cpu'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.hidden_dims = hidden_dims
        self.device = device

        # Create encoder and decoder
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)

        # Threshold for anomaly detection
        self.threshold = None

    def _apply(self, fn):
        """Override _apply to update self.device when model is moved."""
        super()._apply(fn)
        # Update self.device to match the actual device of parameters
        if len(list(self.parameters())) > 0:
            self.device = next(self.parameters()).device
        return self

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * sigma."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Reconstruction, mean, and log variance
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VAE loss function = Reconstruction loss + KL divergence.

        Args:
            recon_x: Reconstructed data
            x: Original data
            mu: Mean from encoder
            logvar: Log variance from encoder
            beta: Weight for KL divergence term

        Returns:
            Total loss, reconstruction loss, KL divergence
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        # Total loss
        loss = recon_loss + beta * kl_div

        return loss, recon_loss, kl_div

    def score_samples(self, x: torch.Tensor, device='cuda') -> torch.Tensor:
        """
        Compute anomaly scores for data points.
        Lower scores indicate higher likelihood of being anomalous.

        Uses negative reconstruction error as the score.
        """
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)

        # Move to model's device
        model_device = next(self.parameters()).device
        x = x.to(model_device)

        self.eval()
        with torch.no_grad():
            recon, mu, logvar = self.forward(x)
            # Negative reconstruction error per sample
            recon_error = F.mse_loss(recon, x, reduction='none').sum(dim=1)
            # Return negative error (higher is better/more normal)
            scores = -recon_error

        return scores.cpu().numpy()

    def sample(self, num_samples: int) -> torch.Tensor:
        """Generate samples from the model."""
        with torch.no_grad():
            model_device = next(self.parameters()).device
            z = torch.randn(num_samples, self.latent_dim, device=model_device)
            samples = self.decoder(z)

        return samples

    def fit(
        self,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        test_data: Optional[torch.Tensor] = None,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        beta: float = 1.0,
        patience: int = 15,
        verbose: bool = True
    ) -> dict:
        """
        Train the VAE model.

        Args:
            train_data: Training dataset
            val_data: Validation dataset
            test_data: Optional test dataset for evaluation
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            beta: Weight for KL divergence term
            patience: Early stopping patience
            verbose: Print training progress

        Returns:
            Training history dictionary
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2
        )

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_data),
            batch_size=batch_size,
            shuffle=True
        )

        history = {
            'train_loss': [], 'train_recon': [], 'train_kl': [],
            'val_loss': [], 'val_recon': [], 'val_kl': []
        }

        if test_data is not None:
            history['test_loss'] = []
            history['test_recon'] = []
            history['test_kl'] = []

        best_val_loss = float('inf')
        patience_counter = 0

        self.train()
        for epoch in range(epochs):
            train_loss = 0.0
            train_recon = 0.0
            train_kl = 0.0
            num_batches = 0

            for batch_data, in train_loader:
                batch_data = batch_data.to(self.device)
                optimizer.zero_grad()

                # Forward pass
                recon, mu, logvar = self.forward(batch_data)
                loss, recon_loss, kl_div = self.loss_function(recon, batch_data, mu, logvar, beta)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_recon += recon_loss.item()
                train_kl += kl_div.item()
                num_batches += 1

            train_loss /= num_batches
            train_recon /= num_batches
            train_kl /= num_batches

            # Validation
            self.eval()
            with torch.no_grad():
                val_recon, val_mu, val_logvar = self.forward(val_data.to(self.device))
                val_loss, val_recon_loss, val_kl_div = self.loss_function(
                    val_recon, val_data.to(self.device), val_mu, val_logvar, beta
                )
                val_loss = val_loss.item()
                val_recon_loss = val_recon_loss.item()
                val_kl_div = val_kl_div.item()

                # Evaluate on test set if provided
                if test_data is not None:
                    test_recon, test_mu, test_logvar = self.forward(test_data.to(self.device))
                    test_loss, test_recon_loss, test_kl_div = self.loss_function(
                        test_recon, test_data.to(self.device), test_mu, test_logvar, beta
                    )
                    history['test_loss'].append(test_loss.item())
                    history['test_recon'].append(test_recon_loss.item())
                    history['test_kl'].append(test_kl_div.item())

            self.train()

            history['train_loss'].append(train_loss)
            history['train_recon'].append(train_recon)
            history['train_kl'].append(train_kl)
            history['val_loss'].append(val_loss)
            history['val_recon'].append(val_recon_loss)
            history['val_kl'].append(val_kl_div)

            scheduler.step(val_loss)

            if verbose and epoch % 5 == 0:
                log_msg = f'Epoch {epoch}: Train Loss = {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}), '
                log_msg += f'Val Loss = {val_loss:.4f} (Recon: {val_recon_loss:.4f}, KL: {val_kl_div:.4f})'
                if test_data is not None:
                    log_msg += f', Test Loss = {history["test_loss"][-1]:.4f}'
                print(log_msg)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch}')
                break

        # Set threshold based on validation data
        self.set_threshold(val_data)

        return history

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
        self.eval()
        with torch.no_grad():
            scores = self.score_samples(val_data.to(self.device))

        # Set threshold as percentile of validation scores
        self.threshold = np.quantile(scores, anomaly_fraction)
        self.threshold = np.quantile(scores, anomaly_fraction)

        print(f'Threshold set to {self.threshold:.4f} '
              f'(marking {anomaly_fraction*100:.1f}% of validation data as anomalies)')

    def predict_anomaly(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict anomalies based on score threshold.

        Args:
            x: Input data

        Returns:
            Boolean tensor indicating anomalies (True = anomaly)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")

        self.eval()
        with torch.no_grad():
            scores = self.score_samples(x.to(self.device))

        return scores < self.threshold

    def predict(self, X):
        """
        Predict anomalies based on threshold.

        Args:
            X: Test data

        Returns:
            predictions: 1 for normal, -1 for anomaly
        """
        scores = self.score_samples(X)
        return np.where(scores.cpu() >= self.threshold, 1, -1)

    def evaluate_anomaly_detection(
        self,
        normal_data: torch.Tensor,
        anomaly_data: torch.Tensor,
        plot: bool = True
    ) -> dict:
        """
        Evaluate anomaly detection performance.
        """
        self.eval()

        # Get the true device of the model
        model_device = next(self.parameters()).device

        # Move data to the same device as the model
        normal_data = normal_data.to(model_device)
        anomaly_data = anomaly_data.to(model_device)

        print("Model device:", model_device)
        print("Normal data device:", normal_data.device)

        with torch.no_grad():
            normal_scores = self.score_samples(normal_data).cpu().numpy()
            anomaly_scores = self.score_samples(anomaly_data).cpu().numpy()

        # Create labels (0 = normal, 1 = anomaly)
        y_true = np.concatenate([
            np.zeros(len(normal_scores)),
            np.ones(len(anomaly_scores))
        ])

        # Use negative score as anomaly score (lower reconstruction = higher anomaly)
        y_scores = np.concatenate([-normal_scores, -anomaly_scores])

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Compute accuracy with current threshold
        if self.threshold is not None:
            predictions = np.concatenate([
                normal_scores < self.threshold,
                anomaly_scores < self.threshold
            ])
            accuracy = (predictions == y_true).mean()
        else:
            accuracy = None

        results = {
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'normal_score_mean': normal_scores.mean(),
            'normal_score_std': normal_scores.std(),
            'anomaly_score_mean': anomaly_scores.mean(),
            'anomaly_score_std': anomaly_scores.std()
        }

        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for Anomaly Detection')
            plt.legend()
            plt.grid(True)
            plt.show()

        return results

    def save_model(self, save_path: str, train_data: torch.Tensor):
        """
        Save the VAE model and metadata.

        Args:
            save_path: Base path for saving (without extension)
            train_data: Training data for computing statistics
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save model state dict
        torch.save(self.state_dict(), f"{save_path}_model.pth")

        # Calculate the scores on training data
        self.eval()
        with torch.no_grad():
            # Ensure train_data is a tensor
            if not isinstance(train_data, torch.Tensor):
                train_data = torch.FloatTensor(train_data)

            # Move to model's device
            model_device = next(self.parameters()).device
            train_data = train_data.to(model_device)

            train_scores = self.score_samples(train_data)
            # Ensure train_data is a tensor
            if not isinstance(train_data, torch.Tensor):
                train_data = torch.FloatTensor(train_data)

            # Move to model's device
            model_device = next(self.parameters()).device
            train_data = train_data.to(model_device)

            train_scores = self.score_samples(train_data)

        # Save metadata (threshold and config)
        # Handle both numpy arrays and tensors
        if isinstance(train_scores, np.ndarray):
            mean_score = float(np.mean(train_scores))
            std_score = float(np.std(train_scores))
        else:
            mean_score = train_scores.cpu().mean().item()
            std_score = train_scores.cpu().std().item()

        # Handle both numpy arrays and tensors
        if isinstance(train_scores, np.ndarray):
            mean_score = float(np.mean(train_scores))
            std_score = float(np.std(train_scores))
        else:
            mean_score = train_scores.cpu().mean().item()
            std_score = train_scores.cpu().std().item()

        metadata = {
            'threshold': self.threshold,
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims,
            'device': str(next(self.parameters()).device),
            "mean": float(train_scores.mean()),
            "std": float(train_scores.std()),
            'hidden_dims': self.hidden_dims,
            'device': str(next(self.parameters()).device),
            "mean": float(train_scores.mean()),
            "std": float(train_scores.std())
        }

        with open(f"{save_path}_meta_data.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Model saved to: {save_path}_model.pth")
        print(f"Metadata saved to: {save_path}_meta_data.pkl")

    @classmethod
    def load_model(cls, save_path: str, hidden_dims: List[int] = [256, 128]):
        """
        Load a saved VAE model.

        Args:
            save_path: Base path for loading (without extension)
            hidden_dims: Hidden layer dimensions (must match saved model)

        Returns:
            Dictionary with loaded VAE model and metadata
        """
        # Load metadata
        with open(f"{save_path}_meta_data.pkl", 'rb') as f:
            metadata = pickle.load(f)

        # Create model with saved configuration
        model = cls(
            input_dim=metadata['input_dim'],
            latent_dim=metadata['latent_dim'],
            hidden_dims=hidden_dims,
            device=metadata['device']
        )

        # Load model state dict
        model.load_state_dict(torch.load(f"{save_path}_model.pth", map_location=metadata['device']))

        # Restore threshold
        model.threshold = metadata['threshold']

        print(f"Model loaded from: {save_path}_model.pth")
        print(f"Metadata loaded from: {save_path}_meta_data.pkl")
        print(f"Threshold: {model.threshold}")

        model_dict = {
            'model': model.to(metadata['device']),
            'thr': model.threshold,
            'mean': metadata["mean"],
            'std': metadata["std"]
        }

        return model_dict


def load_rl_data_for_vae(args, env=None, val_split_ratio=0.2, test_split_ratio=0.2):
    """
    Load RL dataset and prepare next_observations + actions for VAE training.

    Args:
        args: Arguments containing data_path, obs_shape, action_dim
        env: Environment object (for Abiomed datasets)
        val_split_ratio: Fraction of data for validation
        test_split_ratio: Fraction of data for test

    Returns:
        Tuple of (train_data, val_data, test_data, vae_input_dim)
    """
    # Load dataset using the existing utility function
    dataset_result = load_dataset_with_validation_split(
        args=args,
        env=env,
        val_split_ratio=val_split_ratio,
        test_split_ratio=test_split_ratio
    )

    train_dataset = dataset_result['train_data']
    val_dataset = dataset_result['val_data']
    test_dataset = dataset_result['test_data']
    buffer_len = dataset_result['buffer_len']

    # Extract concatenated next_observations + actions for VAE
    train_next_obs = torch.FloatTensor(train_dataset['next_observations'])
    train_actions = torch.FloatTensor(train_dataset['actions'])

    # Concatenate next_observations and actions for training data
    train_vae_input = torch.cat([train_next_obs, train_actions], dim=1)

    val_next_obs = torch.FloatTensor(val_dataset['next_observations'])
    val_actions = torch.FloatTensor(val_dataset['actions'])

    test_next_obs = torch.FloatTensor(test_dataset['next_observations'])
    test_actions = torch.FloatTensor(test_dataset['actions'])

    # Concatenate next_observations and actions for validation and test data
    val_vae_input = torch.cat([val_next_obs, val_actions], dim=1)
    test_vae_input = torch.cat([test_next_obs, test_actions], dim=1)

    # Calculate input dimension for VAE model
    vae_input_dim = train_vae_input.shape[1]

    print(f"VAE input shape: {train_vae_input.shape}")
    print(f"VAE input dimension: {vae_input_dim}")
    print(f"Next observations dim: {train_next_obs.shape[1]}, Actions dim: {train_actions.shape[1]}")

    return train_vae_input, val_vae_input, test_vae_input, vae_input_dim


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def parse_args():
    """Parse command line arguments."""
    print("Running", __file__)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="configs/vae/hopper.yaml")
    config_args, remaining_argv = config_parser.parse_known_args()
    if config_args.config:
        with open(config_args.config, "r") as f:
            config = yaml.safe_load(f)
            config = {k.replace("-", "_"): v for k, v in config.items()}
    else:
        config = {}

    parser = argparse.ArgumentParser(parents=[config_parser])

    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). Overrides config file.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs. Overrides config file.')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output. Overrides config file.')

    # RL dataset specific arguments
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to RL dataset file (pickle or npz)')
    parser.add_argument('--task', type=str, default='hopper-medium-v2',
                        help='Task name (e.g., abiomed, halfcheetah-medium-v2)')
    parser.add_argument('--obs_dim', type=int, nargs='+', default=[11],
                        help='Observation shape (default: [11] for Hopper)')
    parser.add_argument('--action_dim', type=int, default=3,
                        help='Action dimension (default: 3 for Hopper)')
    parser.add_argument('--model_save_path', type=str, default=None,
                        help='Path to save or load model')
    parser.add_argument('--fig_save_path', type=str, default=None,
                        help='Path to save figures')
    parser.add_argument('--seed', type=int, default=42,)
    parser.set_defaults(**config)
    args = parser.parse_args(remaining_argv)
    args.config = config

    return args


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    args.obs_dim = (args.obs_dim,)

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = args.config

    # Override config with command line arguments if provided
    if args.device is not None:
        config['device'] = args.device
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.verbose:
        config['verbose'] = True

    # Override config with RL-specific arguments
    if args.data_path is not None:
        args.data_path = args.data_path
    if args.task != 'synthetic':
        args.task = args.task
    args.obs_shape = tuple(args.obs_dim)
    args.action_dim = args.action_dim

    # Set random seed for reproducibility
    if 'seed' in config:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Determine device
    cli_device = args.device
    cfg_device = config.get('device', None)
    device = cli_device if cli_device is not None else (cfg_device or 'cpu')

    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        print(f"{device} not available, falling back to CPU")
        device = "cpu"

    args.device = device
    config["device"] = device
    print(f"Using device: {device}")

    # Choose data loading mode
    use_rl_data = config.get('use_rl_data', args.task != 'synthetic')

    # Load data
    env = None
    if use_rl_data and args.data_path is None:
        try:
            env = gym.make(args.task)
        except Exception as e:
            print(f"Warning: Could not create environment {args.task}: {e}")

    if use_rl_data:
        print("Loading RL dataset for VAE training...")
        train_data, val_data, test_data, vae_input_dim = load_rl_data_for_vae(
            args=args,
            env=env,
            val_split_ratio=config.get('val_ratio', 0.2),
            test_split_ratio=config.get('test_ratio', 0.2)
        )
        config['input_dim'] = vae_input_dim
    else:
        print("Creating synthetic data...")
        normal_data, anomaly_data = create_synthetic_data(
            n_samples=config.get('n_samples', 2000),
            dim=config.get('input_dim', 2),
            anomaly_type=config.get('anomaly_type', 'outlier'),
            magn = 3
        )

        # Split normal data
        train_ratio = config.get('train_ratio', 0.6)
        val_ratio = config.get('val_ratio', 0.2)

        n_train = int(train_ratio * len(normal_data))
        n_val = int(val_ratio * len(normal_data))

        train_data = normal_data[:n_train]
        val_data = normal_data[n_train:n_train+n_val]
        test_data = normal_data[n_train+n_val:]

    if config.get('verbose', True):
        print(f"Data shapes - Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")

    # Create and train model
    print("Creating VAE model...")
    model = VAE(
        input_dim=config.get('input_dim', 2),
        latent_dim=config.get('latent_dim', 16),
        hidden_dims=config.get('hidden_dims', [256, 256]),
        device=device
    ).to(device)

    # print("Training VAE model...")
    history = model.fit(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        epochs=config.get('epochs', 100),
        batch_size=config.get('batch_size', 128),
        lr=config.get('lr', 1e-3),
        beta=config.get('beta', 1.0),
        patience=config.get('patience', 15),
        verbose=config.get('verbose', True)
    )
    

    # Plot training curves
    plot_training_curves(history, save_path=f"figures/{args.task}/vae_training.png")

    # Save model if requested
    if config.get('model_save_path', False):
        save_path = args.model_save_path 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save_model(save_path, train_data)
        print(f"Model saved to: {save_path}_model.pth")
    # dict_model = model.load_model(args.model_save_path, hidden_dims=config.get('hidden_dims', [256, 256]))
    # model = dict_model['model'].to(device)
    # print(dict_model['mean'])
    # Evaluate on test data

    # plot_likelihood_distributions(
    #                     model,
    #                     train_data.to(device),
    #                     val_data.to(device),
    #                     ood_data=test_data.to(device),
    #                     thr = model.threshold,
    #                     title="Likelihood Distribution",
    #                     savepath=args.fig_save_path,
    #                     bins=50
    #                 )
    print("\nEvaluating VAE on test set...")
    test_scores = model.score_samples(test_data.to(device))
    print(f"Test scores - Mean: {test_scores.mean():.4f}, Std: {test_scores.std():.4f}")
    train_scores = model.score_samples(train_data.to(device))
    print(f"Train scores - Mean: {train_scores.mean():.4f}, Std: {train_scores.std():.4f}")


    print("\nVAE training and evaluation completed!")

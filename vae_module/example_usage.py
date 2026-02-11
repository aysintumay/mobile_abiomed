#!/usr/bin/env python3
"""
Example usage of the VAE module for density estimation and anomaly detection.
This shows how to integrate the VAE into training scripts and use it for RL datasets.
"""

import argparse
import numpy as np
import os
import sys
import torch

# Add parent directory to path to import from common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vae_module.vae import VAE, create_synthetic_data, load_rl_data_for_vae, plot_training_curves
from common.util import load_dataset_with_validation_split


def example_synthetic_data():
    """
    Example 1: Training VAE on synthetic data for anomaly detection.
    """
    print("=" * 70)
    print("Example 1: Training VAE on Synthetic Data")
    print("=" * 70)

    # Create synthetic data
    normal_data, anomaly_data = create_synthetic_data(
        n_samples=2000,
        dim=5,
        anomaly_type='outlier'
    )

    print(f"Normal data shape: {normal_data.shape}")
    print(f"Anomaly data shape: {anomaly_data.shape}")

    # Split normal data into train/val/test
    train_ratio = 0.6
    val_ratio = 0.2

    n_train = int(train_ratio * len(normal_data))
    n_val = int(val_ratio * len(normal_data))

    train_data = normal_data[:n_train]
    val_data = normal_data[n_train:n_train+n_val]
    test_data = normal_data[n_train+n_val:]

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    # Create VAE model
    model = VAE(
        input_dim=5,
        latent_dim=3,
        hidden_dims=[128, 64],
        device='cpu'
    )

    print("\nTraining VAE...")
    history = model.fit(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        epochs=50,
        batch_size=128,
        lr=1e-3,
        beta=1.0,
        patience=10,
        verbose=True
    )

    # Evaluate on test data
    print("\nEvaluating on test data...")
    results = model.evaluate_anomaly_detection(
        normal_data=test_data,
        anomaly_data=anomaly_data,
        plot=False
    )

    print(f"\nResults:")
    print(f"  ROC AUC: {results['roc_auc']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Normal score: {results['normal_score_mean']:.4f} ± {results['normal_score_std']:.4f}")
    print(f"  Anomaly score: {results['anomaly_score_mean']:.4f} ± {results['anomaly_score_std']:.4f}")

    return model, history


def example_rl_dataset():
    """
    Example 2: Training VAE on RL dataset (e.g., D4RL).
    """
    print("\n" + "=" * 70)
    print("Example 2: Training VAE on RL Dataset")
    print("=" * 70)

    # Simulate command line arguments for RL dataset
    args = argparse.Namespace(
        data_path=None,
        task='hopper-medium-v2',
        obs_dim=(11,),
        action_dim=3,
        obs_shape=(11,),
        device='cpu'
    )

    # Mock environment for this example
    class MockD4RLEnv:
        def get_dataset(self):
            # Simulate D4RL dataset structure
            n_samples = 5000
            return {
                'observations': np.random.randn(n_samples, 11).astype(np.float32),
                'actions': np.random.randn(n_samples, 3).astype(np.float32),
                'rewards': np.random.randn(n_samples).astype(np.float32),
                'terminals': np.random.choice([0, 1], n_samples).astype(bool),
                'next_observations': np.random.randn(n_samples, 11).astype(np.float32)
            }

    env = MockD4RLEnv()

    # Load dataset with train/val/test splits
    print("Loading RL dataset...")
    train_data, val_data, test_data, vae_input_dim = load_rl_data_for_vae(
        args=args,
        env=env,
        val_split_ratio=0.15,
        test_split_ratio=0.15
    )

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Input dimension: {vae_input_dim}")

    # Create VAE model
    model = VAE(
        input_dim=vae_input_dim,
        latent_dim=8,
        hidden_dims=[256, 128],
        device='cpu'
    )

    print("\nTraining VAE on RL data...")
    history = model.fit(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        epochs=30,
        batch_size=256,
        lr=1e-3,
        beta=0.5,  # Lower beta for RL data
        patience=10,
        verbose=True
    )

    # Evaluate on test set
    print("\nTest set performance:")
    test_scores = model.score_samples(test_data)
    print(f"  Mean score: {test_scores.mean():.4f}")
    print(f"  Std score: {test_scores.std():.4f}")

    return model, history


def example_save_and_load():
    """
    Example 3: Saving and loading trained VAE models.
    """
    print("\n" + "=" * 70)
    print("Example 3: Saving and Loading VAE Models")
    print("=" * 70)

    # Create and train a simple model
    normal_data, _ = create_synthetic_data(n_samples=1000, dim=4)

    train_data = normal_data[:600]
    val_data = normal_data[600:800]

    print("Training a VAE model...")
    model = VAE(
        input_dim=4,
        latent_dim=2,
        hidden_dims=[64, 32],
        device='cpu'
    )

    model.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=20,
        batch_size=64,
        verbose=False
    )

    # Save model
    save_path = 'saved_models/vae/example_vae'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"\nSaving model to {save_path}...")
    model.save_model(save_path, train_data)

    # Load model
    print(f"Loading model from {save_path}...")
    model_dict = VAE.load_model(save_path, hidden_dims=[64, 32])
    loaded_model = model_dict['model']

    print(f"Loaded model threshold: {model_dict['thr']:.4f}")
    print(f"Loaded model mean score: {model_dict['mean']:.4f}")
    print(f"Loaded model std score: {model_dict['std']:.4f}")

    # Verify loaded model works
    test_input = torch.randn(10, 4)
    scores = loaded_model.score_samples(test_input)
    print(f"\nTest scores from loaded model: Mean = {scores.mean():.4f}, Std = {scores.std():.4f}")

    return loaded_model


def example_latent_space_analysis():
    """
    Example 4: Analyzing the latent space of a trained VAE.
    """
    print("\n" + "=" * 70)
    print("Example 4: Latent Space Analysis")
    print("=" * 70)

    # Create 2D data for easy visualization
    normal_data, anomaly_data = create_synthetic_data(
        n_samples=1500,
        dim=6,
        anomaly_type='outlier'
    )

    train_data = normal_data[:900]
    val_data = normal_data[900:1200]
    test_data = normal_data[1200:]

    # Train VAE with 2D latent space
    model = VAE(
        input_dim=6,
        latent_dim=2,
        hidden_dims=[128, 64],
        device='cpu'
    )

    print("Training VAE with 2D latent space...")
    model.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=30,
        batch_size=128,
        verbose=False
    )

    # Encode data to latent space
    model.eval()
    with torch.no_grad():
        test_mu, test_logvar = model.encoder(test_data)
        anomaly_mu, anomaly_logvar = model.encoder(anomaly_data)

    print(f"\nLatent space statistics:")
    print(f"Normal data latent mean:")
    print(f"  Dim 0: {test_mu[:, 0].mean():.4f} ± {test_mu[:, 0].std():.4f}")
    print(f"  Dim 1: {test_mu[:, 1].mean():.4f} ± {test_mu[:, 1].std():.4f}")

    print(f"Anomaly data latent mean:")
    print(f"  Dim 0: {anomaly_mu[:, 0].mean():.4f} ± {anomaly_mu[:, 0].std():.4f}")
    print(f"  Dim 1: {anomaly_mu[:, 1].mean():.4f} ± {anomaly_mu[:, 1].std():.4f}")

    # Generate samples from latent space
    print(f"\nGenerating {100} samples from latent space...")
    samples = model.sample(100)
    print(f"Generated samples shape: {samples.shape}")
    print(f"Generated samples mean: {samples.mean().item():.4f}")
    print(f"Generated samples std: {samples.std().item():.4f}")

    return model


def example_beta_vae():
    """
    Example 5: Training β-VAE with different β values.
    """
    print("\n" + "=" * 70)
    print("Example 5: β-VAE with Different β Values")
    print("=" * 70)

    # Create data
    normal_data, _ = create_synthetic_data(n_samples=1200, dim=4)

    train_data = normal_data[:720]
    val_data = normal_data[720:960]
    test_data = normal_data[960:]

    beta_values = [0.1, 1.0, 4.0]

    for beta in beta_values:
        print(f"\nTraining VAE with β = {beta}...")

        model = VAE(
            input_dim=4,
            latent_dim=2,
            hidden_dims=[64, 32],
            device='cpu'
        )

        history = model.fit(
            train_data=train_data,
            val_data=val_data,
            epochs=20,
            batch_size=128,
            beta=beta,
            verbose=False
        )

        final_recon = history['val_recon'][-1]
        final_kl = history['val_kl'][-1]

        print(f"  Final validation reconstruction loss: {final_recon:.4f}")
        print(f"  Final validation KL divergence: {final_kl:.4f}")
        print(f"  Ratio (Recon/KL): {final_recon/final_kl:.4f}")


def main():
    """Run all example demonstrations."""

    print("\n" + "🎯" * 35)
    print("VAE Module - Usage Examples")
    print("🎯" * 35)

    # Example 1: Synthetic data
    model1, history1 = example_synthetic_data()

    # Example 2: RL dataset
    model2, history2 = example_rl_dataset()

    # Example 3: Save and load
    loaded_model = example_save_and_load()

    # Example 4: Latent space analysis
    model4 = example_latent_space_analysis()

    # Example 5: β-VAE
    example_beta_vae()

    print("\n" + "=" * 70)
    print("✅ All examples completed successfully!")
    print("=" * 70)

    print("\n📋 Summary of VAE Module Features:")
    print("  ✓ Encoder-decoder architecture for density estimation")
    print("  ✓ Anomaly detection via reconstruction error")
    print("  ✓ Training with train/val/test evaluation")
    print("  ✓ Support for RL datasets (state-action pairs)")
    print("  ✓ Model saving and loading with metadata")
    print("  ✓ Latent space analysis and sampling")
    print("  ✓ β-VAE for disentangled representations")
    print("  ✓ ROC AUC and accuracy metrics")
    print("  ✓ Configurable architecture and hyperparameters")


if __name__ == "__main__":
    main()

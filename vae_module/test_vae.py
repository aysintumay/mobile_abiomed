#!/usr/bin/env python3
"""
Test script for the VAE module.
This demonstrates usage patterns and validates the VAE works correctly.
"""

import numpy as np
import torch
import pickle
import os
import sys
import tempfile
from argparse import Namespace

# Add parent directory to path to import from vae_module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vae_module.vae import VAE, create_synthetic_data, load_rl_data_for_vae


def test_vae_initialization():
    """Test VAE model initialization."""
    print("=" * 50)
    print("Testing VAE Initialization")
    print("=" * 50)

    # Test basic initialization
    model = VAE(input_dim=10, latent_dim=5, hidden_dims=[64, 32], device='cpu')

    print(f"Input dim: {model.input_dim}")
    print(f"Latent dim: {model.latent_dim}")
    print(f"Device: {model.device}")

    # Check encoder and decoder
    assert hasattr(model, 'encoder'), "Model should have encoder"
    assert hasattr(model, 'decoder'), "Model should have decoder"

    print("✓ VAE initialization test passed!")


def test_vae_forward_pass():
    """Test VAE forward pass."""
    print("\n" + "=" * 50)
    print("Testing VAE Forward Pass")
    print("=" * 50)

    model = VAE(input_dim=8, latent_dim=4, hidden_dims=[32, 16], device='cpu')

    # Create dummy input
    batch_size = 16
    x = torch.randn(batch_size, 8)

    # Forward pass
    recon, mu, logvar = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")

    # Check shapes
    assert recon.shape == x.shape, f"Reconstruction shape {recon.shape} != input shape {x.shape}"
    assert mu.shape == (batch_size, 4), f"Mu shape {mu.shape} != expected (16, 4)"
    assert logvar.shape == (batch_size, 4), f"Logvar shape {logvar.shape} != expected (16, 4)"

    print("✓ Forward pass test passed!")


def test_vae_loss():
    """Test VAE loss computation."""
    print("\n" + "=" * 50)
    print("Testing VAE Loss Computation")
    print("=" * 50)

    model = VAE(input_dim=6, latent_dim=3, hidden_dims=[32], device='cpu')

    x = torch.randn(32, 6)
    recon, mu, logvar = model(x)

    # Compute loss
    loss, recon_loss, kl_div = model.loss_function(recon, x, mu, logvar, beta=1.0)

    print(f"Total loss: {loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL divergence: {kl_div.item():.4f}")

    # Check that losses are positive
    assert loss.item() > 0, "Total loss should be positive"
    assert recon_loss.item() > 0, "Reconstruction loss should be positive"
    assert kl_div.item() >= 0, "KL divergence should be non-negative"

    print("✓ Loss computation test passed!")


def test_vae_training():
    """Test VAE training on synthetic data."""
    print("\n" + "=" * 50)
    print("Testing VAE Training")
    print("=" * 50)

    # Create synthetic data
    normal_data, _ = create_synthetic_data(n_samples=500, dim=4, anomaly_type='outlier')

    # Split data
    train_data = normal_data[:300]
    val_data = normal_data[300:400]
    test_data = normal_data[400:]

    print(f"Train data shape: {train_data.shape}")
    print(f"Val data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # Create and train model
    model = VAE(input_dim=4, latent_dim=2, hidden_dims=[32, 16], device='cpu')

    history = model.fit(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        epochs=10,
        batch_size=32,
        lr=1e-3,
        beta=1.0,
        patience=5,
        verbose=True
    )

    # Check history
    assert 'train_loss' in history, "History should contain train_loss"
    assert 'val_loss' in history, "History should contain val_loss"
    assert 'test_loss' in history, "History should contain test_loss"
    assert len(history['train_loss']) > 0, "Training should run at least one epoch"

    # Check that training reduces loss
    initial_loss = history['train_loss'][0]
    final_loss = history['train_loss'][-1]
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")

    print("✓ Training test passed!")


def test_vae_score_samples():
    """Test VAE score_samples method."""
    print("\n" + "=" * 50)
    print("Testing VAE Score Samples")
    print("=" * 50)

    # Create and train model
    normal_data, anomaly_data = create_synthetic_data(n_samples=400, dim=3, anomaly_type='outlier')

    train_data = normal_data[:240]
    val_data = normal_data[240:320]
    test_data = normal_data[320:]

    model = VAE(input_dim=3, latent_dim=2, hidden_dims=[32], device='cpu')

    model.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=5,
        batch_size=32,
        verbose=False
    )

    # Score samples
    normal_scores = model.score_samples(test_data)
    anomaly_scores = model.score_samples(anomaly_data)

    print(f"Normal scores - Mean: {normal_scores.mean():.4f}, Std: {normal_scores.std():.4f}")
    print(f"Anomaly scores - Mean: {anomaly_scores.mean():.4f}, Std: {anomaly_scores.std():.4f}")

    # Normal data should have higher scores (lower reconstruction error) than anomaly data
    print(f"Normal scores higher than anomaly scores: {normal_scores.mean() > anomaly_scores.mean()}")

    print("✓ Score samples test passed!")


def test_vae_anomaly_detection():
    """Test VAE anomaly detection with threshold."""
    print("\n" + "=" * 50)
    print("Testing VAE Anomaly Detection")
    print("=" * 50)

    # Create synthetic data
    normal_data, anomaly_data = create_synthetic_data(n_samples=600, dim=5, anomaly_type='outlier')

    train_data = normal_data[:360]
    val_data = normal_data[360:480]
    test_normal = normal_data[480:]

    # Create and train model
    model = VAE(input_dim=5, latent_dim=3, hidden_dims=[64, 32], device='cpu')

    model.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=10,
        batch_size=32,
        verbose=False
    )

    # Set threshold
    model.set_threshold(val_data, anomaly_fraction=0.05)

    print(f"Threshold: {model.threshold:.4f}")

    # Predict anomalies
    normal_predictions = model.predict_anomaly(test_normal)
    anomaly_predictions = model.predict_anomaly(anomaly_data)

    # Calculate accuracy
    normal_correct = (~normal_predictions).sum().item() / len(normal_predictions)
    anomaly_correct = anomaly_predictions.sum().item() / len(anomaly_predictions)

    print(f"Normal data correctly classified: {normal_correct:.2%}")
    print(f"Anomaly data correctly classified: {anomaly_correct:.2%}")

    print("✓ Anomaly detection test passed!")


def test_vae_save_load():
    """Test VAE model saving and loading."""
    print("\n" + "=" * 50)
    print("Testing VAE Save/Load")
    print("=" * 50)

    # Create and train model
    normal_data, _ = create_synthetic_data(n_samples=400, dim=4, anomaly_type='outlier')

    train_data = normal_data[:240]
    val_data = normal_data[240:320]

    model = VAE(input_dim=4, latent_dim=2, hidden_dims=[32, 16], device='cpu')

    model.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=5,
        batch_size=32,
        verbose=False
    )

    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'test_vae')

        model.save_model(save_path, train_data)

        # Load model
        model_dict = VAE.load_model(save_path, hidden_dims=[32, 16])
        loaded_model = model_dict['model']

        print(f"Original threshold: {model.threshold:.4f}")
        print(f"Loaded threshold: {loaded_model.threshold:.4f}")

        # Check that models produce same output
        test_input = torch.randn(10, 4)

        original_scores = model.score_samples(test_input)
        loaded_scores = loaded_model.score_samples(test_input)

        score_diff = torch.abs(original_scores - loaded_scores).max().item()
        print(f"Max score difference: {score_diff:.6f}")

        assert score_diff < 1e-5, f"Loaded model produces different scores: {score_diff}"

    print("✓ Save/load test passed!")


def test_vae_sampling():
    """Test VAE sampling capability."""
    print("\n" + "=" * 50)
    print("Testing VAE Sampling")
    print("=" * 50)

    # Create and train model
    normal_data, _ = create_synthetic_data(n_samples=300, dim=3, anomaly_type='outlier')

    model = VAE(input_dim=3, latent_dim=2, hidden_dims=[32], device='cpu')

    model.fit(
        train_data=normal_data[:180],
        val_data=normal_data[180:240],
        epochs=5,
        batch_size=32,
        verbose=False
    )

    # Generate samples
    num_samples = 50
    samples = model.sample(num_samples)

    print(f"Generated samples shape: {samples.shape}")
    print(f"Samples mean: {samples.mean().item():.4f}")
    print(f"Samples std: {samples.std().item():.4f}")

    assert samples.shape == (num_samples, 3), f"Samples shape {samples.shape} != expected (50, 3)"

    print("✓ Sampling test passed!")


def test_vae_evaluate_anomaly_detection():
    """Test comprehensive anomaly detection evaluation."""
    print("\n" + "=" * 50)
    print("Testing Anomaly Detection Evaluation")
    print("=" * 50)

    # Create synthetic data
    normal_data, anomaly_data = create_synthetic_data(n_samples=500, dim=4, anomaly_type='outlier')

    train_data = normal_data[:300]
    val_data = normal_data[300:400]
    test_normal = normal_data[400:]

    # Create and train model
    model = VAE(input_dim=4, latent_dim=2, hidden_dims=[64, 32], device='cpu')

    model.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=10,
        batch_size=32,
        verbose=False
    )

    # Evaluate
    results = model.evaluate_anomaly_detection(
        normal_data=test_normal,
        anomaly_data=anomaly_data,
        plot=False
    )

    print(f"ROC AUC: {results['roc_auc']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Normal score mean: {results['normal_score_mean']:.4f}")
    print(f"Anomaly score mean: {results['anomaly_score_mean']:.4f}")

    # ROC AUC should be > 0.5 (better than random)
    assert results['roc_auc'] > 0.5, f"ROC AUC {results['roc_auc']} should be > 0.5"

    print("✓ Evaluation test passed!")


def main():
    """Run all tests."""
    print("Testing VAE Module")
    print("=" * 70)

    try:
        test_vae_initialization()
        test_vae_forward_pass()
        test_vae_loss()
        test_vae_training()
        test_vae_score_samples()
        test_vae_anomaly_detection()
        test_vae_save_load()
        test_vae_sampling()
        test_vae_evaluate_anomaly_detection()

        print("\n" + "=" * 70)
        print("🎉 All tests passed! The VAE module is working correctly.")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

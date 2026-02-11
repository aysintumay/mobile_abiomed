# Loading Pretrained Diffusion Models and Computing ELBO NLL

This guide explains how to load pretrained unconditional diffusion models and compute ELBO-based Negative Log-Likelihood (NLL) for density estimation and OOD detection.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Model Loading](#model-loading)
- [ELBO NLL Computation](#elbo-nll-computation)
- [OOD Detection](#ood-detection)
- [Complete Examples](#complete-examples)
- [Troubleshooting](#troubleshooting)
- [Configuration](#configuration-files)

---

## Overview

The unconditional diffusion models in this codebase can:
1. **Generate samples** from learned distributions (e.g., action sequences)
2. **Compute log-likelihoods** using ELBO (Evidence Lower Bound)
3. **Detect OOD samples** by comparing log-likelihoods

This README focuses on **loading models** and **computing ELBO NLL** for evaluation and OOD detection.

---

## Quick Start

```python
import torch
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from ddim_training_unconditional import UnconditionalEpsilonMLP, log_prob_elbo

# 1. Load pretrained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load('path/to/checkpoint.pt', map_location=device)

model = UnconditionalEpsilonMLP(
    target_dim=ckpt['target_dim'],
    hidden_dim=ckpt['cfg']['hidden_dim'],
    time_embed_dim=ckpt['cfg']['time_embed_dim'],
    num_hidden_layers=ckpt['cfg']['num_hidden_layers'],
)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()

# 2. Load scheduler
scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='linear')

# 3. Prepare data (must be normalized like training data)
test_data = torch.randn(100, ckpt['target_dim']).to(device)  # Example

# 4. Compute ELBO NLL
log_probs = log_prob_elbo(model, scheduler, test_data, device)
nll_total = -log_probs  # Total NLL per sample

print(f"Mean NLL (total): {nll_total.mean().item():.2f} nats")
# Expected for ID data (17 dims): 10-50 nats total
# Expected for OOD data (17 dims): 200-1000 nats total
```

### Using Config File

You can also use the `monte_carlo_sampling_unconditional.py` script with config files:

```bash
# Example using HalfCheetah config
python diffusion/monte_carlo_sampling_unconditional.py \
    --config diffusion/configs/test/halfcheetah_mlp_ddpm_unconditional.yaml

# Example using Hopper config
python diffusion/monte_carlo_sampling_unconditional.py \
    --config diffusion/configs/test/hopper_mlp_ddpm_unconditional.yaml
```

Config files specify:
- Model directory with checkpoint
- Test data path
- Output directory for results
- Batch size, inference steps, etc.

---

## Model Loading

### Basic Model Loading Function

```python
from typing import Tuple
import torch
from torch import nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from ddim_training_unconditional import (
    UnconditionalEpsilonMLP,
    UnconditionalEpsilonTransformer,
)


def build_model_from_ckpt(ckpt_path: str, device: str) -> Tuple[nn.Module, DDPMScheduler, dict]:
    """
    Load pretrained unconditional diffusion model from checkpoint.

    Args:
        ckpt_path: Path to checkpoint.pt file
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        model: Loaded model in eval mode
        scheduler: DDPM/DDIM scheduler with same config as training
        cfg: Configuration dict from checkpoint
    """
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    target_dim = ckpt.get("target_dim")
    model_type = cfg.get("model_type", "mlp")
    time_embed_dim = cfg.get("time_embed_dim", 128)

    # Build model architecture
    if model_type == "mlp":
        model = UnconditionalEpsilonMLP(
            target_dim=target_dim,
            hidden_dim=cfg.get("hidden_dim", 512),
            time_embed_dim=time_embed_dim,
            num_hidden_layers=cfg.get("num_hidden_layers", 3),
            dropout=cfg.get("dropout", 0.0),
        )
    else:  # transformer
        model = UnconditionalEpsilonTransformer(
            target_dim=target_dim,
            d_model=cfg.get("d_model", 256),
            nhead=cfg.get("nhead", 8),
            num_layers=cfg.get("tf_layers", 4),
            dim_feedforward=cfg.get("ff_dim", 512),
            dropout=cfg.get("dropout", 0.1),
            time_embed_dim=time_embed_dim,
        )

    # Load weights
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    # Create scheduler (use same config as training)
    num_train_timesteps = cfg.get("num_train_timesteps", 1000)
    beta_schedule = cfg.get("beta_schedule", "linear")
    beta_start = cfg.get("beta_start", 0.0001)
    beta_end = cfg.get("beta_end", 0.02)

    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule,
        beta_start=beta_start,
        beta_end=beta_end,
    )

    print(f"Loaded {model_type} model with {target_dim} dimensions")
    print(f"Scheduler: {num_train_timesteps} timesteps, {beta_schedule} schedule")

    return model, scheduler, cfg


# Usage
model, scheduler, cfg = build_model_from_ckpt(
    ckpt_path='/path/to/checkpoint.pt',
    device='cuda'
)
```

### Loading from Directory Structure

If your model is saved in a directory with `checkpoint.pt` and `scheduler/` subdirectory:

```python
import os
import joblib
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


def load_model_and_scheduler_from_dir(model_dir: str, device: str):
    """
    Load model and scheduler from directory structure.

    Directory structure:
        model_dir/
            checkpoint.pt
            scheduler/
                scheduler_config.json
    """
    # Load model
    ckpt_path = os.path.join(model_dir, "checkpoint.pt")
    model, _, cfg = build_model_from_ckpt(ckpt_path, device)

    # Load scheduler from saved config
    scheduler_path = os.path.join(model_dir, "scheduler")
    if os.path.exists(scheduler_path):
        scheduler = DDPMScheduler.from_pretrained(scheduler_path)
        print(f"Loaded scheduler from {scheduler_path}")
    else:
        # Fallback: create scheduler from checkpoint config
        scheduler = DDPMScheduler(
            num_train_timesteps=cfg.get("num_train_timesteps", 1000),
            beta_schedule=cfg.get("beta_schedule", "linear"),
        )
        print("Created scheduler from checkpoint config")

    return model, scheduler, cfg


# Usage
model, scheduler, cfg = load_model_and_scheduler_from_dir(
    model_dir='/data/sparse_d4rl/pretrained/unconditional/ddim_mlp_halfcheetah_medium_v2',
    device='cuda'
)
```

---

## ELBO NLL Computation

### Understanding ELBO NLL

The ELBO (Evidence Lower Bound) provides a **lower bound** on the true log-likelihood:

```
log p(x₀) ≥ ELBO(x₀) = E_q[log p(x₀|x₁)] - Σ KL(q(x_{t-1}|x_t,x₀) || p_θ(x_{t-1}|x_t)) - KL(q(x_T|x₀) || p(x_T))
```

**Properties:**
- Returns `log p(x₀)` for each sample (higher = more likely)
- NLL = `-log p(x₀)` (lower = more likely)
- Typically reported **per dimension** for interpretability

### Basic ELBO Computation

```python
from ddim_training_unconditional import log_prob_elbo

# Prepare data (MUST be normalized like training data!)
data = torch.randn(100, 17).to('cuda')  # Example: 100 samples, 17 dims

# Compute ELBO
with torch.no_grad():
    log_probs = log_prob_elbo(
        model=model,
        scheduler=scheduler,
        x0=data,
        device='cuda'
    )

# Convert to NLL
nll_total = -log_probs  # Shape: [100], total NLL per sample (summed over dims)

print(f"Mean NLL (total): {nll_total.mean().item():.2f} nats")
print(f"Std NLL: {nll_total.std().item():.2f} nats")
print(f"Range: [{nll_total.min().item():.2f}, {nll_total.max().item():.2f}]")
```

### Batch Processing for Large Datasets

```python
def compute_nll_batched(model, scheduler, data, device, batch_size=512):
    """
    Compute NLL for large datasets with batching.

    Args:
        model: Pretrained diffusion model
        scheduler: DDPM/DDIM scheduler
        data: Tensor of shape [N, dim] or numpy array
        device: Device to run on
        batch_size: Batch size for processing

    Returns:
        nll_per_sample: Total NLL for each sample (summed over dimensions), shape [N]
    """
    # Convert to tensor if needed
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()

    data = data.to(device)
    num_samples = data.shape[0]
    dim = data.shape[1]

    nll_list = []

    model.eval()
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = data[i:i + batch_size]

            # Compute log prob
            log_probs = log_prob_elbo(model, scheduler, batch, device)

            # Convert to NLL (total per sample, summed over dimensions)
            nll_batch = -log_probs
            nll_list.append(nll_batch.cpu())

    # Concatenate results
    nll_per_sample = torch.cat(nll_list, dim=0).numpy()

    return nll_per_sample


# Usage
data = np.random.randn(10000, 17)  # Large dataset
nll = compute_nll_batched(model, scheduler, data, 'cuda', batch_size=512)

print(f"Computed NLL for {len(nll)} samples")
print(f"Mean: {nll.mean():.4f}, Std: {nll.std():.4f}")
print(f"Min: {nll.min():.4f}, Max: {nll.max():.4f}")
```

### Expected NLL Values

For a **17-dimensional action space** (e.g., HalfCheetah):

| Data Type | Total NLL (summed) | Interpretation |
|-----------|-------------------|----------------|
| **In-distribution (ID)** | 10 - 50 nats | High likelihood |
| **Near-distribution** | 50 - 200 nats | Moderate likelihood |
| **Out-of-distribution (OOD)** | 200 - 1000 nats | Low likelihood |
| **Random noise** | > 1000 nats | Very low likelihood |

**Note:**
- These are **total NLL** values (summed over all dimensions)
- For per-dimension comparison, divide by number of dimensions: `nll_per_dim = nll_total / dim`
- Values assume data is properly normalized (mean=0, std=1)

---

## OOD Detection

### Setting Up OOD Detection

```python
import numpy as np
import torch


def setup_ood_detector(model, scheduler, id_data, device, percentile=95):
    """
    Set up OOD detector by computing threshold on ID validation data.

    Args:
        model: Pretrained model
        scheduler: DDPM scheduler
        id_data: In-distribution validation data (normalized)
        device: Device
        percentile: Percentile for threshold (default: 95)

    Returns:
        threshold: NLL threshold for OOD detection
        id_nll_stats: Statistics dict for ID data
    """
    # Compute NLL for ID data
    id_nll = compute_nll_batched(model, scheduler, id_data, device)

    # Set threshold at percentile
    threshold = np.percentile(id_nll, percentile)

    stats = {
        'mean': id_nll.mean(),
        'std': id_nll.std(),
        'min': id_nll.min(),
        'max': id_nll.max(),
        'median': np.median(id_nll),
        'threshold': threshold,
        'percentile': percentile,
    }

    print(f"ID NLL Statistics (total NLL per sample):")
    print(f"  Mean: {stats['mean']:.2f} ± {stats['std']:.2f} nats")
    print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}] nats")
    print(f"  Threshold ({percentile}th percentile): {threshold:.2f} nats")

    return threshold, stats


# Usage
id_val_data = np.load('path/to/validation_data.npz')['action']
threshold, id_stats = setup_ood_detector(
    model, scheduler, id_val_data, 'cuda', percentile=95
)
```

### Detecting OOD Samples

```python
def detect_ood(model, scheduler, test_data, threshold, device):
    """
    Detect OOD samples based on NLL threshold.

    Args:
        model: Pretrained model
        scheduler: DDPM scheduler
        test_data: Test data to evaluate
        threshold: NLL threshold from ID validation data
        device: Device

    Returns:
        is_ood: Boolean array indicating OOD samples
        nll: NLL for each test sample
        ood_scores: OOD scores (higher = more OOD)
    """
    # Compute NLL
    nll = compute_nll_batched(model, scheduler, test_data, device)

    # Flag samples above threshold as OOD
    is_ood = nll > threshold

    # OOD score: distance from threshold
    ood_scores = nll - threshold

    return is_ood, nll, ood_scores


# Usage
test_data = np.load('path/to/test_data.npz')['action']
is_ood, nll, ood_scores = detect_ood(model, scheduler, test_data, threshold, 'cuda')

print(f"OOD Detection Results:")
print(f"  Total samples: {len(test_data)}")
print(f"  OOD samples: {is_ood.sum()} ({is_ood.mean():.1%})")
print(f"  ID samples: {(~is_ood).sum()} ({(~is_ood).mean():.1%})")
```

### Computing ROC AUC for OOD Detection

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def evaluate_ood_detection(model, scheduler, id_data, ood_data, device):
    """
    Evaluate OOD detection performance using ROC AUC.

    Args:
        model: Pretrained model
        scheduler: DDPM scheduler
        id_data: In-distribution test data
        ood_data: Out-of-distribution test data
        device: Device

    Returns:
        roc_auc: ROC AUC score
        results: Dictionary with detailed metrics
    """
    # Compute NLL for both datasets
    id_nll = compute_nll_batched(model, scheduler, id_data, device)
    ood_nll = compute_nll_batched(model, scheduler, ood_data, device)

    # Create labels (0 = ID, 1 = OOD)
    y_true = np.concatenate([
        np.zeros(len(id_nll)),
        np.ones(len(ood_nll))
    ])

    # Use NLL as OOD score (higher = more OOD)
    y_scores = np.concatenate([id_nll, ood_nll])

    # Compute ROC AUC
    roc_auc = roc_auc_score(y_true, y_scores)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    results = {
        'roc_auc': roc_auc,
        'id_nll_mean': id_nll.mean(),
        'id_nll_std': id_nll.std(),
        'ood_nll_mean': ood_nll.mean(),
        'ood_nll_std': ood_nll.std(),
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
    }

    print(f"OOD Detection Performance:")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  ID NLL: {results['id_nll_mean']:.2f} ± {results['id_nll_std']:.2f} nats")
    print(f"  OOD NLL: {results['ood_nll_mean']:.2f} ± {results['ood_nll_std']:.2f} nats")
    print(f"  Separation: {results['ood_nll_mean'] - results['id_nll_mean']:.2f} nats")

    return roc_auc, results


# Usage
id_test = np.load('halfcheetah_test.npz')['action']
ood_test = np.load('walker2d_test.npz')['action']  # Different env as OOD

roc_auc, results = evaluate_ood_detection(model, scheduler, id_test, ood_test, 'cuda')
```

---

## Complete Examples

### Example 1: Evaluate Pretrained Model on Test Set

```python
import os
import numpy as np
import torch
import joblib
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from ddim_training_unconditional import log_prob_elbo


def main():
    # Configuration
    model_dir = '/data/sparse_d4rl/pretrained/unconditional/ddim_mlp_halfcheetah_medium_v2'
    test_npz = '/public/d4rl/diffusion_processed/halfcheetah-medium-v2_test.npz'
    scaler_path = '/public/d4rl/diffusion_processed/halfcheetah-medium-v2_scaler.pkl'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Load model
    print("Loading model...")
    model, scheduler, cfg = load_model_and_scheduler_from_dir(model_dir, device)
    target_dim = cfg['target_dim']

    # 2. Load and preprocess test data
    print("Loading test data...")
    data = np.load(test_npz)
    test_actions = data['action']  # Assume already normalized

    # If using scaler for normalization:
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        test_actions = scaler.transform(test_actions)
        print(f"Applied StandardScaler from {scaler_path}")

    print(f"Loaded {len(test_actions)} test samples, dim={test_actions.shape[1]}")

    # 3. Compute NLL
    print("\nComputing ELBO NLL...")
    nll = compute_nll_batched(model, scheduler, test_actions, device, batch_size=512)

    # 4. Report statistics
    print(f"\nNLL Statistics (total per sample):")
    print(f"  Mean: {nll.mean():.2f} ± {nll.std():.2f} nats")
    print(f"  Median: {np.median(nll):.2f} nats")
    print(f"  Range: [{nll.min():.2f}, {nll.max():.2f}] nats")
    print(f"  25th percentile: {np.percentile(nll, 25):.2f}")
    print(f"  75th percentile: {np.percentile(nll, 75):.2f}")

    # 5. Save results
    output_path = os.path.join(model_dir, 'test_nll_results.npz')
    np.savez(
        output_path,
        nll_per_sample=nll,
        nll_mean=nll.mean(),
        nll_std=nll.std(),
        num_samples=len(nll),
    )
    print(f"\nSaved results to {output_path}")


if __name__ == '__main__':
    main()
```

### Example 2: Using monte_carlo_sampling_unconditional.py

The `monte_carlo_sampling_unconditional.py` script provides a complete pipeline for:
- Loading pretrained models
- Computing ELBO NLL on test data
- Comparing NLL distributions

```bash
# Run with config file
python diffusion/monte_carlo_sampling_unconditional.py \
    --config diffusion/configs/test/halfcheetah_mlp_ddpm_unconditional.yaml

# Or specify parameters directly
python diffusion/monte_carlo_sampling_unconditional.py \
    --model-dir /data/models/halfcheetah_unconditional \
    --test-npz /data/halfcheetah_test.npz \
    --output-dir results/halfcheetah_nll \
    --batch-size 1000 \
    --device cuda
```

### Example 3: Custom OOD Detection Pipeline

```python
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def ood_detection_pipeline():
    # Configuration
    model_dir = '/data/sparse_d4rl/pretrained/unconditional/ddim_mlp_hopper_medium_v2'
    id_npz = '/public/d4rl/diffusion_processed/hopper-medium-v2_test.npz'
    ood_npz = '/public/d4rl/diffusion_processed/walker2d-medium-v2_test.npz'
    device = 'cuda'

    # 1. Load model
    print("=" * 60)
    print("STEP 1: Loading Model")
    print("=" * 60)
    model, scheduler, cfg = load_model_and_scheduler_from_dir(model_dir, device)

    # 2. Load data
    print("\n" + "=" * 60)
    print("STEP 2: Loading Data")
    print("=" * 60)
    id_data = np.load(id_npz)['action']
    ood_data = np.load(ood_npz)['action']
    print(f"ID data: {len(id_data)} samples")
    print(f"OOD data: {len(ood_data)} samples")

    # Split ID data into validation (for threshold) and test
    split_idx = int(0.5 * len(id_data))
    id_val = id_data[:split_idx]
    id_test = id_data[split_idx:]

    # 3. Set threshold on validation data
    print("\n" + "=" * 60)
    print("STEP 3: Setting OOD Threshold")
    print("=" * 60)
    threshold, id_stats = setup_ood_detector(
        model, scheduler, id_val, device, percentile=95
    )

    # 4. Evaluate on test data
    print("\n" + "=" * 60)
    print("STEP 4: Evaluating OOD Detection")
    print("=" * 60)

    id_test_nll = compute_nll_batched(model, scheduler, id_test, device)
    ood_test_nll = compute_nll_batched(model, scheduler, ood_data, device)

    # Compute metrics
    y_true = np.concatenate([np.zeros(len(id_test_nll)), np.ones(len(ood_test_nll))])
    y_scores = np.concatenate([id_test_nll, ood_test_nll])
    roc_auc = roc_auc_score(y_true, y_scores)

    # Compute accuracy using threshold
    id_predictions = id_test_nll > threshold  # Should be False (ID)
    ood_predictions = ood_test_nll > threshold  # Should be True (OOD)

    id_accuracy = (~id_predictions).mean()  # Correctly classified as ID
    ood_accuracy = ood_predictions.mean()  # Correctly classified as OOD
    overall_accuracy = (len(id_test) * id_accuracy + len(ood_data) * ood_accuracy) / (len(id_test) + len(ood_data))

    print(f"\nResults:")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  ID Test NLL: {id_test_nll.mean():.2f} ± {id_test_nll.std():.2f} nats")
    print(f"  OOD Test NLL: {ood_test_nll.mean():.2f} ± {ood_test_nll.std():.2f} nats")
    print(f"  NLL Separation: {ood_test_nll.mean() - id_test_nll.mean():.2f} nats")
    print(f"\nAccuracy (with threshold={threshold:.2f} nats):")
    print(f"  ID accuracy: {id_accuracy:.1%}")
    print(f"  OOD accuracy: {ood_accuracy:.1%}")
    print(f"  Overall accuracy: {overall_accuracy:.1%}")

    # 5. Plot results
    print("\n" + "=" * 60)
    print("STEP 5: Plotting Results")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot NLL distributions
    axes[0].hist(id_test_nll, bins=50, alpha=0.6, label='ID', density=True)
    axes[0].hist(ood_test_nll, bins=50, alpha=0.6, label='OOD', density=True)
    axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
    axes[0].set_xlabel('NLL (nats/dim)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('NLL Distributions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    axes[1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ood_detection_results.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot to ood_detection_results.png")

    return roc_auc, threshold


if __name__ == '__main__':
    roc_auc, threshold = ood_detection_pipeline()
```

---

## Troubleshooting

### Issue: NLL values are extremely large (> 10,000 nats total for 17 dims)

**Causes:**
1. Data not normalized properly
2. Model not trained well
3. Using wrong variance in ELBO (should be fixed in latest code)

**Solutions:**
```python
# Check data normalization
print(f"Data mean: {test_data.mean()}, std: {test_data.std()}")
# Should be close to 0 and 1 if using StandardScaler

# Check if you're using the scaler from training
scaler = joblib.load('scaler.pkl')
test_data_normalized = scaler.transform(test_data)
```

### Issue: Out of memory during NLL computation

**Solution:** Reduce batch size
```python
nll = compute_nll_batched(model, scheduler, data, device, batch_size=128)
# Or even smaller: batch_size=64
```

### Issue: ID and OOD have similar NLL values

**Causes:**
1. OOD data not different enough from ID
2. Model has poor calibration

**Solutions:**
```python
# Try more extreme OOD data
ood_data = id_data + np.random.randn(*id_data.shape) * 3.0  # Large noise

# Or use data from very different domain
ood_data = np.load('different_environment.npz')['action']
```

### Issue: ELBO computation is slow

**Expected:** ELBO requires forward pass through the entire diffusion chain (T=1000 timesteps), so it's inherently slower than sampling.

**Optimization:**
- Use GPU: `device='cuda'`
- Increase batch size: `batch_size=1024` (if memory allows)
- Reduce number of timesteps (requires retraining): `num_train_timesteps=100`

---

## Key Takeaways

1. **Always normalize data** the same way as training (use saved scaler)
2. **ELBO returns log p(x)** - negate for NLL (total summed over dimensions)
3. **Expected NLL range** for well-trained models (17 dims): 10-50 nats total (ID), 200-1000 nats total (OOD)
4. **Set threshold** on validation data, not test data
5. **ROC AUC** is the best metric for OOD detection performance
6. **Total NLL scales with dimensionality** - higher dims → larger NLL values

## Configuration Files

Example configs are available in `diffusion/configs/test/`:

**DDPM (stochastic sampling):**
- `halfcheetah_mlp_ddpm_unconditional.yaml`
- `hopper_mlp_ddpm_unconditional.yaml`
- `walker2d_mlp_ddpm_unconditional.yaml`

**DDIM (deterministic sampling):**
- `halfcheetah_mlp_ddim_unconditional.yaml`
- `hopper_mlp_ddim_unconditional.yaml`
- `walker2d_mlp_ddim_unconditional.yaml`

Each config specifies:
```yaml
out: /path/to/model/directory          # Model checkpoint directory
test_npz: /path/to/test_data.npz       # Test data with 'action' key
model_dir: /path/to/model/directory    # Same as 'out'
num_samples: 10000                      # Number of samples to generate
batch_size: 1000                        # Batch size for processing
device: cuda                            # Device (cuda/cpu)
inference_steps: 50                     # Number of denoising steps
scheduler_type: ddpm                    # Scheduler type (ddpm/ddim)
output_dir: results/                    # Where to save results
max_test_samples: 1000                  # Max test samples for NLL eval
```

### Running with Config

```bash
# Compute ELBO NLL on test data
python diffusion/monte_carlo_sampling_unconditional.py \
    --config diffusion/configs/test/halfcheetah_mlp_ddpm_unconditional.yaml

# Override specific parameters
python diffusion/monte_carlo_sampling_unconditional.py \
    --config diffusion/configs/test/hopper_mlp_ddpm_unconditional.yaml \
    --batch-size 2000 \
    --max-test-samples 500
```

For more details, see:
- `monte_carlo_sampling_unconditional.py` - Complete sampling and NLL evaluation script
- `ddim_training_unconditional.py` - Model definitions and ELBO computation

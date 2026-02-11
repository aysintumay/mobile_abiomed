# Loading Pretrained Neural ODE Models and Computing NLL

This guide explains how to load pretrained Continuous Normalizing Flow (Neural ODE) models and compute log-likelihoods for density estimation and OOD detection.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Model Loading](#model-loading)
- [NLL Computation](#nll-computation)
- [OOD Detection](#ood-detection)
- [Complete Examples](#complete-examples)
- [Configuration Files](#configuration-files)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Neural ODE (Continuous Normalizing Flow) models learn the probability density `p(x)` using ordinary differential equations. Unlike diffusion models, Neural ODE:
1. **Computes exact log-likelihoods** (not a lower bound like ELBO)
2. **Faster inference** (single ODE solve vs. 1000 diffusion steps)
3. **Invertible transformations** for both sampling and likelihood computation

This README focuses on **loading models** and **computing log-likelihoods** for evaluation and OOD detection.

---

## Quick Start

```python
import torch
import numpy as np
from neuralODE.neural_ode_density import ODEFunc, ContinuousNormalizingFlow

# 1. Load pretrained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = '/path/to/model.pt'

# Load checkpoint
ckpt = torch.load(model_path, map_location=device)
cfg = ckpt.get('cfg', {})
target_dim = ckpt.get('target_dim')

# Create model
odefunc = ODEFunc(
    dim=target_dim,
    hidden_dims=cfg.get('hidden_dims', (512, 512)),
    activation=cfg.get('activation', 'silu'),
    time_dependent=cfg.get('time_dependent', True),
)

flow = ContinuousNormalizingFlow(
    func=odefunc,
    t0=cfg.get('t0', 0.0),
    t1=cfg.get('t1', 1.0),
    solver=cfg.get('solver', 'dopri5'),
    rtol=cfg.get('rtol', 1e-5),
    atol=cfg.get('atol', 1e-5),
)

# Load weights
state_dict = ckpt.get('model_state_dict', ckpt)
flow.load_state_dict(state_dict)
flow.to(device)
flow.eval()

# 2. Prepare data (must be normalized like training data)
test_data = torch.randn(100, target_dim).to(device)

# 3. Compute log-likelihood (EXACT, not a bound!)
with torch.no_grad():
    log_probs = flow.log_prob(test_data)

nll = -log_probs  # Total NLL per sample

print(f"Mean NLL: {nll.mean().item():.2f} nats")
# Expected for ID data (17 dims): 5-30 nats total
# Expected for OOD data (17 dims): 50-500 nats total
```

### Using Config File

You can use the `neural_ode_inference.py` script with config files:

```bash
# Compute NLL on test data
python neuralODE/neural_ode_inference.py \
    --config neuralODE/configs/test/hopper.yaml

# For other environments
python neuralODE/neural_ode_inference.py \
    --config neuralODE/configs/test/halfcheetah.yaml
```

---

## Model Loading

### Basic Model Loading Function

```python
import torch
from neuralODE.neural_ode_density import ODEFunc, ContinuousNormalizingFlow
from typing import Tuple


def load_neural_ode_model(
    model_path: str,
    device: str = 'cuda'
) -> Tuple[ContinuousNormalizingFlow, dict]:
    """
    Load pretrained Neural ODE model from checkpoint.

    Args:
        model_path: Path to model.pt file
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        flow: Loaded ContinuousNormalizingFlow model
        cfg: Configuration dict from checkpoint
    """
    # Load checkpoint
    ckpt = torch.load(model_path, map_location=device)

    # Extract config and dimensions
    cfg = ckpt.get('cfg', {})
    target_dim = ckpt.get('target_dim')

    if target_dim is None:
        raise ValueError("Checkpoint missing 'target_dim'. Old checkpoint format?")

    # Create ODE function
    odefunc = ODEFunc(
        dim=target_dim,
        hidden_dims=cfg.get('hidden_dims', (512, 512)),
        activation=cfg.get('activation', 'silu'),
        time_dependent=cfg.get('time_dependent', True),
    )

    # Create flow model
    flow = ContinuousNormalizingFlow(
        func=odefunc,
        t0=cfg.get('t0', 0.0),
        t1=cfg.get('t1', 1.0),
        solver=cfg.get('solver', 'dopri5'),
        rtol=cfg.get('rtol', 1e-5),
        atol=cfg.get('atol', 1e-5),
    )

    # Load state dict
    state_dict = ckpt.get('model_state_dict', ckpt)
    flow.load_state_dict(state_dict)
    flow.to(device)
    flow.eval()

    print(f"Loaded Neural ODE model:")
    print(f"  Target dim: {target_dim}")
    print(f"  Hidden dims: {cfg.get('hidden_dims', (512, 512))}")
    print(f"  Solver: {cfg.get('solver', 'dopri5')}")
    print(f"  Time dependent: {cfg.get('time_dependent', True)}")

    return flow, cfg


# Usage
flow, cfg = load_neural_ode_model(
    model_path='/data/sparse_d4rl/pretrained/neural_ode/hopper_medium_v2/model.pt',
    device='cuda'
)
```

### Alternative: Load from Config File

```python
from dataclasses import dataclass
from typing import Tuple
import yaml


@dataclass
class EvalConfig:
    model_path: str
    npz_path: str = ""
    hidden_dims: Tuple[int, ...] = (512, 512)
    activation: str = "silu"
    time_dependent: bool = True
    solver: str = "dopri5"
    t0: float = 0.0
    t1: float = 1.0
    rtol: float = 1e-5
    atol: float = 1e-5
    device: str = "cuda"


def load_from_config(config_path: str) -> Tuple[ContinuousNormalizingFlow, EvalConfig]:
    """Load model using YAML config."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Create config
    cfg = EvalConfig(
        model_path=config_dict['model'],
        npz_path=config_dict.get('npz', ''),
        hidden_dims=tuple(config_dict.get('hidden_dims', [512, 512])),
        activation=config_dict.get('activation', 'silu'),
        time_dependent=config_dict.get('time_dependent', True),
        solver=config_dict.get('solver', 'dopri5'),
        t0=config_dict.get('t0', 0.0),
        t1=config_dict.get('t1', 1.0),
        rtol=config_dict.get('rtol', 1e-5),
        atol=config_dict.get('atol', 1e-5),
        device=config_dict.get('device', 'cuda'),
    )

    # Load model using config
    flow, _ = load_neural_ode_model(cfg.model_path, cfg.device)

    return flow, cfg


# Usage
flow, cfg = load_from_config('neuralODE/configs/test/hopper.yaml')
```

---

## NLL Computation

### Understanding Neural ODE Log-Likelihood

Neural ODE computes **exact log-likelihood** using the instantaneous change of variables formula:

```
log p(x) = log p(z) + ∫[0→1] Tr(∂f/∂z) dt
```

where:
- `z = f(x, 1)` is the latent representation
- `log p(z)` is the prior (standard Gaussian)
- The integral tracks how density changes along the ODE trajectory

**Key differences from Diffusion ELBO:**
- ✅ **Exact likelihood** (not a lower bound)
- ✅ **Faster** (single ODE solve vs. 1000 steps)
- ❌ **Requires more computation** per ODE solve (Jacobian trace)

### Basic NLL Computation

```python
from neuralODE.neural_ode_density import ContinuousNormalizingFlow

# Load model
flow, cfg = load_neural_ode_model('model.pt', 'cuda')

# Prepare data
data = torch.randn(100, 17).to('cuda')  # Example: 100 samples, 17 dims

# Compute log-likelihood
with torch.no_grad():
    log_probs = flow.log_prob(data)

# Convert to NLL
nll = -log_probs  # Shape: [100], total NLL per sample

print(f"Mean NLL: {nll.mean().item():.2f} nats")
print(f"Std NLL: {nll.std().item():.2f} nats")
print(f"Range: [{nll.min().item():.2f}, {nll.max().item():.2f}]")
```

### Batch Processing for Large Datasets

```python
def compute_nll_batched(flow, data, device='cuda', batch_size=512):
    """
    Compute NLL for large datasets with batching.

    Args:
        flow: Pretrained ContinuousNormalizingFlow model
        data: Tensor of shape [N, dim] or numpy array
        device: Device to run on
        batch_size: Batch size for processing

    Returns:
        nll_per_sample: Total NLL for each sample, shape [N]
    """
    # Convert to tensor if needed
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()

    data = data.to(device)
    num_samples = data.shape[0]

    nll_list = []

    flow.eval()
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = data[i:i + batch_size]

            # Compute log prob
            log_probs = flow.log_prob(batch)

            # Convert to NLL
            nll_batch = -log_probs
            nll_list.append(nll_batch.cpu())

    # Concatenate results
    nll_per_sample = torch.cat(nll_list, dim=0).numpy()

    return nll_per_sample


# Usage
import numpy as np
data = np.random.randn(10000, 17)  # Large dataset
nll = compute_nll_batched(flow, data, 'cuda', batch_size=512)

print(f"Computed NLL for {len(nll)} samples")
print(f"Mean: {nll.mean():.2f}, Std: {nll.std():.2f}")
```

### Expected NLL Values

For a **17-dimensional action space** (e.g., HalfCheetah):

| Data Type | Total NLL (Neural ODE) | Interpretation |
|-----------|------------------------|----------------|
| **In-distribution (ID)** | 5 - 30 nats | High likelihood |
| **Near-distribution** | 30 - 100 nats | Moderate likelihood |
| **Out-of-distribution (OOD)** | 100 - 500 nats | Low likelihood |
| **Random noise** | > 500 nats | Very low likelihood |

**Note:** Neural ODE typically gives **lower NLL** than Diffusion ELBO for the same data because it computes exact likelihood (not a lower bound).

---

## OOD Detection

### Setting Up OOD Detection

```python
def setup_ood_detector(flow, id_data, device='cuda', percentile=95):
    """
    Set up OOD detector by computing threshold on ID validation data.

    Args:
        flow: Pretrained Neural ODE model
        id_data: In-distribution validation data (normalized)
        device: Device
        percentile: Percentile for threshold (default: 95)

    Returns:
        threshold: NLL threshold for OOD detection
        id_nll_stats: Statistics dict for ID data
    """
    # Compute NLL for ID data
    id_nll = compute_nll_batched(flow, id_data, device)

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

    print(f"ID NLL Statistics:")
    print(f"  Mean: {stats['mean']:.2f} ± {stats['std']:.2f} nats")
    print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}] nats")
    print(f"  Threshold ({percentile}th percentile): {threshold:.2f} nats")

    return threshold, stats


# Usage
id_val_data = np.load('hopper_validation.npz')['action']
threshold, id_stats = setup_ood_detector(flow, id_val_data, 'cuda', percentile=95)
```

### Detecting OOD Samples

```python
def detect_ood(flow, test_data, threshold, device='cuda'):
    """
    Detect OOD samples based on NLL threshold.

    Args:
        flow: Pretrained Neural ODE model
        test_data: Test data to evaluate
        threshold: NLL threshold from ID validation data
        device: Device

    Returns:
        is_ood: Boolean array indicating OOD samples
        nll: NLL for each test sample
        ood_scores: OOD scores (higher = more OOD)
    """
    # Compute NLL
    nll = compute_nll_batched(flow, test_data, device)

    # Flag samples above threshold as OOD
    is_ood = nll > threshold

    # OOD score: distance from threshold
    ood_scores = nll - threshold

    return is_ood, nll, ood_scores


# Usage
test_data = np.load('test_data.npz')['action']
is_ood, nll, ood_scores = detect_ood(flow, test_data, threshold, 'cuda')

print(f"OOD Detection Results:")
print(f"  Total samples: {len(test_data)}")
print(f"  OOD samples: {is_ood.sum()} ({is_ood.mean():.1%})")
print(f"  ID samples: {(~is_ood).sum()} ({(~is_ood).mean():.1%})")
```

### Computing ROC AUC for OOD Detection

```python
from sklearn.metrics import roc_auc_score, roc_curve


def evaluate_ood_detection(flow, id_data, ood_data, device='cuda'):
    """
    Evaluate OOD detection performance using ROC AUC.

    Args:
        flow: Pretrained Neural ODE model
        id_data: In-distribution test data
        ood_data: Out-of-distribution test data
        device: Device

    Returns:
        roc_auc: ROC AUC score
        results: Dictionary with detailed metrics
    """
    # Compute NLL for both datasets
    id_nll = compute_nll_batched(flow, id_data, device)
    ood_nll = compute_nll_batched(flow, ood_data, device)

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
id_test = np.load('hopper_test.npz')['action']
ood_test = np.load('walker2d_test.npz')['action']  # Different env as OOD

roc_auc, results = evaluate_ood_detection(flow, id_test, ood_test, 'cuda')
```

---

## Complete Examples

### Example 1: Evaluate Pretrained Model on Test Set

```python
import os
import numpy as np
import torch


def main():
    # Configuration
    model_path = '/data/sparse_d4rl/pretrained/neural_ode/hopper_medium_v2/model.pt'
    test_npz = '/public/d4rl/neuralODE_processed/hopper-medium-v2_test.npz'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Load model
    print("Loading model...")
    flow, cfg = load_neural_ode_model(model_path, device)

    # 2. Load test data
    print("Loading test data...")
    from neuralODE.neural_ode_density import NPZTargetDataset
    dataset = NPZTargetDataset(test_npz)
    test_data = dataset.target.numpy()

    print(f"Loaded {len(test_data)} test samples, dim={test_data.shape[1]}")

    # 3. Compute NLL
    print("\nComputing NLL...")
    nll = compute_nll_batched(flow, test_data, device, batch_size=512)

    # 4. Report statistics
    print(f"\nNLL Statistics:")
    print(f"  Mean: {nll.mean():.2f} ± {nll.std():.2f} nats")
    print(f"  Median: {np.median(nll):.2f} nats")
    print(f"  Range: [{nll.min():.2f}, {nll.max():.2f}] nats")
    print(f"  25th percentile: {np.percentile(nll, 25):.2f}")
    print(f"  75th percentile: {np.percentile(nll, 75):.2f}")

    # 5. Save results
    output_dir = os.path.dirname(model_path)
    output_path = os.path.join(output_dir, 'test_nll_results.npz')
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

### Example 2: Using neural_ode_inference.py Script

```bash
# Compute NLL on test data using config file
python neuralODE/neural_ode_inference.py \
    --config neuralODE/configs/test/hopper.yaml

# The script will output:
# - Mean NLL and per-sample statistics
# - Save metrics to JSON file (if save_metrics specified in config)
# - Save per-sample log probs to NPY file (if save_logp specified)
```

### Example 3: OOD Detection Pipeline

```python
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def ood_detection_pipeline():
    # Configuration
    model_path = '/data/sparse_d4rl/pretrained/neural_ode/hopper_medium_v2/model.pt'
    id_npz = '/public/d4rl/neuralODE_processed/hopper-medium-v2_test.npz'
    ood_npz = '/public/d4rl/neuralODE_processed/walker2d-medium-v2_test.npz'
    device = 'cuda'

    # 1. Load model
    print("=" * 60)
    print("STEP 1: Loading Model")
    print("=" * 60)
    flow, cfg = load_neural_ode_model(model_path, device)

    # 2. Load data
    print("\n" + "=" * 60)
    print("STEP 2: Loading Data")
    print("=" * 60)
    from neuralODE.neural_ode_density import NPZTargetDataset
    id_dataset = NPZTargetDataset(id_npz)
    ood_dataset = NPZTargetDataset(ood_npz)

    id_data = id_dataset.target.numpy()
    ood_data = ood_dataset.target.numpy()

    print(f"ID data: {len(id_data)} samples")
    print(f"OOD data: {len(ood_data)} samples")

    # Split ID data into validation and test
    split_idx = int(0.5 * len(id_data))
    id_val = id_data[:split_idx]
    id_test = id_data[split_idx:]

    # 3. Set threshold on validation data
    print("\n" + "=" * 60)
    print("STEP 3: Setting OOD Threshold")
    print("=" * 60)
    threshold, id_stats = setup_ood_detector(flow, id_val, device, percentile=95)

    # 4. Evaluate on test data
    print("\n" + "=" * 60)
    print("STEP 4: Evaluating OOD Detection")
    print("=" * 60)

    roc_auc, results = evaluate_ood_detection(flow, id_test, ood_data, device)

    # 5. Plot results
    print("\n" + "=" * 60)
    print("STEP 5: Plotting Results")
    print("=" * 60)

    id_test_nll = compute_nll_batched(flow, id_test, device)
    ood_test_nll = compute_nll_batched(flow, ood_data, device)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot NLL distributions
    axes[0].hist(id_test_nll, bins=50, alpha=0.6, label='ID', density=True)
    axes[0].hist(ood_test_nll, bins=50, alpha=0.6, label='OOD', density=True)
    axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
    axes[0].set_xlabel('NLL (nats)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('NLL Distributions (Neural ODE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot ROC curve
    axes[1].plot(results['fpr'], results['tpr'], linewidth=2,
                 label=f'ROC (AUC={roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('neural_ode_ood_results.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot to neural_ode_ood_results.png")

    return roc_auc, threshold


if __name__ == '__main__':
    roc_auc, threshold = ood_detection_pipeline()
```

---

## Configuration Files

Example configs are available in `neuralODE/configs/`:

**Training configs:**
- `hopper_mlp.yaml`
- `halfcheetah_mlp.yaml`
- `walker2d_mlp.yaml`

**Test/Inference configs:**
- `test/hopper.yaml`
- `test/halfcheetah.yaml`
- `test/walker2d.yaml`

### Config Structure

```yaml
# Test config example
npz: /path/to/test_data.npz           # Test data path
model: /path/to/model.pt               # Trained model checkpoint

# Output paths (optional)
save_metrics: neuralODE/test/metrics.json  # Save metrics to JSON
save_logp: neuralODE/test/logp.npy         # Save per-sample log probs
max_samples: 1000                           # Limit evaluation samples

# Model architecture (must match training)
hidden_dims: [512, 512]
activation: silu
time_dependent: true

# ODE solver settings
solver: dopri5
t0: 0.0
t1: 1.0
rtol: 1.0e-5
atol: 1.0e-5

# Training settings (ignored during inference)
epochs: 200
batch: 512
lr: 1.0e-3

# Other
device: cuda
seed: 0
```

### Running with Config

```bash
# Inference on test data
python neuralODE/neural_ode_inference.py \
    --config neuralODE/configs/test/hopper.yaml

# Override specific parameters
python neuralODE/neural_ode_inference.py \
    --config neuralODE/configs/test/halfcheetah.yaml \
    --max-samples 500 \
    --device cpu
```

---

## Troubleshooting

### Issue: NLL values are extremely large (> 1000 nats for 17 dims)

**Causes:**
1. Data not normalized properly
2. Model not trained well
3. ODE solver tolerance too loose

**Solutions:**
```python
# Check data normalization
print(f"Data mean: {test_data.mean()}, std: {test_data.std()}")
# Should be close to 0 and 1 if using StandardScaler

# Try tighter solver tolerances
flow = ContinuousNormalizingFlow(
    func=odefunc,
    solver='dopri5',
    rtol=1e-6,  # Tighter tolerance
    atol=1e-6,
)
```

### Issue: Out of memory during NLL computation

**Solution:** Reduce batch size
```python
nll = compute_nll_batched(flow, data, device, batch_size=128)
# Or even smaller: batch_size=64
```

### Issue: Very slow inference

**Causes:**
- ODE solver taking many function evaluations
- Solver tolerances too tight

**Solutions:**
```python
# Use faster solver (less accurate)
flow = ContinuousNormalizingFlow(
    func=odefunc,
    solver='euler',  # Faster but less accurate
    # Or: solver='rk4'
)

# Looser tolerances
flow = ContinuousNormalizingFlow(
    func=odefunc,
    solver='dopri5',
    rtol=1e-4,  # Looser
    atol=1e-4,
)
```

### Issue: ID and OOD have similar NLL values

**Causes:**
1. OOD data not different enough from ID
2. Model has poor calibration

**Solutions:**
```python
# Try more extreme OOD data
ood_data = id_data + np.random.randn(*id_data.shape) * 3.0

# Or use data from very different domain
ood_data = np.load('different_environment.npz')['action']
```

---

## Key Takeaways

1. **Neural ODE computes exact log p(x)** - not a lower bound like diffusion ELBO
2. **Faster inference** than diffusion (single ODE solve vs. 1000 steps)
3. **Lower NLL values** than diffusion ELBO for same data (exact vs. bound)
4. **Expected NLL range** for 17 dims: 5-30 nats (ID), 100-500 nats (OOD)
5. **Always normalize data** the same way as training
6. **Solver tolerance** affects both speed and accuracy

## Comparison: Neural ODE vs Diffusion

| Aspect | Neural ODE | Diffusion ELBO |
|--------|-----------|----------------|
| **Likelihood** | Exact | Lower bound |
| **NLL (ID, 17 dims)** | 5-30 nats | 10-50 nats |
| **Inference speed** | Fast (1 ODE solve) | Slow (1000 steps) |
| **Memory** | Moderate | Lower |
| **OOD separation** | Good | Good |
| **Training stability** | Can be tricky | Generally stable |

For more details, see:
- `neural_ode_density.py` - Model definition and training
- `neural_ode_inference.py` - Inference and NLL computation script

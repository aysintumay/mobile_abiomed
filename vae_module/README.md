# VAE Module

Variational Autoencoder (VAE) implementation for density estimation and anomaly detection in the GORMPO framework.

## Overview

The VAE module provides a generative model for learning data distributions and detecting out-of-distribution (OOD) samples. It follows the same design patterns as the RealNVP module, making it easy to swap between different density estimation approaches.

## Key Features

- **Encoder-Decoder Architecture**: Learn latent representations of state-action pairs
- **Anomaly Detection**: Identify OOD samples via reconstruction error
- **Train/Val/Test Evaluation**: Built-in support for proper data splits
- **RL Dataset Integration**: Seamless integration with D4RL and custom datasets
- **Model Persistence**: Save and load trained models with metadata
- **β-VAE Support**: Configurable β parameter for disentangled representations
- **Latent Space Sampling**: Generate new samples from the learned distribution

## Architecture

### Encoder
- Input: State-action pairs (or arbitrary dimensional data)
- Hidden layers: Configurable MLP with ReLU activations
- Output: Mean (μ) and log-variance (log σ²) of latent distribution

### Decoder
- Input: Latent vector z
- Hidden layers: Reverse of encoder architecture
- Output: Reconstructed data

### Loss Function
```
Total Loss = Reconstruction Loss + β × KL Divergence
```

- **Reconstruction Loss**: MSE between input and reconstruction
- **KL Divergence**: KL(q(z|x) || p(z)) where p(z) ~ N(0, I)
- **β**: Weight for KL term (β=1 for standard VAE, β>1 for β-VAE)

## Installation

The VAE module is part of the GORMPO repository. All dependencies are included in the main `requirements.txt`.

```bash
cd GORMPO
pip install -r requirements.txt
```

## Usage

### Training on D4RL Datasets

```bash
# Using YAML config
python vae_module/vae.py --config configs/vae/hopper.yaml

# Using command-line arguments
python vae_module/vae.py \
    --task hopper-medium-v2 \
    --obs_dim 11 \
    --action_dim 3 \
    --latent_dim 8 \
    --epochs 100 \
    --device cuda
```

### Training on Custom Datasets

```bash
python vae_module/vae.py \
    --data_path /path/to/dataset.pkl \
    --task custom \
    --obs_dim 10 \
    --action_dim 4 \
    --latent_dim 6
```

### Training on Synthetic Data

```python
from vae_module.vae import VAE, create_synthetic_data

# Create data
normal_data, anomaly_data = create_synthetic_data(
    n_samples=2000,
    dim=5,
    anomaly_type='outlier'
)

# Split data
train_data = normal_data[:1200]
val_data = normal_data[1200:1600]
test_data = normal_data[1600:]

# Create and train model
model = VAE(
    input_dim=5,
    latent_dim=3,
    hidden_dims=[128, 64],
    device='cuda'
)

history = model.fit(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    epochs=100,
    batch_size=128,
    lr=1e-3,
    beta=1.0
)
```

## Configuration

### YAML Configuration Files

Configuration files are located in `configs/vae/`. Example structure:

```yaml
# Task and data
task: hopper-medium-v2
use_rl_data: true

# Model architecture
input_dim: 14
latent_dim: 8
hidden_dims: [256, 128]

# Training
epochs: 100
batch_size: 256
lr: 0.001
beta: 1.0
patience: 15

# Data splits
val_ratio: 0.2
test_ratio: 0.2

# Device
device: cuda

# Model saving
model_save_path: saved_models/vae/hopper_medium_v2
```

### Key Parameters

- `input_dim`: Dimension of input data (obs_dim + action_dim for RL)
- `latent_dim`: Dimension of latent space
- `hidden_dims`: List of hidden layer sizes (e.g., [256, 128])
- `beta`: Weight for KL divergence term (1.0 for standard VAE)
- `anomaly_fraction`: Percentile threshold for anomaly detection

## Evaluation Metrics

The VAE module provides comprehensive evaluation:

### Training Metrics (per epoch)
- Total Loss (Reconstruction + β × KL)
- Reconstruction Loss (MSE)
- KL Divergence
- Evaluated on train, validation, and test sets

### Anomaly Detection Metrics
- **ROC AUC**: Area under ROC curve
- **Accuracy**: Overall classification accuracy
- **Score Statistics**: Mean and std of reconstruction scores

### Usage

```python
# Evaluate anomaly detection performance
results = model.evaluate_anomaly_detection(
    normal_data=test_normal,
    anomaly_data=test_anomaly,
    plot=True
)

print(f"ROC AUC: {results['roc_auc']:.4f}")
print(f"Accuracy: {results['accuracy']:.4f}")
```

## Model Saving and Loading

### Saving

```python
model.save_model('saved_models/vae/my_model', train_data)
```

This saves:
- `my_model_model.pth`: Model state dict
- `my_model_meta_data.pkl`: Metadata (threshold, dimensions, statistics)

### Loading

```python
from vae_module.vae import VAE

model_dict = VAE.load_model(
    'saved_models/vae/my_model',
    hidden_dims=[256, 128]  # Must match training config
)

model = model_dict['model']
threshold = model_dict['thr']
mean_score = model_dict['mean']
std_score = model_dict['std']
```

## Integration with GORMPO

The VAE can be used as a drop-in replacement for RealNVP in the GORMPO framework:

1. Train VAE on offline dataset:
   ```bash
   python vae_module/vae.py --config configs/vae/hopper.yaml
   ```

2. Use in MOPO training by modifying `transition_model.py` to use VAE scores instead of RealNVP log-probabilities.

## Testing

Run the test suite:

```bash
cd vae_module
python test_vae.py
```

Tests cover:
- Model initialization
- Forward/backward passes
- Loss computation
- Training loop
- Anomaly detection
- Model save/load
- Latent space sampling

## Examples

Run example scripts:

```bash
# Basic examples
python vae_module/example_usage.py

# Individual examples
python -c "from vae_module.example_usage import example_synthetic_data; example_synthetic_data()"
python -c "from vae_module.example_usage import example_rl_dataset; example_rl_dataset()"
```

## Advanced Usage

### β-VAE for Disentanglement

Train with β > 1 to learn disentangled representations:

```python
model = VAE(input_dim=10, latent_dim=5, hidden_dims=[128, 64])
history = model.fit(train_data, val_data, beta=4.0)
```

Higher β values encourage independence in latent dimensions.

### Latent Space Analysis

```python
# Encode data to latent space
model.eval()
with torch.no_grad():
    mu, logvar = model.encoder(data)

# Generate samples from latent space
samples = model.sample(num_samples=100)
```

### Custom Anomaly Thresholds

```python
# Set threshold based on validation data
model.set_threshold(val_data, anomaly_fraction=0.05)

# Or set manually
model.threshold = -50.0

# Predict anomalies
is_anomaly = model.predict_anomaly(test_data)
```

## Comparison with RealNVP

| Feature | VAE | RealNVP |
|---------|-----|---------|
| Architecture | Encoder-Decoder | Coupling Layers |
| Latent Space | Explicit (Gaussian) | Implicit (via flow) |
| Sampling | Direct from latent | Inverse transformation |
| Training | Reconstruction + KL | Negative log-likelihood |
| Anomaly Detection | Reconstruction error | Log probability |
| Invertibility | No | Yes |
| Expressiveness | Limited by encoder capacity | High (with sufficient layers) |

**When to use VAE:**
- Need explicit latent representations
- Want to generate diverse samples
- Prefer interpretable latent dimensions (β-VAE)
- Working with high-dimensional data

**When to use RealNVP:**
- Need exact likelihood computation
- Want invertible transformations
- Prefer density-based anomaly scores
- Working with complex distributions

## File Structure

```
vae_module/
├── vae.py                 # Main VAE implementation
├── test_vae.py           # Test suite
├── example_usage.py      # Usage examples
└── README.md             # This file

configs/vae/
├── hopper.yaml           # Hopper-medium-v2 config
├── walker2d.yaml         # Walker2d-medium-v2 config
└── halfcheetah.yaml      # HalfCheetah-medium-v2 config
```

## API Reference

### VAE Class

```python
VAE(
    input_dim: int,
    latent_dim: int,
    hidden_dims: List[int],
    device: str = 'cpu'
)
```

**Methods:**

- `forward(x)`: Encode, sample latent, decode
- `loss_function(recon_x, x, mu, logvar, beta)`: Compute VAE loss
- `fit(train_data, val_data, test_data, ...)`: Train model
- `score_samples(x)`: Compute anomaly scores
- `predict_anomaly(x)`: Binary anomaly predictions
- `predict(X)`: Predictions (1=normal, -1=anomaly)
- `sample(num_samples)`: Generate samples
- `set_threshold(val_data, anomaly_fraction)`: Set anomaly threshold
- `save_model(save_path, train_data)`: Save model and metadata
- `load_model(save_path, hidden_dims)`: Load saved model

### Utility Functions

- `create_synthetic_data(n_samples, dim, anomaly_type)`: Generate synthetic data
- `load_rl_data_for_vae(args, env, val_split_ratio, test_split_ratio)`: Load RL datasets
- `plot_training_curves(history, save_path)`: Visualize training progress

## Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
- **Solution**: Reduce `batch_size` or `hidden_dims`

**Issue**: High reconstruction loss, low KL divergence
- **Solution**: Reduce `beta` (KL weight) or increase model capacity

**Issue**: Low reconstruction loss, high KL divergence
- **Solution**: Increase `beta` or reduce `latent_dim`

**Issue**: Poor anomaly detection performance
- **Solution**:
  - Increase training epochs
  - Adjust `anomaly_fraction` threshold
  - Increase model capacity
  - Check data normalization

## Citation

If you use this VAE module in your research, please cite:

```bibtex
@misc{gormpo_vae,
  title={VAE Module for GORMPO},
  author={GORMPO Contributors},
  year={2025},
  howpublished={\url{https://github.com/yourusername/GORMPO}}
}
```

## References

- Kingma & Welling (2014). "Auto-Encoding Variational Bayes"
- Higgins et al. (2017). "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
- An & Cho (2015). "Variational Autoencoder based Anomaly Detection using Reconstruction Probability"

## License

This module is part of the GORMPO project and follows the same license.

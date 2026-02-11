# Neural ODE Density Model

`neuralODE/neural_ode_density.py` trains a Continuous Normalizing Flow (CNF) to learn the marginal density `p(x)` from any NPZ/PKL dataset used elsewhere in this repo.

## Requirements

- PyTorch (already required by other scripts)
- `torchdiffeq` (`pip install torchdiffeq`)

## YAML Config Example

```yaml
# neuralODE/configs/hopper_mlp.yaml
npz: /public/d4rl/neuralODE_processed/hopper-medium-v2_train.npz
out: /data/sparse_d4rl/pretrained/neural_ode/hopper_medium_v2

epochs: 200
batch: 512
lr: 1.0e-3
wd: 0.0

hidden_dims: [512, 512]
activation: silu
time_dependent: true
solver: dopri5
rtol: 1e-5
atol: 1e-5
log_every: 100
checkpoint_every: 20
device: cuda
```

## Training

```bash
python neuralODE/neural_ode_density.py \
  --config neuralODE/configs/hopper_mlp.yaml
```

Key CLI flags (overridable even when using YAML):

| Flag | Description |
| --- | --- |
| `--npz` | Path to dataset (supports `.npz`, `.npy`, `.pkl`) |
| `--out` | Output directory for checkpoints and `model.pt` |
| `--hidden-dims` | Hidden layer sizes for the ODE function (space-separated) |
| `--solver` | ODE solver passed to `torchdiffeq.odeint` (e.g., `dopri5`, `rk4`) |
| `--t0`, `--t1` | Integration start/end times |
| `--rtol`, `--atol` | Adaptive solver tolerances |

## Sampling & Evaluation

### Dataset NLL / Inference

Compute evaluation-set NLLs (optionally saving metrics/per-sample logp):

```bash
python neuralODE/neural_ode_inference.py --config neuralODE/configs/test/hopper.yaml
```

You can reuse the same YAML file that was used for training—the evaluator will pull matching keys (hidden dims, solver, tolerances, device, etc.). Just point `--model` to the saved weights and override `--npz` if you evaluate a different split.

### Manual Sampling

After training:

```python
import torch
from neuralODE.neural_ode_density import ODEFunc, ContinuousNormalizingFlow

state = torch.load("/path/to/model.pt", map_location="cpu")
odefunc = ODEFunc(dim=target_dim, hidden_dims=(512, 512))
flow = ContinuousNormalizingFlow(odefunc)
flow.load_state_dict(state)
flow.eval()

# Sample data points
samples = flow.sample(num_samples=1024, device="cpu")

# Log probability of a batch
logp = flow.log_prob(samples)
```

This README is intentionally minimal—extend it with experiment notes as you benchmark the CNF against DDIM/DDPM baselines.


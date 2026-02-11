import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional
try:
    from torchdiffeq import odeint
except Exception as e:  # pragma: no cover - dependency check
    raise ImportError(
        "Missing dependency 'torchdiffeq'. Install via `pip install torchdiffeq`."
    ) from e

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

class NPZTargetDataset(Dataset):
    def __init__(self, npz_path: str, dtype: torch.dtype = torch.float32):
        super().__init__()
        # Robust loader: supports .npz/.npy with object arrays and .pkl pickled dicts
        data = None
        try:
            data = np.load(npz_path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object and data.size == 1:
                data = data.item()
        except Exception:
            try:
                with open(npz_path, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                raise ValueError(
                    f"Failed to load dataset from {npz_path}. Tried numpy.load and pickle.load. Error: {e}"
                )

        def available_keys(obj):
            if isinstance(obj, dict):
                return list(obj.keys())
            try:
                return list(obj.keys())  # NpzFile supports .keys()
            except Exception:
                try:
                    return list(obj.files)  # type: ignore[attr-defined]
                except Exception:
                    return []

        def get_key(key):
            """Try to get a key from data, return None if not found."""
            try:
                if key in data:
                    return data[key]
            except Exception:
                try:
                    if hasattr(data, "files") and key in data.files:
                        return data[key]
                except Exception:
                    pass
            return None

        # Check for D4RL format (observations + actions)
        observations = get_key("observations")
        actions = get_key("actions")

        if observations is not None and actions is not None:
            # D4RL format: concatenate observations and actions
            print(f"Detected D4RL format: observations shape={observations.shape}, actions shape={actions.shape}")
            self.target_np = np.concatenate([observations, actions], axis=1)
            print(f"Concatenated target shape: {self.target_np.shape}")
        else:
            # Fall back to original single-key format
            possible_target_keys = [
                "target",
                "y",
                "action",
                "next",
                "outputs",
                "X_target",
                "data",
                "x",
                "samples",
            ]

            def pick(keys):
                for k in keys:
                    val = get_key(k)
                    if val is not None:
                        return val
                raise KeyError(
                    f"Could not find any of keys {keys} in {available_keys(data)}. "
                    "For D4RL datasets, ensure both 'observations' and 'actions' keys exist. "
                    "Otherwise, rename your arrays or pass a small shim."
                )

            self.target_np = pick(possible_target_keys)

        self.target = torch.from_numpy(self.target_np).to(dtype)

        # Flatten trailing dims into feature vectors for generic MLP handling
        self.target = self.target.view(self.target.size(0), -1)

        self.num_samples = self.target.size(0)
        self.target_dim = self.target.size(1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.target[idx]


class D4RLTargetDataset(Dataset):
    """Dataset that loads directly from D4RL task names."""

    def __init__(self, task: str, dtype: torch.dtype = torch.float32):
        super().__init__()
        try:
            import gym
            import d4rl
        except ImportError as e:
            raise ImportError(
                "D4RL and gym are required for loading from task names. "
                "Install with: pip install gym d4rl"
            ) from e

        print(f"Loading D4RL dataset for task: {task}")
        env = gym.make(task)
        dataset = d4rl.qlearning_dataset(env)

        observations = dataset['observations']
        actions = dataset['actions']

        print(f"D4RL dataset loaded: observations shape={observations.shape}, actions shape={actions.shape}")
        self.target_np = np.concatenate([observations, actions], axis=1).astype(np.float32)
        print(f"Concatenated target shape: {self.target_np.shape}")

        self.target = torch.from_numpy(self.target_np).to(dtype)
        self.target = self.target.view(self.target.size(0), -1)

        self.num_samples = self.target.size(0)
        self.target_dim = self.target.size(1)

        env.close()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.target[idx]


def divergence_bruteforce(f: torch.Tensor, z: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """
    Computes divergence of f with respect to z using autograd.

    Args:
        f: Output tensor from the ODE function
        z: Input tensor
        create_graph: Whether to create graph for higher-order gradients (False for inference)
    """
    divergence = torch.zeros(z.size(0), device=z.device)
    for i in range(z.size(1)):
        grad = torch.autograd.grad(
            f[:, i].sum(), z, create_graph=create_graph, retain_graph=True
        )[0][:, i]
        divergence = divergence + grad
    return divergence


class ODEFunc(nn.Module):
    """
    Time-conditioned neural ODE function f(z, t).
    """

    def __init__(
        self,
        dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "silu",
        time_dependent: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.time_dependent = time_dependent
        self._inference_mode = False  # Set to True during inference to reduce memory

        act = nn.SiLU if activation == "silu" else nn.Tanh
        input_dim = dim + (1 if time_dependent else 0)
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]):
        z, logp = states
        if not z.requires_grad:
            z = z.requires_grad_(True)
        if self.time_dependent:
            t_input = torch.full((z.size(0), 1), t.item(), device=z.device, dtype=z.dtype)
            h = torch.cat([z, t_input], dim=1)
        else:
            h = z
        dz_dt = self.net(h)
        # Use create_graph=False during inference to save memory
        div = divergence_bruteforce(dz_dt, z, create_graph=not self._inference_mode)
        dlogp_dt = -div
        return dz_dt, dlogp_dt


class ContinuousNormalizingFlow(nn.Module):
    """
    Continuous normalizing flow (CNF) wrapper using torchdiffeq.
    """

    def __init__(
        self,
        func: ODEFunc,
        t0: float = 0.0,
        t1: float = 1.0,
        solver: str = "dopri5",
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> None:
        super().__init__()
        self.func = func
        self.register_buffer("integration_times", torch.tensor([t0, t1], dtype=torch.float32))
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

    def _odeint(
        self,
        z0: torch.Tensor,
        logp0: torch.Tensor,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        times = self.integration_times
        if reverse:
            times = torch.flip(times, dims=[0])
        z_t, logp_t = odeint(
            self.func,
            (z0, logp0),
            times,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )
        return z_t[-1], logp_t[-1]

    def log_prob(self, x: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
        """
        Compute log probability of data points.

        Args:
            x: Input tensor of shape (batch_size, dim)
            requires_grad: Whether to compute gradients (default: False for inference)

        Returns:
            Log probabilities for each sample
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.integration_times.device)

        # Set inference mode on the ODE function to reduce memory usage
        self.func._inference_mode = not requires_grad

        # For divergence computation in ODE, we always need input gradients
        # But we can avoid creating the computation graph during inference
        x = x.requires_grad_(True)
        logp0 = torch.zeros(x.size(0), device=x.device)
        z1, logp1 = self._odeint(x, logp0, reverse=False)
        logpz = -0.5 * (
            z1.pow(2).sum(dim=1) + z1.size(1) * math.log(2 * math.pi)
        )
        result = logpz - logp1

        # Reset inference mode
        self.func._inference_mode = False

        return result

    def score_samples(self, x: torch.Tensor, device: str = 'cuda', batch_size: int = 100) -> np.ndarray:
        """
        Compute log probability of data points (alias for log_prob for compatibility).

        Args:
            x: Input tensor of shape (batch_size, dim)
            device: Device to use (ignored, uses model's device)
            batch_size: Batch size for processing to avoid OOM (default: 100)

        Returns:
            Log probabilities as numpy array
        """
        self.eval()

        # Ensure x is a tensor on the correct device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Move to model's device
        model_device = self.integration_times.device
        x = x.to(model_device)

        # Process in batches to avoid OOM during ODE integration
        n_samples = x.shape[0]
        log_probs_list = []

        for i in range(0, n_samples, batch_size):
            batch = x[i:i+batch_size].detach().clone()

            # Compute log prob - requires_grad=False for inference
            batch_log_probs = self.log_prob(batch, requires_grad=False)
            log_probs_list.append(batch_log_probs.detach().cpu())

            # Delete intermediate tensors and clear GPU cache
            del batch, batch_log_probs
            if model_device.type == 'cuda':
                torch.cuda.empty_cache()

        log_probs = torch.cat(log_probs_list, dim=0)
        return log_probs.numpy()

    def sample(self, num_samples: int, device: str) -> torch.Tensor:
        z = torch.randn(num_samples, self.func.dim, device=device)
        logp = torch.zeros(num_samples, device=device)
        x, _ = self._odeint(z, logp, reverse=True)
        return x


@dataclass
class TrainConfig:
    npz_path: Optional[str]  # Path to NPZ/pkl file, or None if using task
    out_dir: str
    task: Optional[str] = None  # D4RL task name (e.g., "halfcheetah-medium-expert-v2")
    batch_size: int = 512
    num_epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    hidden_dims: Tuple[int, ...] = (512, 512)
    activation: str = "silu"
    time_dependent: bool = True
    solver: str = "dopri5"
    t0: float = 0.0
    t1: float = 1.0
    rtol: float = 1e-5
    atol: float = 1e-5
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 100
    checkpoint_every: int = 0
    config_path: str = ""


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Load dataset from task name or npz path
    if cfg.task is not None:
        print(f"Loading dataset from D4RL task: {cfg.task}")
        dataset = D4RLTargetDataset(cfg.task)
    elif cfg.npz_path is not None:
        print(f"Loading dataset from file: {cfg.npz_path}")
        dataset = NPZTargetDataset(cfg.npz_path)
    else:
        raise ValueError("Either --task or --npz must be specified")

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    odefunc = ODEFunc(
        dim=dataset.target_dim,
        hidden_dims=cfg.hidden_dims,
        activation=cfg.activation,
        time_dependent=cfg.time_dependent,
    ).to(cfg.device)

    flow = ContinuousNormalizingFlow(
        func=odefunc,
        t0=cfg.t0,
        t1=cfg.t1,
        solver=cfg.solver,
        rtol=cfg.rtol,
        atol=cfg.atol,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(flow.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    global_step = 0
    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0
        for batch in loader:
            x = batch.to(cfg.device)
            log_px = flow.log_prob(x)
            loss = -log_px.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            if (global_step + 1) % cfg.log_every == 0:
                print(
                    f"step {global_step+1}  epoch {epoch+1}/{cfg.num_epochs}  "
                    f"loss {loss.item():.6f}  mean_logp {log_px.mean().item():.6f}"
                )
            global_step += 1

        epoch_loss /= len(dataset)
        print(f"[CNF] epoch {epoch+1}/{cfg.num_epochs}  NLL {epoch_loss:.6f}")

        if cfg.checkpoint_every and ((epoch + 1) % cfg.checkpoint_every == 0):
            ckpt = {
                "model_state_dict": flow.state_dict(),
                "cfg": cfg.__dict__,
                "target_dim": dataset.target_dim,
            }
            torch.save(ckpt, os.path.join(cfg.out_dir, f"checkpoint_epoch_{epoch+1}.pt"))

    # Compute and save metadata (threshold, mean, std) for OOD detection
    print("Computing metadata for OOD detection...")
    flow.eval()
    all_log_probs = []
    with torch.no_grad():
        for batch in loader:
            x = batch.to(cfg.device)
            log_px = flow.log_prob(x)
            all_log_probs.append(log_px.cpu().numpy())
    all_log_probs = np.concatenate(all_log_probs)

    # Set threshold as 1st percentile (marking 1% as anomalies)
    threshold = float(np.percentile(all_log_probs, 1.0))
    mean_logp = float(np.mean(all_log_probs))
    std_logp = float(np.std(all_log_probs))

    metadata = {
        'threshold': threshold,
        'mean': mean_logp,
        'std': std_logp,
        'device': cfg.device,
        'target_dim': dataset.target_dim,
        'hidden_dims': cfg.hidden_dims,
        'activation': cfg.activation,
        'time_dependent': cfg.time_dependent,
        'solver': cfg.solver,
        't0': cfg.t0,
        't1': cfg.t1,
        'rtol': cfg.rtol,
        'atol': cfg.atol,
    }

    # Save model.pt with embedded metadata
    model_ckpt = {
        'model_state_dict': flow.state_dict(),
        'threshold': threshold,
        'mean': mean_logp,
        'std': std_logp,
        'target_dim': dataset.target_dim,
        'cfg': cfg.__dict__,
    }
    torch.save(model_ckpt, os.path.join(cfg.out_dir, "model.pt"))

    # Also save metadata.pkl for backwards compatibility
    metadata_path = os.path.join(cfg.out_dir, "metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"Model saved to: {os.path.join(cfg.out_dir, 'model.pt')}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Mean log prob: {mean_logp:.4f}")
    print(f"  Std log prob: {std_logp:.4f}")


def parse_args() -> TrainConfig:
    config_only = argparse.ArgumentParser(add_help=False)
    config_only.add_argument("--config", type=str, default="")
    known, _ = config_only.parse_known_args()

    yaml_defaults = {}
    if getattr(known, "config", "") and yaml is not None:
        try:
            with open(known.config, "r") as f:
                y = yaml.safe_load(f)
            if isinstance(y, dict):
                yaml_defaults = y
        except Exception as e:
            print(f"Failed to load YAML config: {e}")

    def dget(key, default):
        return yaml_defaults.get(key, default)

    parser = argparse.ArgumentParser(
        description="Train a neural ODE (CNF) density model on NPZ data or D4RL task",
        parents=[config_only],
    )
    # Either --npz or --task must be provided
    has_data_source = "npz" in yaml_defaults or "task" in yaml_defaults
    parser.add_argument("--npz", required=False, default=dget("npz", None),
                        help="Path to NPZ/pkl dataset file")
    parser.add_argument("--task", required=False, default=dget("task", None),
                        help="D4RL task name (e.g., halfcheetah-medium-expert-v2)")
    parser.add_argument("--out", required=("out" not in yaml_defaults), default=dget("out", None))
    parser.add_argument("--epochs", type=int, default=dget("epochs", 200))
    parser.add_argument("--batch", type=int, default=dget("batch", 512))
    parser.add_argument("--lr", type=float, default=dget("lr", 1e-3))
    parser.add_argument("--wd", type=float, default=dget("wd", 0.0))
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=dget("hidden_dims", [512, 512]))
    parser.add_argument("--activation", type=str, default=dget("activation", "silu"), choices=["silu", "tanh"])
    parser.add_argument("--time-dependent", dest="time_dependent", action="store_true")
    parser.add_argument("--no-time-dependent", dest="time_dependent", action="store_false")
    parser.add_argument("--solver", type=str, default=dget("solver", "dopri5"))
    parser.add_argument("--t0", type=float, default=dget("t0", 0.0))
    parser.add_argument("--t1", type=float, default=dget("t1", 1.0))
    parser.add_argument("--rtol", type=float, default=dget("rtol", 1e-5))
    parser.add_argument("--atol", type=float, default=dget("atol", 1e-5))
    parser.add_argument("--seed", type=int, default=dget("seed", 0))
    parser.add_argument("--device", type=str, default=dget("device", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--log-every", type=int, default=dget("log_every", 100))
    parser.add_argument("--ckpt-every", type=int, default=dget("checkpoint_every", 0))

    parser.set_defaults(time_dependent=dget("time_dependent", True))

    args = parser.parse_args()

    # Validate that either npz or task is provided
    if args.npz is None and args.task is None:
        parser.error("Either --npz or --task must be specified")

    print("Parsed configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    hidden_dims = tuple(args.hidden_dims) if isinstance(args.hidden_dims, list) else tuple([args.hidden_dims])

    return TrainConfig(
        npz_path=args.npz,
        out_dir=args.out,
        task=args.task,
        num_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=args.wd,
        hidden_dims=hidden_dims,
        activation=args.activation,
        time_dependent=args.time_dependent,
        solver=args.solver,
        t0=args.t0,
        t1=args.t1,
        rtol=args.rtol,
        atol=args.atol,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
        checkpoint_every=args.ckpt_every,
        config_path=getattr(args, "config", ""),
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)


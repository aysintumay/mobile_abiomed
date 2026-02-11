import argparse
import math
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
 
try:
    import wandb  # type: ignore
except Exception:
    wandb = None

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


# -----------------------------
# Data
# -----------------------------


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
        # Expect common key names; fall back heuristics
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

        def pick(keys):
            for k in keys:
                try:
                    if k in data:
                        return data[k]
                except Exception:
                    # For NpzFile mapping interface
                    try:
                        if hasattr(data, "files") and k in data.files:
                            return data[k]
                    except Exception:
                        pass
            raise KeyError(
                f"Could not find any of keys {keys} in {available_keys(data)}. "
                "Please rename your arrays or pass a small shim."
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


# -----------------------------
# Model: Unconditional noise predictor (epsilon net)
# -----------------------------


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps expected in [0, T), shape [batch]
        half_dim = self.embedding_dim // 2
        device = timesteps.device
        exponent = -math.log(10000.0) / max(1, (half_dim - 1))
        freqs = torch.exp(torch.arange(half_dim, device=device) * exponent)
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class UnconditionalEpsilonMLP(nn.Module):
    def __init__(
        self,
        target_dim: int,
        hidden_dim: int = 512,
        time_embed_dim: int = 128,
        num_hidden_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        layers = []
        in_dim = target_dim + time_embed_dim
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, target_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_t: [B, target_dim], t: [B]
        t_emb = self.time_embed(t)
        h = torch.cat([x_t, t_emb], dim=1)
        h = self.backbone(h)
        eps = self.head(h)
        return eps


# -----------------------------
# Model: Transformer variant
# -----------------------------


class UnconditionalEpsilonTransformer(nn.Module):
    def __init__(
        self,
        target_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        time_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.target_dim = target_dim
        self.d_model = d_model

        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_to_model = nn.Linear(time_embed_dim, d_model)

        # Represent x_t as a sequence of length L = target_dim with scalar tokens → project to d_model
        self.x_proj = nn.Linear(1, d_model)
        # Positional embedding for L positions
        self.pos_embed = nn.Parameter(torch.zeros(1, target_dim, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Predict per-position noise and then flatten back to vector
        self.out_head = nn.Linear(d_model, 1)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        bsz, dim = x_t.shape
        assert dim == self.target_dim, "x_t must have target_dim features"

        # Build token sequence for x_t: shape [B, L, 1] → proj to [B, L, d_model]
        x_tokens = x_t.unsqueeze(-1)
        x_tokens = self.x_proj(x_tokens)

        # Time embedding added to all tokens
        t_emb = self.time_to_model(self.time_embed(t)).unsqueeze(1)  # [B,1,d_model]
        tokens = x_tokens + self.pos_embed + t_emb

        h = self.encoder(tokens)
        eps_tokens = self.out_head(h)  # [B, L, 1]
        eps = eps_tokens.squeeze(-1)  # [B, L]
        return eps


# -----------------------------
# Training
# -----------------------------


@dataclass
class TrainConfig:
    npz_path: str
    out_dir: str
    batch_size: int = 512
    num_epochs: int = 50
    lr: float = 2e-4
    weight_decay: float = 0.0
    hidden_dim: int = 512
    time_embed_dim: int = 128
    num_hidden_layers: int = 3
    dropout: float = 0.0
    num_train_timesteps: int = 1000
    ddim_eta: float = 0.0  # 0 = deterministic DDIM
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_type: str = "mlp"  # mlp | transformer
    # Transformer hyperparams
    d_model: int = 256
    nhead: int = 8
    tf_layers: int = 4
    ff_dim: int = 512
    # Logging
    use_wandb: bool = False
    wandb_project: str = "GORMPO"
    wandb_entity: str = ""
    wandb_run: str = ""
    log_every: int = 100
    samples_every: int = 0  # epochs; 0 disables
    # External config
    config_path: str = ""
    # Checkpointing (epoch-based)
    checkpoint_every_epochs: int = 1


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Save a copy of provided YAML config (if any)
    config_yaml = None
    if cfg.config_path:
        if yaml is None:
            print("PyYAML not installed; can't read --config. Proceeding without it.")
        else:
            try:
                with open(cfg.config_path, "r") as f:
                    config_yaml = yaml.safe_load(f)
                # Save a copy to out directory for reproducibility
                os.makedirs(cfg.out_dir, exist_ok=True)
                with open(os.path.join(cfg.out_dir, "used_config.yaml"), "w") as f:
                    yaml.safe_dump(config_yaml, f, sort_keys=False)
            except Exception as e:
                print(f"Failed to load YAML config: {e}")

    dataset = NPZTargetDataset(cfg.npz_path)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    if cfg.model_type == "mlp":
        model = UnconditionalEpsilonMLP(
            target_dim=dataset.target_dim,
            hidden_dim=cfg.hidden_dim,
            time_embed_dim=cfg.time_embed_dim,
            num_hidden_layers=cfg.num_hidden_layers,
            dropout=cfg.dropout,
        ).to(cfg.device)
    elif cfg.model_type == "transformer":
        model = UnconditionalEpsilonTransformer(
            target_dim=dataset.target_dim,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.tf_layers,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            time_embed_dim=cfg.time_embed_dim,
        ).to(cfg.device)
    else:
        raise ValueError("--model must be 'mlp' or 'transformer'")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Use diffusers' DDIMScheduler. We train with an epsilon objective as usual.
    scheduler = DDIMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule="linear",
        prediction_type="epsilon",
    )

    # Move scheduler config to disk for later sampling reuse
    scheduler.save_pretrained(os.path.join(cfg.out_dir, "scheduler"))

    # Weights & Biases logging (optional)
    wb_run = None
    if cfg.use_wandb:
        if wandb is None:
            print("wandb not installed; disable --wandb or pip install wandb")
        else:
            wb_kwargs = {
                "project": cfg.wandb_project,
                "config": {**cfg.__dict__, "target_dim": dataset.target_dim},
            }
            if cfg.wandb_entity:
                wb_kwargs["entity"] = cfg.wandb_entity
            if cfg.wandb_run:
                wb_kwargs["name"] = cfg.wandb_run
            wb_run = wandb.init(**wb_kwargs)
            wandb.watch(model, log="gradients", log_freq=cfg.log_every)
            if config_yaml is not None:
                # Log YAML parameters as a nested config
                try:
                    wandb.config.update({"yaml_config": config_yaml}, allow_val_change=True)
                except Exception:
                    pass

    model.train()
    global_step = 0
    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0
        step = 0
        for target in loader:
            clean = target.to(cfg.device)

            # Sample random timesteps per batch element
            bsz = clean.size(0)
            t = torch.randint(0, cfg.num_train_timesteps, (bsz,), device=cfg.device)

            noise = torch.randn_like(clean)
            # Add noise according to the forward process
            noisy = scheduler.add_noise(clean, noise, t)

            pred_noise = model(noisy, t)
            loss = torch.nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * bsz
            step += 1
            global_step += 1
            if cfg.use_wandb and wandb is not None and (step % max(1, cfg.log_every) == 0):
                wandb.log({
                    "loss/train_iter": loss.item(),
                    "epoch": epoch + 1,
                    "step": global_step,
                })

        epoch_loss /= len(dataset)
        print(f"epoch {epoch+1}/{cfg.num_epochs}  loss {epoch_loss:.6f}")
        if cfg.use_wandb and wandb is not None:
            wandb.log({"loss/train_epoch": epoch_loss, "epoch": epoch + 1})

        # Lightweight checkpoint each epoch
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "cfg": cfg.__dict__,
                "target_dim": dataset.target_dim,
            },
            os.path.join(cfg.out_dir, "checkpoint.pt"),
        )

        # Optional sampling preview to wandb
        if cfg.samples_every and ((epoch + 1) % cfg.samples_every == 0):
            try:
                model.eval()
                with torch.no_grad():
                    num_samples = min(16, cfg.batch_size)
                    sch = DDIMScheduler.from_pretrained(os.path.join(cfg.out_dir, "scheduler"))
                    preds = sample(
                        model=model,
                        scheduler=sch,
                        num_samples=num_samples,
                        num_inference_steps=50,
                        eta=0.0,
                        device=cfg.device,
                    ).detach().cpu().numpy()
                # Save to disk and log to wandb if available
                npy_path = os.path.join(cfg.out_dir, f"samples_epoch_{epoch+1}.npy")
                np.save(npy_path, preds)
                if cfg.use_wandb and wandb is not None:
                    wandb.log({"samples/npy": wandb.Artifact(f"samples_epoch_{epoch+1}", type="dataset")})
            except Exception:
                pass

        # Epoch-level checkpointing
        if cfg.checkpoint_every_epochs and ((epoch + 1) % cfg.checkpoint_every_epochs == 0):
            ckpt_dir = os.path.join(cfg.out_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "cfg": cfg.__dict__,
                    "target_dim": dataset.target_dim,
                },
                ckpt_path,
            )

    # Final save
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model.pt"))


# -----------------------------
# Sampling (unconditional generation) using deterministic DDIM
# -----------------------------


@torch.no_grad()
def sample(
    model: nn.Module,
    scheduler: DDIMScheduler,
    num_samples: int = 1,
    num_inference_steps: int = 50,
    eta: float = 0.0,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()

    scheduler.set_timesteps(num_inference_steps)
    # Get target_dim from model
    if hasattr(model, "target_dim"):
        target_dim = model.target_dim
    elif hasattr(model, "head"):
        target_dim = model.head.out_features
    else:
        raise ValueError("Cannot determine target_dim from model")
    
    x = torch.randn(num_samples, target_dim, device=device)

    for t in scheduler.timesteps:
        # Predict noise
        eps = model(x, t.expand(num_samples))
        # DDIM step
        out = scheduler.step(model_output=eps, timestep=t, sample=x, eta=eta)
        x = out.prev_sample
    return x


# -----------------------------
# Density Estimation: Computing log p(x) from diffusion model
# -----------------------------


def gaussian_kl(mean1: torch.Tensor, var1: torch.Tensor, mean2: torch.Tensor, var2: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between two diagonal Gaussian distributions.
    KL(N(mean1, var1) || N(mean2, var2)) for diagonal covariances.
    """
    eps = 1e-12
    var1 = torch.clamp(var1, min=eps)
    var2 = torch.clamp(var2, min=eps)
    return 0.5 * (
        ((mean2 - mean1) ** 2) / var2 + (var1 / var2) - 1.0 + torch.log(var2) - torch.log(var1)
    ).sum(dim=1)


@torch.no_grad()
def log_prob_elbo(
    model: nn.Module,
    scheduler: DDIMScheduler,
    x0: torch.Tensor,
    device: str = "cpu",
    num_inference_steps: int = None,
) -> torch.Tensor:
    """
    Compute log p(x0) using the ELBO (Evidence Lower Bound) from the diffusion process.

    This gives a lower bound on log p(x0) by using the variational bound:
    log p(x0) >= ELBO = E_q[log p(x0|x1)] - sum_t KL(q(x_{t-1}|x_t,x0) || p_theta(x_{t-1}|x_t)) - KL(q(x_T|x0) || p(x_T))

    Args:
        model: Unconditional epsilon prediction model
        scheduler: DDIMScheduler (or DDPMScheduler) with diffusion parameters
        x0: Data samples of shape [batch_size, target_dim]
        device: Device to run computation on
        num_inference_steps: Number of timesteps to use (default: None = use all T timesteps)
            If specified, uniformly subsample timesteps for faster approximation

    Returns:
        log_prob: Lower bound on log p(x0) for each sample, shape [batch_size]
    """
    model.eval()
    x0 = x0.to(device)
    
    # Get diffusion parameters
    betas = scheduler.betas.to(device).to(torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    T = betas.shape[0]
    bsz, dim = x0.shape

    # Determine which timesteps to use
    if num_inference_steps is not None and num_inference_steps < T:
        # Uniformly subsample timesteps for faster approximation
        # Use linspace to get evenly spaced timesteps
        timesteps = torch.linspace(0, T - 1, num_inference_steps, dtype=torch.long, device=device)
        # Ensure we include T-1 (last timestep) and 0 (first timestep)
        if timesteps[-1] != T - 1:
            timesteps[-1] = T - 1
        if timesteps[0] != 0:
            timesteps[0] = 0
        # Convert to list and add 1 (since loop uses 1-indexed)
        timestep_list = [int(t) + 1 for t in reversed(timesteps.cpu().tolist())]
    else:
        # Use all timesteps (original behavior)
        timestep_list = list(reversed(range(1, T + 1)))

    # Sample a single q trajectory for expectation (Monte Carlo)
    eps = torch.randn(bsz, dim, device=device)
    x_t = torch.sqrt(alphas_cumprod[-1]) * x0 + torch.sqrt(1.0 - alphas_cumprod[-1]) * eps

    total = torch.zeros(bsz, device=device)

    # L_T = KL(q(x_T|x0) || N(0, I))
    mean_T = torch.sqrt(alphas_cumprod[-1]) * x0
    var_T = (1.0 - alphas_cumprod[-1]) * torch.ones_like(x0)
    mean_p = torch.zeros_like(x0)
    var_p = torch.ones_like(x0)
    total = total + gaussian_kl(mean_T, var_T, mean_p, var_p)

    # Loop through selected timesteps for KL terms, and handle t=1 as NLL of p_theta(x0|x1)
    for t in timestep_list:
        at = alphas[t - 1]
        a_bar_t = alphas_cumprod[t - 1]
        if t > 1:
            a_bar_prev = alphas_cumprod[t - 2]
        else:
            a_bar_prev = torch.tensor(1.0, device=device)
        
        beta_t = betas[t - 1]
        # Posterior variance (tilde beta_t) with clamp
        beta_t_tilde = (1.0 - a_bar_prev) / (1.0 - a_bar_t) * beta_t
        beta_t_tilde = torch.clamp(beta_t_tilde, min=1e-20)
        
        # Predict epsilon at timestep t
        t_batch = torch.full((bsz,), t - 1, device=device, dtype=torch.long)
        eps_theta = model(x_t, t_batch)
        
        # Predict mean of p_theta(x_{t-1}|x_t) for epsilon parameterization
        coef1 = 1.0 / torch.sqrt(at)
        coef2 = beta_t / torch.sqrt(1.0 - a_bar_t)
        mu_theta = coef1 * (x_t - coef2 * eps_theta)
        
        # True posterior q(x_{t-1}|x_t, x0)
        mu_q = (
            (torch.sqrt(a_bar_prev) * beta_t / (1.0 - a_bar_t)) * x0
            + (torch.sqrt(at) * (1.0 - a_bar_prev) / (1.0 - a_bar_t)) * x_t
        )
        var_q = beta_t_tilde * torch.ones_like(x0)
        
        if t > 1:
            # KL term
            total = total + gaussian_kl(mu_q, var_q, mu_theta, var_q)
            # Sample x_{t-1} from q to continue the trajectory
            noise = torch.randn_like(x0)
            x_t = mu_q + torch.sqrt(beta_t_tilde) * noise
        else:
            # Reconstruction term: -log p_theta(x0|x1)
            # At t=1, beta_t_tilde → 0, so use beta_1 instead as in DDPM paper
            # This avoids numerical instability from dividing by near-zero variance
            var = beta_t  # Use beta_1 directly instead of posterior variance
            var = torch.clamp(var, min=1e-20)
            # Proper Gaussian NLL formula: NLL = 0.5 * (log(2πσ²) + (x-μ)²/σ²) summed over dims
            # Since var is scalar (same for all dims), factor out the log term
            nll0 = 0.5 * (
                dim * torch.log(2 * torch.pi * var) +  # Constant log term × dim
                ((x0 - mu_theta) ** 2 / var).sum(dim=1)  # Squared error term summed
            )
            total = total + nll0
    
    # ELBO is a lower bound, so we return it as log_prob (negative NLL)
    log_prob = -total
    return log_prob  # shape [bsz]


@torch.no_grad()
def score_function(
    model: nn.Module,
    scheduler: DDIMScheduler,
    x_t: torch.Tensor,
    t: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute the score function (gradient of log probability) at timestep t.
    
    The score function is: ∇_x log p_t(x_t) = -eps_theta(x_t, t) / sqrt(1 - alpha_bar_t)
    
    Args:
        model: Unconditional epsilon prediction model
        scheduler: DDIMScheduler with diffusion parameters
        x_t: Noisy samples at timestep t, shape [batch_size, target_dim]
        t: Timestep values, shape [batch_size] with values in [0, T-1]
        device: Device to run computation on
        
    Returns:
        score: Score function values, shape [batch_size, target_dim]
    """
    model.eval()
    x_t = x_t.to(device)
    t = t.to(device)
    
    # Get diffusion parameters
    betas = scheduler.betas.to(device).to(torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Predict epsilon
    eps_theta = model(x_t, t)
    
    # Compute alpha_bar_t for each sample
    # t is in [0, T-1], so we need to index into alphas_cumprod
    t_idx = t.long()
    t_idx = torch.clamp(t_idx, 0, len(alphas_cumprod) - 1)
    alpha_bar_t = alphas_cumprod[t_idx].unsqueeze(-1)  # [batch_size, 1]
    
    # Score = -eps / sqrt(1 - alpha_bar_t)
    score = -eps_theta / torch.sqrt(1.0 - alpha_bar_t + 1e-8)
    return score


@torch.no_grad()
def log_prob(
    model: nn.Module,
    scheduler: DDIMScheduler,
    x0: torch.Tensor,
    device: str = "cpu",
    method: str = "elbo",
) -> torch.Tensor:
    """
    Compute log p(x0) from the unconditional diffusion model.
    
    Args:
        model: Unconditional epsilon prediction model
        scheduler: DDIMScheduler with diffusion parameters
        x0: Data samples of shape [batch_size, target_dim]
        device: Device to run computation on
        method: Method to use. Currently only "elbo" is supported.
        
    Returns:
        log_prob: Log probability (or lower bound) for each sample, shape [batch_size]
    """
    if method == "elbo":
        return log_prob_elbo(model, scheduler, x0, device)
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods: 'elbo'")


def evaluate_log_prob(
    model: nn.Module,
    scheduler: DDIMScheduler,
    data: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 512,
) -> Tuple[torch.Tensor, dict]:
    """
    Evaluate log probabilities for a dataset.
    
    Args:
        model: Unconditional epsilon prediction model
        scheduler: DDIMScheduler with diffusion parameters
        data: Dataset tensor of shape [num_samples, target_dim]
        device: Device to run computation on
        batch_size: Batch size for evaluation
        
    Returns:
        log_probs: Log probabilities for each sample, shape [num_samples]
        stats: Dictionary with statistics (mean, std, min, max)
    """
    model.eval()
    data = data.to(device)
    
    log_probs_list = []
    num_samples = data.shape[0]
    
    for i in range(0, num_samples, batch_size):
        batch = data[i:i + batch_size]
        batch_log_probs = log_prob(model, scheduler, batch, device)
        log_probs_list.append(batch_log_probs)
    
    log_probs = torch.cat(log_probs_list, dim=0)
    
    stats = {
        "mean": log_probs.mean().item(),
        "std": log_probs.std().item(),
        "min": log_probs.min().item(),
        "max": log_probs.max().item(),
    }
    
    return log_probs, stats


def parse_args() -> TrainConfig:
    # Stage 1: parse only --config to get YAML path
    config_only = argparse.ArgumentParser(add_help=False)
    config_only.add_argument("--config", type=str, default="")
    known, _ = config_only.parse_known_args()

    # Load YAML if provided and available
    yaml_defaults = {}
    if getattr(known, "config", "") and yaml is not None:
        try:
            with open(known.config, "r") as f:
                y = yaml.safe_load(f)
            if isinstance(y, dict):
                yaml_defaults = y
        except Exception as e:
            print(f"Warning: failed to read YAML config: {e}")

    # Stage 2: Build full parser, inject YAML values as defaults so CLI overrides them
    parser = argparse.ArgumentParser(description="Train DDIM (diffusers) unconditional on NPZ", parents=[config_only])

    def dget(key, default):
        return yaml_defaults.get(key, default)

    parser.add_argument("--npz", required=("npz" not in yaml_defaults), default=yaml_defaults.get("npz", None), help="Path to data .npz file")
    parser.add_argument("--out", required=("out" not in yaml_defaults), default=yaml_defaults.get("out", None), help="Output directory")
    parser.add_argument("--epochs", type=int, default=dget("epochs", 50))
    parser.add_argument("--batch", type=int, default=dget("batch", 512))
    parser.add_argument("--lr", type=float, default=dget("lr", 2e-4))
    parser.add_argument("--wd", type=float, default=dget("wd", 0.0))
    parser.add_argument("--hidden", type=int, default=dget("hidden", 512))
    parser.add_argument("--time-emb", type=int, default=dget("time_emb", 128))
    parser.add_argument("--layers", type=int, default=dget("layers", 3))
    parser.add_argument("--dropout", type=float, default=dget("dropout", 0.0))
    parser.add_argument("--timesteps", type=int, default=dget("timesteps", 1000))
    parser.add_argument("--seed", type=int, default=dget("seed", 0))
    parser.add_argument("--device", type=str, default=dget("device", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--model", type=str, choices=["mlp", "transformer"], default=dget("model", "mlp"))
    parser.add_argument("--d-model", type=int, default=dget("d_model", 256))
    parser.add_argument("--nhead", type=int, default=dget("nhead", 8))
    parser.add_argument("--tf-layers", type=int, default=dget("tf_layers", 4))
    parser.add_argument("--ff-dim", type=int, default=dget("ff_dim", 512))
    # Logging
    parser.add_argument("--wandb", action="store_true", default=dget("wandb", False))
    parser.add_argument("--wandb-project", type=str, default=dget("wandb_project", "GORMPO"))
    parser.add_argument("--wandb-entity", type=str, default=dget("wandb_entity", ""))
    parser.add_argument("--wandb-run", type=str, default=dget("wandb_run", ""))
    parser.add_argument("--log-every", type=int, default=dget("log_every", 100))
    parser.add_argument("--samples-every", type=int, default=dget("samples_every", 0))
    parser.add_argument("--ckpt-every-epochs", type=int, default=dget("checkpoint_every_epochs", 1))

    args = parser.parse_args()
    
    # Print all parsed configs (arguments)
    print("Parsed configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    return TrainConfig(
        npz_path=args.npz,
        out_dir=args.out,
        num_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=args.wd,
        hidden_dim=args.hidden,
        time_embed_dim=args.time_emb,
        num_hidden_layers=args.layers,
        dropout=args.dropout,
        num_train_timesteps=args.timesteps,
        seed=args.seed,
        device=args.device,
        model_type=args.model,
        d_model=args.d_model,
        nhead=args.nhead,
        tf_layers=args.tf_layers,
        ff_dim=args.ff_dim,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run=args.wandb_run,
        log_every=args.log_every,
        samples_every=args.samples_every,
        config_path=getattr(args, "config", ""),
        checkpoint_every_epochs=args.ckpt_every_epochs,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)


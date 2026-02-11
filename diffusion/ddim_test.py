import argparse
import os
from typing import Tuple

import numpy as np
import torch
from torch import nn

from diffusers.schedulers.scheduling_ddim import DDIMScheduler

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    import joblib  # type: ignore
except Exception:
    joblib = None


from ddim_training import (
    ConditionalEpsilonMLP,
    ConditionalEpsilonTransformer,
)


def gaussian_kl(mean1: torch.Tensor, var1: torch.Tensor, mean2: torch.Tensor, var2: torch.Tensor) -> torch.Tensor:
    # All diagonal covariances; numerically stable with clamping
    eps = 1e-12
    var1 = torch.clamp(var1, min=eps)
    var2 = torch.clamp(var2, min=eps)
    return 0.5 * (
        ((mean2 - mean1) ** 2) / var2 + (var1 / var2) - 1.0 + torch.log(var2) - torch.log(var1)
    ).sum(dim=1)


def build_model_from_ckpt(ckpt_path: str, device: str) -> Tuple[nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    target_dim = ckpt.get("target_dim")
    cond_dim = ckpt.get("cond_dim")
    model_type = cfg.get("model_type", "mlp")
    time_embed_dim = cfg.get("time_embed_dim", 128)

    if model_type == "mlp":
        model = ConditionalEpsilonMLP(
            target_dim=target_dim,
            cond_dim=cond_dim,
            hidden_dim=cfg.get("hidden_dim", 512),
            time_embed_dim=time_embed_dim,
            num_hidden_layers=cfg.get("num_hidden_layers", 3),
            dropout=cfg.get("dropout", 0.0),
        )
    else:
        model = ConditionalEpsilonTransformer(
            target_dim=target_dim,
            cond_dim=cond_dim,
            d_model=cfg.get("d_model", 256),
            nhead=cfg.get("nhead", 8),
            num_layers=cfg.get("tf_layers", 4),
            dim_feedforward=cfg.get("ff_dim", 512),
            dropout=cfg.get("dropout", 0.1),
            time_embed_dim=time_embed_dim,
        )

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, cfg


def estimate_nll_bound(
    model: nn.Module,
    scheduler: DDIMScheduler,
    cond: torch.Tensor,
    x0: torch.Tensor,
    device: str,
) -> torch.Tensor:
    # Uses DDPM ELBO (variational) bound with fixed variance equal to posterior variance
    betas = scheduler.betas.to(device).to(torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    T = betas.shape[0]
    bsz, dim = x0.shape

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

    # Loop t=T..2 for KL terms, and handle t=1 as NLL of p_theta(x0|x1)
    for t in reversed(range(1, T + 1)):
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
        eps_theta = model(x_t, cond, t_batch)

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

    return total  # shape [bsz]


@torch.no_grad()
def ddim_deterministic_sample(
    model: nn.Module,
    scheduler: DDIMScheduler,
    cond: torch.Tensor,
    target_shape: torch.Size,
    num_inference_steps: int = 50,
    device: str = "cpu",
) -> torch.Tensor:
    # Deterministic DDIM (eta=0) sampling for cond→target
    scheduler.set_timesteps(num_inference_steps)
    x = torch.randn(target_shape, device=device)
    for t in scheduler.timesteps:
        t_batch = t.to(device).expand(cond.size(0)).long()
        eps = model(x, cond, t_batch)
        eps = torch.nan_to_num(eps, nan=0.0, posinf=0.0, neginf=0.0)
        out = scheduler.step(model_output=eps, timestep=t, sample=x, eta=0.0)
        x = out.prev_sample
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def inverse_scale_target(x_norm: torch.Tensor, mean_arr: np.ndarray | None, std_arr: np.ndarray | None) -> torch.Tensor:
    if mean_arr is None or std_arr is None:
        return x_norm
    mean_t = torch.as_tensor(mean_arr, dtype=x_norm.dtype, device=x_norm.device).view(1, -1)
    std_t = torch.as_tensor(std_arr, dtype=x_norm.dtype, device=x_norm.device).view(1, -1)
    return x_norm * std_t + mean_t


def main():
    # Stage 1: parse only --config
    config_only = argparse.ArgumentParser(add_help=False)
    config_only.add_argument("--config", type=str, default="", help="YAML with paths and defaults")
    known, _ = config_only.parse_known_args()

    # Load YAML defaults if provided
    yaml_defaults = {}
    if getattr(known, "config", "") and yaml is not None:
        try:
            with open(known.config, "r") as f:
                y = yaml.safe_load(f)
            if isinstance(y, dict):
                yaml_defaults = y
        except Exception:
            pass

    def dget(key, default):
        return yaml_defaults.get(key, default)

    # Stage 2: full parser with YAML-provided defaults (CLI overrides)
    parser = argparse.ArgumentParser(
        description="Evaluate NLL (variational bound) on test NPZ",
        parents=[config_only],
    )
    parser.add_argument("--model-dir", type=str, default=dget("out", ""), help="Dir with checkpoint.pt and scheduler/")
    parser.add_argument("--test-npz", type=str, default=dget("test_npz", ""), help="Path to test NPZ with X_cond/X_target")
    parser.add_argument("--scaler", type=str, default=dget("scaler", ""), help="Joblib StandardScaler for de-normalization adj")
    parser.add_argument("--num-samples", type=int, default=dget("num_samples", 10))
    parser.add_argument("--inference-steps", type=int, default=dget("inference_steps", None) or dget("timesteps", 1000))
    parser.add_argument("--device", type=str, default=dget("device", "cuda" if torch.cuda.is_available() else "cpu"))

    args = parser.parse_args()

    model_dir = args.model_dir
    test_npz = args.test_npz
    scaler_path = args.scaler
    device = args.device

    if not model_dir:
        raise ValueError("Please provide --model-dir or put 'out' in the YAML config used for training.")
    ckpt_path = os.path.join(model_dir, "checkpoint.pt")
    sched_dir = os.path.join(model_dir, "scheduler")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint at {ckpt_path}")
    if not os.path.exists(sched_dir):
        raise FileNotFoundError(f"Missing scheduler directory at {sched_dir}")
    if not test_npz:
        raise ValueError("Please provide --test-npz or 'test_npz' in YAML config.")

    # Load test data
    data = np.load(test_npz)
    # Accept keys X_cond/X_target or cond/target
    cond = data["X_cond"] if "X_cond" in data else data["cond"]
    target = data["X_target"] if "X_target" in data else data["target"]
    N = min(args.num_samples, cond.shape[0])
    cond = torch.from_numpy(cond[:N]).float().view(N, -1).to(device)
    x0 = torch.from_numpy(target[:N]).float().view(N, -1).to(device)

    # Load scaler (optional) for original-scale NLL adjustment and rel-L2 on original scale
    std_sum_log = 0.0
    mean_target_arr = None
    std_target_arr = None
    if scaler_path:
        if joblib is None:
            print("joblib not installed; skipping scaler adjustment")
        else:
            try:
                scaler = joblib.load(scaler_path)
                # StandardScaler: x_norm = (x - mean)/std. log|det J| = -sum log std, so
                # log p_orig = log p_norm - sum log std; hence NLL_orig = NLL_norm + sum log std
                std = np.asarray(getattr(scaler, "scale_", None))
                if std is None:
                    std = np.asarray(getattr(scaler, "std_", None))
                if std is not None:
                    std = np.asarray(std, dtype=np.float64).reshape(-1)
                    # Replace zeros/negatives to avoid -inf/NaN in log-det; treat as no scaling
                    std = np.where(std <= 1e-12, 1.0, std)
                    # Use target dimensions only; if scaler fitted on combined dims, assume last dims match target size
                    target_dim = x0.shape[1]
                    std_target = std[-target_dim:]
                    std_sum_log = float(np.log(std_target).sum())
                    # Also fetch mean for inverse-transform on targets
                    mean = np.asarray(getattr(scaler, "mean_", None))
                    if mean is not None:
                        mean = np.asarray(mean, dtype=np.float64).reshape(-1)
                        mean_target_arr = mean[-target_dim:]
                    std_target_arr = std_target
                else:
                    print("Scaler missing scale_/std_; skipping adjustment")
            except Exception as e:
                print(f"Failed to load scaler: {e}")

    # Build model and scheduler
    model, train_cfg = build_model_from_ckpt(ckpt_path, device)
    scheduler = DDIMScheduler.from_pretrained(sched_dir)

    # Relative L2 error using deterministic DDIM samples
    with torch.no_grad():
        x_pred = ddim_deterministic_sample(
            model=model,
            scheduler=scheduler,
            cond=cond,
            target_shape=x0.shape,
            num_inference_steps=args.inference_steps,
            device=device,
        )

    # Optionally invert scaling for both x0 and x_pred to original scale
    x0_orig = inverse_scale_target(x0, mean_target_arr, std_target_arr)
    x_pred_orig = inverse_scale_target(x_pred, mean_target_arr, std_target_arr)

    # rel-L2 = ||pred - gt||2 / max(||gt||2, eps)
    eps = 1e-12
    # Sanitize any non-finite values before computing norms
    x0_orig = torch.nan_to_num(x0_orig, nan=0.0, posinf=1e6, neginf=-1e6)
    x_pred_orig = torch.nan_to_num(x_pred_orig, nan=0.0, posinf=1e6, neginf=-1e6)

    num = torch.linalg.vector_norm(x_pred_orig - x0_orig, dim=1)
    den = torch.maximum(torch.linalg.vector_norm(x0_orig, dim=1), torch.tensor(eps, device=device))
    rel_l2 = (num / den).mean().item()
    print(f"Relative L2 error (mean over {N} samples): {rel_l2:.6f}")


if __name__ == "__main__":
    main()



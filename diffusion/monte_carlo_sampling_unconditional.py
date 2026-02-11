import argparse
import json
import os
from typing import Tuple, Optional

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable
#change the path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    import joblib  # type: ignore
except Exception:
    joblib = None


from diffusion.ddim_training_unconditional import (
    UnconditionalEpsilonMLP,
    UnconditionalEpsilonTransformer,
    log_prob_elbo,
)


def build_model_from_ckpt(ckpt_path: str, device: str) -> Tuple[nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    target_dim = ckpt.get("target_dim")
    model_type = cfg.get("model_type", "mlp")
    time_embed_dim = cfg.get("time_embed_dim", 128)

    if model_type == "mlp":
        model = UnconditionalEpsilonMLP(
            target_dim=target_dim,
            hidden_dim=cfg.get("hidden_dim", 512),
            time_embed_dim=time_embed_dim,
            num_hidden_layers=cfg.get("num_hidden_layers", 3),
            dropout=cfg.get("dropout", 0.0),
        )
    else:
        model = UnconditionalEpsilonTransformer(
            target_dim=target_dim,
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


@torch.no_grad()
def ddpm_stochastic_sample(
    model: nn.Module,
    scheduler: DDPMScheduler,
    num_samples: int,
    target_dim: int,
    num_inference_steps: int = 1000,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    DDPM stochastic sampling for unconditional generation with optional generator for reproducibility.
    """
    scheduler.set_timesteps(num_inference_steps)
    
    # Generate initial noise with optional generator
    if generator is not None:
        x = torch.randn(num_samples, target_dim, device=device, generator=generator)
    else:
        x = torch.randn(num_samples, target_dim, device=device)
    
    for t in scheduler.timesteps:
        t_batch = t.to(device).expand(num_samples).long()
        eps = model(x, t_batch)
        eps = torch.nan_to_num(eps, nan=0.0, posinf=0.0, neginf=0.0)
        # DDPM step adds noise at each timestep (stochastic)
        out = scheduler.step(model_output=eps, timestep=t, sample=x, generator=generator)
        x = out.prev_sample
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    return x


@torch.no_grad()
def ddim_deterministic_sample(
    model: nn.Module,
    scheduler: DDIMScheduler,
    num_samples: int,
    target_dim: int,
    num_inference_steps: int = 50,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    DDIM deterministic sampling (eta=0) with optional generator for initial noise.
    """
    scheduler.set_timesteps(num_inference_steps)
    
    # Generate initial noise with optional generator
    if generator is not None:
        x = torch.randn(num_samples, target_dim, device=device, generator=generator)
    else:
        x = torch.randn(num_samples, target_dim, device=device)
    
    for t in scheduler.timesteps:
        t_batch = t.to(device).expand(num_samples).long()
        eps = model(x, t_batch)
        eps = torch.nan_to_num(eps, nan=0.0, posinf=0.0, neginf=0.0)
        out = scheduler.step(model_output=eps, timestep=t, sample=x, eta=0.0)
        x = out.prev_sample
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


@torch.no_grad()
def ddpm_stochastic_sample_batch(
    model: nn.Module,
    scheduler: DDPMScheduler,
    batch_size: int,
    target_dim: int,
    num_inference_steps: int = 1000,
    device: str = "cpu",
    seed_offset: int = 0,
) -> torch.Tensor:
    """
    Batched DDPM stochastic sampling - processes multiple samples in parallel.
    
    Args:
        model: The noise prediction model
        scheduler: DDPMScheduler instance
        batch_size: Number of samples to generate in parallel
        target_dim: Dimension of target/output
        num_inference_steps: Number of denoising steps
        device: Device to run on
        seed_offset: Offset for random seed generation
        
    Returns:
        Tensor of shape (batch_size, target_dim) with all samples
    """
    scheduler.set_timesteps(num_inference_steps)
    
    # Generate initial noise for all samples with different seeds
    # Use different seeds for each sample in the batch
    x = torch.zeros(batch_size, target_dim, device=device)
    for i in range(batch_size):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed_offset + i)
        x[i] = torch.randn(target_dim, device=device, generator=generator)
    
    # Get scheduler parameters
    betas = scheduler.betas.to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Process all samples through denoising steps
    for step_idx, t in enumerate(scheduler.timesteps):
        t_batch = t.to(device).expand(batch_size).long()
        eps = model(x, t_batch)
        eps = torch.nan_to_num(eps, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get timestep index - scheduler timesteps are in descending order (999, 998, ..., 0)
        # Map to actual beta index
        t_val = int(t.item())
        
        # Use scheduler's step_index if available, otherwise compute from timestep
        if hasattr(scheduler, '_index_for_timestep'):
            t_idx = scheduler._index_for_timestep(t, scheduler.timesteps)
        else:
            # Map timestep value to index - assuming timesteps go from num_train_timesteps-1 down to 0
            t_idx = t_val
        
        # Ensure t_idx is within bounds
        t_idx = min(max(0, t_idx), len(betas) - 1)
        
        alpha_t = alphas[t_idx]
        alpha_bar_t = alphas_cumprod[t_idx]
        
        # Compute variance for this step (posterior variance)
        if t_idx > 0:
            alpha_bar_prev = alphas_cumprod[t_idx - 1]
            beta_tilde = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * betas[t_idx]
        else:
            beta_tilde = betas[t_idx]
        
        # Predict x_{t-1} mean: 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1 - alpha_bar_t) * eps)
        pred_prev_sample_mean = (1.0 / torch.sqrt(alpha_t)) * (x - (betas[t_idx] / torch.sqrt(1.0 - alpha_bar_t)) * eps)
        
        # Generate independent noise for each sample in the batch
        # Use manual seed setting for reproducibility while generating in parallel
        noise = torch.zeros_like(x)
        # Generate noise sequentially with different seeds to ensure independence
        # This is fast compared to model forward pass
        for i in range(batch_size):
            generator = torch.Generator(device=device)
            generator.manual_seed(seed_offset + i + step_idx * batch_size * 1000)
            noise[i] = torch.randn(target_dim, device=device, generator=generator)
        
        # Add noise: x_{t-1} = mean + sqrt(beta_tilde) * noise
        x = pred_prev_sample_mean + torch.sqrt(beta_tilde.clamp(min=1e-20)) * noise
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    return x


@torch.no_grad()
def ddim_deterministic_sample_batch(
    model: nn.Module,
    scheduler: DDIMScheduler,
    batch_size: int,
    target_dim: int,
    num_inference_steps: int = 50,
    device: str = "cpu",
    seed_offset: int = 0,
) -> torch.Tensor:
    """
    Batched DDIM deterministic sampling - processes multiple samples in parallel.
    Only the initial noise differs; the denoising is deterministic.
    """
    scheduler.set_timesteps(num_inference_steps)
    
    # Generate initial noise for all samples with different seeds
    x = torch.zeros(batch_size, target_dim, device=device)
    for i in range(batch_size):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed_offset + i)
        x[i] = torch.randn(target_dim, device=device, generator=generator)
    
    # Process all samples through denoising steps
    for t in scheduler.timesteps:
        t_batch = t.to(device).expand(batch_size).long()
        eps = model(x, t_batch)
        eps = torch.nan_to_num(eps, nan=0.0, posinf=0.0, neginf=0.0)
        
        # DDIM is deterministic, so we can process all at once
        out = scheduler.step(model_output=eps, timestep=t, sample=x, eta=0.0)
        x = torch.nan_to_num(out.prev_sample, nan=0.0, posinf=0.0, neginf=0.0)
    
    return x


def inverse_scale_target(x_norm: torch.Tensor,
    mean_arr: Optional[np.ndarray],
    std_arr: Optional[np.ndarray]) -> torch.Tensor:
    if mean_arr is None or std_arr is None:
        return x_norm
    mean_t = torch.as_tensor(mean_arr, dtype=x_norm.dtype, device=x_norm.device).view(1, -1)
    std_t = torch.as_tensor(std_arr, dtype=x_norm.dtype, device=x_norm.device).view(1, -1)
    return x_norm * std_t + mean_t


def calculate_nll_binned(
    samples: np.ndarray,
    target: np.ndarray,
    bin_width: float = 0.1,
    smoothing: float = 1e-10,
) -> Tuple[float, np.ndarray, dict]:
    """
    Calculate Negative Log-Likelihood (NLL) using histogram binning.
    
    Args:
        samples: Array of shape (num_samples, target_dim) with Monte Carlo samples
        target: Ground truth target of shape (target_dim,)
        bin_width: Width of each bin (default 0.1)
        smoothing: Small value added to avoid log(0) (default 1e-10)
        
    Returns:
        total_nll: Total NLL summed over all dimensions
        nll_per_dim: NLL for each dimension
        stats: Dictionary with statistics about the binning
    """
    num_samples, target_dim = samples.shape
    target = target.flatten()
    
    # Find the range of values across all samples and target
    all_values = np.concatenate([samples.flatten(), target])
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    
    # Create bins: extend slightly beyond min/max to ensure target is always within range
    margin = bin_width
    bin_edges = np.arange(min_val - margin, max_val + margin + bin_width, bin_width)
    num_bins = len(bin_edges) - 1
    
    nll_per_dim = np.zeros(target_dim)
    bin_probs_per_dim = []
    target_bins_per_dim = []
    
    for dim_idx in range(target_dim):
        # Create histogram for this dimension
        counts, bin_edges_actual = np.histogram(
            samples[:, dim_idx],
            bins=bin_edges,
            density=False
        )
        
        # Convert counts to probability density with smoothing
        # Standard histogram density: density = counts / (total_count * bin_width)
        # With smoothing: density = (counts + smoothing) / (total_smoothed * bin_width)
        # This ensures the integral of density equals 1: sum(density * bin_width) = 1
        smoothed_counts = counts + smoothing
        total_smoothed = smoothed_counts.sum()  # = total_count + smoothing * num_bins
        # Compute density: normalized by total_smoothed and bin_width
        # This removes the influence of bin_width on NLL (NLL = -log(density))
        prob_density = smoothed_counts / (total_smoothed * bin_width)
        # Also compute probability mass for stats (normalized to sum to 1)
        prob_mass = smoothed_counts / total_smoothed
        
        bin_probs_per_dim.append(prob_mass)
        
        # Find which bin the target falls into
        target_val = target[dim_idx]
        # numpy.histogram bins: [edges[i], edges[i+1]) for i < n-1, [edges[n-2], edges[n-1]] for last bin
        # Use side='right' to get the first index where value > target, then subtract 1
        # Special case: if target equals the last edge, it should go in the last bin
        if target_val >= bin_edges_actual[-1]:
            target_bin_idx = len(prob_mass) - 1
        else:
            target_bin_idx = np.searchsorted(bin_edges_actual, target_val, side='right') - 1
            # Clamp to valid range [0, num_bins-1]
            target_bin_idx = max(0, min(target_bin_idx, len(prob_mass) - 1))
        target_bins_per_dim.append(target_bin_idx)
        
        # Calculate NLL from probability density: NLL = -log(pdf)
        # Using density removes the influence of bin_width on NLL values
        # The density is computed as: pdf = (counts + smoothing) / (total * bin_width)
        # So NLL = -log(pdf) is now independent of bin_width (for the same underlying distribution)
        target_density = prob_density[target_bin_idx]
        # Ensure density is positive (smoothing already applied in density calculation)
        nll_per_dim[dim_idx] = -np.log(np.maximum(target_density, 1e-20))
    
    total_nll = nll_per_dim.sum()
    
    stats = {
        'bin_width': bin_width,
        'num_bins': num_bins,
        'bin_edges': bin_edges,
        'bin_probs_per_dim': bin_probs_per_dim,
        'target_bins_per_dim': target_bins_per_dim,
        'value_range': (min_val, max_val),
    }
    
    return total_nll, nll_per_dim, stats


def plot_distributions(
    samples: np.ndarray,
    target: np.ndarray,
    output_path: str,
    num_dims_to_plot: Optional[int] = None,
):
    """
    Plot distribution of Monte Carlo samples.
    
    Args:
        samples: Array of shape (num_samples, target_dim) with all Monte Carlo samples
        target: Ground truth target of shape (target_dim,)
        output_path: Path to save the plot
        num_dims_to_plot: Number of dimensions to plot (None = plot all dimensions)
    """
    num_samples, target_dim = samples.shape
    
    # Limit number of dimensions to plot if specified, otherwise plot all
    if num_dims_to_plot is None:
        dims_to_plot = target_dim
    else:
        dims_to_plot = min(num_dims_to_plot, target_dim)
    
    # Create subplots
    n_cols = 3
    n_rows = (dims_to_plot + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if dims_to_plot > 1 else [axes]
    
    for dim_idx in range(dims_to_plot):
        ax = axes[dim_idx]
        
        # Plot histogram of samples
        ax.hist(samples[:, dim_idx], bins=50, alpha=0.7, density=True, label='Monte Carlo samples')
        
        # Mark ground truth
        ax.axvline(target[dim_idx], color='red', linestyle='--', linewidth=2, label='Ground truth')
        
        # Compute statistics
        mean_sample = samples[:, dim_idx].mean()
        std_sample = samples[:, dim_idx].std()
        
        # Mark mean
        ax.axvline(mean_sample, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_sample:.4f}')
        
        ax.set_xlabel(f'Dimension {dim_idx}')
        ax.set_ylabel('Density')
        ax.set_title(f'Dim {dim_idx}: μ={mean_sample:.4f}, σ={std_sample:.4f}, GT={target[dim_idx]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(dims_to_plot, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved distribution plot to {output_path}")
    plt.close()
    
    # Also create a summary statistics plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Mean and std per dimension
    dims = np.arange(target_dim)
    sample_means = samples.mean(axis=0)
    sample_stds = samples.std(axis=0)
    target_arr = target.flatten()
    
    axes[0].plot(dims, sample_means, 'b-', label='Sample mean', linewidth=2)
    axes[0].plot(dims, target_arr, 'r--', label='Ground truth', linewidth=2)
    axes[0].fill_between(dims, sample_means - sample_stds, sample_means + sample_stds, alpha=0.3, label='±1 std')
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Mean per Dimension')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Standard deviation per dimension
    axes[1].plot(dims, sample_stds, 'g-', linewidth=2)
    axes[1].set_xlabel('Dimension')
    axes[1].set_ylabel('Standard Deviation')
    axes[1].set_title('Std per Dimension')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Error per dimension (mean - ground truth)
    errors = sample_means - target_arr
    axes[2].plot(dims, errors, 'orange', linewidth=2)
    axes[2].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[2].set_xlabel('Dimension')
    axes[2].set_ylabel('Error (Mean - GT)')
    axes[2].set_title('Bias per Dimension')
    axes[2].grid(True, alpha=0.3)
    
    summary_path = output_path.replace('.png', '_summary.png')
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary plot to {summary_path}")
    plt.close()


def plot_logp_distribution(
    logp_values: np.ndarray,
    percentile_value: Optional[float],
    percentile: float,
    save_path: str,
    title: str = "ELBO Log-Likelihood Distribution",
) -> None:
    """
    Plot histogram with KDE overlay and optional threshold line for log probability distribution.
    Similar to neuralODE's plot_logp_distribution function.
    
    Args:
        logp_values: Array of log probability values (ELBO log probabilities)
        percentile_value: The threshold value at the specified percentile (None to skip threshold line)
        percentile: The percentile used (e.g., 1.0 for 1%)
        save_path: Path to save the figure
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    n, bins, patches = ax.hist(
        logp_values,
        bins=50,
        density=False,
        alpha=0.7,
        color='lightblue',
        edgecolor='blue',
        linewidth=1.2,
        label='Test samples'
    )
    
    # Compute and plot KDE
    try:
        kde = gaussian_kde(logp_values)
        x_kde = np.linspace(logp_values.min(), logp_values.max(), 200)
        y_kde = kde(x_kde)
        # Scale KDE to match histogram scale (multiply by number of samples and bin width)
        bin_width = bins[1] - bins[0]
        y_kde_scaled = y_kde * len(logp_values) * bin_width
        ax.plot(x_kde, y_kde_scaled, 'b-', linewidth=2, label='KDE')
    except Exception as e:
        print(f"[Plot] Warning: Could not compute KDE: {e}")
    
    # Draw threshold line if provided
    if percentile_value is not None:
        ax.axvline(
            x=percentile_value,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Threshold ({percentile}% percentile)'
        )
    
    ax.set_xlabel('Log-likelihood (ELBO)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper center', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Plot] Saved likelihood distribution plot to {save_path}")


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
        description="Monte Carlo sampling for unconditional diffusion model",
        parents=[config_only],
    )
    parser.add_argument("--model-dir", type=str, default=dget("out", ""), help="Dir with checkpoint.pt and scheduler/")
    parser.add_argument("--test-npz", type=str, default=dget("test_npz", ""), help="Path to test NPZ with target data (optional, for comparison)")
    parser.add_argument("--scaler", type=str, default=dget("scaler", ""), help="Joblib StandardScaler for de-normalization")
    parser.add_argument("--num-mc-samples", type=int, default=dget("num_mc_samples", 1000), help="Number of Monte Carlo samples")
    parser.add_argument("--batch-size", type=int, default=dget("batch_size", 1000), help="Batch size for parallel sampling")
    parser.add_argument("--inference-steps", type=int, default=dget("inference_steps", None) or dget("timesteps", 1000))
    parser.add_argument("--scheduler-type", type=str, default=dget("scheduler_type", "ddpm"), choices=["ddpm", "ddim"], help="Scheduler type to use")
    parser.add_argument("--output-dir", type=str, default=dget("output_dir", ""), help="Directory to save plots")
    parser.add_argument("--device", type=str, default=dget("device", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--num-dims-to-plot", type=int, default=dget("num_dims_to_plot", None), help="Number of dimensions to plot histograms for (None = plot all dimensions)")
    parser.add_argument("--bin-width", type=float, default=dget("bin_width", 0.1), help="Bin width for NLL calculation (only used if --test-npz is provided)")
    parser.add_argument("--max-test-samples", type=int, default=dget("max_test_samples", None), help="Maximum number of test samples to evaluate NLL on (None = use all samples)")
    parser.add_argument("--percentile", type=float, default=dget("percentile", 1.0), help="Percentile to compute for threshold line in likelihood plot (e.g., 1.0 for 1% percentile)")

    args = parser.parse_args()

    model_dir = args.model_dir
    test_npz = args.test_npz
    scaler_path = args.scaler
    device = args.device
    num_mc_samples = args.num_mc_samples
    scheduler_type = args.scheduler_type.lower()

    if not model_dir:
        raise ValueError("Please provide --model-dir or put 'out' in the YAML config used for training.")
    ckpt_path = os.path.join(model_dir, "checkpoint.pt")
    sched_dir = os.path.join(model_dir, "scheduler")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint at {ckpt_path}")
    if not os.path.exists(sched_dir):
        raise FileNotFoundError(f"Missing scheduler directory at {sched_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data if provided (for comparison/evaluation)
    target_all_np = None
    target_first_np = None
    if test_npz:
        data = np.load(test_npz)
        # Try to find target data
        possible_keys = ["target", "y", "action", "next", "outputs", "X_target", "data", "x", "samples"]
        target = None
        for key in possible_keys:
            if key in data:
                target = data[key]
                break
        if target is None:
            print(f"Warning: Could not find target data in {test_npz}. Available keys: {list(data.keys())}")
            print("Proceeding without target comparison.")
        else:
            # Load test samples (optionally limit the number)
            if len(target.shape) > 1:
                target_all_np = target  # Shape: (num_samples, target_dim)
            else:
                target_all_np = target.reshape(1, -1)  # Reshape to (1, target_dim)
            
            # Limit number of test samples if requested
            max_test_samples = args.max_test_samples
            if max_test_samples is not None and max_test_samples > 0 and target_all_np.shape[0] > max_test_samples:
                print(f"Limiting test samples from {target_all_np.shape[0]} to {max_test_samples}")
                target_all_np = target_all_np[:max_test_samples]
            
            # Also keep first for plotting/comparison
            target_first_np = target_all_np[0].flatten()
            print(f"Loaded {target_all_np.shape[0]} test samples from {test_npz}:")
            print(f"  Target shape: {target_all_np.shape}")

    # Load scaler for de-normalization
    mean_target_arr = None
    std_target_arr = None
    if scaler_path:
        if joblib is None:
            print("joblib not installed; skipping scaler")
        else:
            try:
                scaler = joblib.load(scaler_path)
                std = np.asarray(getattr(scaler, "scale_", None))
                if std is None:
                    std = np.asarray(getattr(scaler, "std_", None))
                if std is not None:
                    std = np.asarray(std, dtype=np.float64).reshape(-1)
                    std = np.where(std <= 1e-12, 1.0, std)
                    # For unconditional, assume scaler is for target only
                    mean = np.asarray(getattr(scaler, "mean_", None))
                    if mean is not None:
                        mean = np.asarray(mean, dtype=np.float64).reshape(-1)
                        mean_target_arr = mean
                    std_target_arr = std
            except Exception as e:
                print(f"Failed to load scaler: {e}")

    # Build model and scheduler
    model, train_cfg = build_model_from_ckpt(ckpt_path, device)
    
    # Get target_dim from checkpoint (already loaded in build_model_from_ckpt, but we need it here)
    ckpt = torch.load(ckpt_path, map_location=device)
    target_dim = ckpt.get("target_dim")
    if target_dim is None:
        raise ValueError("Could not determine target_dim from checkpoint")
    
    # Load scheduler
    if scheduler_type == "ddpm":
        try:
            scheduler = DDPMScheduler.from_pretrained(sched_dir)
        except Exception:
            try:
                from diffusers.schedulers.scheduling_ddim import DDIMScheduler
                ddim_scheduler = DDIMScheduler.from_pretrained(sched_dir)
                scheduler = DDPMScheduler(
                    num_train_timesteps=ddim_scheduler.num_train_timesteps,
                    beta_schedule=ddim_scheduler.beta_schedule if hasattr(ddim_scheduler, 'beta_schedule') else "linear",
                    prediction_type=ddim_scheduler.prediction_type,
                )
                print("Converted DDIMScheduler to DDPMScheduler for inference")
            except Exception as e:
                print(f"Warning: Could not load scheduler: {e}")
                print("Creating default DDPMScheduler")
                scheduler = DDPMScheduler(
                    num_train_timesteps=1000,
                    beta_schedule="linear",
                    prediction_type="epsilon",
                )
        sampling_fn_batch = ddpm_stochastic_sample_batch
    else:  # ddim
        try:
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler
            scheduler = DDIMScheduler.from_pretrained(sched_dir)
        except Exception:
            print("Warning: Could not load DDIMScheduler, creating default")
            scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_schedule="linear",
                prediction_type="epsilon",
            )
        sampling_fn_batch = ddim_deterministic_sample_batch

    print(f"Using {scheduler_type.upper()} scheduler with {args.inference_steps} inference steps")
    print(f"Batch size: {args.batch_size}")
    print(f"Target dimension: {target_dim}")

    # Perform Monte Carlo sampling in batches
    print(f"\nGenerating {num_mc_samples} samples in batches of {args.batch_size}...")
    all_samples = []
    
    num_batches = (num_mc_samples + args.batch_size - 1) // args.batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Sampling batches"):
        # Calculate actual batch size for last batch
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, num_mc_samples)
        current_batch_size = end_idx - start_idx
        
        # Generate batch of samples
        batch_samples = sampling_fn_batch(
            model=model,
            scheduler=scheduler,
            batch_size=current_batch_size,
            target_dim=target_dim,
            num_inference_steps=args.inference_steps,
            device=device,
            seed_offset=start_idx,
        )
        
        # De-normalize if scaler available
        batch_samples_orig = inverse_scale_target(
            batch_samples,
            mean_target_arr,
            std_target_arr
        )
        
        all_samples.append(batch_samples_orig.cpu().numpy())

    all_samples = np.concatenate(all_samples, axis=0)  # Shape: (num_mc_samples, target_dim)

    print(f"\nSample statistics:")
    print(f"  Sample shape: {all_samples.shape}")
    print(f"  Mean across samples: {all_samples.mean():.6f}")
    print(f"  Std across samples: {all_samples.std():.6f}")
    print(f"  Min value: {all_samples.min():.6f}")
    print(f"  Max value: {all_samples.max():.6f}")

    # Compute per-dimension statistics
    sample_means = all_samples.mean(axis=0)
    sample_stds = all_samples.std(axis=0)
    
    print(f"\nPer-dimension statistics:")
    print(f"  Mean std across dims: {sample_stds.mean():.6f}")
    print(f"  Min std: {sample_stds.min():.6f}")
    print(f"  Max std: {sample_stds.max():.6f}")

    # Save samples to file
    samples_path = os.path.join(args.output_dir, "monte_carlo_samples.npy")
    np.save(samples_path, all_samples)
    print(f"\nSaved samples to {samples_path}")

    # If target provided, compute NLL and plot comparisons
    if target_all_np is not None:
        # For plotting, use first sample
        target_orig = inverse_scale_target(
            torch.from_numpy(target_first_np).float().unsqueeze(0),
            mean_target_arr,
            std_target_arr
        ).cpu().numpy().flatten()
        
        errors = sample_means - target_orig
        print(f"\nComparison with ground truth (first sample):")
        print(f"  Mean error: {errors.mean():.6f} ± {errors.std():.6f}")
        print(f"  Max absolute error: {np.abs(errors).max():.6f}")

        # Calculate NLL using ELBO (log_prob_elbo) for all test samples
        print(f"\nCalculating NLL using ELBO (log_prob_elbo) for all {target_all_np.shape[0]} test samples...")
        # Use normalized targets for model (model expects normalized inputs)
        target_all_normalized = torch.from_numpy(target_all_np).float().to(device)
        
        # Process in batches to avoid memory issues
        eval_batch_size = min(512, target_all_np.shape[0])
        nll_elbo_list = []
        log_prob_elbo_list = []
        
        for i in range(0, target_all_np.shape[0], eval_batch_size):
            batch_targets = target_all_normalized[i:i + eval_batch_size]
            batch_log_probs = log_prob_elbo(
                model=model,
                scheduler=scheduler,
                x0=batch_targets,
                device=device,
            )
            # NLL = -log_prob (this is summed over dimensions)
            batch_nll_total = -batch_log_probs
            batch_nll_per_dim = batch_nll_total
            nll_elbo_list.append(batch_nll_per_dim.cpu())
            log_prob_elbo_list.append(batch_log_probs.cpu())
        
        # Concatenate and compute statistics
        nll_elbo_all = np.concatenate([nll.numpy() for nll in nll_elbo_list], axis=0)
        log_prob_elbo_all = np.concatenate([lp.numpy() for lp in log_prob_elbo_list], axis=0)
        nll_elbo_mean = nll_elbo_all.mean()
        nll_elbo_std = nll_elbo_all.std()
        
        print(f"\nELBO-based NLL per sample (computed for {target_all_np.shape[0]} test samples):")
        print(f"  NOTE: Each sample gets its own NLL (NOT averaged inside batch)")
        print(f"  NOTE: NLL is normalized by dividing by {target_dim} dimensions (average per dimension)")
        print(f"  Target dimension: {target_dim}")
        print(f"  Mean NLL per sample (per dimension, averaged): {nll_elbo_mean:.6f} ± {nll_elbo_std:.6f}")
        print(f"  Min NLL per sample: {nll_elbo_all.min():.6f}")
        print(f"  Max NLL per sample: {nll_elbo_all.max():.6f}")
        print(f"  Median NLL per sample: {np.median(nll_elbo_all):.6f}")
        print(f"  Mean Log probability per sample: {log_prob_elbo_all.mean():.6f} ± {log_prob_elbo_all.std():.6f}")
        print(f"  First 10 NLL values per sample (per dimension): {nll_elbo_all[:10]}")
        
        # Compute percentile threshold for likelihood distribution plot
        percentile_logp = None
        elbo_mean = log_prob_elbo_all.mean()
        if len(log_prob_elbo_all) > 0:
            percentile_logp = np.percentile(log_prob_elbo_all, args.percentile)
            print(f"  {args.percentile}% percentile log-likelihood: {percentile_logp:.6f}")
        
        # Save ELBO mean and threshold to JSON file
        metrics = {
            "num_samples": int(len(log_prob_elbo_all)),
            "elbo_mean": float(elbo_mean),
            "elbo_std": float(log_prob_elbo_all.std()),
            "nll_mean": float(nll_elbo_mean),
            "nll_std": float(nll_elbo_std),
        }
        if percentile_logp is not None:
            metrics[f"percentile_{args.percentile}_logp"] = float(percentile_logp)
            metrics["threshold"] = float(percentile_logp)  # Also store as "threshold" for easy access
        
        metrics_path = os.path.join(args.output_dir, "elbo_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved ELBO metrics to {metrics_path}")
        
        # Plot likelihood distribution (similar to neuralODE)
        likelihood_plot_path = os.path.join(args.output_dir, "likelihood_distribution.png")
        plot_logp_distribution(
            logp_values=log_prob_elbo_all,
            percentile_value=percentile_logp,
            percentile=args.percentile,
            save_path=likelihood_plot_path,
            title="ELBO Log-Likelihood Distribution"
        )
        
        # Calculate NLL using binning (for comparison)
        print(f"\nCalculating NLL with binning (bin width={args.bin_width})...")
        total_nll, nll_per_dim, nll_stats = calculate_nll_binned(
            samples=all_samples,
            target=target_orig,
            bin_width=args.bin_width,
        )
        
        print(f"\nBinning-based NLL Statistics:")
        print(f"  Total NLL: {total_nll:.6f}")
        print(f"  Mean NLL per dimension: {nll_per_dim.mean():.6f} ± {nll_per_dim.std():.6f}")
        print(f"  Min NLL per dimension: {nll_per_dim.min():.6f}")
        print(f"  Max NLL per dimension: {nll_per_dim.max():.6f}")
        print(f"  Number of bins used: {nll_stats['num_bins']}")
        print(f"  Value range: [{nll_stats['value_range'][0]:.4f}, {nll_stats['value_range'][1]:.4f}]")
        
        print(f"\nComparison (first sample only for binning):")
        print(f"  ELBO NLL per dimension (mean over all test samples): {nll_elbo_mean:.6f}")
        print(f"  Binning NLL per dimension (first sample, mean): {nll_per_dim.mean():.6f}")
        print(f"  Binning NLL per dimension (first sample, total): {total_nll / target_dim:.6f}")
        print(f"  Note: Binning NLL is computed only for the first test sample")
        print(f"  Note: Both methods now report NLL per dimension (averaged)")

        # Save NLL results
        nll_results_path = os.path.join(args.output_dir, "nll_results.npz")
        np.savez(
            nll_results_path,
            nll_elbo_mean=nll_elbo_mean,
            nll_elbo_std=nll_elbo_std,
            nll_elbo_per_sample=nll_elbo_all,  # NLL for each test sample
            log_prob_elbo_mean=log_prob_elbo_all.mean(),
            log_prob_elbo_per_sample=log_prob_elbo_all,  # Log prob for each test sample
            total_nll=total_nll,
            nll_per_dim=nll_per_dim,
            bin_width=args.bin_width,
            num_bins=nll_stats['num_bins'],
            value_range=nll_stats['value_range'],
        )
        print(f"Saved NLL results to {nll_results_path}")
        print(f"  - nll_elbo_per_sample: NLL for each of {target_all_np.shape[0]} test samples (shape: {nll_elbo_all.shape})")
        
        # Also save per-sample NLL to a text file for easy inspection
        nll_per_sample_path = os.path.join(args.output_dir, "nll_per_sample.txt")
        with open(nll_per_sample_path, "w") as f:
            f.write(f"NLL per sample (ELBO method)\n")
            f.write(f"Total samples: {len(nll_elbo_all)}\n")
            f.write(f"Mean: {nll_elbo_mean:.6f}, Std: {nll_elbo_std:.6f}\n")
            f.write(f"Min: {nll_elbo_all.min():.6f}, Max: {nll_elbo_all.max():.6f}\n")
            f.write(f"\nNLL per sample:\n")
            for i, nll in enumerate(nll_elbo_all):
                f.write(f"Sample {i}: {nll:.6f}\n")
        print(f"Saved per-sample NLL values to {nll_per_sample_path}")

        # Plot distributions
        plot_path = os.path.join(args.output_dir, "monte_carlo_distributions.png")
        plot_distributions(
            samples=all_samples,
            target=target_orig,
            output_path=plot_path,
            num_dims_to_plot=args.num_dims_to_plot,
        )

        # Plot NLL per dimension
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: NLL per dimension
        dims = np.arange(target_dim)
        axes[0].bar(dims, nll_per_dim, alpha=0.7)
        axes[0].axhline(nll_per_dim.mean(), color='red', linestyle='--', linewidth=2, 
                        label=f'Mean: {nll_per_dim.mean():.4f}')
        axes[0].set_xlabel('Dimension')
        axes[0].set_ylabel('NLL')
        axes[0].set_title(f'NLL per Dimension (bin width={args.bin_width})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Histogram of NLL values
        axes[1].hist(nll_per_dim, bins=30, alpha=0.7, edgecolor='black')
        axes[1].axvline(nll_per_dim.mean(), color='red', linestyle='--', linewidth=2, 
                        label=f'Mean: {nll_per_dim.mean():.4f}')
        axes[1].axvline(np.median(nll_per_dim), color='green', linestyle='--', linewidth=2, 
                        label=f'Median: {np.median(nll_per_dim):.4f}')
        axes[1].set_xlabel('NLL')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of NLL Values')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        nll_plot_path = os.path.join(args.output_dir, "nll_per_dimension.png")
        plt.savefig(nll_plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved NLL plot to {nll_plot_path}")
        plt.close()
    else:
        # Plot distributions without target comparison
        print("\nNo target provided, generating distribution plots without comparison...")
        # Create a simple distribution plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        dims = np.arange(target_dim)
        sample_means = all_samples.mean(axis=0)
        sample_stds = all_samples.std(axis=0)
        
        # Plot 1: Mean and std per dimension
        axes[0].plot(dims, sample_means, 'b-', label='Sample mean', linewidth=2)
        axes[0].fill_between(dims, sample_means - sample_stds, sample_means + sample_stds, alpha=0.3, label='±1 std')
        axes[0].set_xlabel('Dimension')
        axes[0].set_ylabel('Value')
        axes[0].set_title('Mean per Dimension')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Standard deviation per dimension
        axes[1].plot(dims, sample_stds, 'g-', linewidth=2)
        axes[1].set_xlabel('Dimension')
        axes[1].set_ylabel('Standard Deviation')
        axes[1].set_title('Std per Dimension')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "monte_carlo_distributions.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved distribution plot to {plot_path}")
        plt.close()

    print(f"\nMonte Carlo sampling complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python
"""
Script to compute and save metadata for an existing trained Neural ODE model.
Usage: python neuralODE/save_metadata.py --model_dir /path/to/model --npz /path/to/data.npz
"""
import argparse
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from neural_ode_density import ContinuousNormalizingFlow, ODEFunc, NPZTargetDataset


def main():
    parser = argparse.ArgumentParser(description="Save metadata for existing Neural ODE model")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model.pt")
    parser.add_argument("--npz", type=str, required=True, help="Path to training data NPZ/PKL file")
    parser.add_argument("--batch", type=int, default=512, help="Batch size for computing log probs")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--activation", type=str, default="silu")
    parser.add_argument("--time_dependent", action="store_true", default=True)
    parser.add_argument("--solver", type=str, default="dopri5")
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--t1", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--percentile", type=float, default=1.0, help="Percentile for threshold (default 1%)")
    parser.add_argument("--max_samples", type=int, default=50000, help="Max samples to use (default 50000, 0=all)")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading data from {args.npz}...")
    dataset = NPZTargetDataset(args.npz)
    target_dim = dataset.target_dim
    print(f"  Target dim: {target_dim}")
    print(f"  Total samples: {len(dataset)}")

    # Subsample if needed
    if args.max_samples > 0 and len(dataset) > args.max_samples:
        indices = np.random.choice(len(dataset), args.max_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"  Subsampled to {args.max_samples} samples")

    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False)

    # Build model
    print(f"Building model...")
    odefunc = ODEFunc(
        dim=target_dim,
        hidden_dims=tuple(args.hidden_dims),
        activation=args.activation,
        time_dependent=args.time_dependent,
    ).to(args.device)

    flow = ContinuousNormalizingFlow(
        func=odefunc,
        t0=args.t0,
        t1=args.t1,
        solver=args.solver,
        rtol=args.rtol,
        atol=args.atol,
    ).to(args.device)

    # Load weights (model.pt or latest checkpoint)
    import glob
    model_path = os.path.join(args.model_dir, "model.pt")
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
    else:
        # Find latest checkpoint
        ckpt_pattern = os.path.join(args.model_dir, "checkpoint_epoch_*.pt")
        ckpts = glob.glob(ckpt_pattern)
        if ckpts:
            ckpts.sort(key=lambda x: int(x.split('_epoch_')[-1].replace('.pt', '')))
            model_path = ckpts[-1]
            print(f"model.pt not found, using latest checkpoint: {model_path}")
        else:
            raise FileNotFoundError(f"No model.pt or checkpoint found in {args.model_dir}")

    # Load state dict (handle both checkpoint format and direct state dict)
    ckpt = torch.load(model_path, map_location=args.device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        flow.load_state_dict(ckpt['model_state_dict'])
    else:
        flow.load_state_dict(ckpt)
    flow.eval()

    # Compute log probabilities
    # Note: log_prob needs gradients for divergence computation, so we can't use torch.no_grad()
    print("Computing log probabilities on training data...")
    all_log_probs = []
    for i, batch in enumerate(loader):
        x = batch.to(args.device)
        log_px = flow.log_prob(x)
        all_log_probs.append(log_px.detach().cpu().numpy())
        if (i + 1) % 10 == 0:
            print(f"  Processed {(i+1) * args.batch} / {len(dataset)} samples")

    all_log_probs = np.concatenate(all_log_probs)

    # Compute statistics
    threshold = float(np.percentile(all_log_probs, args.percentile))
    mean_logp = float(np.mean(all_log_probs))
    std_logp = float(np.std(all_log_probs))

    print(f"\nStatistics:")
    print(f"  Threshold ({args.percentile}%): {threshold:.4f}")
    print(f"  Mean log prob: {mean_logp:.4f}")
    print(f"  Std log prob: {std_logp:.4f}")

    # Save metadata
    metadata = {
        'threshold': threshold,
        'mean': mean_logp,
        'std': std_logp,
        'device': args.device,
        'target_dim': target_dim,
        'hidden_dims': tuple(args.hidden_dims),
        'activation': args.activation,
        'time_dependent': args.time_dependent,
        'solver': args.solver,
        't0': args.t0,
        't1': args.t1,
        'rtol': args.rtol,
        'atol': args.atol,
    }

    metadata_path = os.path.join(args.model_dir, "metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    # Also update model.pt with embedded metadata
    model_ckpt = {
        'model_state_dict': flow.state_dict(),
        'threshold': threshold,
        'mean': mean_logp,
        'std': std_logp,
        'target_dim': target_dim,
    }
    model_pt_path = os.path.join(args.model_dir, "model.pt")
    torch.save(model_ckpt, model_pt_path)

    print(f"\nMetadata saved to: {metadata_path}")
    print(f"Model updated with embedded metadata: {model_pt_path}")


if __name__ == "__main__":
    main()

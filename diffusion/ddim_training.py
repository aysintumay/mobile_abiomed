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


class NPZCondTargetDataset(Dataset):
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
        possible_cond_keys = [
            "cond",
            "conditions",
            "x",
            "state",
            "inputs",
            "X_cond",
        ]
        possible_target_keys = [
            "target",
            "y",
            "action",
            "next",
            "outputs",
            "X_target",
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

        self.cond_np = pick(possible_cond_keys)
        self.target_np = pick(possible_target_keys)

        if self.cond_np.shape[0] != self.target_np.shape[0]:
            raise ValueError(
                f"Mismatched lengths: cond {self.cond_np.shape} vs target {self.target_np.shape}"
            )

        self.cond = torch.from_numpy(self.cond_np).to(dtype)
        self.target = torch.from_numpy(self.target_np).to(dtype)

        # Flatten trailing dims into feature vectors for generic MLP handling
        self.cond = self.cond.view(self.cond.size(0), -1)
        self.target = self.target.view(self.target.size(0), -1)

        self.num_samples = self.cond.size(0)
        self.cond_dim = self.cond.size(1)
        self.target_dim = self.target.size(1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cond[idx], self.target[idx]


# -----------------------------
# Model: Conditional noise predictor (epsilon net)
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


class ConditionalEpsilonMLP(nn.Module):
    def __init__(
        self,
        target_dim: int,
        cond_dim: int,
        hidden_dim: int = 512,
        time_embed_dim: int = 128,
        num_hidden_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        layers = []
        in_dim = target_dim + cond_dim + time_embed_dim
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, target_dim)

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_t: [B, target_dim], cond: [B, cond_dim], t: [B]
        t_emb = self.time_embed(t)
        h = torch.cat([x_t, cond, t_emb], dim=1)
        h = self.backbone(h)
        eps = self.head(h)
        return eps


# -----------------------------
# Model: Transformer variant
# -----------------------------


class ConditionalEpsilonTransformer(nn.Module):
    def __init__(
        self,
        target_dim: int,
        cond_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        time_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.target_dim = target_dim
        self.cond_dim = cond_dim
        self.d_model = d_model

        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_to_model = nn.Linear(time_embed_dim, d_model)

        # Represent x_t as a sequence of length L = target_dim with scalar tokens → project to d_model
        self.x_proj = nn.Linear(1, d_model)
        # Positional embedding for L + 1 (cond token)
        self.pos_embed = nn.Parameter(torch.zeros(1, target_dim + 1, d_model))

        # Condition token from cond vector
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

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

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        bsz, dim = x_t.shape
        assert dim == self.target_dim, "x_t must have target_dim features"

        # Build token sequence for x_t: shape [B, L, 1] → proj to [B, L, d_model]
        x_tokens = x_t.unsqueeze(-1)
        x_tokens = self.x_proj(x_tokens)

        # Condition token
        c_token = self.cond_proj(cond).unsqueeze(1)  # [B, 1, d_model]

        # Time embedding added to all tokens
        t_emb = self.time_to_model(self.time_embed(t)).unsqueeze(1)  # [B,1,d_model]
        tokens = torch.cat([c_token, x_tokens], dim=1)  # [B, 1+L, d_model]
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :] + t_emb

        h = self.encoder(tokens)
        # Discard cond token, keep L positions
        h_x = h[:, 1:, :]
        eps_tokens = self.out_head(h_x)  # [B, L, 1]
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

    dataset = NPZCondTargetDataset(cfg.npz_path)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    if cfg.model_type == "mlp":
        model = ConditionalEpsilonMLP(
            target_dim=dataset.target_dim,
            cond_dim=dataset.cond_dim,
            hidden_dim=cfg.hidden_dim,
            time_embed_dim=cfg.time_embed_dim,
            num_hidden_layers=cfg.num_hidden_layers,
            dropout=cfg.dropout,
        ).to(cfg.device)
    elif cfg.model_type == "transformer":
        model = ConditionalEpsilonTransformer(
            target_dim=dataset.target_dim,
            cond_dim=dataset.cond_dim,
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
                "config": {**cfg.__dict__, "cond_dim": dataset.cond_dim, "target_dim": dataset.target_dim},
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
        for cond, target in loader:
            cond = cond.to(cfg.device)
            clean = target.to(cfg.device)

            # Sample random timesteps per batch element
            bsz = clean.size(0)
            t = torch.randint(0, cfg.num_train_timesteps, (bsz,), device=cfg.device)

            noise = torch.randn_like(clean)
            # Add noise according to the forward process
            noisy = scheduler.add_noise(clean, noise, t)

            pred_noise = model(noisy, cond, t)
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
                "cond_dim": dataset.cond_dim,
                "target_dim": dataset.target_dim,
            },
            os.path.join(cfg.out_dir, "checkpoint.pt"),
        )

        # Optional sampling preview to wandb
        if cfg.samples_every and ((epoch + 1) % cfg.samples_every == 0):
            try:
                model.eval()
                with torch.no_grad():
                    sample_cond = next(iter(loader))[0][: min(16, cfg.batch_size)].to(cfg.device)
                    sch = DDIMScheduler.from_pretrained(os.path.join(cfg.out_dir, "scheduler"))
                    preds = sample(
                        model=model,
                        scheduler=sch,
                        cond=sample_cond,
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
                    "cond_dim": dataset.cond_dim,
                    "target_dim": dataset.target_dim,
                },
                ckpt_path,
            )

    # Final save
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model.pt"))


# -----------------------------
# Sampling (cond -> target) using deterministic DDIM
# -----------------------------


@torch.no_grad()
def sample(
    model: ConditionalEpsilonMLP,
    scheduler: DDIMScheduler,
    cond: torch.Tensor,
    num_inference_steps: int = 50,
    eta: float = 0.0,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()
    cond = cond.to(device)

    scheduler.set_timesteps(num_inference_steps)
    x = torch.randn(cond.size(0), model.head.out_features, device=device)

    for t in scheduler.timesteps:
        # Predict noise
        eps = model(x, cond, t.expand(cond.size(0)))
        # DDIM step
        out = scheduler.step(model_output=eps, timestep=t, sample=x, eta=eta)
        x = out.prev_sample
    return x


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
    parser = argparse.ArgumentParser(description="Train DDIM (diffusers) cond->target on NPZ", parents=[config_only])

    def dget(key, default):
        return yaml_defaults.get(key, default)

    parser.add_argument("--npz", required=("npz" not in yaml_defaults), default=yaml_defaults.get("npz", None), help="Path to hopper_one_step_train.npz")
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



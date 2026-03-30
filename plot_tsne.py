#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from models.nets import MLP
from models.actor_critic import ActorProb, Critic
from models.dist import TanhDiagGaussian
from policies import MOBILEPolicy


DEFAULT_ABIOMED_POLICIES = [
    {
        "label": "MBPO",
        "path": "/public/gormpo/models/rl/abiomed/realnvp/seed_42_0302_234230-abiomed_mbpo/policy_abiomed.pth",
    },
    {
        "label": "Reg-MBPO-KDE",
        "path": "/public/gormpo/models/rl/abiomed/kde/seed_42_0302_234240-abiomed_mbpo_kde/policy_abiomed.pth",
    },
    {
        "label": "Reg-MBPO-VAE",
        "path": "/public/gormpo/models/rl/abiomed/vae/seed_42_0303_000341-abiomed_mbpo_vae/policy_abiomed.pth",
    },
    {
        "label": "Reg-MBPO-RealNVP",
        "path": "/public/gormpo/models/rl/abiomed/realnvp/seed_42_0302_234510-abiomed_mbpo_realnvp/policy_abiomed.pth",
    },
    {
        "label": "Reg-MBPO-Diffusion",
        "path": "/public/gormpo/models/rl/abiomed/seed_42_0303_000226-abiomed_mbpo_diffusion/policy_abiomed.pth",
    },
    {
        "label": "Reg-MBPO-NeuralODE",
        "path": "/public/gormpo/models/rl/abiomed/seed_42_0302_234847-abiomed_mbpo_neuralode/policy_abiomed.pth",
    },
    {
        "label": "Pen-MBPO-KDE",
        "path": "/public/gormpo/models/rl/abiomed/kde/seed_42_0303_000216-abiomed_mbpo_kde/policy_abiomed.pth",
    },
    {
        "label": "Pen-MBPO-VAE",
        "path": "/public/gormpo/models/rl/abiomed/vae/seed_42_0302_234248-abiomed_mbpo_vae/policy_abiomed.pth",
    },
    {
        "label": "Pen-MBPO-RealNVP",
        "path": "/public/gormpo/models/rl/abiomed/realnvp/seed_42_0303_000316-abiomed_mbpo_realnvp/policy_abiomed.pth",
    },
]


@dataclass
class PolicySpec:
    label: str
    path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate t-SNE plots for offline dataset support vs policy support."
    )
    parser.add_argument("--task", type=str, default="halfcheetah-medium-expert-v2")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        help='Policy spec as "label=/abs/path/to/policy.pth". Can be repeated.',
    )
    parser.add_argument(
        "--policy-json",
        type=str,
        default=None,
        help="JSON file with list of {'label': ..., 'path': ...}.",
    )
    parser.add_argument(
        "--use-abiomed-defaults",
        action="store_true",
        help="Use built-in Abiomed policy list from the rebuttal setup.",
    )
    # Match notebook defaults.
    parser.add_argument("--n-rollout-episodes", type=int, default=100)
    parser.add_argument("--max-offline-samples", type=int, default=100000)
    # Notebook does not explicitly cap rollout transitions; use a large default.
    parser.add_argument("--max-rollout-samples", type=int, default=100000)
    parser.add_argument("--perplexity", type=float, default=40.0)
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--num-q-ensemble", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--support-source",
        type=str,
        choices=["rollout", "offline_policy"],
        default="offline_policy",
        help="Use real env rollouts or infer policy support from offline states.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=None,
        help="Gym env id for rollout mode. Defaults to --task.",
    )
    parser.add_argument("--output-dir", type=str, default="results/tsne_policy_vs_dataset")
    parser.add_argument("--panel-cols", type=int, default=6)
    parser.add_argument("--save-pdf", action="store_true")
    # Match plotting aesthetics from `tsne_policy_vs_dataset.ipynb`.
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--point-size", type=float, default=8.0)  # offline uses this directly
    parser.add_argument("--offline-alpha", type=float, default=0.35)
    parser.add_argument("--policy-alpha", type=float, default=0.55)
    return parser.parse_args()


def clean_path(path: str) -> str:
    return path.strip().strip('"').strip("'").replace("\n", "").replace("\r", "")


def parse_policy_specs(args: argparse.Namespace) -> List[PolicySpec]:
    specs: List[PolicySpec] = []

    if args.use_abiomed_defaults:
        specs.extend([PolicySpec(s["label"], s["path"]) for s in DEFAULT_ABIOMED_POLICIES])

    if args.policy_json:
        with open(args.policy_json, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        for entry in loaded:
            specs.append(PolicySpec(entry["label"], clean_path(entry["path"])))

    for raw in args.policy:
        if "=" not in raw:
            raise ValueError(f'Invalid --policy value "{raw}", expected "label=path"')
        label, path = raw.split("=", 1)
        specs.append(PolicySpec(label.strip(), clean_path(path)))

    if not specs:
        raise ValueError("No policies provided. Use --policy, --policy-json, or --use-abiomed-defaults.")

    return specs


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
            "figure.titlesize": 14,
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
        }
    )


def _normalize_loaded_dataset(dataset_obj):
    if isinstance(dataset_obj, dict):
        return dataset_obj
    if isinstance(dataset_obj, list):
        raise ValueError(
            "Loaded dataset is a list; expected dict with keys like "
            "'observations', 'next_observations', and 'actions'."
        )
    raise ValueError(f"Unsupported dataset object type: {type(dataset_obj)}")


def _load_dataset_file(dataset_path: str) -> dict:
    suffix = Path(dataset_path).suffix.lower()
    if suffix == ".npz":
        npz_data = np.load(dataset_path, allow_pickle=True)
        return {k: npz_data[k] for k in npz_data.files}
    if suffix in {".pkl", ".pickle"}:
        with open(dataset_path, "rb") as f:
            obj = pickle.load(f)
        return _normalize_loaded_dataset(obj)

    # Fallback: try pickle first, then npz
    try:
        with open(dataset_path, "rb") as f:
            obj = pickle.load(f)
        return _normalize_loaded_dataset(obj)
    except Exception:
        npz_data = np.load(dataset_path, allow_pickle=True)
        return {k: npz_data[k] for k in npz_data.files}


def _extract_key(dataset: dict, candidates: Sequence[str], required: bool = True):
    for key in candidates:
        if key in dataset:
            return dataset[key], key
    if required:
        raise KeyError(f"None of keys found: {candidates}")
    return None, None


def load_dataset(task: str, dataset_path: Optional[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import gym

    if dataset_path is not None and os.path.exists(dataset_path):
        dataset = _load_dataset_file(dataset_path)
        print(f"[dataset] Loaded sparse/custom dataset: {dataset_path}")
    else:
        try:
            import d4rl  # noqa: F401  # required to register D4RL env datasets

            env_tmp = gym.make(task)
            dataset = d4rl.qlearning_dataset(env_tmp)
            env_tmp.close()
            print(f"[dataset] Loaded D4RL dataset for task: {task}")
        except Exception as exc:
            raise RuntimeError(
                f"Could not auto-load dataset for task '{task}'. "
                "For non-D4RL tasks (e.g., Abiomed), pass --dataset-path to a pickle "
                "or npz-like dataset dict containing keys: observations, next_observations, actions."
            ) from exc
    obs, obs_key = _extract_key(dataset, ("observations", "obs", "states"))
    next_obs, next_obs_key = _extract_key(dataset, ("next_observations", "next_obs", "next_states"))
    actions, act_key = _extract_key(dataset, ("actions", "acts", "action"))
    print(f"[dataset] Using keys: obs='{obs_key}', next_obs='{next_obs_key}', actions='{act_key}'")
    return obs, next_obs, actions


def build_policy(obs_dim: int, action_dim: int, num_q_ensemble: int = 2, device: str = "cpu") -> MOBILEPolicy:
    actor_backbone = MLP(input_dim=obs_dim, hidden_dims=[256, 256])
    dist = TanhDiagGaussian(
        latent_dim=actor_backbone.output_dim,
        output_dim=action_dim,
        unbounded=True,
        conditioned_sigma=True,
    )
    actor = ActorProb(actor_backbone, dist, device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-4)

    critics = torch.nn.ModuleList([
        Critic(MLP(input_dim=obs_dim + action_dim, hidden_dims=[256, 256]), device)
        for _ in range(num_q_ensemble)
    ])
    critics_optim = torch.optim.Adam(critics.parameters(), lr=3e-4)

    return MOBILEPolicy(
        dynamics=None,
        actor=actor,
        critics=critics,
        actor_optim=actor_optim,
        critics_optim=critics_optim,
    )


def collect_rollouts(
    policy: MOBILEPolicy, env, n_episodes: int, deterministic: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    all_next_obs: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []

    for ep in range(n_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        truncated = False

        while not done and not truncated:
            action = policy.select_action(obs, deterministic=deterministic)
            step_out = env.step(action)

            if len(step_out) == 5:
                next_obs, _, done, truncated, _ = step_out
            else:
                next_obs, _, done, _ = step_out
                truncated = False

            all_next_obs.append(next_obs.copy())
            all_actions.append(np.asarray(action).copy())
            obs = next_obs

        if (ep + 1) % 10 == 0:
            print(f"[rollout] finished episode {ep + 1}/{n_episodes}")

    return np.asarray(all_next_obs), np.asarray(all_actions)




def collect_policy_support_from_offline(
    policy: MOBILEPolicy,
    offline_obs: np.ndarray,
    offline_next_obs: np.ndarray,
    max_samples: int,
    deterministic: bool,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    n = offline_obs.shape[0]
    k = min(max_samples, n)
    idx = rng.choice(n, size=k, replace=False)
    obs_batch = offline_obs[idx]
    next_obs_batch = offline_next_obs[idx]
    actions = policy.select_action(obs_batch, deterministic=deterministic)
    return next_obs_batch, np.asarray(actions)


def prepare_tsne_input(
    offline_next_obs: np.ndarray,
    offline_actions: np.ndarray,
    rollout_next_obs: np.ndarray,
    rollout_actions: np.ndarray,
    max_offline_samples: int,
    max_rollout_samples: int,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)

    n_off = min(max_offline_samples, offline_next_obs.shape[0])
    off_idx = rng.choice(offline_next_obs.shape[0], size=n_off, replace=False)
    offline_sa = np.concatenate([offline_next_obs[off_idx], offline_actions[off_idx]], axis=1)

    if rollout_next_obs.shape[0] > max_rollout_samples:
        roll_idx = rng.choice(rollout_next_obs.shape[0], size=max_rollout_samples, replace=False)
        rollout_sa = np.concatenate([rollout_next_obs[roll_idx], rollout_actions[roll_idx]], axis=1)
    else:
        rollout_sa = np.concatenate([rollout_next_obs, rollout_actions], axis=1)

    all_sa = np.concatenate([offline_sa, rollout_sa], axis=0)
    labels = np.concatenate(
        [np.zeros(offline_sa.shape[0], dtype=int), np.ones(rollout_sa.shape[0], dtype=int)],
        axis=0,
    )
    print(
        f"[tsne] offline transitions used={offline_sa.shape[0]:,} "
        f"rollout transitions used={rollout_sa.shape[0]:,} "
        f"total={all_sa.shape[0]:,}"
    )
    return all_sa, labels


def run_tsne(all_sa: np.ndarray, labels: np.ndarray, perplexity: float, n_iter: int, random_seed: int):
    scaler = StandardScaler()
    all_sa_scaled = scaler.fit_transform(all_sa)

    print(
        f"[tsne] points={all_sa_scaled.shape[0]:,}, perplexity={perplexity}, n_iter={n_iter}, seed={random_seed}"
    )
    # sklearn changed TSNE arg name from n_iter -> max_iter in newer versions.
    tsne_kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        random_state=random_seed,
        verbose=1,
        init="pca",
        learning_rate="auto",
    )
    try:
        tsne = TSNE(**tsne_kwargs, n_iter=n_iter)
    except TypeError:
        tsne = TSNE(**tsne_kwargs, max_iter=n_iter)
    emb = tsne.fit_transform(all_sa_scaled)
    emb_offline = emb[labels == 0]
    emb_rollout = emb[labels == 1]
    return emb_offline, emb_rollout


def draw_panel(
    ax,
    emb_offline: np.ndarray,
    emb_rollout: np.ndarray,
    title: str,
    point_size: float,
    offline_alpha: float,
    policy_alpha: float,
):
    # Match the notebook scatter styling:
    # offline: s=8 alpha=0.35, policy: s=12 alpha=0.55
    # (point_size default is 8.0, so policy uses 1.5x).
    offline_s = point_size
    policy_s = point_size * 1.5

    ax.scatter(
        emb_offline[:, 0],
        emb_offline[:, 1],
        c="steelblue",
        alpha=offline_alpha,
        s=offline_s,
        label="Offline dataset",
        linewidths=0,
    )
    ax.scatter(
        emb_rollout[:, 0],
        emb_rollout[:, 1],
        c="tomato",
        alpha=policy_alpha,
        s=policy_s,
        label="Policy rollout",
        linewidths=0,
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    # Notebook doesn't enable grid.


def save_single_figure(
    out_path: Path,
    emb_offline: np.ndarray,
    emb_rollout: np.ndarray,
    title: str,
    args: argparse.Namespace,
):
    fig, ax = plt.subplots(figsize=(10, 8))
    draw_panel(
        ax=ax,
        emb_offline=emb_offline,
        emb_rollout=emb_rollout,
        title=title,
        point_size=args.point_size,
        offline_alpha=args.offline_alpha,
        policy_alpha=args.policy_alpha,
    )
    ax.legend(markerscale=3, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    if args.save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def save_combined_figure(
    out_path: Path,
    task: str,
    model_results: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    panel_cols: int,
    args: argparse.Namespace,
):
    n = len(model_results)
    ncols = max(1, panel_cols)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.6 * ncols, 4.0 * nrows),
        squeeze=False,
    )

    flat_axes = axes.flatten()
    for i, (label, emb_off, emb_roll) in enumerate(model_results):
        draw_panel(
            ax=flat_axes[i],
            emb_offline=emb_off,
            emb_rollout=emb_roll,
            title=label,
            point_size=args.point_size,
            offline_alpha=args.offline_alpha,
            policy_alpha=args.policy_alpha,
        )

    for j in range(n, len(flat_axes)):
        flat_axes[j].axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=2,
        markerscale=3,
        fontsize=11,
        frameon=True,
    )
    fig.suptitle(f"t-SNE Support Overlap: {task}", y=1.02, fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    if args.save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("_")


def main() -> None:
    import gym

    args = parse_args()
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    apply_plot_style()

    specs = parse_policy_specs(args)
    output_dir = Path(args.output_dir) / sanitize_filename(args.task)
    output_dir.mkdir(parents=True, exist_ok=True)

    offline_obs, offline_next_obs, offline_actions = load_dataset(args.task, args.dataset_path)
    env = None
    obs_dim = offline_obs.shape[1]
    action_dim = offline_actions.shape[1]

    if args.support_source == "rollout":
        # D4RL registers env IDs on import; required even if dataset is loaded from file.
        # `mujoco_py` sometimes requires /usr/lib/nvidia to be present in LD_LIBRARY_PATH.
        # We patch it in automatically to avoid runtime failures.
        nvidia_lib = "/usr/lib/nvidia"
        if os.path.isdir(nvidia_lib):
            ld = os.environ.get("LD_LIBRARY_PATH", "")
            if nvidia_lib not in ld.split(":"):
                os.environ["LD_LIBRARY_PATH"] = (ld + (":" if ld else "") + nvidia_lib).strip(":")

        try:
            import d4rl  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "Failed to import d4rl. Rollout mode requires D4RL to register Gym env IDs."
            ) from exc

        env_id = args.env_id or args.task
        env = gym.make(env_id)
        obs_dim = env.observation_space.shape[0]
        action_dim = int(np.prod(env.action_space.shape))

    model_results: List[Tuple[str, np.ndarray, np.ndarray]] = []

    for spec in specs:
        policy_path = clean_path(spec.path)
        if not os.path.exists(policy_path):
            print(f"[skip] missing policy checkpoint: {policy_path}")
            continue

        print(f"[policy] {spec.label} -> {policy_path}")
        if os.path.getsize(policy_path) == 0:
            print(f"[skip] empty policy checkpoint: {policy_path}")
            continue

        policy = build_policy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_q_ensemble=args.num_q_ensemble,
            device=args.device,
        )
        try:
            try:
                state_dict = torch.load(policy_path, map_location=args.device, weights_only=False)
            except TypeError:
                state_dict = torch.load(policy_path, map_location=args.device)
            policy.load_state_dict(state_dict, strict=False)
            policy.eval()
        except Exception as exc:
            size_bytes = os.path.getsize(policy_path)
            print(
                f"[skip] failed to load policy '{spec.label}' "
                f"from '{policy_path}' (size={size_bytes} bytes): "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        if args.support_source == "rollout":
            rollout_next_obs, rollout_actions = collect_rollouts(
                policy=policy,
                env=env,
                n_episodes=args.n_rollout_episodes,
                deterministic=args.deterministic,
            )
        else:
            rollout_next_obs, rollout_actions = collect_policy_support_from_offline(
                policy=policy,
                offline_obs=offline_obs,
                offline_next_obs=offline_next_obs,
                max_samples=args.max_rollout_samples,
                deterministic=args.deterministic,
                random_seed=args.random_seed,
            )
        all_sa, labels = prepare_tsne_input(
            offline_next_obs=offline_next_obs,
            offline_actions=offline_actions,
            rollout_next_obs=rollout_next_obs,
            rollout_actions=rollout_actions,
            max_offline_samples=args.max_offline_samples,
            max_rollout_samples=args.max_rollout_samples,
            random_seed=args.random_seed,
        )
        emb_offline, emb_rollout = run_tsne(
            all_sa=all_sa,
            labels=labels,
            perplexity=args.perplexity,
            n_iter=args.n_iter,
            random_seed=args.random_seed,
        )

        single_name = f"{sanitize_filename(spec.label)}_tsne.png"
        single_path = output_dir / single_name
        save_single_figure(
            out_path=single_path,
            emb_offline=emb_offline,
            emb_rollout=emb_rollout,
            title=f"{spec.label} ({args.task})",
            args=args,
        )
        print(f"[saved] {single_path}")
        model_results.append((spec.label, emb_offline, emb_rollout))

    if env is not None:
        env.close()

    if not model_results:
        raise RuntimeError("No figures generated. Check policy paths and environment setup.")

    panel_path = output_dir / "combined_panel.png"
    save_combined_figure(
        out_path=panel_path,
        task=args.task,
        model_results=model_results,
        panel_cols=args.panel_cols,
        args=args,
    )
    print(f"[saved] {panel_path}")

    manifest = {
        "task": args.task,
        "output_dir": str(output_dir),
        "num_models": len(model_results),
        "models": [{"label": m[0]} for m in model_results],
        "combined_panel": str(panel_path),
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[saved] {manifest_path}")


if __name__ == "__main__":
    main()

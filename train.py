import argparse
import random
import pickle

import gym
import d4rl
import neorl
import json
import os
import numpy as np
import torch


from models.nets import MLP
from models.actor_critic import ActorProb, Critic
from models.dist import TanhDiagGaussian
from models.dynamics_model import EnsembleDynamicsModel
from dynamics import EnsembleDynamics
from utils.scaler import StandardScaler
from utils.termination_fns import get_termination_fn
from utils.load_dataset import qlearning_dataset, load_neorl_dataset, normalize_rewards
from utils.buffer import ReplayBuffer
from utils.logger import Logger, make_log_dirs
from utils.policy_trainer import PolicyTrainer
from policies import MOBILEPolicy
from configs import loaded_args
from vae_module.vae import VAE
from realnvp_module.realnvp import RealNVP
from kde_module.kde import PercentileThresholdKDE


from neuralODE.neural_ode_density import ContinuousNormalizingFlow, ODEFunc
from neuralODE.neural_ode_ood import NeuralODEOOD

from diffusion.monte_carlo_sampling_unconditional import build_model_from_ckpt
from diffusion.ddim_training_unconditional import log_prob_elbo
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import yaml

def get_args():
    print("Running", __file__)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="config_gormpo/hopper_medium_expert_78.yaml")
    config_args, remaining_argv = config_parser.parse_known_args()
    if config_args.config:
        with open(config_args.config, "r") as f:
            config = yaml.safe_load(f)
            config = {k.replace("-", "_"): v for k, v in config.items()}
    else:
        config = {}
    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--algo-name", type=str, default="mobile_gormpo")
    parser.add_argument("--task", type=str, default="hopper-medium-expert-v2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:3" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to saved d4rl dataset pickle file")
    parser.add_argument('--dynamics_path', type=str, default="")
    parser.add_argument('--reward_penalty_coef', type=float, default=0.5)
    parser.add_argument('--classifier_model_name', type=str, default='')
    parser.add_argument('--use_wandb', action='store_true', default=True)
    parser.add_argument('--wandb_project', type=str, default='mobile-gormpo')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.set_defaults(**config)
    known_args, _ = parser.parse_known_args()
    default_args = loaded_args[known_args.task]
    for arg_key, default_value in default_args.items():
        parser.add_argument(f'--{arg_key}', default=default_value, type=type(default_value))
    # 5. Final parse (command line still wins over YAML)
    args = parser.parse_args(remaining_argv)
    args.config = config_args.config
    print(args.config)

    return args



class DiffusionDensityWrapper:
    """Wrapper for diffusion model to provide score_samples interface."""

    def __init__(self, model, scheduler, target_dim, device):
        self.model = model
        self.scheduler = scheduler
        self.target_dim = target_dim
        self.device = device

    @torch.no_grad()
    def score_samples(self, x, device=None):
        """
        Compute log probability using ELBO from unconditional diffusion model.

        Args:
            x: Input samples (numpy array or tensor) of shape (batch_size, target_dim)
            device: Device to use (optional)

        Returns:
            Log probabilities normalized by target dimension
        """
        if device is None:
            device = self.device

        # Convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        x = x.to(device)

        # Compute log probability using ELBO
        log_probs = log_prob_elbo(
            model=self.model,
            scheduler=self.scheduler,
            x0=x,
            num_inference_steps=100,
            device=device,
        )

        # Normalize by target dimension for consistency
        log_probs_per_dim = log_probs

        return log_probs_per_dim


def train(args=get_args()):
    # create env and dataset
    if args.dataset_path:
        # Load dataset from pickle file
        with open(args.dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        # Still need to create env for environment specs
        if hasattr(args, 'domain') and args.domain == "neorl":
            task, version, data_type = tuple(args.task.split("-"))
            env = neorl.make(task+'-'+version)
        else:
            env = gym.make(args.task)
    else:
        assert args.domain in ["gym", "adroit", "neorl"]
        if args.domain == "neorl":
            task, version, data_type = tuple(args.task.split("-"))
            env = neorl.make(task+'-'+version)
            dataset = load_neorl_dataset(env, data_type)
        else:
            env = gym.make(args.task)
            dataset = qlearning_dataset(env)
    if args.norm_reward:
        # dataset = normalize_rewards(dataset)
        r_mean, r_std = dataset["rewards"].mean(), dataset["rewards"].std()
        dataset["rewards"] = (dataset["rewards"] - r_mean) / (r_std + 1e-3)

    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critics = []
    for i in range(args.num_q_ensemble):
        critic_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
        critics.append(Critic(critic_backbone, args.device))
    critics = torch.nn.ModuleList(critics)
    critics_optim = torch.optim.Adam(critics.parameters(), lr=args.critic_lr)

    if args.lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)
    else:
        lr_scheduler = None

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create dynamics
    load_dynamics_model = True if args.load_dynamics_path else False
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=args.device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )
    scaler = StandardScaler()
    termination_fn = get_termination_fn(task=args.task)

    print("reward_penalty_coef:", args.reward_penalty_coef)
    print("penalty_coef:", args.penalty_coef)

    if "vae" in args.classifier_model_name:
        classifier = VAE(
            # hidden_dims= args.vae_hidden_dims,
            device=args.device 
        ).to(args.device)
        classifier_dict = classifier.load_model(args.classifier_model_name, device=args.device)
        print("vae laoded")
    elif "realnvp" in args.classifier_model_name:
        classifier = RealNVP(
        device=args.device 
        ).to(args.device )
        classifier_dict = classifier.load_model(args.classifier_model_name, device=args.device)
    elif "kde" in args.classifier_model_name:
        # Extract device ID from args.device (e.g., "cuda:0" -> 0)
        # devid = int(args.device.split(":")[-1]) if "cuda" in args.device else 0
        devid = 2
        classifier = PercentileThresholdKDE(
        devid=devid,
        
        )
        classifier_dict = classifier.load_model(args.classifier_model_name, use_gpu=True , devid=devid)
    elif "neuralODE" in args.classifier_model_name:
        print("Loading Neural ODE based classifier... for task:", args.task)
        # Use the new NeuralODEOOD.load_model interface
        device = args.device if torch.cuda.is_available() else "cpu"

        # Load model using NeuralODEOOD wrapper
        # target_dim is read from metadata, no need to pass it explicitly
        classifier_dict = NeuralODEOOD.load_model(
            save_path=args.classifier_model_name.replace('_model.pt', ''),
            device=device
        )
        # classifier_dict now contains: {'model': ood_model, 'threshold': ..., 'mean': ..., 'std': ...}
        # Rename 'threshold' to 'thr' for compatibility with transition_model
        classifier_dict['thr'] = classifier_dict['threshold']
    elif "diffusion" in args.classifier_model_name:
        print("Loading Diffusion based classifier... for task:", args.task)
        # Load model using build_model_from_ckpt from monte_carlo_sampling_unconditional
        device = args.device
        ckpt_path = args.classifier_model_name
        sched_dir = os.path.dirname(ckpt_path) + "/scheduler"

        # Build model
        model, cfg = build_model_from_ckpt(ckpt_path, device)

        # Get target dimension
        ckpt = torch.load(ckpt_path, map_location=device)
        target_dim = ckpt.get("target_dim")

        # Load scheduler
        try:
            scheduler = DDIMScheduler.from_pretrained(sched_dir)
        except Exception:
            try:
                scheduler = DDPMScheduler.from_pretrained(sched_dir)
            except Exception as e:
                print(f"Warning: Could not load scheduler: {e}")
                scheduler = DDIMScheduler(
                    num_train_timesteps=1000,
                    beta_schedule="linear",
                    prediction_type="epsilon",
                )

        # Wrap in our interface
        diffusion_wrapper = DiffusionDensityWrapper(model, scheduler, target_dim, device)

        # Load threshold from metrics if available
        thr_path = f"diffusion/monte_carlo_results/{args.task.lower().split('_')[0].split('-')[0]}_unconditional_ddpm/elbo_metrics.json"
        if os.path.exists(thr_path):
            with open(thr_path, 'r') as f:
                metrics = json.load(f)
            thr = metrics.get("percentile_1.0_logp", 0.0)
        else:
            print(f"Warning: Threshold file not found at {thr_path}, using default threshold 0.0")
            thr = 0.0

        classifier_dict = {'model': diffusion_wrapper, 'thr': thr}
    

    dynamics = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn,
        classifier=classifier_dict,
        penalty_coef = args.reward_penalty_coef,
        device=args.device

    )

    if args.load_dynamics_path:
        dynamics.load(args.load_dynamics_path)

    # create policy
    policy = MOBILEPolicy(
        dynamics,
        actor,
        critics,
        actor_optim,
        critics_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        penalty_coef=args.penalty_coef,
        num_samples=args.num_samples,
        deterministic_backup=args.deterministic_backup,
        max_q_backup=args.max_q_backup
    )

    # create buffer
    real_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    real_buffer.load_dataset(dataset)

    fake_buffer = ReplayBuffer(
        buffer_size=args.rollout_batch_size*args.rollout_length*args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )

    # log
    _clf = args.classifier_model_name
    if "diffusion" in _clf:
        args.classifier_type = "diffusion"
    elif "realnvp" in _clf:
        args.classifier_type = "realnvp"
    elif "vae" in _clf:
        args.classifier_type = "vae"
    elif "kde" in _clf:
        args.classifier_type = "kde"
    elif "neuralODE" in _clf:
        args.classifier_type = "neuralODE"
    else:
        args.classifier_type = ""
    log_dirs = make_log_dirs(
        args.task, args.algo_name, args.seed, vars(args),
        record_params=["penalty_coef", "rollout_length", "classifier_type"]
    )
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.task}__{args.algo_name}__seed{args.seed}",
            config=vars(args),
        )
        output_config["wandb_run"] = "wandb"
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # create policy trainer
    policy_trainer = PolicyTrainer(
        policy=policy,
        eval_env=env,
        real_buffer=real_buffer,
        fake_buffer=fake_buffer,
        logger=logger,
        rollout_setting=(args.rollout_freq, args.rollout_batch_size, args.rollout_length),
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler
    )

    # train
    if args.dynamics_path == "":
        dynamics.train(
            real_buffer.sample_all(),
            logger,
            max_epochs_since_update=args.max_epochs_since_update,
            max_epochs=args.dynamics_max_epochs
        )
    else:
        dynamics.load(args.dynamics_path)
    
    policy_trainer.train()


if __name__ == "__main__":
    train()
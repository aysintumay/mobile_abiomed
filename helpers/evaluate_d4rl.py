import argparse
import datetime
import os
import random
import time
import importlib
import wandb 
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import sys
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import gymnasium as gym
import gym
# import d4rl

import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from GORMPO.models.policy_models  import MLP, ActorProb, Critic, DiagGaussian
from GORMPO.algo.sac import SACPolicy


import warnings
warnings.filterwarnings("ignore")
from wrapper import (
                        RandomNormalNoisyActions, 
                        RandomNormalNoisyTransitions,
                        RandomNormalNoisyTransitionsActions
                        )
                    


"""
This script supports gymnasium environments but not D4RL package. See the difference:
gymnasium: >= 0.26.3
    observation, info = env.reset()
    observation, reward, terminal,truncated, info =  env.step(action)

gym: < 0.26.3
    observation = env.reset()
    observation, reward, terminal, info = env.step(action)

D4RL uses old gym whereas gymnasium uses the new Gymnasium API.
"""
def get_mopo(args):


    # import configs
    task = args.task.split('-')[0]
    import_path = f"static_fns.{task.lower()}"
    # static_fns = importlib.import_module(import_path).StaticFns
    config_path = f"config.{task.lower()}"
    # config = importlib.import_module(config_path).default_config
    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=[256, 256])
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=[256, 256])
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )

    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        # target_entropy = args.target_entropy if args.target_entropy \
        #     else -np.prod(env.action_space.shape)
        target_entropy = -np.prod(env.action_space.shape)
        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    # create policy
    sac_policy = SACPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        dist=dist,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        device=args.device
    )
    print('mopo is migrated to', args.device)
    policy_state_dict = torch.load(args.policy_path, map_location=args.device)
    sac_policy.load_state_dict(policy_state_dict)
    return sac_policy


def _evaluate(policy, eval_env, episodes, args, plot=None):
        policy.eval()
        obs = eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < episodes:
            action = policy.sample_action(obs, deterministic=True)
            next_obs, reward, terminal, info = eval_env.step(action) #next_obs = world model forecast
            episode_reward += reward
            episode_length += 1

            obs = next_obs  #next_obs = world model forecast

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )

                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = eval_env.reset()
        eval_info = {
                        "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
                        "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
                    }
        ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
        ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
        return {
            "mean_return":ep_reward_mean,
            "std_return": ep_reward_std,
            "mean_length": ep_length_mean,
            "std_length": ep_length_std
        }


def get_env():
   
    env = gym.make(args.task)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)

    if args.action and not args.transition:
        print("Environment with noisy actions")
        env = RandomNormalNoisyActions(env=env, noise_rate=args.noise_rate_action, loc = args.loc, scale = args.scale_action)
    elif args.transition and not args.action:
        print("Environment with noisy transitions")
        env = RandomNormalNoisyTransitions(env=env, noise_rate=args.noise_rate_transition, loc = args.loc, scale = args.scale_transition)
    elif args.transition and args.action:
        print("Environment with noisy actions and transitions")
        env = RandomNormalNoisyTransitionsActions(env=env, noise_rate_action=args.noise_rate_action, loc = args.loc, scale_action = args.scale_action,\
                                                        noise_rate_transition=args.noise_rate_transition, scale_transition = args.scale_transition)
    else:
        print("Environment without noise")
        env = env

    return env

def mopo_args(parser):

    g = parser.add_argument_group("MOPO hyperparameters")
    g.add_argument("--actor-lr", type=float, default=3e-4)
    g.add_argument("--critic-lr", type=float, default=3e-4)
    g.add_argument("--gamma", type=float, default=0.99)
    g.add_argument("--tau", type=float, default=0.005)
    g.add_argument("--alpha", type=float, default=0.2)
    g.add_argument('--auto-alpha', default=True)
    g.add_argument('--target-entropy', type=int, default=-1) #-action_dim
    g.add_argument('--alpha-lr', type=float, default=3e-4)

    # dynamics model's arguments
    g.add_argument("--dynamics-lr", type=float, default=0.001)
    g.add_argument("--n-ensembles", type=int, default=7)
    g.add_argument("--n-elites", type=int, default=5)
    g.add_argument("--reward-penalty-coef", type=float, default=5e-3) #1e=6
    g.add_argument("--rollout-length", type=int, default=5) #1 
    g.add_argument("--rollout-batch-size", type=int, default=5000) #50000
    g.add_argument("--rollout-freq", type=int, default=1000)
    g.add_argument("--model-retain-epochs", type=int, default=5)
    g.add_argument("--real-ratio", type=float, default=0.05)
    g.add_argument("--dynamics-model-dir", type=str, default=None)
    g.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")


    g.add_argument("--epoch", type=int, default=600) #1000
    g.add_argument("--step-per-epoch", type=int, default=1000) 
    #1000
    g.add_argument("--batch-size", type=int, default=256)
    return parser

def bcq_args(parser):
    parser.add_argument("--eval_freq", default=2e4, type=float)     # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment or train for (this defines buffer size)
    parser.add_argument("--rand_action_p", default=0.3, type=float) # Probability of selecting random action during batch generation
    parser.add_argument("--gaussian_std", default=0.3, type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
    parser.add_argument("--batch_size", default=100, type=int)      # Mini batch size for networks
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--lmbda", default=0.75)                    # Weighting for clipped double Q-learning in BCQ
    parser.add_argument("--phi", default=0.05)                      # Max perturbation hyper-parameter for BCQ
    return parser
    
if __name__ == "__main__":
    print("Running", __file__)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining_argv = config_parser.parse_known_args()
    if config_args.config:
        with open(config_args.config, "r") as f:
            config = yaml.safe_load(f)
            config = {k.replace("-", "_"): v for k, v in config.items()}
    else:
        config = {}
    base = argparse.ArgumentParser(parents=[config_parser], add_help=False)
    base.add_argument(
        "--algo-name",
        choices=["mbpo","mopo",'gormpo',"bcq","bc","physician"],
        default="mopo",
        help="Which algorithm’s flags to load"
    )
    args_partial, remaining_argv = base.parse_known_args()
    parser = argparse.ArgumentParser(
        # keep the base flags and auto‐help
        parents=[base],
        description="Test your RL method"
    )

    parser.add_argument("--task", type=str, default="abiomed")
    parser.add_argument("--policy_path" , type=str,
                         default="/abiomed/models/policy_models/mopo/abiomed/seed_1_0902_191735-abiomed_mopo/policy_abiomed.pth")
    parser.add_argument("--model_path" , type=str, default="saved_models")
    parser.add_argument(
                    "--devid", 
                    type=int,
                    default=7,
                    help="Which GPU device index to use"
                )

    parser.add_argument("--seeds", type=int, nargs='+', default=[1,2,3,4,5], help="List of seeds for evaluation")
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--log-freq", type=int, default=1000)
    
    #===============Noisy Environment Arguments================
    parser.add_argument("--noise_rate_action", type=float, help="Portion of action to be noisy with probability", default=0.01)
    parser.add_argument("--noise_rate_transition", type=float, help="Portion of transitions to be noisy with probability", default=0.01)
    parser.add_argument("--loc", type=float, default=0.0, help="Mean of the noise distribution")
    parser.add_argument("--scale_action", type=float, default=0.001, help="Standard deviation of the action noise distribution")
    parser.add_argument("--scale_transition", type=float, default=0.001, help="Standard deviation of the transition noise distribution")
    parser.add_argument("--action", action='store_true', help="Create dataset with noisy actions")
    parser.add_argument("--transition", action='store_true', help="Create dataset with noisy transitions")

    
    
    if args_partial.algo_name == "mopo":
        mopo_args(parser)
    elif args_partial.algo_name == "mbpo":
        mopo_args(parser)
        
    elif args_partial.algo_name == "gormpo":
        mopo_args(parser)
    elif args_partial.algo_name == "physician":
        mopo_args(parser)
    elif args_partial.algo_name == "bcq":
        bcq_args(parser)
    # elif args_partial.algo_name == "bcq":
    #     bcq_args(parser)
    # else:
    #     bc_args(parser)
    parser.set_defaults(**config)

    # 5. Final parse (command line still wins over YAML)
    args = parser.parse_args(remaining_argv)
    args.config = config_args.config
    print(args.config)

    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    wandb.init(
    project="mopo-eval",
    name=f"eval_{args.task}_{args.algo_name}_{t0}",
    config=vars(args)
        )
    
    results = []
    for seed in args.seeds:
        args.seed = seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

       
        
        log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_{args.algo_name}'
        log_path = os.path.join(args.logdir, args.task, args.algo_name, log_file)

        model_path = os.path.join(args.model_path, args.task, args.algo_name, log_file)
       
        args.device = f'cuda:{args.devid}'

        
        env = get_env() 
        policy = get_mopo(args)
        eval_info = _evaluate(policy, env, args.eval_episodes, args,plot=True)
        
        mean_return = eval_info["mean_return"]
        std_return = eval_info["std_return"]
        mean_length = eval_info["mean_length"]
        std_length = eval_info["std_length"]

    
        results.append({
            'seed': seed,
            'mean_return': mean_return,
            'std_return': std_return,
            'mean_length': mean_length,
            'std_length': std_length,
           
        })
        
        print(f"Mean Return: {mean_return:.2f} ± {std_return:.2f}")
        # Save results to CSV
    os.makedirs(os.path.join('results', args.task, args.algo_name), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_path = os.path.join('results', args.task, args.algo_name, f"{args.task}_results_{t0}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
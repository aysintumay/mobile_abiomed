import numpy as np
import torch

import os
import sys
import random
import imageio
import gym
import tqdm
from tqdm import trange
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from noisy_mujoco.abiomed_env.rl_env import AbiomedRLEnvFactory

from noisy_mujoco.abiomed_env.cost_func import (compute_acp_cost, 
                                                unstable_percentage_model_merged,
                                                unstable_percentage_model_gradient,
                                                weaning_score_model_merged,
                                                compute_acp_cost_model,
                                                weaning_score_model_gradient,
                                                compute_air_aggregate_gradient_threshold,
                                                compute_map_model_air, 
                                                compute_hr_model_air,
                                                compute_pulsatility_model_air,
                                                aggregate_air_model, 
                                                weaning_score_model, 
                                                unstable_percentage_model, 
                                                super_metric)



def eval_policy_simple(policy, env_name, seed,  eval_episodes=10, args=None, seed_offset=100):
    """
    Simplified evaluation function that doesn't require logger or video recorder.
    Compatible with both standard gym environments and abiomed environment.

    Args:
        policy: The policy to evaluate
        env_name: Name of the environment
        seed: Random seed
        mean: State normalization mean
        std: State normalization std
        eval_episodes: Number of episodes to evaluate (default: 10)
        args: Optional args object for abiomed environment (required if env_name == "abiomed")
        seed_offset: Seed offset for evaluation (default: 100)

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    if env_name == "abiomed":
        if args is None:
            raise ValueError("args parameter is required for abiomed environment")

        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from noisy_mujoco.abiomed_env.rl_env import AbiomedRLEnvFactory

        eval_env = AbiomedRLEnvFactory.create_env(
            model_name=args.model_name,
            model_path=args.model_path,
            data_path=args.data_path,
            max_steps=args.max_steps,
            gamma1=args.gamma1,
            gamma2=args.gamma2,
            gamma3=args.gamma3,
            action_space_type=args.action_space_type,
            reward_type="smooth",
            normalize_rewards=True,
            noise_rate=args.noise_rate,
            noise_scale=args.noise_scale,
            seed=seed + seed_offset,
            device=f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu",
        )

        avg_reward = 0.0
        avg_acp = 0.0
        total_map_air_sum = 0.0
        total_hr_air_sum = 0.0
        total_pulsatility_air_sum = 0.0
        total_aggregate_air_sum = 0.0
        total_unstable_percentage_sum = 0.0
        total_super_sum = 0.0
        wean_score_sum = 0.0
        ws_merged_sum = 0.0
        ws_thr_sum = 0.0
        total_unstable_percentage_gradient_sum = 0.0
        total_unstable_percentage_merged_sum = 0.0

        for k in trange(eval_episodes, desc="Evaluating"):
            ep_states = []
            (state, info), done = eval_env.reset(), False
            truncated = False

            while not (done or truncated):
                action = policy.select_action(s_norm)
                next_state, reward, done, truncated, _ = eval_env.step(action)
                avg_reward += reward
                ep_states.append(state)
                state = next_state

            ep_states_np = np.asarray(ep_states, dtype=np.float32)
            wm = eval_env.world_model

            avg_acp += compute_acp_cost_model(wm, eval_env.episode_actions, ep_states_np)
            total_map_air_sum += compute_acp_cost_model(wm, eval_env.episode_actions, ep_states_np)
            total_hr_air_sum += compute_hr_model_air(wm, ep_states_np, eval_env.episode_actions)
            total_pulsatility_air_sum += compute_pulsatility_model_air(wm, ep_states_np, eval_env.episode_actions)
            total_aggregate_air_sum += compute_air_aggregate_gradient_threshold(wm, ep_states_np, eval_env.episode_actions)
            total_super_sum += super_metric(wm, ep_states_np, eval_env.episode_actions)
            wean_score_sum += weaning_score_model_gradient(wm, ep_states_np, eval_env.episode_actions)[0]
            ws_merged_sum += weaning_score_model_merged(wm, ep_states_np, eval_env.episode_actions)
            ws_thr_sum += weaning_score_model(wm, ep_states_np, eval_env.episode_actions)
            total_unstable_percentage_sum += unstable_percentage_model(wm, ep_states_np)
            total_unstable_percentage_gradient_sum += unstable_percentage_model_gradient(wm, ep_states_np)
            total_unstable_percentage_merged_sum += unstable_percentage_model_merged(wm, ep_states_np)

        avg_reward /= eval_episodes
        acp_mean = avg_acp / eval_episodes
        map_air_mean = total_map_air_sum / eval_episodes
        hr_air_mean = total_hr_air_sum / eval_episodes
        puls_air_mean = total_pulsatility_air_sum / eval_episodes
        aggregate_air_mean = total_aggregate_air_sum / eval_episodes
        unstable_hours_mean = total_unstable_percentage_sum / eval_episodes
        total_unstable_percentage_gradient_mean = total_unstable_percentage_gradient_sum / eval_episodes
        total_unstable_percentage_merged_mean = total_unstable_percentage_merged_sum / eval_episodes
        weaning_score_mean = wean_score_sum / eval_episodes
        wean_merged = ws_merged_sum / eval_episodes
        ws_thr_mean = ws_thr_sum / eval_episodes
        super_mean = total_super_sum / eval_episodes

        results = {
            'avg_reward': avg_reward,
            'acp': acp_mean,
            'map_air': map_air_mean,
            'hr_air': hr_air_mean,
            'pulsatility_air': puls_air_mean,
            'aggregate_air': aggregate_air_mean,
            'unstable_hours_pct': unstable_hours_mean,
            'unstable_hours_gradient_pct': total_unstable_percentage_gradient_mean,
            'unstable_hours_merged_pct': total_unstable_percentage_merged_mean,
            'weaning_score': weaning_score_mean,
            'weaning_score_merged': wean_merged,
            'weaning_score_threshold': ws_thr_mean,
            'super_metric': super_mean,
        }

        print("\n" + "="*80)
        print(f"ABIOMED EVALUATION RESULTS OVER {eval_episodes} EPISODES")
        print("="*80)
        print(f"Average Return:    {avg_reward:.3f}")
        print(f"ACP:               {acp_mean:.4f}")
        print(f"MAP AIR/ep:        {map_air_mean:.5f}")
        print(f"HR AIR/ep:         {hr_air_mean:.5f}")
        print(f"Pulsatility AIR/ep: {puls_air_mean:.5f}")
        print(f"Aggregate AIR/ep:  {aggregate_air_mean:.5f}")
        print(f"Unstable hours (%): {unstable_hours_mean:.2f}")
        print(f"Unstable hours gradient (%): {total_unstable_percentage_gradient_mean:.2f}")
        print(f"Unstable hours merged (%): {total_unstable_percentage_merged_mean:.2f}")
        print(f"Weaning score:     {weaning_score_mean:.5f}")
        print(f"Weaning score merged: {wean_merged:.5f}")
        print(f"Weaning score threshold: {ws_thr_mean:.5f}")

       

        return results
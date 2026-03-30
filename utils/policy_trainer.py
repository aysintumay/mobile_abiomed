import time
import os

import numpy as np
import torch
import gym
import matplotlib.pyplot as plt

from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
from collections import deque
from utils.buffer import ReplayBuffer
from utils.logger import Logger
from policies import BasePolicy

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


# model-based policy trainer
class PolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        real_buffer: ReplayBuffer,
        fake_buffer: ReplayBuffer,
        logger: Logger,
        rollout_setting: Tuple[int, int, int],
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        real_ratio: float = 0.05,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self.policy = policy
        self.eval_env = eval_env
        self.real_buffer = real_buffer
        self.fake_buffer = fake_buffer
        self.logger = logger

        self._rollout_freq, self._rollout_batch_size, \
            self._rollout_length = rollout_setting

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._real_ratio = real_ratio
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler

    def _log_reward_plot(self, reward_history: List[Tuple], num_timesteps: int) -> None:
        """Log a matplotlib reward curve (mean ± std) to wandb."""
        if _wandb is None or _wandb.run is None:
            return
        steps = [r[0] for r in reward_history]
        means = np.array([r[1] for r in reward_history])
        stds  = np.array([r[2] for r in reward_history])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, means, linewidth=2, label="mean")
        ax.fill_between(steps, means - stds, means + stds, alpha=0.25, label="±1 std")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Episode Reward")
        ax.set_title("Eval Reward over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _wandb.log({"eval/reward_curve": _wandb.Image(fig)}, step=num_timesteps)
        plt.close(fig)

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        reward_history: List[Tuple] = []  # (timestep, mean, std)
        # train loop
        for e in range(1, self._epoch + 1):

            self.policy.train()

            pbar = tqdm(range(self._step_per_epoch), desc=f"Epoch #{e}/{self._epoch}")
            for it in pbar:
                if num_timesteps % self._rollout_freq == 0:
                    init_obss = self.real_buffer.sample(self._rollout_batch_size)["observations"].cpu().numpy()
                    rollout_transitions, rollout_info = self.policy.rollout(init_obss, self._rollout_length)
                    self.fake_buffer.add_batch(**rollout_transitions)
                    self.logger.log(
                        "num rollout transitions: {}, reward mean: {:.4f}".\
                            format(rollout_info["num_transitions"], rollout_info["reward_mean"])
                    )
                    for _key, _value in rollout_info.items():
                        self.logger.logkv_mean("rollout_info/"+_key, _value)

                real_sample_size = int(self._batch_size * self._real_ratio)
                fake_sample_size = self._batch_size - real_sample_size
                real_batch = self.real_buffer.sample(batch_size=real_sample_size)
                fake_batch = self.fake_buffer.sample(batch_size=fake_sample_size)
                batch = {"real": real_batch, "fake": fake_batch}
                loss = self.policy.learn(batch)
                pbar.set_postfix(**loss)

                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)
                
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            if e % 50 == 0: #change to 50
                # evaluate current policy
                eval_info = self._evaluate()
                ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
                ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
                if hasattr(self.eval_env, "get_normalized_score"):
                    norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
                    norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
                    last_10_performance.append(norm_ep_rew_mean)
                    self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
                    self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
                    reward_history.append((num_timesteps, norm_ep_rew_mean, norm_ep_rew_std))
                else:
                    last_10_performance.append(ep_reward_mean)
                    self.logger.logkv("eval/episode_reward", ep_reward_mean)
                    self.logger.logkv("eval/episode_reward_std", ep_reward_std)
                    reward_history.append((num_timesteps, ep_reward_mean, ep_reward_std))
                self.logger.logkv("eval/episode_length", ep_length_mean)
                self.logger.logkv("eval/episode_length_std", ep_length_std)
                self.logger.set_timestep(num_timesteps)
                self.logger.dumpkvs(exclude=["dynamics_training_progress"])

                # log reward curve and episode histogram to wandb
                if _wandb is not None and _wandb.run is not None:
                    self._log_reward_plot(reward_history, num_timesteps)
                    _wandb.log(
                        {"eval/episode_reward_hist": _wandb.Histogram(eval_info["eval/episode_reward"])},
                        step=num_timesteps,
                    )
            
                # save checkpoint
                torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.policy.dynamics.save(self.logger.model_dir)
        self.policy.plot_pessimism_ratio()
        self.logger.close()
        return {"last_10_performance": np.mean(last_10_performance)}

    def _evaluate(self) -> Dict[str, List[float]]:
        self.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            action = self.policy.select_action(obs, deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
import numpy as np
import torch

from typing import Optional, Union, Tuple, Dict
import sys
import os
import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", '..')))

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

def get_env_data(args, val=None):
    if args.env[0].isupper():
        env = gym.make(args.env)
        with open(args.data_path, "rb") as f:
            print("opening")
            dataset = pickle.load(f)
        return env, dataset
    elif args.env == "abiomed":
        env = AbiomedRLEnvFactory.create_env(
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
            seed=args.seed,
            device=f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu",
        )
        if args.data_path is None:
            dataset1 = env.world_model.data_train
            dataset2 = env.world_model.data_val
            dataset3 = env.world_model.data_test
            dataset = [dataset1, dataset2, dataset3]
        else:
            try:
                with open(args.data_path, "rb") as f:
                    dataset = pickle.load(f)
                print("opened the pickle file for synthetic dataset")
            except:
                dataset = np.load(args.data_path)
                dataset = {k: dataset[k] for k in dataset.files}
                print("opened the npz file for synthetic dataset")

        if val:
            print("Validation data is supported for Abiomed dataset")
            dataset_val = env.world_model.data_test
            return env, dataset, dataset_val
        else:
            print("No validation data for Abiomed dataset")
            dataset_val = None
            return env, dataset


class ReplayBufferAbiomed(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_size=int(1e6),
        timesteps=6,
        feature_dim=12,
        device="cpu",
    ):
        self._max_size = max_size
        self._ptr = 0
        self._size = 0

        self.observations = np.zeros((max_size, state_dim))
        self.actions = np.zeros((max_size, action_dim))
        self.next_observations = np.zeros((max_size, state_dim))
        self.rewards = np.zeros((max_size, 1))
        self.terminals = np.zeros((max_size, 1))
        self.timesteps = timesteps
        self.feature_dim = feature_dim

        self.device = device
        self.action_space_type = "continuous" if action_dim == 1 else "discrete"

    def add(self, state, action, next_state, reward, done):
        self.observations[self._ptr] = state
        self.actions[self._ptr] = action
        self.next_observations[self._ptr] = next_state
        self.rewards[self._ptr] = reward
        self.terminals[self._ptr] = 1.0 - done

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def add_batch(self, obss, actions, next_obss, rewards, terminals):
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = obss
        self.actions[indexes] = actions.reshape(-1, 1) if actions.ndim == 1 else actions
        self.next_observations[indexes] = next_obss
        self.rewards[indexes] = rewards.reshape(-1, 1) if rewards.ndim == 1 else rewards
        self.terminals[indexes] = terminals.reshape(-1, 1) if terminals.ndim == 1 else terminals

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self._size, size=batch_size)

        return {
            "observations": torch.FloatTensor(self.observations[ind]).to(self.device),
            "actions": torch.FloatTensor(self.actions[ind]).to(self.device),
            "next_observations": torch.FloatTensor(self.next_observations[ind]).to(self.device),
            "rewards": torch.FloatTensor(self.rewards[ind]).to(self.device),
            "terminals": torch.FloatTensor(self.terminals[ind]).to(self.device),
        }

    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy()
        }
    def convert_abiomed(self, dataset, env, shuffle=None):
        if isinstance(dataset, dict):  # check if data is already in D4RL format
            self.observations = dataset["observations"]
            # self.actions = dataset["actions"]
            self.next_observations = dataset["next_observations"]
            self.rewards = dataset["rewards"].reshape(-1, 1)
            self.terminals = 1.0 - dataset["terminals"].reshape(-1, 1)
            self._size = self.observations.shape[0]
            # normalize actions
            if np.rint(np.array((env.world_model.unnorm_pl(dataset['actions'])).max() - (env.world_model.unnorm_pl(dataset['actions'])).min())) == 8:

                self.actions = dataset["actions"].reshape(-1,1)
            else:
                self.actions = np.asarray(
                    env.world_model.normalize_pl(torch.Tensor(dataset["actions"]))
                ).reshape(-1, 1)
            print("D4RL dataset for Abiomed")
        else:  # convert abiomed data into D4RL type
            if isinstance(dataset, list):
                all_x = torch.cat(
                    [dataset[0].data, dataset[1].data, dataset[2].data], axis=0
                )
                all_pl = torch.cat(
                    [dataset[0].pl, dataset[1].pl, dataset[2].pl], axis=0
                )
                all_labels = torch.cat(
                    [dataset[0].labels, dataset[1].labels, dataset[2].labels], axis=0
                )

                print(all_x.shape, all_pl.shape, all_labels.shape)

            else:
                all_x = dataset.data
                all_pl = dataset.pl
                all_labels = dataset.labels

            reward_l = []
            done_l = []
            observation = all_x.reshape(-1, self.timesteps * (self.feature_dim))
            next_observation = torch.cat(
                [
                    all_labels.reshape(-1, self.timesteps, self.feature_dim - 1),
                    all_pl.reshape(-1, self.timesteps, 1),
                ],
                axis=2,
            )
            next_observation = next_observation.reshape(
                -1, self.timesteps * (self.feature_dim)
            )

            action = all_pl
            # take one number with majority voting among 6 numbers
            action_unnorm = np.array(env.world_model.unnorm_pl(action))
            action_1 = np.array(
                [np.bincount(np.rint(a).astype(int)).argmax() for a in action_unnorm]
            ).reshape(-1, 1)
            # normalize back
            action = env.world_model.normalize_pl(torch.Tensor(action_1))
            obs_reshaped = (
                observation.reshape(-1, self.timesteps, self.feature_dim)
            ).clone()  # shape (1,6,12)
            for i in tqdm.tqdm(range(action.shape[0])):
                if (env.gamma1 != 0.0) or (env.gamma2 != 0.0) or (env.gamma3 != 0.0):
                    # change the last column of obs_reshaped with all_pl[i-1] after i==0
                    if i > 0:
                        obs_reshaped[i, :, -1] = all_pl[i - 1]
                    reward = env._compute_reward(
                        next_observation[i].reshape(
                            -1, self.timesteps, self.feature_dim
                        ),
                        obs_reshaped[i],
                        action_1[i],
                    )
                    # action2 = [action2[1], action_1[i+1]] if (i+1)< action.shape[0] else None

                else:
                    reward = env._compute_reward(
                        next_observation[i].reshape(
                            -1, self.timesteps, self.feature_dim
                        )
                    )

                reward_l.append(reward)
                done_l.append(np.array([0]))

            self.observations = np.array(observation)
            self.actions = np.array(action)
            self.next_observations = np.array(next_observation)
            self.rewards = np.array(reward_l).reshape(-1, 1)
            self.terminals = 1.0 - np.array(done_l).reshape(-1, 1)
            self._size = self.observations.shape[0]

            # print(self.observations.min(), self.observations.max())
            # print(self.actions.min(), self.actions.max())
            # print(self.next_observations.min(), self.next_observations.max())
            # print(self.rewards.min(), self.rewards.max())
            # print(self.terminals.min(), self.terminals.max())

    def normalize_states(self, eps=1e-3):
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean) / std
        # do not take p-level mean
        self.next_observations = (self.next_observations - mean) / std
        return mean, std


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self._ptr = 0
        self._size = 0
        
        self.observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)

        self.device = torch.device(device)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)
    
    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        self._ptr = len(observations)
        self._size = len(observations)
    
    def normalize_reward(self) -> None:
        min, max = self.rewards.min(), self.rewards.max()
        self.rewards = (self.rewards - min) / (max - min)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        
        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device)
        }
    
    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy()
        }
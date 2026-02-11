# %%


# %%
# load /public/sparse_d4rl/hopper-medium-v2.pkl
import pickle

file_path = '/public/sparse_d4rl/hopper-medium-v2.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Print the type and, if possible, the shape/length of the main elements
print(f"Type of loaded data: {type(data)}")
if isinstance(data, dict):
    for key, value in data.items():
        if hasattr(value, 'shape'):
            print(f"{key}: shape = {value.shape}")
        elif hasattr(value, '__len__'):
            print(f"{key}: length = {len(value)}")
        else:
            print(f"{key}: type = {type(value)}")
elif hasattr(data, 'shape'):
    print(f"Shape: {data.shape}")
elif hasattr(data, '__len__'):
    print(f"Length: {len(data)}")


# %%
# We want to check if, for some action taken at time t, the resulting next_observations[t] matches
# any observation (possibly at t+1). The typical structure of D4RL datasets is that:
# data['observations'][t] + data['actions'][t] => data['next_observations'][t]

# Let's check if any next_observation appears as an observation elsewhere,
# and if so, print the corresponding indices and actions.

import numpy as np

obs = data.get('observations')
next_obs = data.get('next_observations')
actions = data.get('actions')

if obs is not None and next_obs is not None and actions is not None:
    # reshape to (n, -1) for robust matching (flatten)
    flat_obs = obs.reshape(obs.shape[0], -1)
    flat_next_obs = next_obs.reshape(next_obs.shape[0], -1)

    # Build a mapping from flattened observation to index for fast lookup
    obs_map = {}
    for i, o in enumerate(flat_obs):
        obs_map[tuple(o.tolist())] = i

    matches_found = 0
    for i, nobs in enumerate(flat_next_obs):
        key = tuple(nobs.tolist())
        if key in obs_map:
            match_idx = obs_map[key]
            print(f"next_observations[{i}] matches observations[{match_idx}], action taken: {actions[i]}")
            matches_found += 1
    if matches_found == 0:
        print("No matches found between next_observations and observations.")
    else:
        print(f"{matches_found} matches found between next_observations and observations.")
else:
    print("One of 'observations', 'next_observations', or 'actions' not found in data.")


# %%
import os
import pickle

# Path where we will store reorganized dataset
output_dir = "/public/sparse_d4rl/mapped"
output_path = os.path.join(output_dir, "hopper-medium-v2.pkl")

# Map observation -> index for fast lookup
if obs is not None and next_obs is not None and actions is not None:
    # Build mapping: flattened obs -> index
    flat_obs = obs.reshape(obs.shape[0], -1)
    flat_next_obs = next_obs.reshape(next_obs.shape[0], -1)
    obs_map = {tuple(o.tolist()): i for i, o in enumerate(flat_obs)}

    reorganized_data = []
    for i, nobs in enumerate(flat_next_obs):
        key = tuple(nobs.tolist())
        if key in obs_map and (obs_map[key]+1 < len(actions)):  # Check valid indices
            match_idx = obs_map[key]
            item = {
                'observation': obs[i],
                'action': actions[i],
                'next_observation': next_obs[i],
                'next_action': actions[match_idx]
            }
            reorganized_data.append(item)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(reorganized_data, f)

    print(f"Saved reorganized dataset with {len(reorganized_data)} matching pairs to {output_path}")
else:
    print("One of 'observations', 'next_observations', or 'actions' not found, cannot reorganize.")


# %%

# Additional environments to map: halfcheetah and walker2d
import glob

envs = ["hopper", "halfcheetah", "walker2d"]
base_input_dir = "/public/sparse_d4rl"
base_output_dir = "/public/sparse_d4rl/mapped"

for env in envs:
    # Find a d4rl dataset file for this env (try _medium-v2 for consistency)
    input_paths = sorted(glob.glob(os.path.join(base_input_dir, f"{env}-medium-v2.pkl")))
    if not input_paths:
        print(f"No dataset found for {env}-medium-v2 in {base_input_dir}")
        continue
    input_path = input_paths[0]
    output_path = os.path.join(base_output_dir, f"{env}-medium-v2.pkl")

    # Load original env dataset
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    # Try to find required arrays
    obs = data.get('observations', None)
    next_obs = data.get('next_observations', None)
    actions = data.get('actions', None)

    if obs is not None and next_obs is not None and actions is not None:
        flat_obs = obs.reshape(obs.shape[0], -1)
        flat_next_obs = next_obs.reshape(next_obs.shape[0], -1)
        obs_map = {tuple(o.tolist()): i for i, o in enumerate(flat_obs)}

        reorganized_data = []
        for i, nobs in enumerate(flat_next_obs):
            key = tuple(nobs.tolist())
            if key in obs_map and (obs_map[key]+1 < len(actions)):
                match_idx = obs_map[key]
                item = {
                    'observation': obs[i],
                    'action': actions[i],
                    'next_observation': next_obs[i],
                    'next_action': actions[match_idx]
                }
                reorganized_data.append(item)

        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(reorganized_data, f)

        print(f"{env}: Saved reorganized dataset with {len(reorganized_data)} entries to {output_path}")
    else:
        print(f"{env}: 'observations', 'next_observations', or 'actions' missing. Skipped.")




# %%
# Print one sample from the reorganized dataset
if os.path.exists(output_path):
    with open(output_path, 'rb') as f:
        loaded_data = pickle.load(f)
    if len(loaded_data) > 0:
        import pprint
        print("Sample record from reorganized dataset:")
        pprint.pprint(loaded_data[0])
    else:
        print("Reorganized dataset is empty.")
else:
    print(f"File {output_path} does not exist.")


# %%
# Concatenate dataset for diffuser usage

import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def load_reorganized_dataset(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} samples from {pkl_path}")
    return data

def build_condition_target_arrays(data):
    cond_list = []
    target_list = []
    for item in data:
        obs_t    = item['observation']
        act_t    = item['action']
        obs_tp1  = item['next_observation']
        act_tp1  = item['next_action']

        x_cond   = np.concatenate([obs_t,   act_t],   axis=-1)
        x_target = np.concatenate([obs_tp1, act_tp1], axis=-1)

        cond_list.append(x_cond)
        target_list.append(x_target)

    X_cond   = np.array(cond_list,   dtype=np.float32)
    X_target = np.array(target_list, dtype=np.float32)
    print(f"Built X_cond shape = {X_cond.shape}, X_target shape = {X_target.shape}")
    return X_cond, X_target

def fit_and_save_scaler(X_cond, X_target, scaler_path):
    scaler = StandardScaler()
    combined = np.vstack([X_cond, X_target])
    scaler.fit(combined)
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    return scaler

def transform_data(scaler, X_cond, X_target):
    X_cond_norm   = scaler.transform(X_cond)
    X_target_norm = scaler.transform(X_target)
    return X_cond_norm, X_target_norm

def split_train_val_test(X_cond_norm, X_target_norm, train_ratio=0.6, val_ratio=0.2, seed=42):
    assert train_ratio + val_ratio < 1.0, "train_ratio + val_ratio must be < 1.0"
    N = X_cond_norm.shape[0]
    np.random.seed(seed)
    perm = np.random.permutation(N)

    n_train = int(N * train_ratio)
    n_val   = int(N * val_ratio)
    # rest is test
    train_idx = perm[:n_train]
    val_idx   = perm[n_train : n_train + n_val]
    test_idx  = perm[n_train + n_val :]

    Xc_train   = X_cond_norm[train_idx]
    Xt_train   = X_target_norm[train_idx]
    Xc_val     = X_cond_norm[val_idx]
    Xt_val     = X_target_norm[val_idx]
    Xc_test    = X_cond_norm[test_idx]
    Xt_test    = X_target_norm[test_idx]

    print(f"Split sizes → train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    return (Xc_train, Xt_train), (Xc_val, Xt_val), (Xc_test, Xt_test)

def save_npz(dataset: dict, file_path: str):
    np.savez(file_path, **dataset)
    print(f"Saved dataset to {file_path}")

def main():
    # === Config ===
    pkl_path      = "/public/sparse_d4rl/mapped/hopper-medium-v2.pkl"
    output_dir    = "/public/sparse_d4rl/mapped/concat"
    scaler_fname  = "scaler_hopper_joblib.pkl"
    train_fname   = "hopper_one_step_train.npz"
    val_fname     = "hopper_one_step_val.npz"
    test_fname    = "hopper_one_step_test.npz"
    train_ratio   = 0.6
    val_ratio     = 0.2

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    scaler_path = os.path.join(output_dir, scaler_fname)
    train_npz   = os.path.join(output_dir, train_fname)
    val_npz     = os.path.join(output_dir, val_fname)
    test_npz    = os.path.join(output_dir, test_fname)

    # === Steps ===
    data        = load_reorganized_dataset(pkl_path)
    X_cond, X_target = build_condition_target_arrays(data)
    scaler      = fit_and_save_scaler(X_cond, X_target, scaler_path)
    Xc_norm, Xt_norm = transform_data(scaler, X_cond, X_target)

    (Xc_train, Xt_train), (Xc_val, Xt_val), (Xc_test, Xt_test) = \
        split_train_val_test(Xc_norm, Xt_norm, train_ratio=train_ratio, val_ratio=val_ratio)

    save_npz({"X_cond": Xc_train, "X_target": Xt_train}, train_npz)
    save_npz({"X_cond": Xc_val,   "X_target": Xt_val},   val_npz)
    save_npz({"X_cond": Xc_test,  "X_target": Xt_test},  test_npz)

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()

# %%
# Process for walker2d and halfcheetah datasets, mirroring the hopper pipeline.

def process_env(env_name):
    # Adjust paths for each environment
    pkl_path = f"/public/sparse_d4rl/mapped/{env_name}-medium-v2.pkl"
    output_dir = f"/public/sparse_d4rl/mapped/concat"
    scaler_fname = f"scaler_{env_name}_joblib.pkl"
    train_fname = f"{env_name}_one_step_train.npz"
    val_fname   = f"{env_name}_one_step_val.npz"
    test_fname  = f"{env_name}_one_step_test.npz"
    train_ratio = 0.6
    val_ratio = 0.2

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    scaler_path = os.path.join(output_dir, scaler_fname)
    train_npz = os.path.join(output_dir, train_fname)
    val_npz = os.path.join(output_dir, val_fname)
    test_npz = os.path.join(output_dir, test_fname)

    # === Steps ===
    data = load_reorganized_dataset(pkl_path)
    X_cond, X_target = build_condition_target_arrays(data)
    scaler = fit_and_save_scaler(X_cond, X_target, scaler_path)
    Xc_norm, Xt_norm = transform_data(scaler, X_cond, X_target)

    (Xc_train, Xt_train), (Xc_val, Xt_val), (Xc_test, Xt_test) = \
        split_train_val_test(Xc_norm, Xt_norm, train_ratio=train_ratio, val_ratio=val_ratio)

    save_npz({"X_cond": Xc_train, "X_target": Xt_train}, train_npz)
    save_npz({"X_cond": Xc_val,   "X_target": Xt_val},   val_npz)
    save_npz({"X_cond": Xc_test,  "X_target": Xt_test},  test_npz)

    print(f"{env_name}: Preprocessing complete.")

# For walker2d
process_env("walker2d")

# For halfcheetah
process_env("halfcheetah")





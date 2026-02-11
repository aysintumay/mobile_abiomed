import sys
import numpy as np
import csv
import pickle
import d4rl

#TODO: find the minimum and maximum scores for each task 
# def find_min_max_reward(data_path):
#     try:
#         with open(data_path, "rb") as f:
#             dataset = pickle.load(f)
#             print('opened the pickle file for synthetic dataset')
#     except:
#         dataset = np.load(data_path)
#         dataset = {k: dataset[k] for k in dataset.files}
#         print('opened the npz file for synthetic dataset')
    
#     rwd_min = dataset['rewards'].min()
#     rwd_max = dataset['rewards'].max()
#     return rwd_min, rwd_max


# MINIMUM = {'hopper-medium-v2': -20.272305 ,
#            'halfcheetah-medium-v2': -280.178953,
#            'walker2d-medium-v2':1.629008,
# }
           
# MAXIMUM = {'hopper-medium-v2':3234.3,
#               'halfcheetah-medium-v2':12135.0  ,
#               'walker2d-medium-v2':  4592.3,
#     }

# def normalized_score(score, task):

#     return (score - MINIMUM[task]) / (MAXIMUM[task] - MINIMUM[task])


# def normalized_score_sparse(score, min_rwd, max_rwd):
#     print(min_rwd, max_rwd)
#     return (score - min_rwd) / (max_rwd - min_rwd)

def compute_normalized_score(csv_path, env_name):
    print("Script has started")
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        returns = [float(row['mean_return']) for row in reader]
    env_name_norm = env_name.split('-')[0]
    normalized_mean_scores = [d4rl.get_normalized_score(env_name, np.array(r) ) * 100 for r in returns]
    mean_score = np.mean(normalized_mean_scores)
    std_score = np.std(normalized_mean_scores)

    print(f"Computed Mean Normalized Score: {mean_score:.2f} ± {std_score:.2f}")


# def compute_normalized_score_sparse(csv_path, min_rwd, max_rwd):

#     print("Script has started")
#     with open(csv_path, 'r') as f:
#         reader = csv.DictReader(f)
#         returns = [float(row['mean_return']) for row in reader]

#     normalized_mean_scores = [normalized_score_sparse(np.array(r), min_rwd, max_rwd) * 100 for r in returns]
#     mean_score = np.mean(normalized_mean_scores)
#     std_score = np.std(normalized_mean_scores)

    # print(f"Computed Mean Normalized Score: {mean_score:.2f} ± {std_score:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python normalizer.py <csv_path> <env_name>")
    else:
        csv_path = sys.argv[1]
        env_name = sys.argv[2]
        # if len(sys.argv) == 4:
        #     print("Using sparse normalization with provided data path")
        #     data_path = sys.argv[3]
        #     min_rwd, max_rwd = find_min_max_reward(data_path)
        #     compute_normalized_score_sparse(csv_path, min_rwd, max_rwd)
        # else:
        print("Using standard normalization")
        compute_normalized_score(csv_path, env_name)
#!/bin/bash

TASK="halfcheetah-medium-expert-v2"
DATASET_PATH="/public/d4rl/sparse_datasets/halfcheetah_medium_expert_sparse_72.5.pkl"
DEVICE="cuda:4"
DATASET_NAME=$(basename "$DATASET_PATH" .pkl)
RESULTS_FILE="results_${DATASET_NAME}.csv"

# Dynamics path dictionary - map each seed to its dynamics model path
declare -A DYNAMICS_PATHS
DYNAMICS_PATHS[42]="log/halfcheetah-medium-expert-v2/mobile&penalty_coef=1.0&rollout_length=5/seed_42&timestamp_26-0116-223826/model"
DYNAMICS_PATHS[123]="log/halfcheetah-medium-expert-v2/mobile&penalty_coef=1.0&rollout_length=5/seed_123&timestamp_26-0117-063103/model"
DYNAMICS_PATHS[456]="log/halfcheetah-medium-expert-v2/mobile&penalty_coef=1.0&rollout_length=5/seed_456&timestamp_26-0117-131326/model"

# Create results file with header if it doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo "task,dataset,seed,final_normalized_reward,final_reward_std" > "$RESULTS_FILE"
fi

for seed in 42 123 456
do
    DYNAMICS_PATH="${DYNAMICS_PATHS[$seed]}"
    echo "Running MOBILE with seed $seed"
    echo "Using dynamics path: $DYNAMICS_PATH"
    python train.py \
    --dataset-path "$DATASET_PATH" \
    --device $DEVICE \
    --seed $seed \
    --epoch 1000 \
    --task $TASK \
    --dynamics_path "$DYNAMICS_PATH"

       # Find the most recent log directory for this seed (sort by modification time)
    LOG_DIR=$(find log/$TASK/"mobile&penalty_coef=1.5&rollout_length=5" -type d -name "seed_${seed}&timestamp_*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)

    if [ -n "$LOG_DIR" ]; then
        CSV_FILE="$LOG_DIR/record/policy_training_progress.csv"
        if [ -f "$CSV_FILE" ]; then
            # Extract the last row's eval/normalized_episode_reward value
            # Need both header and last data row for csv.DictReader
            FINAL_REWARD=$( (head -n 1 "$CSV_FILE"; tail -n 1 "$CSV_FILE") | python3 -c "
import sys
import csv
reader = csv.DictReader(sys.stdin)
for row in reader:
    reward = row.get('eval/normalized_episode_reward', 'N/A')
    reward_std = row.get('eval/normalized_episode_reward_std', 'N/A')
    print(f'{reward},{reward_std}')
" 2>/dev/null)

            if [ -n "$FINAL_REWARD" ]; then
                echo "$TASK,$DATASET_NAME,$seed,$FINAL_REWARD" >> "$RESULTS_FILE"
                echo "Results appended: $TASK, seed $seed -> $FINAL_REWARD"
            else
                echo "Warning: Could not extract results for seed $seed"
            fi
        else
            echo "Warning: CSV file not found at $CSV_FILE"
        fi
    else
        echo "Warning: Log directory not found for seed $seed"
    fi
done

echo ""
echo "All runs completed! Results saved to $RESULTS_FILE"


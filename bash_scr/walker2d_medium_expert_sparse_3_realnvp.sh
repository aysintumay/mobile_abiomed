#!/bin/bash

TASK="walker2d-medium-expert-v2"
DATASET_PATH="/public/d4rl/sparse_datasets/walker2d_medium_expert_sparse_73.pkl"
DEVICE="cuda:7"
CLASSIFIER='realnvp'
DATASET_NAME=$(basename "$DATASET_PATH" .pkl)
RESULTS_FILE="results_${DATASET_NAME}_${CLASSIFIER}.csv"

# Dynamics path dictionary - map each seed to its dynamics model path
declare -A DYNAMICS_PATHS
DYNAMICS_PATHS[42]="log/walker2d-medium-expert-v2/seed_42&timestamp_26-0112-030600/model"
DYNAMICS_PATHS[123]="log/walker2d-medium-expert-v2/seed_123&timestamp_26-0112-030558/model"
DYNAMICS_PATHS[456]="log/walker2d-medium-expert-v2/seed_456&timestamp_26-0112-030559/model"

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
    --config config_gormpo/walker2d_medium_expert_72.5_realnvp.yaml \
    --seed $seed \
    --epoch 3000 \
    --reward_penalty_coef 0.5 \
    --task $TASK \
    --classifier_model_name /public/gormpo/models/walker2d_medium_expert_sparse_3/realnvp \
    --dynamics_path "$DYNAMICS_PATH"

    # Find the most recent log directory for this seed (sort by modification time)
    LOG_DIR=$(find "log/$TASK/" -mindepth 2 -maxdepth 2 -type d -name "seed_${seed}&timestamp_*" -path "*classifier_type=${CLASSIFIER}*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)

    if [ -n "$LOG_DIR" ]; then
        CSV_FILE="$LOG_DIR/record/policy_training_progress.csv"
        if [ -f "$CSV_FILE" ]; then
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

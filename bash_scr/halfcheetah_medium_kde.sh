#!/bin/bash

TASK="halfcheetah-medium-v2"
DEVICE="cuda:0"
CLASSIFIER='kde'
DATASET_NAME=$(echo "$TASK" | tr '-' '_')
RESULTS_FILE="results_${DATASET_NAME}_${CLASSIFIER}.csv"

DYNAMICS_PATHS[42]="log/halfcheetah-medium-v2/mobile_gormpo&penalty_coef=0.5&rollout_length=5&classifier_type=diffusion/seed_42&timestamp_26-0323-063251/model"
DYNAMICS_PATHS[123]="log/halfcheetah-medium-v2/mobile_gormpo&penalty_coef=0.5&rollout_length=5&classifier_type=diffusion/seed_123&timestamp_26-0323-093934/model"
DYNAMICS_PATHS[456]="log/halfcheetah-medium-v2/mobile_gormpo&penalty_coef=0.5&rollout_length=5&classifier_type=diffusion/seed_456&timestamp_26-0323-143055/model"

# Create results file with header if it doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo "task,dataset,seed,final_normalized_reward,final_reward_std" > "$RESULTS_FILE"
fi

for seed in 42 123 456
do
    echo "Running MOBILE with seed $seed"
    python train.py \
    --device $DEVICE \
    --config config_gormpo/halfcheetah_medium_kde.yaml \
    --seed $seed \
    --classifier_model_name /public/gormpo/models/halfcheetah_normal/kde \
    --dynamics_path "${DYNAMICS_PATHS[$seed]}"

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

#!/bin/bash

# Script to run train.py with multiple seeds
# Usage: ./run_multi_seed.sh --task <task_name> --dataset-path <path> --device <device>

# Default configuration
TASK="halfcheetah-medium-expert-v2"
SEEDS=(42 123 456)
DATASET_PATH=""
DEVICE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run_multi_seed.sh --task <task_name> --dataset-path <path> --device <device>"
            exit 1
            ;;
    esac
done

echo "Running training for task: $TASK with seeds: ${SEEDS[@]}"
echo "Dataset path: $DATASET_PATH"
echo "Device: $DEVICE"
echo "=========================================="

# Run training for each seed
for SEED in "${SEEDS[@]}"
do
    echo ""
    echo "Starting training with seed $SEED..."
    echo "=========================================="

    TRAIN_ARGS="--task $TASK --seed $SEED"

    if [ -n "$DATASET_PATH" ]; then
        TRAIN_ARGS="$TRAIN_ARGS --dataset-path $DATASET_PATH"
    fi

    if [ -n "$DEVICE" ]; then
        TRAIN_ARGS="$TRAIN_ARGS --device $DEVICE"
    fi

    python train.py $TRAIN_ARGS

    if [ $? -eq 0 ]; then
        echo "Seed $SEED completed successfully!"
    else
        echo "ERROR: Seed $SEED failed!"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All seeds completed! Aggregating results..."
echo "=========================================="

# Aggregate results
python aggregate_results.py --task "$TASK"

echo "Done! Check results.csv for aggregated evaluation results."

#!/bin/bash

# Script to run train.py with multiple seeds in parallel
# Usage: ./run_parallel_seeds.sh --task <task_name> --dataset-path <path> --device <device> [additional_args]

# Default configuration
TASK="halfcheetah-medium-expert-v2"
DATASET_PATH=""
DEVICE=""
SEEDS=(42 123 456)
DYNAMICS_PATH=""
ADDITIONAL_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --dynamics_path)
            DYNAMICS_PATH="$2"
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
            ADDITIONAL_ARGS="$ADDITIONAL_ARGS $1"
            shift
            ;;
    esac
done

# Determine log directory based on whether dataset contains "sparse"
if [[ "$DATASET_PATH" == *"sparse"* ]]; then
    LOG_DIR="${TASK}_sparse3"
else
    LOG_DIR="$TASK"
fi

echo "Running training in parallel for task: $TASK with seeds: ${SEEDS[@]}"
echo "Dataset path: $DATASET_PATH"
echo "Device: $DEVICE"
echo "Log directory: $LOG_DIR"
echo "Additional arguments: $ADDITIONAL_ARGS"
echo "=========================================="

# Array to store background process IDs
PIDS=()

# Run training for each seed in parallel
for SEED in "${SEEDS[@]}"
do
    echo "Starting training with seed $SEED in background..."

    # Create a log file for this seed
    LOG_FILE="$LOG_DIR/train_seed_${SEED}.log"

    # Build the command with optional parameters
    CMD="python train.py --task \"$TASK\" --seed \"$SEED\""

    if [ -n "$DATASET_PATH" ]; then
        CMD="$CMD --dataset-path \"$DATASET_PATH\""
    fi

    if [ -n "$DYNAMICS_PATH" ]; then
        CMD="$CMD --dynamics_path \"$DATASET_PATH\""
    fi

    if [ -n "$DEVICE" ]; then
        CMD="$CMD --device \"$DEVICE\""
    fi

    if [ -n "$ADDITIONAL_ARGS" ]; then
        CMD="$CMD $ADDITIONAL_ARGS"
    fi

    # Run in background and redirect output to log file
    eval $CMD > "$LOG_FILE" 2>&1 &

    # Store the process ID
    PIDS+=($!)
    echo "  PID: ${PIDS[-1]} | Log: $LOG_FILE"
done

echo ""
echo "All training runs started in parallel!"
echo "Waiting for all runs to complete..."
echo "=========================================="

# Wait for all background processes
FAILED=0
for i in "${!PIDS[@]}"
do
    PID=${PIDS[$i]}
    SEED=${SEEDS[$i]}

    wait $PID
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Seed ${SEED} (PID: $PID) completed successfully!"
    else
        echo "ERROR: Seed ${SEED} (PID: $PID) failed with exit code $EXIT_CODE!"
        FAILED=1
    fi
done

if [ $FAILED -eq 1 ]; then
    echo ""
    echo "Some runs failed. Check the log files for details."
    exit 1
fi

echo ""
echo "=========================================="
echo "All seeds completed! Aggregating results..."
echo "=========================================="

# Aggregate results
python aggregate_results.py --task "$LOG_DIR"

echo "Done! Check results.csv for aggregated evaluation results."

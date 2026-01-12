# Multi-Seed Training Scripts

This directory contains scripts to run training with multiple seeds and aggregate the results.

## Files

1. **run_multi_seed.sh** - Run training sequentially for multiple seeds
2. **run_parallel_seeds.sh** - Run training in parallel for multiple seeds (faster)
3. **aggregate_results.py** - Aggregate evaluation results from multiple runs into CSV

## Quick Start

### Sequential Execution (Recommended for limited resources)

```bash
./run_multi_seed.sh
```

This will run training for the default task (`halfcheetah-medium-expert-v2`) with seeds 0, 1, 2.

### Parallel Execution (Faster, requires more resources)

```bash
./run_parallel_seeds.sh
```

This runs all seeds simultaneously. Each seed's output is saved to `train_seed_X.log`.

## Usage Examples

### Basic Usage

```bash
# Run with default task (halfcheetah-medium-expert-v2)
./run_multi_seed.sh

# Run with a specific task
./run_multi_seed.sh walker2d-medium-expert-v2

# Run with additional arguments
./run_multi_seed.sh hopper-medium-v2 --epoch 1000 --batch_size 512
```

### Parallel Execution

```bash
# Run in parallel with default task
./run_parallel_seeds.sh

# Run in parallel with specific task
./run_parallel_seeds.sh walker2d-medium-expert-v2

# Run in parallel with additional arguments
./run_parallel_seeds.sh hopper-medium-v2 --epoch 1000
```

### Manual Aggregation

If you want to aggregate results from existing runs:

```bash
python aggregate_results.py --task halfcheetah-medium-expert-v2
```

## Customizing Seeds

To use different seeds, edit the script and modify the `SEEDS` array:

```bash
# In run_multi_seed.sh or run_parallel_seeds.sh
SEEDS=(0 1 2)  # Change to your desired seeds, e.g., SEEDS=(42 123 999)
```

## Output Files

After running the scripts, you'll get:

1. **results.csv** - Combined evaluation results from all seeds
   - Contains timestep, seed, and all evaluation metrics
   - Rows are sorted by timestep and seed

2. **results_summary.csv** - Summary statistics across seeds
   - Contains mean and std of evaluation metrics across seeds for each timestep
   - Useful for plotting with error bars

## Results Format

### results.csv
```csv
timestep,seed,eval/normalized_episode_reward,eval/normalized_episode_reward_std,...
10000,0,45.2,12.3,...
10000,1,47.1,11.8,...
10000,2,46.5,12.1,...
20000,0,52.3,10.2,...
...
```

### results_summary.csv
```csv
timestep,eval/normalized_episode_reward_mean,eval/normalized_episode_reward_std,...
10000,46.27,0.95,...
20000,53.12,1.23,...
...
```

## Troubleshooting

### Checking Individual Logs (Parallel Mode)

```bash
# View logs for each seed
tail -f train_seed_0.log
tail -f train_seed_1.log
tail -f train_seed_2.log
```

### Finding Existing Runs

```bash
# List all runs for a task
ls -l log/halfcheetah-medium-expert-v2/
```

### Manual Cleanup

```bash
# Remove log files
rm train_seed_*.log

# Remove results
rm results.csv results_summary.csv
```

## Notes

- Sequential execution (`run_multi_seed.sh`) is safer for systems with limited GPU memory
- Parallel execution (`run_parallel_seeds.sh`) is faster but requires enough GPU memory for multiple runs
- The aggregation script automatically finds the latest runs for each seed
- Training progress is logged to CSV files in the `log/` directory structure

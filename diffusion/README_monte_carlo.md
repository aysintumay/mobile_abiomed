# Monte Carlo Sampling for Diffusion Models

This script performs Monte Carlo sampling using trained DDPM or DDIM models to evaluate the uncertainty and distribution of predictions. It generates multiple samples from the model for a given test case and computes statistics, visualizations, and Negative Log-Likelihood (NLL) metrics.

## Features

- **Batched Sampling**: Efficiently processes multiple samples in parallel (default: 1000 samples per batch)
- **DDPM/DDIM Support**: Works with both stochastic DDPM and deterministic DDIM schedulers
- **Distribution Visualization**: Creates histograms showing the distribution of samples across all dimensions
- **NLL Calculation**: Computes Negative Log-Likelihood using histogram binning
- **Statistical Analysis**: Provides mean, std, error metrics, and per-dimension statistics

## Usage

### Example

```bash
python monte_carlo_sampling.py \
    --config configs/test/hopper_mlp_ddpm.yaml \
    --model-dir /data/sparse_d4rl/pretrained/ddim_mlp_hopper_medium_v2 \
    --test-npz /public/sparse_d4rl/mapped/concat/hopper_one_step_test.npz \
    --output-dir ./monte_carlo_results/hopper_ddpm
```

## Command-Line Arguments

### Required Arguments

- `--model-dir`: Directory containing `checkpoint.pt` and `scheduler/` subdirectory
- `--test-npz`: Path to test NPZ file with `X_cond`/`cond` and `X_target`/`target` arrays

### Optional Arguments

- `--config`: Path to YAML configuration file (can provide defaults for other arguments)
- `--scaler`: Path to joblib StandardScaler for de-normalization
- `--num-mc-samples`: Number of Monte Carlo samples to generate (default: 1000)
- `--batch-size`: Batch size for parallel sampling (default: 1000)
  - Increase for faster processing if you have sufficient GPU memory
  - Decrease if you run into memory issues
- `--inference-steps`: Number of denoising steps during inference (default: from config or 1000)
- `--scheduler-type`: Type of scheduler to use - `ddpm` or `ddim` (default: `ddpm`)
- `--output-dir`: Directory to save results and plots (default: `./monte_carlo_results`)
- `--device`: Device to run on - `cuda` or `cpu` (default: `cuda` if available)
- `--num-dims-to-plot`: Number of dimensions to plot histograms for (default: `None` = plot all dimensions)
- `--bin-width`: Bin width for NLL calculation (default: 0.1)

### YAML Config File

You can provide a YAML configuration file with default values. The config file should contain:

```yaml
out: /path/to/model/directory
test_npz: /path/to/test.npz
scaler: /path/to/scaler.pkl
num_samples: 1000
device: cuda
inference_steps: 100
```

Command-line arguments will override values from the YAML file.

## Output

The script generates the following outputs in the specified `--output-dir`:

### Files

1. **`monte_carlo_samples.npy`**: NumPy array containing all generated samples
   - Shape: `(num_samples, target_dim)`
   - Contains the raw (de-normalized if scaler provided) samples

2. **`nll_results.npz`**: NumPy compressed archive with NLL calculation results
   - `total_nll`: Total NLL summed over all dimensions
   - `nll_per_dim`: NLL for each dimension
   - `bin_width`: Bin width used
   - `num_bins`: Number of bins used
   - `value_range`: Min and max values in the data

### Visualizations

1. **`monte_carlo_distributions.png`**: Histogram plots for each dimension
   - Shows distribution of Monte Carlo samples
   - Marks ground truth value (red line)
   - Marks sample mean (green line)
   - Includes statistics (mean, std, ground truth value)

2. **`monte_carlo_distributions_summary.png`**: Summary statistics plots
   - Mean per dimension (with ±1 std band)
   - Standard deviation per dimension
   - Error (bias) per dimension

3. **`nll_per_dimension.png`**: NLL analysis plots
   - Bar plot of NLL per dimension
   - Histogram of NLL values distribution
   - Shows mean and median NLL

### Console Output

The script prints:
- Sample statistics (mean, std, ground truth)
- Per-dimension statistics (mean error, max error, std)
- NLL statistics (total, mean, min, max per dimension)
- Information about binning (number of bins, value range)

## How It Works

1. **Model Loading**: Loads the trained model from checkpoint and scheduler configuration
2. **Test Case Selection**: Uses the first test case from the provided NPZ file
3. **Monte Carlo Sampling**: Generates multiple samples with different random seeds:
   - For DDPM: Each sample uses independent noise at every denoising step (stochastic)
   - For DDIM: Only initial noise differs, denoising is deterministic
4. **Batched Processing**: Processes samples in batches for efficiency
5. **Statistical Analysis**: Computes mean, std, and error metrics
6. **NLL Calculation**: Uses histogram binning to estimate probability distribution and compute NLL
7. **Visualization**: Creates plots showing distributions and statistics

## NLL Calculation

The NLL is calculated using histogram binning:

1. Creates bins with specified width (default: 0.1)
2. For each dimension, builds a histogram of Monte Carlo samples
3. Converts counts to probabilities with smoothing (to avoid log(0))
4. Finds which bin the ground truth falls into
5. Computes NLL = -log(P(target in that bin))

The total NLL is the sum of NLL across all dimensions.

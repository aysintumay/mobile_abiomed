# Borrow a lot from tianshou:
# https://github.com/thu-ml/tianshou/blob/master/examples/mujoco/plotter.py
import csv
import os
import re

import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import argparse
import wandb
import torch
from tensorboard.backend.event_processing import event_accumulator
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE

COLORS = (
    [
        # deepmind style
        '#0072B2',
        '#009E73',
        '#D55E00',
        '#CC79A7',
        # '#F0E442',
        '#d73027',  # RED
        # built-in color
        'blue',
        'red',
        'pink',
        'cyan',
        'magenta',
        'yellow',
        'black',
        'purple',
        'brown',
        'orange',
        'teal',
        'lightblue',
        'lime',
        'lavender',
        'turquoise',
        'darkgreen',
        'tan',
        'salmon',
        'gold',
        'darkred',
        'darkblue',
        'green',
        # personal color
        '#313695',  # DARK BLUE
        '#74add1',  # LIGHT BLUE
        '#f46d43',  # ORANGE
        '#4daf4a',  # GREEN
        '#984ea3',  # PURPLE
        '#f781bf',  # PINK
        '#ffc832',  # YELLOW
        '#000000',  # BLACK
    ]
)


def convert_tfenvents_to_csv(root_dir, xlabel, ylabel):
    """Recursively convert test/metric from all tfevent file under root_dir to csv."""
    tfevent_files = []
    for dirname, _, files in os.walk(root_dir):
        for f in files:
            absolute_path = os.path.join(dirname, f)
            if re.match(re.compile(r"^.*tfevents.*$"), absolute_path):
                tfevent_files.append(absolute_path)
    print(f"Converting {len(tfevent_files)} tfevents files under {root_dir} ...")
    result = {}
    with tqdm.tqdm(tfevent_files) as t:
        for tfevent_file in t:
            t.set_postfix(file=tfevent_file)
            output_file = os.path.join(os.path.split(tfevent_file)[0], ylabel+'.csv')
            ea = event_accumulator.EventAccumulator(tfevent_file)
            ea.Reload()
            content = [[xlabel, ylabel]]
            for test_rew in ea.scalars.Items('eval/'+ylabel):
                content.append(
                    [
                        round(test_rew.step, 4),
                        round(test_rew.value, 4),
                    ]
                )
            csv.writer(open(output_file, 'w')).writerows(content)
            result[output_file] = content
    return result


def merge_csv(csv_files, root_dir, xlabel, ylabel):
    """Merge result in csv_files into a single csv file."""
    assert len(csv_files) > 0
    sorted_keys = sorted(csv_files.keys())
    sorted_values = [csv_files[k][1:] for k in sorted_keys]
    content = [
        [xlabel, ylabel+'_mean', ylabel+'_std']
    ]
    for rows in zip(*sorted_values):
        array = np.array(rows)
        assert len(set(array[:, 0])) == 1, (set(array[:, 0]), array[:, 0])
        line = [rows[0][0], round(array[:, 1].mean(), 4), round(array[:, 1].std(), 4)]
        content.append(line)
    output_path = os.path.join(root_dir, ylabel+".csv")
    print(f"Output merged csv file to {output_path} with {len(content[1:])} lines.")
    csv.writer(open(output_path, "w")).writerows(content)


def csv2numpy(file_path):
    df = pd.read_csv(file_path)
    step = df.iloc[:,0].to_numpy()
    mean = df.iloc[:,1].to_numpy()
    std = df.iloc[:,2].to_numpy()
    return step, mean, std


def smooth(y, radius=0):
    convkernel = np.ones(2 * radius + 1)
    out = np.convolve(y, convkernel, mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
    return out


def plot_figure(root_dir, task, algo_list, x_label, y_label, title, smooth_radius, color_list=None):
    fig, ax = plt.subplots()
    if color_list is None:
        color_list = [COLORS[i] for i in range(len(algo_list))]
    for i, algo in enumerate(algo_list):
        x, y, shaded = csv2numpy(os.path.join(root_dir, task, algo, y_label+'.csv'))
        # y = smooth(y, smooth_radius)
        # shaded = smooth(shaded, smooth_radius)
        # x=smooth(x, smooth_radius)
        ax.plot(x, y, color=color_list[i], label=algo_list[i])
        ax.fill_between(x, y-shaded, y+shaded, color=color_list[i], alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()


def plot_histogram(data, y_label,):
    color = [f'tab:{COLORS[0]}', f'tab:{COLORS[1]}', f'tab:{COLORS[2]}']
    """
    Plot histograms of data[1][y_label], data[2][y_label], data[3][y_label],
    each in its own color.
    
    Parameters
    ----------
    data : dict of DataFrame-like
        data[i][y_label] should be iterable of values.
    y_label : str
        Column/key to plot.
    """
    plt.figure(figsize=(8, 5))
    
    # Loop over the three iterations
    for i, color in zip(np.arange(4), COLORS):
        plt.hist(
            data[i][y_label],
            bins=50,
            density=True,
            alpha=0.6,        # semi‐transparent so overlaps show
            color=color,
            label=f'Iteration {i}' if i != 0 else 'Ground Truth',
        )
    
    plt.xlabel(y_label)
    plt.ylabel('Count')
    plt.title(f'Histogram of {y_label} over {i} iterations')
    plt.legend()
    plt.tight_layout()
    out_file = os.path.join(args.output_path, f'{y_label}.png')
    plt.savefig(out_file, dpi=args.dpi, bbox_inches='tight')
    plt.show()





def plot_policy(eval_env, state, all_states, title, legend=False):
    """
    Plots observed vs predicted MAP, HR, and PULSAT with recommended vs input PL levels.
    Each metric is plotted separately and logged to wandb.
    """
    # --- Colors ---
    colors = {
        "MAP": {"gt": "tab:red", "pred": "tab:red"},
        "HR": {"gt": "tab:orange", "pred": "tab:orange"},
        "PULSAT": {"gt": "tab:green", "pred": "tab:green"},
        "PL_input": "tab:blue",
        "PL_pred": "tab:blue",
    }

    # --- Setup ---
    max_steps = eval_env.max_steps
    forecast_n = eval_env.world_model.forecast_horizon
    action_unnorm = np.repeat(eval_env.episode_actions, forecast_n)
    state_unnorm = eval_env.world_model.unnorm_output(np.array(state).reshape(max_steps, forecast_n, -1))
    all_state_unnorm = eval_env.world_model.unnorm_output(np.array(all_states).reshape(max_steps + 1, forecast_n, -1))
    x1 = len(all_state_unnorm[0, :, 0].reshape(-1, 1))
    x2 = len(all_state_unnorm[1:, :, 0].reshape(-1, 1))

    # --- Metric index mapping ---
    metric_indices = {
        "MAP": 0,
        "HR": 9,
        "PULSAT": 7,
    }

    limits = {
        "MAP": (10, 150),
        "HR": (40, 160),
        "PULSAT": (10, 80),
    }

    for metric_name, idx in metric_indices.items():
        fig, ax1 = plt.subplots(figsize=(5.5, 5), dpi=300, layout='constrained')

        # x-axis ticks
        default_x_ticks = range(0, 181, 18)
        x_ticks = np.array(list(range(0, 31, 3)))
        plt.xticks(default_x_ticks, x_ticks)

        # Vertical divider line
        ax1.axvline(x=x1, linestyle='--', c='black', alpha=0.7)

        # Observed metric (historical)
        line_obs, = ax1.plot(
            range(0, x1 + x2),
            all_state_unnorm[:, :, idx].reshape(-1, 1),
            '--',
            label=f'Observed {metric_name}',
            alpha=0.5,
            color=colors[metric_name]["gt"],
            linewidth=2.0,
        )

        # Predicted metric
        line_pred, = ax1.plot(
            range(x1, x1 + x2),
            state_unnorm[:, :, idx].reshape(-1, 1),
            label=f'Predicted {metric_name}',
            color=colors[metric_name]["pred"],
            linewidth=3,
        )

        # Secondary axis for PL
        ax2 = ax1.twinx()
        line_pl1, = ax2.plot(
            range(0, x1 + x2),
            all_state_unnorm[:, :, -1].reshape(-1),
            '--',
            label='Input PL',
            alpha=0.5,
            color=colors["PL_input"],
            linewidth=2.0,
        )
        line_pl2, = ax2.plot(
            range(x1, x1 + x2),
            action_unnorm.reshape(-1, 1),
            label='Recommended PL',
            color=colors["PL_pred"],
            linewidth=3,
        )

        # --- Legend ---
        if legend:
            lines = [line_obs, line_pred, line_pl1, line_pl2]
            labels = [
                f'Observed {metric_name}',
                f'Predicted {metric_name}',
                'Input PL',
                'Recommended PL',
            ]
            ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(1.0, 1.35),
                       fancybox=True, ncol=2, fontsize='medium')

        # --- Axis styling ---
        ax1.set_xlabel('Time (hour)', fontsize=22)
        ax1.tick_params(axis='x', labelsize=16)
        ax1.tick_params(axis='y', colors=colors[metric_name]["pred"], labelsize=16)
        ax2.tick_params(axis='y', colors=colors["PL_pred"], labelsize=16)

        ax2.set_ylim(1, 10)
        ax1.set_ylim(limits[metric_name])
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)

        # --- Title ---
        # ax1.set_title(f"{metric_name} Forecast vs Observed ({title})", fontsize=20, fontweight="bold")

        # --- Log to wandb ---
        wandb.log({f"eval_sample_{metric_name}_{title}": wandb.Image(fig)})
        plt.savefig(os.path.join('figures', f'eval_sample_{metric_name}_{title}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)


    # ax1.set_ylabel('MAP (mmHg)', size="x-large", color='tab:red')

def plot_score_histograms(acp_list, ws_list, rwd_list, title):
    col_w_in   = 3.25   # one-column width (IEEE ~3.45", NeurIPS 3.25")
    row_h_in   = 1.8    # height per subplot (increase to make each panel taller)
    v_gap_in   = 0.8   # vertical gap between rows (inches)

    fig_h_in = 3 * row_h_in + 2 * v_gap_in

    fig, axes = plt.subplots(
        nrows=3, ncols=1,
        figsize=(col_w_in, fig_h_in),
        dpi=300,
        constrained_layout=False
    )
    # Control margins + gaps explicitly
    fig.subplots_adjust(left=0.18, right=0.92, top=0.92, bottom=0.12,
                    hspace=v_gap_in / row_h_in)

    

    configs = [
        ("ACP values",    acp_list, (0, 5)),
        ("WS values",     ws_list,  (-0.5, 1)),
        ("Reward values", rwd_list, (-12, 4)),
    ]

    for ax, (xlabel, vals, xlim) in zip(axes, configs):
        ax.hist(vals, edgecolor='black', linewidth=0.4)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=14)

    # # Single y-label for all
    # try:
    #     fig.supylabel("Episode Count", fontsize=10)
    # except AttributeError:
    #     for ax in axes: ax.set_ylabel("Episode Count", fontsize=28)

    # fig.suptitle(f"Distributions for {title.upper()}", fontsize=12, fontweight="bold", y=0.98)

    # Log once to W&B
    import wandb
    wandb.log({f"eval_distributions_{title}": wandb.Image(fig)})
    plt.close(fig)


# =============================================================================
# Density Model Plotting Functions (KDE, VAE, RealNVP)
# =============================================================================

def plot_likelihood_distributions(
    model,
    train_data,
    val_data,
    ood_data=None,
    thr=None,
    title="Likelihood Distribution",
    savepath=None,
    bins=50,
    use_detach=True
):
    """
    Visualize log-likelihood distributions for train, val, and OOD data.

    Args:
        model: density model with .score_samples(X) method (returns log probs)
        train_data: np.ndarray or torch.Tensor, in-distribution training set
        val_data: np.ndarray or torch.Tensor, held-out validation set
        ood_data: np.ndarray or torch.Tensor, optional OOD dataset
        thr: float, threshold value to indicate on the plot
        title: str, title for the plot
        savepath: str, optional path to save figure
        bins: int, number of histogram bins
        use_detach: bool, whether to detach tensors (for PyTorch models)
    """
    os.makedirs("figures", exist_ok=True)

    # Compute log-likelihoods
    logp_train = model.score_samples(train_data)
    logp_val = model.score_samples(val_data)

    if use_detach and hasattr(logp_train, 'detach'):
        logp_train = logp_train.detach().cpu().numpy()
        logp_val = logp_val.detach().cpu().numpy()
    elif hasattr(logp_train, 'cpu'):
        logp_train = logp_train.cpu().numpy()
        logp_val = logp_val.cpu().numpy()

    logp_ood = None
    if ood_data is not None:
        logp_ood = model.score_samples(ood_data)
        if use_detach and hasattr(logp_ood, 'detach'):
            logp_ood = logp_ood.detach().cpu().numpy()
        elif hasattr(logp_ood, 'cpu'):
            logp_ood = logp_ood.cpu().numpy()

    # Plot train/val distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(logp_train, bins=bins, color="blue", alpha=0.4, label="Train", kde=True)
    sns.histplot(logp_val, bins=bins, color="green", alpha=0.4, label="Validation", kde=True)
    if thr is not None:
        plt.axvline(x=thr, color='tab:red', linestyle='--', label='Threshold')
    plt.xlabel("Log-likelihood", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.tight_layout(pad=2.0)
    os.makedirs(savepath, exist_ok=True)

    plt.savefig(f"{savepath}/train_distribution.png", dpi=300, bbox_inches="tight")
    print(f"Saved figure at {savepath}/train_distribution.png")
    plt.close()

    # Plot OOD distribution if provided
    if logp_ood is not None:
        plt.figure(figsize=(8, 5))
        sns.histplot(logp_ood, bins=bins, color="blue", alpha=0.4, label="Test", kde=True)
        if thr is not None:
            plt.axvline(x=thr, color='tab:red', linestyle='--', label=f'Threshold: {thr:.3f}')
        plt.xlabel("Log-likelihood", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.tight_layout(pad=2.0)
        plt.savefig(f"{savepath}/ood_distribution.png", dpi=300, bbox_inches="tight")
        print(f"Saved figure at {savepath}/ood_distribution.png")
        plt.close()


def plot_val_test_id_distribution(
    model,
    val_data,
    test_id_data,
    thr=None,
    title="Validation vs Test ID Log-Likelihood Distribution",
    savepath="figures/val_test_id_distribution.png",
    bins=50
):
    """
    Visualize log-likelihood distributions for validation and test ID data in one plot.

    Args:
        model: density model with .score_samples(X) method
        val_data: np.ndarray or torch.Tensor, validation set
        test_id_data: np.ndarray or torch.Tensor, in-distribution test set
        thr: float, threshold value to indicate on the plot
        title: str, title for the plot
        savepath: str, path to save figure
        bins: int, number of histogram bins
    """
    logp_val = model.score_samples(val_data)
    logp_test_id = model.score_samples(test_id_data)

    if hasattr(logp_val, 'detach'):
        logp_val = logp_val.detach().cpu().numpy()
        logp_test_id = logp_test_id.detach().cpu().numpy()

    plt.figure(figsize=(7, 6))
    sns.histplot(logp_val, bins=bins, color="green", alpha=0.5, label="Validation", kde=True, stat="density")
    sns.histplot(logp_test_id, bins=bins, color="blue", alpha=0.5, label="Test ID", kde=True, stat="density")

    if thr is not None:
        plt.axvline(x=thr, color='tab:red', linestyle='--', linewidth=2, label=f'Threshold ({thr:.3f})')

    plt.xlabel("Log-Likelihood", fontsize=16, labelpad=10)
    plt.ylabel("Density", fontsize=16, labelpad=10)
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.legend(fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout(pad=2.5)

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    print(f"Saved figure at {savepath}")
    plt.close()


def plot_tsne(tsne_data, preds, title, save_dir="figures"):
    """
    Plot t-SNE visualization with ID/OOD predictions.

    Args:
        tsne_data: np.ndarray, 2D t-SNE transformed data
        preds: np.ndarray, predictions (1=ID, -1=OOD)
        title: str, title for the plot
        save_dir: str, directory to save figure
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_data[preds == 1, 0], tsne_data[preds == 1, 1],
                color='blue', label='ID', alpha=0.5)
    plt.scatter(tsne_data[preds == -1, 0], tsne_data[preds == -1, 1],
                color='red', label='OOD', alpha=0.5)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('TSNE Dimension 1', fontsize=14)
    plt.ylabel('TSNE Dimension 2', fontsize=14)
    plt.tick_params(labelsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(save_dir, f"{title.replace(' ', '_')}.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_curves(history: dict, save_path: str = "figures/training_curves.png"):
    """
    Plot training and validation curves for VAE/density models.

    Args:
        history: dict containing 'train_loss', 'val_loss', etc.
        save_path: str, path to save the figure
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Total loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    if 'test_loss' in history and history['test_loss']:
        axes[0].plot(history['test_loss'], label='Test')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Reconstruction loss
    if 'train_recon' in history:
        axes[1].plot(history['train_recon'], label='Train')
        axes[1].plot(history['val_recon'], label='Val')
        if 'test_recon' in history and history['test_recon']:
            axes[1].plot(history['test_recon'], label='Test')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Reconstruction Loss')
        axes[1].set_title('Reconstruction Loss')
        axes[1].legend()
        axes[1].grid(True)

    # KL divergence
    if 'train_kl' in history:
        axes[2].plot(history['train_kl'], label='Train')
        axes[2].plot(history['val_kl'], label='Val')
        if 'test_kl' in history and history['test_kl']:
            axes[2].plot(history['test_kl'], label='Test')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('KL Divergence')
        axes[2].set_title('KL Divergence')
        axes[2].legend()
        axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to: {save_path}")
    plt.close()


def plot_ood_metrics(res_dict, test_id_score=None, save_dir="figures"):
    """
    Plot comprehensive OOD detection metrics.

    Args:
        res_dict: Dictionary with OOD percentages as keys and metrics as values
        test_id_score: Optional mean score for pure ID test data (0% OOD)
        save_dir: Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)

    percentages = sorted(res_dict.keys())
    if test_id_score is not None:
        percentages = [0.0] + percentages

    mean_scores = []
    roc_aucs = []
    accuracies = []
    id_accuracies = []
    ood_accuracies = []

    for perc in percentages:
        if perc == 0.0 and test_id_score is not None:
            mean_scores.append(test_id_score)
            roc_aucs.append(None)
            accuracies.append(None)
            id_accuracies.append(None)
            ood_accuracies.append(None)
        else:
            metrics = res_dict[perc]
            mean_scores.append(metrics['mean_score'])
            roc_aucs.append(metrics['roc_auc'])
            accuracies.append(metrics['accuracy'])
            id_accuracies.append(metrics['id_accuracy'])
            ood_accuracies.append(metrics['ood_accuracy'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('OOD Detection Performance Metrics', fontsize=18, fontweight='bold', y=0.995)

    # Plot 1: Mean Log-Likelihood
    ax1 = axes[0, 0]
    ax1.plot([p * 100 for p in percentages], mean_scores, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel('OOD Percentage (%)', fontsize=14)
    ax1.set_ylabel('Mean Log-Likelihood', fontsize=14)
    ax1.set_title('Log-Likelihood vs OOD Ratio', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([p * 100 for p in percentages])
    ax1.tick_params(labelsize=12)

    # Plot 2: ROC AUC
    ax2 = axes[0, 1]
    valid_percentages = [p for p, a in zip(percentages, roc_aucs) if a is not None]
    valid_aucs = [a for a in roc_aucs if a is not None]
    if valid_aucs:
        ax2.plot([p * 100 for p in valid_percentages], valid_aucs, 'o-', linewidth=2, markersize=8, color='forestgreen')
        ax2.axhline(y=0.5, color='r', linestyle='--', label='Random', alpha=0.7)
        ax2.set_xlabel('OOD Percentage (%)', fontsize=14)
        ax2.set_ylabel('ROC AUC', fontsize=14)
        ax2.set_title('ROC AUC vs OOD Ratio', fontsize=15, fontweight='bold')
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
        ax2.tick_params(labelsize=12)

    # Plot 3: Accuracy Metrics
    ax3 = axes[1, 0]
    valid_acc = [a for a in accuracies if a is not None]
    valid_id_acc = [a for a in id_accuracies if a is not None]
    valid_ood_acc = [a for a in ood_accuracies if a is not None]
    valid_perc_acc = [p for p, a in zip(percentages, accuracies) if a is not None]

    if valid_acc:
        ax3.plot([p * 100 for p in valid_perc_acc], valid_acc, 'o-', linewidth=2, markersize=8, label='Overall', color='purple')
        ax3.plot([p * 100 for p in valid_perc_acc], valid_id_acc, 's-', linewidth=2, markersize=8, label='ID', color='blue')
        ax3.plot([p * 100 for p in valid_perc_acc], valid_ood_acc, '^-', linewidth=2, markersize=8, label='OOD', color='red')
        ax3.set_xlabel('OOD Percentage (%)', fontsize=14)
        ax3.set_ylabel('Accuracy', fontsize=14)
        ax3.set_title('Accuracy vs OOD Ratio', fontsize=15, fontweight='bold')
        ax3.set_ylim([0, 1.05])
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=12)
        ax3.tick_params(labelsize=12)

    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    plt.tight_layout(pad=2.5)
    save_path = os.path.join(save_dir, 'ood_metrics_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved OOD metrics to: {save_path}")
    plt.close()


def plot_roc_curves(model, test_ood_dict, res_ood_dict, save_dir="figures"):
    """
    Plot ROC curves for different OOD ratios.

    Args:
        model: Trained density model
        test_ood_dict: Dictionary with OOD percentages as keys and test data as values
        res_ood_dict: Dictionary with metrics
        save_dir: Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(test_ood_dict)))

    for i, (perc, test_data) in enumerate(sorted(test_ood_dict.items())):
        metrics = res_ood_dict[perc]
        n_id = metrics['n_id']
        n_ood = metrics['n_ood']

        if n_id > 0 and n_ood > 0:
            with torch.no_grad():
                if hasattr(model, 'device'):
                    scores = model.score_samples(test_data.to(model.device))
                else:
                    scores = model.score_samples(test_data)
                if hasattr(scores, 'cpu'):
                    scores = scores.cpu().numpy()

            y_true = np.concatenate([np.zeros(n_id), np.ones(n_ood)])
            y_scores = -scores

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc_val = auc(fpr, tpr)

            plt.plot(fpr, tpr, linewidth=2, color=colors[i],
                    label=f'{perc:.1%} OOD (AUC = {roc_auc_val:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random', alpha=0.5)
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title('ROC Curves for Different OOD Ratios', fontsize=17, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(labelsize=13)
    plt.tight_layout(pad=2.0)
    save_path = os.path.join(save_dir, 'roc_curves_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved ROC curves to: {save_path}")
    plt.close()


def plot_likelihood_histograms(model, test_ood_dict, test_id_data=None, save_dir="figures"):
    """
    Plot likelihood distributions for different OOD ratios.

    Args:
        model: Trained density model
        test_ood_dict: Dictionary with OOD percentages as keys and test data as values
        test_id_data: Optional pure ID test data
        save_dir: Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Log-Likelihood Distributions for Different OOD Ratios', fontsize=18, fontweight='bold', y=0.995)

    axes = axes.flatten()
    percentages = sorted(test_ood_dict.keys())

    plot_idx = 0
    if test_id_data is not None:
        with torch.no_grad():
            if hasattr(model, 'device'):
                id_scores = model.score_samples(test_id_data.to(model.device))
            else:
                id_scores = model.score_samples(test_id_data)
            if hasattr(id_scores, 'cpu'):
                id_scores = id_scores.cpu().numpy()

        ax = axes[0]
        ax.hist(id_scores, bins=50, color='blue', alpha=0.6, label='ID (0% OOD)', edgecolor='black')
        if hasattr(model, 'threshold') and model.threshold is not None:
            ax.axvline(x=model.threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax.set_xlabel('Log-Likelihood', fontsize=13)
        ax.set_ylabel('Frequency', fontsize=13)
        ax.set_title(f'0% OOD (Pure ID)\nMean: {id_scores.mean():.3f}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plot_idx = 1

    for perc in percentages:
        if plot_idx >= len(axes):
            break

        test_data = test_ood_dict[perc]
        with torch.no_grad():
            if hasattr(model, 'device'):
                scores = model.score_samples(test_data.to(model.device))
            else:
                scores = model.score_samples(test_data)
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()

        ax = axes[plot_idx]
        ax.hist(scores, bins=50, color='purple', alpha=0.6, edgecolor='black')
        if hasattr(model, 'threshold') and model.threshold is not None:
            ax.axvline(x=model.threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax.set_xlabel('Log-Likelihood', fontsize=13)
        ax.set_ylabel('Frequency', fontsize=13)
        ax.set_title(f'{perc:.1%} OOD\nMean: {scores.mean():.3f}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout(pad=2.5)
    save_path = os.path.join(save_dir, 'likelihood_histograms.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved likelihood histograms to: {save_path}")
    plt.close()


def evaluate_anomaly_detection(model, X_test, y_true, verbose=True):
    """
    Evaluate anomaly detection performance.

    Args:
        model: Trained anomaly detection model with predict() and decision_function()
        X_test: Test data
        y_true: True labels (1=normal, -1=anomaly)
        verbose: Print results

    Returns:
        dict: Dictionary with precision, recall, f1, accuracy, confusion matrix
    """
    y_pred = model.predict(X_test)
    anomaly_scores = model.decision_function(X_test) if hasattr(model, 'decision_function') else None

    tp = np.sum((y_true == -1) & (y_pred == -1))
    fp = np.sum((y_true == 1) & (y_pred == -1))
    tn = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == -1) & (y_pred == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(y_true)

    if verbose:
        print(f"\n=== Anomaly Detection Results ===")
        print(f"True anomalies in test set: {np.sum(y_true == -1)}")
        print(f"Predicted anomalies: {np.sum(y_pred == -1)}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Confusion Matrix:")
        print(f"  TP: {tp:4d} | FP: {fp:4d}")
        print(f"  FN: {fn:4d} | TN: {tn:4d}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def evaluate_and_plot_density(model, X_val, X_train, X_test, save_dir="figures"):
    """
    Evaluate the model on validation, train, and test datasets and plot t-SNE results.

    Args:
        model: Trained anomaly detection model with predict() and score_samples()
        X_val: Validation dataset
        X_train: Training dataset
        X_test: Test dataset
        save_dir: Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)

    def evaluate_dataset(data, data_name):
        predictions = model.predict(data)
        scores = model.score_samples(data)
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy() if hasattr(scores, 'numpy') else scores
        anomaly_count = np.sum(predictions == -1)
        anomaly_rate = anomaly_count / len(data)
        print(f"\n=== {data_name} Results ===")
        print(f"Anomalies detected: {anomaly_count}/{len(data)} ({anomaly_rate:.1%})")
        print(f"Density score range: [{scores.min():.4f}, {scores.max():.4f}]")
        return predictions

    print(f"\nEvaluating on validation set...")
    val_predictions = evaluate_dataset(X_val, "Validation Set")

    print(f"\nEvaluating on training set...")
    train_predictions = evaluate_dataset(X_train, "Training Set")

    print(f"\nEvaluating on test set...")
    test_predictions = evaluate_dataset(X_test, "Test Set")

    # t-SNE visualizations
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)

    if hasattr(X_val, 'cpu'):
        X_val_np = X_val.cpu().numpy()
    elif hasattr(X_val, 'numpy'):
        X_val_np = X_val.numpy()
    else:
        X_val_np = X_val

    reduced_val = tsne.fit_transform(X_val_np)
    plot_tsne(reduced_val, val_predictions, "Density_Results_Validation_Set", save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotter')
    parser.add_argument(
        '--root-dir', 
        #default='log/hopper-medium-replay-v0/mopo',
         default='log', help='root dir'
    )
    parser.add_argument(
        '--task', default='abiomed_plot', help='task'
    )
    parser.add_argument(
        '--algos', default=["mopo"], help='algos'
    )
    parser.add_argument(
        '--title', default=None, help='matplotlib figure title (default: None)'
    )
    parser.add_argument(
        '--xlabel', default='Timesteps', help='matplotlib figure xlabel'
    )
    parser.add_argument(
        '--ylabel', default='episode_reward', help='matplotlib figure ylabel'
    )
    parser.add_argument(
        '--hist_ylabel', default='actions', help='matplotlib figure ylabel'
    )
    parser.add_argument(
        '--hist_ylabel2', default='rewards', help='matplotlib figure ylabel'
    )
    parser.add_argument(
        '--smooth', type=int, default=10, help='smooth radius of y axis (default: 0)'
    )
    parser.add_argument(
        '--colors', default=None, help='colors for different algorithms'
    )
    parser.add_argument('--show', action='store_true', help='show figure')
    parser.add_argument(
        '--output-path', type=str, help='figure save path', default="./figure.png"
    )
    parser.add_argument(
        '--dpi', type=int, default=200, help='figure dpi (default: 200)'
    )
    args = parser.parse_args()

    # args.task = 'halfcheetah-expert-v2'
    for algo in args.algos:
        path = os.path.join(args.root_dir, args.task, algo)
        result = convert_tfenvents_to_csv(path, args.xlabel, args.ylabel)
        merge_csv(result, path, args.xlabel, args.ylabel)

    # plt.style.use('seaborn')
    plot_figure(root_dir=args.root_dir, task=args.task, algo_list=args.algos, x_label=args.xlabel, y_label=args.ylabel, title=args.title, smooth_radius=args.smooth, color_list=args.colors)
    if args.output_path:
        plt.savefig(args.output_path, dpi=args.dpi, bbox_inches='tight')
    if args.show:
        plt.show()

    data = {}
    for i in [0,1]:
        test_path = os.path.join(args.data_path, f"dataset_test_{i}.pkl")
        with open(test_path, 'rb') as f:
            data[i] = pickle.load(f)
    plot_histogram(data, args.hist_ylabel)
    
    plot_histogram(data, args.hist_ylabel2)
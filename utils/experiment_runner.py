# This module provides a framework for running and evaluating training experiments using various optimizers.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

def plot_seaborn_style(data, x_values, title, filename, y_label, visuals_dir, xlabel="Epoch", xlimit=None, yscale=None):
    """Create a standardized seaborn plot with consistent styling."""
    plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.gca()
    palette = sns.color_palette("colorblind")

    for i, (optimizer_name, metrics) in enumerate(data.items()):
        # Check if x_values is a dictionary (for walltime plots) or a sequence (for regular plots)
        if isinstance(x_values, dict):
            x_vals = x_values[optimizer_name][:len(metrics)] if optimizer_name in x_values else range(len(metrics))
            y_vals = metrics[:len(x_vals)]
        else:
            y_vals = metrics[:len(x_values)] if xlimit else metrics
            x_vals = x_values[:len(y_vals)]
            
        sns.lineplot(x=x_vals, y=y_vals, label=optimizer_name, color=palette[i], linewidth=2, alpha=0.9)

    plt.xlabel(xlabel, fontweight='bold')
    plt.ylabel(y_label, fontweight='bold')
    plt.title(title, fontweight='bold', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if xlimit is not None:
        plt.xlim(0, xlimit)
    
    if yscale:
        plt.yscale(yscale)
        
    legend = plt.legend(title='Optimizer', frameon=True, fancybox=True, framealpha=0.95, facecolor='white')
    legend.get_title().set_fontweight('bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, filename + '.png'), bbox_inches='tight')
    plt.savefig(os.path.join(visuals_dir, filename + '.pdf'), bbox_inches='tight')
    plt.close()

def save_experiment_results(data, results_dir, csv_filename):
    """Save experiment results to a CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(results_dir, csv_filename), index_label='Epoch')

def plot_f1_scores(f1_scores, optimizer_names, title, filename):
    """Plot F1 scores for multiple optimizers."""
    plt.figure(figsize=(10, 6))
    for optimizer_name in optimizer_names:
        plt.plot(f1_scores[optimizer_name], label=optimizer_name)
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def run_experiments(train_experiment, results_dir, visuals_dir, epochs,
                    optimizer_names=["SGD", "ADAM", "ADAMW"],
                    loss_title="Loss vs. Epoch", loss_ylabel="Average Loss",
                    acc_title="Accuracy vs. Epoch", acc_ylabel="Accuracy (%)",
                    plot_filename="training_curves", csv_filename="metrics.csv",
                    experiment_title="Experiment", cost_xlimit=None, f1_title="F1 Score vs. Epoch",
                    num_runs=1):
    """Run experiments for multiple optimizers and plot averaged results over multiple runs."""
    results, metrics, all_metrics = {}, {}, {}
    iter_results, wall_results = {}, {}
    grad_norms_history = {}
    layer_names_dict = {}
    f1_scores = {}

    for opt in optimizer_names:
        # Initialize lists to store outputs from each run
        runs_losses, runs_accs, runs_f1s = [], [], []
        runs_aucs, runs_iter_costs, runs_wall = [], [], []
        runs_gradients = []
        run_layer_names = None
        # Execute training multiple times for the given optimizer
        for _ in range(num_runs):
            out = train_experiment(opt)
            loss, acc, f1, auc, iter_cost, walltime, grad_norms, ln = out
            runs_losses.append(loss)
            runs_accs.append(acc)
            runs_f1s.append(f1)
            runs_aucs.append(auc)
            runs_iter_costs.append(iter_cost)
            runs_wall.append(walltime)
            runs_gradients.append(grad_norms)
            run_layer_names = ln

        # Average the epoch-level lists elementwise (assumes same length per run)
        avg_loss = [np.mean([run[i] for run in runs_losses]) for i in range(len(runs_losses[0]))]
        avg_acc = [np.mean([run[i] for run in runs_accs]) for i in range(len(runs_accs[0]))]
        avg_f1 = [np.mean([run[i] for run in runs_f1s]) for i in range(len(runs_f1s[0]))]
        avg_auc = [np.mean([run[i] for run in runs_aucs]) for i in range(len(runs_aucs[0]))]
        avg_iter_costs = [np.mean([run[i] for run in runs_iter_costs]) for i in range(len(runs_iter_costs[0]))]
        avg_wall = [np.mean([run[i] for run in runs_wall]) for i in range(len(runs_wall[0]))]
        
        # Average gradient norms per layer (each is a dict mapping layer -> list)
        avg_gradients = {}
        for layer in runs_gradients[0].keys():
            # assume each run returns same length list for given layer
            length = len(runs_gradients[0][layer])
            avg_gradients[layer] = [np.mean([run[layer][i] for run in runs_gradients]) 
                                    for i in range(length)]
        
        results[opt] = avg_loss
        metrics[opt] = avg_acc
        f1_scores[opt] = avg_f1
        all_metrics[opt] = [{"epoch": i+1, "loss": avg_loss[i], "accuracy": avg_acc[i], "f1_score": avg_f1[i]} for i in range(len(avg_loss))]
        iter_results[opt] = avg_iter_costs
        wall_results[opt] = avg_wall
        grad_norms_history[opt] = avg_gradients
        layer_names_dict[opt] = run_layer_names

        print(f"Averaged metrics logged for optimizer {opt} over {num_runs} runs")
        
    # Plotting calls (unchanged)
    plot_seaborn_style(
        results, 
        range(1, epochs+1), 
        f"{experiment_title}: {loss_title}", 
        f"loss_{plot_filename}", 
        loss_ylabel, 
        visuals_dir
    )
    
    plot_seaborn_style(
        metrics, 
        range(1, epochs+1), 
        f"{experiment_title}: {acc_title}", 
        f"accuracy_{plot_filename}", 
        acc_ylabel, 
        visuals_dir
    )
    
    for opt in iter_results:
        if cost_xlimit is not None:
            iter_results[opt] = iter_results[opt][:cost_xlimit]
            
    max_iterations = max([len(costs) for costs in iter_results.values()])
    iteration_range = range(1, max_iterations+1)
    
    plot_seaborn_style(
        iter_results, 
        iteration_range, 
        f"{experiment_title}: Iteration vs. Training Cost", 
        f"iter_training_cost_{plot_filename}", 
        "Batch Loss", 
        visuals_dir,
        xlabel="Iteration",
        xlimit=cost_xlimit
    )

    plot_seaborn_style(
        {opt: iter_results[opt] for opt in iter_results}, 
        {opt: wall_results[opt][:len(iter_results[opt])] for opt in wall_results},
        f"{experiment_title}: Walltime vs. Training Cost", 
        f"walltime_training_cost_{plot_filename}", 
        "Batch Loss", 
        visuals_dir,
        xlabel="Time (s)"
    )
    
    plot_f1_scores(
        f1_scores, 
        optimizer_names, 
        f"{experiment_title}: F1 Score vs. Epoch", 
        os.path.join(visuals_dir, f"f1_score_{plot_filename}.png")
    )
    
    df_metrics = pd.DataFrame([{**{"optimizer": opt}, **m} for opt in all_metrics for m in all_metrics[opt]])
    df_metrics.to_csv(os.path.join(results_dir, csv_filename), index=False)
    
    return results, metrics, all_metrics, iter_results, wall_results

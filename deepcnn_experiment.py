import os
from utils.train_utils import run_training, get_layer_names
from utils.experiment_runner import run_experiments
from utils.network import DeepCNN
from utils.hyperparameter_tuning_utils import tune_hyperparameters

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import seaborn as sns
sns.set_context("paper", font_scale=1.2)
from matplotlib import rcParams

# Set up seaborn for publication-quality plots
sns.set(style="whitegrid", context="paper")
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 12
rcParams['axes.labelsize'] = 11
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 14

# Set a color palette that looks good in publications
palette = sns.color_palette("colorblind")

# Create directories for results and visuals
results_dir = os.path.join(os.path.dirname(__file__), 'results')
visuals_dir = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(visuals_dir, exist_ok=True)

# Import configuration from the local config file
from config import BATCH_SIZE, EPOCHS, PARAM_GRID, OPTIMIZER_PARAMS, SCHEDULER_PARAMS, RUNS_PER_OPTIMIZER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Added device definition

# Using CIFAR-10 dataset
transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

def train_experiment(optimizer_name):
    # Use a more stable initialization for deep networks
    model = DeepCNN().to(device)
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Use Kaiming initialization with small gain to start conservatively
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    
    criterion = nn.CrossEntropyLoss()
    
    # Get layer names
    layer_names = get_layer_names(model)
    
    # Tune hyperparameters
    if optimizer_name.upper() != "LW":
        best_hyperparams = tune_hyperparameters(
            model_fn=DeepCNN,
            optimizer_names=[optimizer_name],
            param_grid=PARAM_GRID,
            train_loader=train_loader,
            device=device,
            experiment_name="DeepCNN",  # Pass experiment name
            task_type="classification", # Pass task type
            epochs=5,  # Reduced epochs for tuning
            num_trials=10  # Reduced trials for faster tuning
        )[optimizer_name]
    
        print(f"Using best hyperparameters for {optimizer_name}: {best_hyperparams}")
    
        # Remove repeated definitions – now look up parameters from config
        params = OPTIMIZER_PARAMS.get(optimizer_name.upper(), {})
        # Merge any tuned hyperparameters (if needed) with the base config
        params.update(best_hyperparams or {})
    else:
        params = OPTIMIZER_PARAMS.get(optimizer_name.upper(), {})
        
    # Create optimizer and scheduler for each type
    scheduler = None
    
    if optimizer_name.upper() == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), **params)
    elif optimizer_name.upper() == "ADAMW":
        optimizer = torch.optim.AdamW(model.parameters(), **params)
    elif optimizer_name.upper() == "SGD": 
        optimizer = torch.optim.SGD(model.parameters(), **params)

    # Set up the scheduler based on the config
    scheduler_config = OPTIMIZER_PARAMS.get(optimizer_name.upper(), {}).get("scheduler")
    if scheduler_config and SCHEDULER_PARAMS[optimizer_name.upper()]["scheduler"] != "None":
        scheduler_type = SCHEDULER_PARAMS[optimizer_name.upper()]["scheduler"]
        scheduler_params = SCHEDULER_PARAMS[optimizer_name.upper()]["params"]
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_type)
        scheduler = scheduler_class(optimizer, **scheduler_params)

    # Unpack four values from run_training
    metrics, norm_walltimes, gradient_norms, iter_costs = run_training(model, train_loader, optimizer, criterion, device, EPOCHS, scheduler=scheduler, layer_names=layer_names)

    # Save the trained model
    model_filename = f"deepcnn_{optimizer_name.lower()}.pth"
    model_path = os.path.join(results_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Return the 8 values expected by run_experiments
    return (
        metrics['loss'],       # costs
        metrics['accuracy'],   # accs
        metrics['f1_score'],   # f1s
        [0.0] * EPOCHS,        # aucs placeholder
        iter_costs,            # iter_costs
        norm_walltimes,        # cumulative walltimes
        gradient_norms,        # gradient norms
        layer_names            # layer names
    )

# Use the centralized run_experiments function from experiment_runner
run_experiments(train_experiment, results_dir, visuals_dir, EPOCHS,
                optimizer_names=["SGD", "ADAM", "ADAMW"], 
                loss_title="Deep CNN: Loss vs. Epoch", acc_title="Deep CNN: Accuracy vs. Epoch",
                plot_filename="deepcnn_training_curves", csv_filename="deepcnn_metrics.csv",
                experiment_title="Deep CNN Experiment", cost_xlimit=None,
                f1_title="Deep CNN: F1 Score vs. Epoch",
                num_runs=RUNS_PER_OPTIMIZER)

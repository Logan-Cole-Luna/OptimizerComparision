# Hyperparameter Tuning Framework for PyTorch Models

# This module provides utilities for training PyTorch models and tuning optimizer hyperparameters
# for classification and regression tasks.
import torch
import torch.nn as nn
import random
import os
import json
from itertools import product

def train_and_evaluate(model, train_loader, optimizer, criterion, device, task_type, epochs=10):
    model.train()
    if task_type == "classification":
        correct = 0
        total = 0
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                # Convert NumPy arrays to PyTorch tensors and move to device
                inputs, labels = torch.tensor(inputs).float().to(device), torch.tensor(labels).long().to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            for inputs, labels in train_loader:
                # Convert NumPy arrays to PyTorch tensors and move to device
                inputs, labels = torch.tensor(inputs).float().to(device), torch.tensor(labels).long().to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total
    elif task_type == "regression":
        total_mse = 0.0
        num_batches = 0
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                # Convert NumPy arrays to PyTorch tensors and move to device
                inputs, labels = torch.tensor(inputs).float().to(device), torch.tensor(labels).float().to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Calculate MSE for evaluation
                total_mse += nn.functional.mse_loss(outputs, labels).item()
                num_batches += 1
        # Calculate average MSE
        avg_mse = total_mse / num_batches
        return -avg_mse  # Return negative MSE as the "accuracy" to maximize
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def tune_hyperparameters(model_fn, optimizer_names, param_grid, train_loader, device,
                         experiment_name="Experiment", task_type="classification",
                         epochs=5, num_trials=5, test_loader=None):
    """
    Tune hyperparameters for multiple optimizers.
    
    Args:
        model_fn: Function that creates a new model instance
        optimizer_names: List of optimizer names to tune
        param_grid: Dictionary of hyperparameter grids for each optimizer
        train_loader: DataLoader for training data
        device: Device to train on
        experiment_name: Name of the experiment (for saving hyperparameters)
        task_type: Type of task (classification or regression)
        epochs: Number of epochs for each trial
        num_trials: Number of trials for each optimizer
        test_loader: DataLoader for test data (optional)
        
    Returns:
        Dictionary of best hyperparameters for each optimizer
    """
    hyperparams_dir = os.path.join(os.path.dirname(__file__), 'hyperparameters')
    os.makedirs(hyperparams_dir, exist_ok=True)
    
    # NEW: Create a logs directory for this experiment
    logs_dir = os.path.join(hyperparams_dir, "logs", experiment_name)
    os.makedirs(logs_dir, exist_ok=True)
    
    best_hyperparams = {}
    
    # Check if optimizer is supported
    for optimizer_name in optimizer_names:
        if optimizer_name not in ["ADAM", "ADAMW", "SGD"]:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    for optimizer_name in optimizer_names:
        # Skip tuning if not in param_grid
        if optimizer_name not in param_grid:
            print(f"Skipping hyperparameter tuning for {optimizer_name} as it is not in the param_grid.")
            best_hyperparams[optimizer_name] = {}
            continue
            
        # Check for existing hyperparameters
        hyperparam_file = os.path.join(hyperparams_dir, f"{experiment_name}_{optimizer_name}_hyperparameters.json")
        if os.path.exists(hyperparam_file):
            with open(hyperparam_file, 'r') as f:
                best_hyperparams[optimizer_name] = json.load(f)
                continue
                
        print(f"Tuning hyperparameters for {optimizer_name}...")
        
        # Create a tuning log file for this optimizer inside the logs folder
        log_file = os.path.join(logs_dir, f"{optimizer_name}_tuning_log.txt")
        with open(log_file, 'w') as lf:
            lf.write(f"Tuning log for {optimizer_name}:\n")
        
        # Create parameter combinations
        param_combinations = list(product(*param_grid[optimizer_name].values()))
        param_keys = list(param_grid[optimizer_name].keys())
        
        best_accuracy = 0
        best_params = {}
        
        # Run trials
        for trial in range(min(num_trials, len(param_combinations))):
            # Select a random parameter combination
            params_idx = random.randint(0, len(param_combinations) - 1)
            params = {key: param_combinations[params_idx][i] for i, key in enumerate(param_keys)}
            
            # Create model and optimizer
            model = model_fn().to(device)
            
            # Handle optimizer creation based on name
            if optimizer_name == "SGD":
                optimizer = torch.optim.SGD(model.parameters(), **params)
            elif optimizer_name == "ADAM" or optimizer_name == "Adam":  # Handle both cases
                optimizer = torch.optim.Adam(model.parameters(), **params)
            elif optimizer_name == "ADAMW" or optimizer_name == "AdamW":  # Handle both cases
                optimizer = torch.optim.AdamW(model.parameters(), **params)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
            # Define criterion based on task type to fix NameError
            criterion = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()
            
            # Train and evaluate the model
            accuracy = train_and_evaluate(model, train_loader, optimizer, criterion, device, task_type, epochs)
            
            # Append trial result to log file
            with open(log_file, 'a') as lf:
                lf.write(f"Trial {trial+1}: {optimizer_name} - Accuracy: {accuracy:.4f}, Params: {params}\n")
            
            # Update the best hyperparameters if the current accuracy is better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
            
            print(f"Trial {trial+1}: {optimizer_name} - Accuracy: {accuracy:.4f}, Params: {params}")
        
        print(f"Best {optimizer_name} Accuracy: {best_accuracy:.4f}, Best Params: {best_params}")
        best_hyperparams[optimizer_name] = best_params
        
        # Save the best hyperparameters to a file
        with open(hyperparam_file, 'w') as f:
            json.dump(best_params, f)
            print(f"Saved hyperparameters for {optimizer_name} to {hyperparam_file}")
    
    return best_hyperparams

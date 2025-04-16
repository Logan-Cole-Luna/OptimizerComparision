"""
Configuration file for Deep CNN experiments.

This file defines the parameter grid for hyperparameter tuning, optimizer-specific 
parameters, and learning rate scheduler configurations.

Attributes:
    PARAM_GRID (dict): A dictionary containing hyperparameter grids for various 
        optimizers. Each optimizer has its own set of tunable parameters such as 
        learning rate, momentum, weight decay, etc.
    BATCH_SIZE (int): The batch size used for training.
    EPOCHS (int): The number of epochs for training.
    LR (float): The default learning rate for optimizers.
    OPTIMIZERS (list): A list of optimizer names available for use in experiments.
    OPTIMIZER_PARAMS (dict): A dictionary containing default parameters for each 
        optimizer. These parameters are used during training.
    SCHEDULER_PARAMS (dict): A dictionary containing learning rate scheduler 
        configurations for each optimizer. Each scheduler has its own parameters 
        for controlling the learning rate schedule.
    RUNS_PER_OPTIMIZER (int): The number of runs to perform for each optimizer.

"""
# Determine runs per optimizer
RUNS_PER_OPTIMIZER = 1
BATCH_SIZE = 64
EPOCHS = 50 
LR = 0.001
OPTIMIZERS = ["SGD", "ADAM", "ADAMW"]

# Deep CNN experiment configuration
PARAM_GRID = {
    'SGD': {
        'lr': [0.0005, 0.001, 0.005],  # Reduced learning rates
        'momentum': [0.9],
        'weight_decay': [0.0001, 0.0005],  # Reduced weight decay
    },
    'ADAM': {
        'lr': [0.0001, 0.0003, 0.001, 0.003],  # Lower learning rates often work better with Adam
        'weight_decay': [0.0001, 0.0005, 0.001, 0.01],
        'betas': [(0.9, 0.999), (0.9, 0.99), (0.8, 0.999)],  # Try different momentum parameters
        'eps': [1e-8, 1e-7],  # Numerical stability term
    },
    'ADAMW': {
        'lr': [0.001, 0.005, 0.01],
        'weight_decay': [0.01, 0.05],
        'betas': [(0.9, 0.999), (0.9, 0.99), (0.8, 0.999)],  # Try different momentum parameters
        'eps': [1e-8, 1e-7],  # Numerical stability term
    }
}

OPTIMIZER_PARAMS = {
    "SGD": {"lr": LR, "momentum": 0.9, "nesterov": True},
    "ADAM": {
        "lr": 0.001,  # Lower default learning rate for Adam
        "betas": (0.9, 0.999),  # Explicitly set momentum parameters
        "eps": 1e-8,
        "weight_decay": 0.001  # Add some regularization
    }
}

SCHEDULER_PARAMS = {
    "ADAM": {
        "scheduler": "ReduceLROnPlateau",
        "params": {"mode": 'min', "factor": 0.5, "patience": 2}
    },
    "ADAMW": {
        "scheduler": "CosineAnnealingLR",
        "params": {"T_max": EPOCHS}
    },
    "SGD": {
        "scheduler": "StepLR",
        "params": {"step_size": 3, "gamma": 0.5}
    }
}

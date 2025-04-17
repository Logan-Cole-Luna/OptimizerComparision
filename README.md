# Optimizer Comparison with DeepCNN on CIFAR-10

This repository contains a PyTorch implementation of a Deep Convolutional Neural Network (DeepCNN) trained on the CIFAR-10 dataset. The primary objective is to compare the performance of different optimizers—SGD, Adam, and AdamW—across multiple training runs. The script includes hyperparameter tuning, custom weight initialization, and training visualization.

## Features

- **Dataset**: Utilizes the CIFAR-10 dataset with data augmentation techniques.
- **Model**: Implements a DeepCNN architecture suitable for image classification tasks.
- **Optimizers**: Compares Stochastic Gradient Descent (SGD), Adam, and AdamW optimizers.
- **Training**: Incorporates multiple training runs for statistical significance.
- **Visualization**: Provides training and validation accuracy/loss plots for performance analysis.

## Prerequisites

- Python 3.6 or higher
- PyTorch
- torchvision
- matplotlib
- numpy

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Logan-Cole-Luna/OptimizerComparision.git
   cd OptimizerComparision
   ```

2. (Optional but recommended) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install the dependencies:

   If a requirements.txt is available:

   ```bash
   pip install -r requirements.txt
   ```

   Otherwise, install manually:

   ```bash
   pip install torch torchvision matplotlib numpy
   ```

## Running the Script

To run the experiment:

```bash
python deepcnn_experiment.py
```

The script will:
- Download the CIFAR-10 dataset (if not already cached)
- Train the DeepCNN model using three optimizers
- Output accuracy and loss metrics
- Generate and save training/validation plots

## Resources

- PyTorch Installation: https://pytorch.org/get-started/locally/
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html

> Note: If a CUDA-compatible GPU is available, the script will use it. Otherwise, it defaults to CPU.

## Results

The script outputs plots comparing:
- Training and validation accuracy
- Training and validation loss

These are saved in the working directory and can be used for optimizer performance comparison.


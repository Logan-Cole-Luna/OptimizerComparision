import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from utils.network import DeepCNN  

# Configuration
OPTIMIZER_NAMES = ["SGD", "ADAM", "ADAMW"]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
IMAGE_INDEX_TO_SHOW = 5  

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 test dataset
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# CIFAR-10 class labels
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Select a single image and its label
image, true_label_idx = test_dataset[IMAGE_INDEX_TO_SHOW]
image_tensor = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
true_label = CLASSES[true_label_idx]

# Function to load model
def load_model(optimizer_name):
    model = DeepCNN().to(device)
    model_filename = f"deepcnn_{optimizer_name.lower()}.pth"
    model_path = os.path.join(RESULTS_DIR, model_filename)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded for {optimizer_name} from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Model file not found for {optimizer_name}: {model_path}")
        return None

# Load models and make predictions
predictions = {}
models = {}
for optimizer_name in OPTIMIZER_NAMES:
    model = load_model(optimizer_name)
    if model:
        models[optimizer_name] = model
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predictions[optimizer_name] = CLASSES[predicted_idx.item()]

# --- Visualization using Matplotlib ---
fig, axes = plt.subplots(1, 4, figsize=(15, 5)) # 1 row, 4 columns

# Subplot 1: Original Image
ax = axes[0]
img_display = image.cpu().numpy().transpose((1, 2, 0)) # Convert CHW to HWC for display
ax.imshow(img_display)
ax.set_title("Original Image")
ax.set_xlabel(f"True Label: {true_label}")
ax.set_xticks([])
ax.set_yticks([])

# Subplots 2-4: Predictions
for i, optimizer_name in enumerate(OPTIMIZER_NAMES):
    ax = axes[i+1]
    if optimizer_name in predictions:
        predicted_label = predictions[optimizer_name]
        ax.imshow(img_display)
        ax.set_title(f"Optimizer: {optimizer_name}")
        ax.set_xlabel(f"Predicted: {predicted_label}")
    else:
        ax.text(0.5, 0.5, 'Model not found', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(f"Optimizer: {optimizer_name}")

    ax.set_xticks([])
    ax.set_yticks([])


plt.tight_layout()
plt.show()

print("\nDemo finished.")

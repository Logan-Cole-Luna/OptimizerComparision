import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
import os
from network import DeepCNN  # Import the DeepCNN class
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Configuration
BATCH_SIZE = 64
OPTIMIZER_NAMES = ["SGD", "ADAM", "ADAMW"]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
VISUALS_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
PREDICTIONS_DIR = os.path.join(VISUALS_DIR, 'predictions')  # Directory for image predictions
NUM_PREDICTIONS = 10

# Create the predictions directory if it doesn't exist
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 test dataset and dataloader
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# CIFAR-10 class labels
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def visualize_prediction(image, prediction, optimizer_name, index):
    """Visualizes the prediction on the image with a caption."""
    image = image.cpu().numpy().transpose((1, 2, 0))
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)

    # Scale up the image
    scale_factor = 4  # Adjust the scale factor as needed
    image = image.resize((image.width * scale_factor, image.height * scale_factor), Image.NEAREST)

    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Use a truetype font
    try:
        font_size = min(width, height) // 8  # Adjust the divisor as needed
        font = ImageFont.truetype("arial.ttf", size=font_size)  # Replace "arial.ttf" with a valid font path
    except IOError:
        font = ImageFont.load_default()

    label = f"Predicted: {CLASSES[prediction]}"
    # Calculate the size of the text
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Calculate the position for the text (centered at the bottom)
    text_x = (width - text_width) // 2
    text_y = height - text_height - 5  # 5 pixels from the bottom

    # Draw a semi-transparent black rectangle behind the text
    draw.rectangle((text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2), fill=(0, 0, 0, 128))

    # Draw the text on the image
    draw.text((text_x, text_y), label, font=font, fill=(255, 255, 255))  # White text

    image.save(os.path.join(PREDICTIONS_DIR, f"prediction_{optimizer_name}_{index}.png"))

# Load models and make predictions
predictions = {}
for optimizer_name in OPTIMIZER_NAMES:
    model = DeepCNN().to(device)
    model_filename = f"deepcnn_{optimizer_name.lower()}.pth"
    model_path = os.path.join(RESULTS_DIR, model_filename)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set the model to evaluation mode
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        continue

    all_preds = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())

            for j in range(images.size(0)):  # Iterate through the batch
                if i * BATCH_SIZE + j < NUM_PREDICTIONS:
                    visualize_prediction(images[j], predicted[j].item(), optimizer_name, i * BATCH_SIZE + j)
                else:
                    break  # Stop after NUM_PREDICTIONS

            if i * BATCH_SIZE + images.size(0) >= NUM_PREDICTIONS:
                all_preds = all_preds[:NUM_PREDICTIONS]
                break  # All predictions done, exit the loop

    predictions[optimizer_name] = all_preds

# Create a DataFrame and save predictions to CSV
# df_predictions = pd.DataFrame(predictions)
# df_predictions.index.name = 'Sample ID'  # Set index name
# df_predictions.to_csv(PREDICTIONS_FILE)
print(f"Predictions with visualizations saved to {PREDICTIONS_DIR}")

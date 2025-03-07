import torch
import torch.nn as nn

# Define the CNN model (same as you have provided)
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 * 28 * 28, 1024),  # Automatically handle input size
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)  # Flatten before feeding into fully connected layers
        x = self.dense_layers(x)
        return x

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the number of classes (adjust according to your dataset)
num_classes = 34

# Initialize the model
model = CNN(num_classes)
model.to(device)

# Load the saved model weights
model.load_state_dict(torch.load(r"C:\Users\muska\OneDrive\Desktop\mushu\Plant-Disease-Detection-main\Flask Deployed App\final_plant_disease_model.pt", map_location=device))

# Set the model to evaluation mode
model.eval()

# Print confirmation
print("✅ Model loaded successfully!")

# Now, you can use this model to make predictions.
def load_model(model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Model prediction
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import pandas as pd

# Load the model
model = load_model(r"C:\Users\muska\OneDrive\Desktop\mushu\Plant-Disease-Detection-main\Flask Deployed App\final_plant_disease_model.pt", num_classes=34)

# Reverse index mapping
data = pd.read_csv(r"C:\Users\muska\OneDrive\Desktop\mushu\Plant-Disease-Detection-main\Flask Deployed App\disease_info.csv", encoding="cp1252")
data['disease_name'] = data['disease_name'].str.replace(" :", "___")  # Standardize format
transform_index_to_disease = data["disease_name"].to_dict()
transform_index_to_disease = dict([(value, key) for key, value in transform_index_to_disease.items()])

# Reverse the mapping correctly
transform_index_to_disease = {v: k for k, v in transform_index_to_disease.items()}

def single_prediction(image_path):
    # Image preprocessing
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Resize the image to match the input size
    
    # Convert to tensor and normalize (if normalization was done during training)
    input_data = TF.to_tensor(image)
    mean = [0.485, 0.456, 0.406]  # Example, update with correct values if needed
    std = [0.229, 0.224, 0.225]   # Example, update with correct values if needed
    input_data = TF.normalize(input_data, mean=mean, std=std)
    input_data = input_data.view((-1, 3, 224, 224))  # Reshape to batch form

    # Model prediction
    output = model(input_data)
    output = output.detach().numpy()

    print("Model output:", output)  # This is the raw output from the model

    # Apply softmax to get probabilities
    softmax_output = torch.nn.functional.softmax(torch.tensor(output), dim=1).numpy()
    print("Softmax output:", softmax_output)  # Probabilities of each class

    # Get the index of the highest probability
    index = np.argmax(softmax_output)
    print("Predicted index:", index)  # Check if the index is reasonable

    # Fetch the disease name based on index
    predicted_disease = transform_index_to_disease.get(index, "Unknown Disease")
    print("Predicted Disease:", predicted_disease)

# Example usage
single_prediction(r"C:\Users\muska\OneDrive\Desktop\mushu\Plant-Disease-Detection-main\test_images\Apple_ceder_apple_rust.JPG")

import numpy as np
from PIL import Image

# Function to calculate affected area
def calculate_severity_pil(prediction_mask_path):
    # Load the segmentation mask as grayscale
    mask = Image.open(prediction_mask_path).convert("L")  # Convert to grayscale
    mask_array = np.array(mask)  # Convert to NumPy array

    # Assuming white (255) represents affected areas and black (0) is healthy
    affected_area = np.sum(mask_array > 128)  # Count pixels above threshold (affected)
    total_area = mask_array.size  # Total pixels in the image

    severity = (affected_area / total_area) * 100  # Severity percentage
    return severity

# Example Usage
severity_percentage = calculate_severity_pil(r"C:\Users\muska\OneDrive\Desktop\mushu\Plant-Disease-Detection-main\test_images\Apple_ceder_apple_rust.JPG")
print(f"Confidence: {severity_percentage:.2f}%")


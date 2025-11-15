import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
# Imports for handling image data and transfer learning
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
# Data utilities and visualization
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# --- 1. DATASET AND DATALOADER SETUP ---

# IMPORTANT: SET YOUR BASE PATH HERE
# Replace this path with the directory where you downloaded and unzipped the LC25000 lung data.
# We assume the structure is: [base_data_dir]/train/lung_n, [base_data_dir]/train/lung_aca, etc.
base_data_dir = r'path/to/your/LC25000/lung_image_sets' 

class HistologyImageDataset(Dataset):
    """
    Custom Dataset class based on ImageFolder for handling histology images.
    It automatically labels data based on subdirectory names.
    """
    def __init__(self, data_dir, transform=None):
        # ImageFolder automatically handles labels based on subfolders (e.g., lung_n, lung_aca, lung_scc)
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        """Returns the list of class names inferred from directory names."""
        return self.data.classes

# --- 2. TRANSFORMATION AND DATA LOADING ---

# The LC25000 images are large (e.g., 768x768). Resize to a common size for EfficientNet.
transform = transforms.Compose([
    transforms.Resize((224, 224)), # EfficientNet_B0 prefers 224x224 input
    transforms.ToTensor(),
    # Normalization parameters from ImageNet (standard practice for pre-trained models)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define paths for the split data
train_folder = os.path.join(base_data_dir, 'train')
valid_folder = os.path.join(base_data_dir, 'valid')
test_folder = os.path.join(base_data_dir, 'test') # Note: LC25000 often uses 'test' for validation

# Create datasets
train_dataset = HistologyImageDataset(train_folder, transform=transform)
valid_dataset = HistologyImageDataset(valid_folder, transform=transform)
test_dataset = HistologyImageDataset(test_folder, transform=transform)

# Determine the number of classes (should be 3: lung_n, lung_aca, lung_scc)
NUM_CLASSES = len(train_dataset.classes)
print(f"Detected Classes: {train_dataset.classes}")
print(f"Number of Output Classes for Model: {NUM_CLASSES}")

# Create DataLoaders
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Map target index to class name for prediction output
target_to_class = {i: c for i, c in enumerate(train_dataset.classes)}

# --- 3. PYTORCH MODEL DEFINITION ---

class HistologyClassifier(nn.Module):
    def __init__(self, num_classes):
        super(HistologyClassifier, self).__init__()

        # Use EfficientNet-B0 as the base model for transfer learning
        # 'pretrained=True' loads weights trained on ImageNet
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
        
        # NOTE: When setting num_classes in timm.create_model, it automatically
        # replaces the final classification layer, making the model ready.
        
    def forward(self, x): 
        # The base_model handles the features and final classification layer
        return self.base_model(x)

# Initialize the model with the detected number of classes (expected 3)
model = HistologyClassifier(num_classes=NUM_CLASSES)

# Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- 4. TRAINING LOOP ---
num_epoch = 5 
train_losses, val_losses = [], []
print(f"Starting training on device: {device}")

for epoch in range(num_epoch):
    # Training Phase
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1} Training'):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * labels.size(0)
    
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation Phase
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc=f'Epoch {epoch+1} Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * labels.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    val_loss = running_loss / len(valid_loader.dataset)
    val_accuracy = correct_predictions / total_samples
    # --- BUG FIX: Append the correct loss variable ---
    val_losses.append(val_loss) 

    print(f"Epoch {epoch+1}/{num_epoch} - Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# --- 5. MODEL SAVING (CRITICAL FOR DEPLOYMENT) ---
model_save_path = 'histology_classifier_final.pth'
torch.save(model.state_dict(), model_save_path)
print(f"\nModel weights successfully saved to: {model_save_path}")
print("This file should be used in your FastAPI backend for deployment.")

# --- 6. PREDICTION INTERPRETATION FOR WEB APP ---

def interpret_binary_result(prediction_index, classes):
    """
    Converts the 3-class prediction into the required binary 'Cancerous' or 'Non-Cancerous' result.
    This logic will be mirrored in your FastAPI service.
    """
    predicted_class = classes[prediction_index]
    
    if predicted_class == 'lung_n':
        return "Non-Cancerous"
    elif predicted_class in ['lung_aca', 'lung_scc']:
        return "Cancerous"
    else:
        return "Uncertain"

print("\n--- Prediction Interpretation Logic ---")
print(f"Example 1: Model predicts index 0 (if index 0 maps to {target_to_class.get(0)}): Result -> {interpret_binary_result(0, target_to_class)}")
print(f"Example 2: Model predicts index 1 (if index 1 maps to {target_to_class.get(1)}): Result -> {interpret_binary_result(1, target_to_class)}")
print(f"Example 3: Model predicts index 2 (if index 2 maps to {target_to_class.get(2)}): Result -> {interpret_binary_result(2, target_to_class)}")
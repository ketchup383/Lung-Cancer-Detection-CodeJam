import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# Handling image data and transfer learning
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

# Data utilities and visualization
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from tqdm import tqdm

# 1. DATASET AND DATALOADER SETUP
# Dataset has been split into train, valid, and test folders
base_data_dir = r'./data_splits' 

# Handle histological slides and data labeling
class HistologyImageDataset(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        # Return class names based on directory name
        return self.data.classes

# 2. TRANSFORMATION AND DATA LOADING 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Apply standard EfficientNet normalization for pre-trained
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define paths for the split data
train_folder = os.path.join(base_data_dir, 'train')
valid_folder = os.path.join(base_data_dir, 'valid')
test_folder = os.path.join(base_data_dir, 'test')

# Create datasets
train_dataset = HistologyImageDataset(train_folder, transform = transform)
valid_dataset = HistologyImageDataset(valid_folder, transform = transform)
test_dataset = HistologyImageDataset(test_folder, transform = transform)

# Determine number of classes (lung_n, lung_aca, lung_scc) --> 3
NUM_CLASSES = len(train_dataset.classes)
print(f"Detected Classes: {train_dataset.classes}")
print(f"Number of Output Classes for Model: {NUM_CLASSES}")

        # Map efficientnet output:
        # lung_n   → benign (0)
        # lung_aca → cancerous (1)
        # lung_scc → cancerous (1)
        self.binary_map = {
            "lung_n": 0,
            "lung_aca": 1,
            "lung_scc": 1
        }

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        image, original_label = self.folder[idx]
        class_name = self.folder.classes[original_label]
        binary_label = self.binary_map[class_name]
        return image, torch.tensor(binary_label)

    @property
    def classes(self):
        return ["benign", "cancerous"]


# Load dataset (no train/valid/test folders needed)
full_dataset = BinaryHistologyDataset(base_data_dir, transform=transform)

# Split 80/10/10
total_len = len(full_dataset)
train_len = int(0.8 * total_len)
valid_len = int(0.1 * total_len)
test_len = total_len - train_len - valid_len

train_dataset, valid_dataset, test_dataset = random_split(
    full_dataset, [train_len, valid_len, test_len]
)

print("Dataset sizes:")
print("Train:", len(train_dataset))
print("Valid:", len(valid_dataset))
print("Test : ", len(test_dataset))


# ---------------------------
# 2. DATA LOADERS
# ---------------------------
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle = False)


# 3. PYTORCH MODEL

class HistologyBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=2  # binary output
        )

        # Use EfficientNet-B0 as model
        self.base_model = timm.create_model('efficientnet_b0', pretrained = True, num_classes=num_classes)
        
    def forward(self, x): 
        return self.base_model(x)

model = HistologyClassifier(num_classes = NUM_CLASSES)

# Loss function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 4. TRAINING LOOP
num_epoch = 5 
train_losses, val_losses = [], []
val_accuracies = []
print(f"Starting training on device: {device}")

# ---------------------------
# 4. TRAINING LOOP
# ---------------------------

EPOCHS = 5
print(f"Training on {device}")

for epoch in range(EPOCHS):
    # --- TRAIN ---
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc = f'Epoch {epoch + 1} Training'):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * y.size(0)

    train_loss /= len(train_loader.dataset)


    # --- VALIDATION ---
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in tqdm(valid_loader, desc=f"Epoch {epoch+1} Validation"):
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            val_loss += loss.item() * y.size(0)

    val_loss = running_loss / len(valid_loader.dataset)
    val_accuracy = correct_predictions / total_samples
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch + 1}/{num_epoch} - Train loss: {train_loss: .4f}, Validation loss: {val_loss: .4f}, Validation Accuracy: {val_accuracy: .4f}")

# 5. MODEL SAVING (CRITICAL FOR DEPLOYMENT)
model_save_path = 'histology_classifier_final.pth'
torch.save(model.state_dict(), model_save_path)
print(f"\nModel weights successfully saved to: {model_save_path}")
print("*Use this file for FastAPI backend.")

# 6. PREDICTION INTERPRETATION (FOR WEB APP)

# def interpret_binary_result(prediction_index, classes):
#     predicted_class = classes[prediction_index]
    
#     if predicted_class == 'lung_n':
#         return "Non-Cancerous"
#     elif predicted_class in ['lung_aca', 'lung_scc']:
#         return "Cancerous"
#     else:
#         return "Uncertain"

# print("\n--- Prediction Interpretation Logic ---")
# print(f"Example 1: Model predicts index 0 (if index 0 maps to {target_to_class.get(0)}): Result -> {interpret_binary_result(0, target_to_class)}")
# print(f"Example 2: Model predicts index 1 (if index 1 maps to {target_to_class.get(1)}): Result -> {interpret_binary_result(1, target_to_class)}")
# print(f"Example 3: Model predicts index 2 (if index 2 maps to {target_to_class.get(2)}): Result -> {interpret_binary_result(2, target_to_class)}")

def plot_training_results(train_losses, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # First plot: Training and Validation Loss
    plt.figure(figsize = (10, 5))
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.title('Loss over Epochs (Detecting Overfitting)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (CrossEntropy)')
    plt.legend()
    plt.grid(True)
    plt.show() # Display loss plot
    
    # Second plot: Validation Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_accuracies, 'g', label='Validation Accuracy', marker='o')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show() # Display accuracy plot

# Call the plotting function after the training loop completes
if num_epoch > 0:
    plot_training_results(train_losses, val_losses, val_accuracies)
    print("\nTraining plots generated.")
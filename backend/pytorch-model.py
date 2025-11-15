import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import os
from tqdm import tqdm


# ---------------------------
# 1. DATASET SETUP
# ---------------------------

base_data_dir = '../lung_image_sets'

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ----- Custom Wrapper to Convert 3 Classes → Binary -----

class BinaryHistologyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.folder = ImageFolder(data_dir, transform=transform)

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
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# ---------------------------
# 3. MODEL SETUP (EfficientNet-B0 Binary)
# ---------------------------

class HistologyBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=2  # binary output
        )

    def forward(self, x):
        return self.model(x)


model = HistologyBinaryClassifier()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# ---------------------------
# 4. TRAINING LOOP
# ---------------------------

EPOCHS = 5
print(f"Training on {device}")

for epoch in range(EPOCHS):
    # --- TRAIN ---
    model.train()
    train_loss = 0

    for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        X, y = X.to(device), y.to(device)

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

            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()

    val_loss /= len(valid_loader.dataset)
    val_acc = correct / len(valid_loader.dataset)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f}")


# ---------------------------
# 5. SAVE MODEL
# ---------------------------

torch.save(model.state_dict(), "binary_histology_classifier.pth")
print("Saved model → binary_histology_classifier.pth")


# ---------------------------
# 6. INTERPRETATION FUNCTION
# ---------------------------

def interpret(pred_idx):
    return "Benign" if pred_idx == 0 else "Cancerous"

print(interpret(0))
print(interpret(1))

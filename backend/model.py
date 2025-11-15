import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
import os

# Model architecture (must match pytorch-model.py)
class HistologyClassifier(nn.Module):
    def __init__(self, num_classes):
        super(HistologyClassifier, self).__init__()
        # Use EfficientNet-B0 as model
        self.base_model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
        
    def forward(self, x): 
        return self.base_model(x)

# Class names (alphabetically sorted as ImageFolder does)
CLASS_NAMES = ['lung_aca', 'lung_n', 'lung_scc']
NUM_CLASSES = len(CLASS_NAMES)

# Same transforms as used in training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Apply standard EfficientNet normalization for pre-trained
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize model and load weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HistologyClassifier(num_classes=NUM_CLASSES)

# Load model weights (check both backend and root directory)
model_path = 'histology_classifier_final.pth'
if not os.path.exists(model_path):
    # Try parent directory (root)
    model_path = '../histology_classifier_final.pth'

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded successfully from {model_path}")
else:
    raise FileNotFoundError(f"Model file not found. Please ensure 'histology_classifier_final.pth' exists in backend/ or root directory.")

def predict_image(image_path: str) -> dict:
    """
    Predict the class of a lung histology image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with 'class' and 'confidence' keys
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        # Get class name and confidence
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_value = confidence.item()
        
        return {
            "class": predicted_class,
            "confidence": round(confidence_value, 4)
        }
    
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")


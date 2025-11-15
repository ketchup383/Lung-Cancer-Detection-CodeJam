import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
import os
import io # Import for handling file objects/bytes

# --- Configuration ---
# Class names (alphabetically sorted as ImageFolder does)
CLASS_NAMES = ['lung_aca', 'lung_n', 'lung_scc']
NUM_CLASSES = len(CLASS_NAMES)
MODEL_FILENAME = 'histology_classifier_final.pth' # Use a constant for the filename

# Model architecture (must match training script)
class HistologyClassifier(nn.Module):
    def __init__(self, num_classes):
        super(HistologyClassifier, self).__init__()
        # Use EfficientNet-B0 as model. pretrained=False since we load weights later.
        self.base_model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
        
    def forward(self, x): 
        return self.base_model(x)

# Same transforms as used in training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Apply standard EfficientNet normalization for pre-trained
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize model and load weights
# Device detection should happen only once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HistologyClassifier(num_classes=NUM_CLASSES)

# --- CRITICAL FIX: Ensure model path is relative to main.py's execution context ---
# Since main.py is running the backend, the relative path logic is correct here.
model_path = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)
# Check outside the current directory as a fallback for flexibility
if not os.path.exists(model_path):
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), MODEL_FILENAME)

if os.path.exists(model_path):
    # Load state dict only once on startup
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    # The print statement is good for debugging startup issues
    print(f"Model loaded successfully from {model_path}")
else:
    # Exit cleanly if the model file is not found (prevents app startup)
    raise FileNotFoundError(f"Model file '{MODEL_FILENAME}' not found. Checked: {model_path}")

def predict_image(image_path: str) -> dict:
    """
    Predict the class of a lung histology image.
    
    Args:
        image_path: Path to the image file saved temporarily on the server.
        
    Returns:
        Dictionary with 'class' and 'confidence' keys
    """
    try:
        # Load and preprocess image using file path
        image = Image.open(image_path).convert('RGB')
        
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            # Use outputs[0] to access the logits for the single image in the batch
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
        # It's better to log the exception and re-raise it for FastAPI to catch
        print(f"Prediction Error in model.py: {str(e)}")
        raise Exception("Failed to process image and run prediction.")
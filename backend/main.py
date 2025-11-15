import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
import shutil
# FIX: Use a relative import to ensure 'model.py' is found within the 'backend' package
from .model import predict_image 

# --- Pydantic Schemas (Re-defined for clarity across files) ---
# These must match the types expected by the Angular frontend (app.ts)

class PredictionResponse(BaseModel):
    image_id: str
    result: str
    probability: float
    is_cancerous: bool
    
class FeedbackRequest(BaseModel):
    image_id: str
    user_id: str
    user_correction: str
    model_prediction: str
    confidence: float
# -------------------------------------------------------------

app = FastAPI(title="Lung Histology Prediction API")

# Allow Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to temporarily save uploaded files
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/status")
def get_status():
    """Checks the health of the API."""
    return {"status": "online", "message": "Cancer Detection API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Receives an uploaded image, saves it, gets the PyTorch prediction, 
    and returns the result in the format the Angular frontend expects.
    """
    # 1. Setup paths and generate a unique ID
    image_uuid = str(uuid.uuid4())
    file_id = f"{image_uuid}_{file.filename}" 
    file_path = os.path.join(UPLOAD_DIR, file_id)

    try:
        # 2. Save the uploaded file locally
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 3. Run prediction using the model.py function
        prediction_result = predict_image(file_path)
        
        # 4. Implement Binary Logic (matching the Angular expected output)
        is_cancerous = prediction_result["class"] in ['lung_aca', 'lung_scc']
        binary_result = "Cancerous" if is_cancerous else "Non-Cancerous"

        # 5. Return the structured response
        return PredictionResponse(
            image_id=image_uuid, 
            result=binary_result,
            probability=prediction_result["confidence"],
            is_cancerous=is_cancerous
        )

    except Exception as e:
        print(f"Prediction or file handling error: {e}")
        # Send a generic 500 error back to the client
        raise HTTPException(status_code=500, detail=f"Prediction failed due to internal server error.")
    finally:
        # 6. Clean up: Delete the temporary file (Optional but recommended)
        # We comment this out so you can inspect the uploaded files, 
        # but in production, you should delete it or move it to permanent storage.
        # os.remove(file_path) 
        pass


@app.post("/feedback")
async def receive_feedback(feedback: FeedbackRequest):
    """
    This endpoint serves as a proxy or logger for the feedback loop. 
    The Angular app is currently set to write directly to Firestore.
    """
    # Log the feedback locally (for demonstration purposes)
    print("\n--- NEW USER FEEDBACK RECEIVED (FastAPI Log) ---")
    print(f"Image ID: {feedback.image_id}")
    print(f"User ID: {feedback.user_id}")
    print(f"User Correction: {feedback.user_correction}")
    print("------------------------------------\n")
    
    return {"message": "Feedback received and logged successfully."}
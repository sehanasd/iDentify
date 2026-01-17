from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os

# Import our Logic Layer
from inference import setup_system, run_inference

# Global Variable to hold models
MODELS = {}

# --- LIFESPAN (The Modern Way to Load Models) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup: Load Models
    print("⏳ Initializing AI Models...")
    yolo_path = "models/best.pt"
    effnet_path = "models/efficientnet_best.pth"
    
    # Check if models exist before loading
    if os.path.exists(yolo_path) and os.path.exists(effnet_path):
        yolo_model, eff_model = setup_system(yolo_path, effnet_path)
        MODELS['yolo'] = yolo_model
        MODELS['effnet'] = eff_model
        print("✅ System Ready!")
    else:
        print("❌ Error: Model files not found in /models folder!")
    
    yield  # The application runs here
    
    # 2. Shutdown: Clean up (if needed)
    print("🛑 Shutting down system...")
    MODELS.clear()

# Initialize App with Lifespan
app = FastAPI(lifespan=lifespan)

# Allow React to talk to us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": "yolo" in MODELS}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if 'yolo' not in MODELS:
        return {"error": "Models not loaded"}
        
    image_bytes = await file.read()
    result = run_inference(image_bytes, MODELS['yolo'], MODELS['effnet'])
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
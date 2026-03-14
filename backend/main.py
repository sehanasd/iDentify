from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os

from inference import setup_system, run_inference, validate_opg

MODELS = {}

YOLO_PATH   = os.path.join(os.path.dirname(__file__), "models", "best.pt")
EFFNET_PATH = os.path.join(os.path.dirname(__file__), "models", "efficientnet_b0_v4_best.pt")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Initializing iDentify AI System...")
    if os.path.exists(YOLO_PATH) and os.path.exists(EFFNET_PATH):
        yolo_model, eff_model, grad_cam = setup_system(YOLO_PATH, EFFNET_PATH)
        MODELS['yolo']     = yolo_model
        MODELS['effnet']   = eff_model
        MODELS['grad_cam'] = grad_cam
        print("✅ iDentify System Ready!")
    else:
        missing = []
        if not os.path.exists(YOLO_PATH):   missing.append(YOLO_PATH)
        if not os.path.exists(EFFNET_PATH): missing.append(EFFNET_PATH)
        print(f"❌ Model files not found: {missing}")
    yield
    print("🛑 Shutting down iDentify...")
    MODELS.clear()

app = FastAPI(
    title="iDentify API",
    description="AI-powered dental pathology detection from OPG radiographs",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "online", "system": "iDentify", "models_loaded": "yolo" in MODELS}

@app.get("/health")
def health():
    return {"status": "ready" if "yolo" in MODELS else "models_not_loaded", "models_loaded": "yolo" in MODELS}

@app.post("/validate")
async def validate_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = validate_opg(image_bytes, file.filename)
    return result

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if 'yolo' not in MODELS:
        raise HTTPException(status_code=503, detail="Models not loaded.")
    image_bytes = await file.read()
    validation = validate_opg(image_bytes, file.filename)
    if not validation['valid']:
        return {"error": True, "reason": validation['reason']}
    result = run_inference(image_bytes, MODELS['yolo'], MODELS['effnet'], MODELS['grad_cam'])
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
import torch
import cv2
import numpy as np
import io
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms, models
import torch.nn as nn
import base64

# --- CONFIGURATION ---
CLASS_NAMES = ['bdc_bdr', 'caries', 'fractured', 'healthy', 'impacted', 'infection']

# Select Device (Supports Mac M4/M1 Metal Performance Shaders)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"🚀 Using Apple MPS (Mac GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 Using CUDA GPU")
else:
    device = torch.device("cpu")
    print(f"⚠️ Using CPU (Slower)")

def setup_system(yolo_path, effnet_path):
    print("Loading models...")
    
    # 1. Load YOLO
    yolo_model = YOLO(yolo_path)
    
    # 2. Load EfficientNet
    eff_model = models.efficientnet_b0(weights=None)
    in_features = eff_model.classifier[1].in_features
    eff_model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    
    # Load Weights safely
    state_dict = torch.load(effnet_path, map_location=device)
    eff_model.load_state_dict(state_dict)
    eff_model.to(device)
    eff_model.eval()
    
    print("✅ Models Loaded Successfully.")
    return yolo_model, eff_model

def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def run_inference(image_bytes, yolo_model, eff_model):
    # 1. Convert bytes to OpenCV Image
    nparr = np.frombuffer(image_bytes, np.uint8)
    original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 2. Preprocess
    processed_img = apply_clahe(original_img)
    output_img = processed_img.copy() # We draw on this
    h_img, w_img, _ = original_img.shape
    
    # 3. Stage 1: YOLO
    results = yolo_model(processed_img, verbose=False)
    
    # EfficientNet Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    detections = []
    
    # 4. Loop through detections
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Padding
            pad = 0.15
            w_box, h_box = x2 - x1, y2 - y1
            x1_p = max(0, int(x1 - w_box * pad))
            y1_p = max(0, int(y1 - h_box * pad))
            x2_p = min(w_img, int(x2 + w_box * pad))
            y2_p = min(h_img, int(y2 + h_box * pad))
            
            # Crop & Stage 2: EfficientNet
            crop = processed_img[y1_p:y2_p, x1_p:x2_p]
            if crop.size == 0: continue
            
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            input_tensor = preprocess(crop_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = eff_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_prob, top_idx = torch.max(probabilities, 1)
                diagnosis = CLASS_NAMES[top_idx.item()]
                confidence = top_prob.item()
            
            detections.append({
                "diagnosis": diagnosis,
                "confidence": float(confidence),
                "box": [x1, y1, x2, y2]
            })

            # --- DRAWING (Thinner Lines) ---
            color = (0, 255, 0) if diagnosis == 'healthy' else (0, 0, 255)
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 1)
            
            label = f"{diagnosis} ({confidence*100:.0f}%)"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            text_y = y1 - 4 if y1 - h - 4 > 0 else y1 + h + 10
            
            cv2.rectangle(output_img, (x1, text_y - h - 2), (x1 + w, text_y + 2), color, -1)
            cv2.putText(output_img, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    # 5. Encode Result Image to Base64 (to send back to React)
    _, buffer = cv2.imencode('.jpg', output_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {"detections": detections, "image_base64": img_base64}
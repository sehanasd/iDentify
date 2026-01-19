import torch
import cv2
import numpy as np
import base64
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F

# --- CONFIGURATION ---
CLASS_NAMES = ['bdc_bdr', 'caries', 'fractured', 'healthy', 'impacted', 'infection']

# Select Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# --- 1. XAI ENGINE (Grad-CAM) ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook into the model to catch data
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        # 1. Zero out previous gradients
        self.model.zero_grad()
        
        # 2. Forward Pass
        output = self.model(input_tensor)
        
        # 3. Backward Pass (Focus on the specific diagnosis class)
        target = output[0][class_idx]
        target.backward()
        
        # 4. Generate Heatmap from gradients
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        # Global Average Pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # 5. ReLU (We only care about positive influence)
        cam = np.maximum(cam, 0)
        
        # 6. Normalize to 0-1
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

# --- 2. SYSTEM SETUP ---
def setup_system(yolo_path, effnet_path):
    print("Loading models...")
    
    # Load YOLO
    yolo_model = YOLO(yolo_path)
    
    # Load EfficientNet
    eff_model = models.efficientnet_b0(weights=None)
    in_features = eff_model.classifier[1].in_features
    eff_model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    
    state_dict = torch.load(effnet_path, map_location=device)
    eff_model.load_state_dict(state_dict)
    eff_model.to(device)
    eff_model.eval()
    
    print("✅ Models Loaded.")
    return yolo_model, eff_model

def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def run_inference(image_bytes, yolo_model, eff_model):
    # Prepare Grad-CAM
    grad_cam = GradCAM(eff_model, eff_model.features[-1])

    # 1. Read Image
    nparr = np.frombuffer(image_bytes, np.uint8)
    original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_img = apply_clahe(original_img)
    h_img, w_img, _ = original_img.shape
    
    # 2. Create TWO copies
    img_standard = processed_img.copy() # For Boxes Only
    img_xai = processed_img.copy()      # For Boxes + Heatmaps

    # 3. YOLO Detection
    results = yolo_model(processed_img, verbose=False)
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    detections = []
    
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
            
            crop = processed_img[y1_p:y2_p, x1_p:x2_p]
            if crop.size == 0: continue
            
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            input_tensor = preprocess(crop_pil).unsqueeze(0).to(device)
            input_tensor.requires_grad = True
            
            eff_model.zero_grad()
            outputs = eff_model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_prob, top_idx = torch.max(probabilities, 1)
            diagnosis = CLASS_NAMES[top_idx.item()]
            confidence = top_prob.item()
            
            if diagnosis == 'healthy': continue

            # --- A. HEATMAP LOGIC (Apply only to img_xai) ---
            heatmap = grad_cam.generate_heatmap(input_tensor, top_idx.item())
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap_resized = cv2.resize(heatmap, (crop.shape[1], crop.shape[0]))
            
            # Blend onto the specific crop location in img_xai
            roi = img_xai[y1_p:y2_p, x1_p:x2_p]
            superimposed = cv2.addWeighted(roi, 0.6, heatmap_resized, 0.4, 0)
            img_xai[y1_p:y2_p, x1_p:x2_p] = superimposed
            # -----------------------------------------------

            detections.append({
                "diagnosis": diagnosis,
                "confidence": float(confidence),
                "box": [x1, y1, x2, y2]
            })

            # --- B. DRAW BOXES (On BOTH images) ---
            color = (0, 0, 255)
            label = f"{diagnosis} ({confidence*100:.0f}%)"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            for target_img in [img_standard, img_xai]:
                cv2.rectangle(target_img, (x1, y1), (x2, y2), color, 1)
                text_y = y1 - 4 if y1 - h - 4 > 0 else y1 + h + 10
                cv2.rectangle(target_img, (x1, text_y - h - 2), (x1 + w, text_y + 2), color, -1)
                cv2.putText(target_img, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    # 4. Encode BOTH images
    _, buf_std = cv2.imencode('.jpg', img_standard)
    _, buf_xai = cv2.imencode('.jpg', img_xai)
    
    return {
        "detections": detections, 
        "image_standard": base64.b64encode(buf_std).decode('utf-8'),
        "image_xai": base64.b64encode(buf_xai).decode('utf-8')
    }
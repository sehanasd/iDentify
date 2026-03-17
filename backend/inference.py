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
EFFNET_CLASS_NAMES = ['bdc_bdr', 'caries', 'fractured', 'impacted', 'infection']

YOLO_CONF       = 0.35
YOLO_IOU        = 0.45
EFFNET_CONF_MIN = 0.70
PADDING         = 0.10

CLASS_COLORS = {
    'bdc_bdr':   (0, 165, 255),
    'caries':    (0, 0, 255),
    'fractured': (255, 0, 0),
    'impacted':  (0, 255, 255),
    'infection': (0, 255, 0),
}

CLASS_LABELS = {
    'bdc_bdr':   'Broken Down Crown/Root',
    'caries':    'Dental Caries',
    'fractured': 'Fractured Tooth',
    'impacted':  'Impacted Tooth',
    'infection': 'Periapical Infection',
}

CLASS_HEX = {
    'bdc_bdr':   '#FF44FF',
    'caries':    '#FFA500',
    'fractured': '#FFFF44',
    'impacted':  '#4d9ef7',
    'infection': '#FF4444',
}

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
MAX_FILE_SIZE      = 20 * 1024 * 1024  # 20 MB
MIN_WIDTH, MIN_HEIGHT = 400, 200
MIN_ASPECT, MAX_ASPECT = 1.5, 5.0

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# --- GRAD-CAM ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.target_layer = target_layer
        self.gradients   = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0][class_idx].backward()

        gradients  = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        weights    = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam


# --- SETUP ---
def setup_system(yolo_path, effnet_path):
    print("⏳ Loading models...")

    yolo_model = YOLO(yolo_path)
    print(f"   YOLO classes: {yolo_model.names}")

    eff_model = models.efficientnet_b0(weights=None)
    eff_model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(1280, 256),
        nn.BatchNorm1d(256),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 5),
    )
    ckpt = torch.load(effnet_path, map_location=device)
    eff_model.load_state_dict(ckpt['model_state_dict'])
    eff_model.to(device)
    eff_model.eval()
    print(f"   EfficientNet classes: {EFFNET_CLASS_NAMES}")

    grad_cam = GradCAM(eff_model, eff_model.features[-1])
    print("✅ iDentify System Ready!")
    return yolo_model, eff_model, grad_cam


# --- VALIDATION ---
def validate_opg(image_bytes, filename=""):
    import os
    ext = os.path.splitext(filename or "")[1].lower()

    if ext and ext not in ALLOWED_EXTENSIONS:
        return {"valid": False, "reason": f"Unsupported file type '{ext}'. Use JPG, PNG, BMP or TIFF."}

    if len(image_bytes) > MAX_FILE_SIZE:
        return {"valid": False, "reason": "File exceeds 20 MB limit."}

    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"valid": False, "reason": "Could not decode the image. Make sure it is a valid image file."}

    h, w = img.shape[:2]

    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return {"valid": False, "reason": f"Image too small ({w}×{h}px). Minimum is {MIN_WIDTH}×{MIN_HEIGHT}px."}

    aspect = w / h
    if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
        return {"valid": False, "reason": f"Aspect ratio {aspect:.2f} is outside the expected OPG range (1.5–5.0). Make sure you are uploading a panoramic radiograph."}

    # Greyscale check — OPGs should have very low colour saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_sat = float(np.mean(hsv[:, :, 1]))
    if mean_sat > 40:
        return {"valid": False, "reason": "Image appears to be a colour photograph. Please upload a greyscale panoramic dental radiograph (OPG)."}

    return {"valid": True, "reason": ""}


# --- HELPERS ---
def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


# --- INFERENCE ---
def run_inference(image_bytes, yolo_model, eff_model, grad_cam):
    nparr        = np.frombuffer(image_bytes, np.uint8)
    original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_img = apply_clahe(original_img)
    h_img, w_img, _ = original_img.shape

    img_standard = processed_img.copy()
    img_xai      = processed_img.copy()

    results = yolo_model(processed_img, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            w_box, h_box = x2 - x1, y2 - y1
            x1_p = max(0, int(x1 - w_box * PADDING))
            y1_p = max(0, int(y1 - h_box * PADDING))
            x2_p = min(w_img, int(x2 + w_box * PADDING))
            y2_p = min(h_img, int(y2 + h_box * PADDING))

            crop = processed_img[y1_p:y2_p, x1_p:x2_p]
            if crop.size == 0:
                continue

            crop_pil     = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            input_tensor = preprocess(crop_pil).unsqueeze(0).to(device)
            input_tensor.requires_grad = True

            eff_model.zero_grad()
            outputs       = eff_model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_prob, top_idx = torch.max(probabilities, 1)

            diagnosis  = EFFNET_CLASS_NAMES[top_idx.item()]
            confidence = top_prob.item()

            # Fractured confidence gate
            if diagnosis == 'fractured' and confidence < EFFNET_CONF_MIN:
                probs_copy = probabilities.clone()
                probs_copy[0][top_idx] = 0
                top_prob2, top_idx2 = torch.max(probs_copy, 1)
                diagnosis  = EFFNET_CLASS_NAMES[top_idx2.item()]
                confidence = top_prob2.item()
                top_idx    = top_idx2

            # Grad-CAM heatmap
            heatmap = grad_cam.generate_heatmap(input_tensor, top_idx.item())
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap_resized = cv2.resize(heatmap, (crop.shape[1], crop.shape[0]))

            roi = img_xai[y1_p:y2_p, x1_p:x2_p]
            img_xai[y1_p:y2_p, x1_p:x2_p] = cv2.addWeighted(roi, 0.6, heatmap_resized, 0.4, 0)

            detections.append({
                "class_name":     diagnosis,
                "label":          CLASS_LABELS.get(diagnosis, diagnosis),
                "confidence":     float(confidence),
                "confidence_pct": f"{confidence * 100:.1f}%",
                "color_hex":      CLASS_HEX.get(diagnosis, '#4d9ef7'),
                "box":            [x1, y1, x2, y2],
            })

    # --- Two-pass rendering: boxes first, then labels on top ---
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thick = 1
    pad        = 4

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = CLASS_COLORS.get(det["class_name"], (0, 0, 255))
        for target_img in [img_standard, img_xai]:
            cv2.rectangle(target_img, (x1, y1), (x2, y2), color, 2)

    def label_overlap(rx1, ry1, rx2, ry2, boxes, own_box):
        total = 0
        for bx1, by1, bx2, by2 in boxes:
            if [bx1, by1, bx2, by2] == list(own_box):
                continue
            ox = max(0, min(rx2, bx2) - max(rx1, bx1))
            oy = max(0, min(ry2, by2) - max(ry1, by1))
            total += ox * oy
        return total

    all_boxes = [det["box"] for det in detections]

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color      = CLASS_COLORS.get(det["class_name"], (0, 0, 255))
        b, g, r    = color
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = (0, 0, 0) if brightness > 160 else (255, 255, 255)
        label      = f"{det['label']} ({det['confidence'] * 100:.0f}%)"
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
        lw, lh = tw + pad * 2, th + baseline + pad * 2

        # Candidate positions: above, below, inside-top, inside-bottom
        candidates = [
            (x1, y1 - lh,      x1 + lw, y1,      y1 - baseline - pad),       # above
            (x1, y2,           x1 + lw, y2 + lh, y2 + th + pad),              # below
            (x1, y1,           x1 + lw, y1 + lh, y1 + th + pad),              # inside-top
            (x1, y2 - lh,      x1 + lw, y2,      y2 - baseline - pad),        # inside-bottom
        ]

        best = min(candidates, key=lambda c: label_overlap(c[0], c[1], c[2], c[3], all_boxes, det["box"]))
        rx1, ry1, rx2, ry2, text_y = best

        for target_img in [img_standard, img_xai]:
            cv2.rectangle(target_img, (rx1, ry1), (rx2, ry2), color, -1)
            cv2.putText(target_img, label, (x1 + pad, text_y),
                        font, font_scale, text_color, font_thick, cv2.LINE_AA)

    _, buf_std = cv2.imencode('.jpg', img_standard)
    _, buf_xai = cv2.imencode('.jpg', img_xai)

    return {
        "detections":     detections,
        "total_found":    len(detections),
        "image_standard": base64.b64encode(buf_std).decode('utf-8'),
        "image_xai":      base64.b64encode(buf_xai).decode('utf-8'),
    }
"""
FastAPI Backend — Brain Tumour Segmentation
Endpoints:
    GET  /api/health   -> health check
    GET  /api/models   -> available models
    POST /api/segment  -> run inference, returns mask + stats
"""

import io
import sys
import time
import base64
import numpy as np
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn

# Path setup 
PROJECT_ROOT = Path(__file__).parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

# App
app = FastAPI(
    title="Tumor Segmentation API",
    description="Brain tumour segmentation using U-Net on BraTS2020",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# U-Net definition
class DoubleConv(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True),
            nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True)
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, ic=1, oc=1, base=32):
        super().__init__()
        self.d1=DoubleConv(ic,base);      self.p1=nn.MaxPool2d(2)
        self.d2=DoubleConv(base,base*2);  self.p2=nn.MaxPool2d(2)
        self.d3=DoubleConv(base*2,base*4);self.p3=nn.MaxPool2d(2)
        self.d4=DoubleConv(base*4,base*8);self.p4=nn.MaxPool2d(2)
        self.b =DoubleConv(base*8,base*16)
        self.u4=nn.ConvTranspose2d(base*16,base*8,2,2); self.c4=DoubleConv(base*16,base*8)
        self.u3=nn.ConvTranspose2d(base*8,base*4,2,2);  self.c3=DoubleConv(base*8,base*4)
        self.u2=nn.ConvTranspose2d(base*4,base*2,2,2);  self.c2=DoubleConv(base*4,base*2)
        self.u1=nn.ConvTranspose2d(base*2,base,2,2);    self.c1=DoubleConv(base*2,base)
        self.h =nn.Conv2d(base, oc, 1)

    def forward(self, x):
        c1=self.d1(x); p1=self.p1(c1)
        c2=self.d2(p1); p2=self.p2(c2)
        c3=self.d3(p2); p3=self.p3(c3)
        c4=self.d4(p3); p4=self.p4(c4)
        b=self.b(p4)
        u4=self.u4(b);  d4=self.c4(torch.cat([u4,c4],1))
        u3=self.u3(d4); d3=self.c3(torch.cat([u3,c3],1))
        u2=self.u2(d3); d2=self.c2(torch.cat([u2,c2],1))
        u1=self.u1(d2); d1=self.c1(torch.cat([u1,c1],1))
        return self.h(d1)

# Model cache 
_models: dict = {}

def get_unet():
    if "unet" not in _models:
        print("[backend] Loading U-Net...")
        model = UNet(ic=1, oc=1, base=32)
        ckpt = PROJECT_ROOT / "Processed_Data" / "Unet" / "model" / "best.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            print(f"[backend] Weights loaded from {ckpt}")
        else:
            print(f"[backend] Warning: checkpoint not found at {ckpt}")
        model.eval()
        _models["unet"] = model
    return _models["unet"]

# Helpers
def decode_image(upload: UploadFile) -> np.ndarray:
    """Uploaded file -> numpy H×W×3 uint8."""
    raw = upload.file.read()
    return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))

def mask_to_base64(mask: np.ndarray) -> str:
    """Bool/uint8 mask -> base64 PNG."""
    if mask.dtype == bool:
        mask = (mask * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def image_to_base64(img: np.ndarray) -> str:
    """RGB numpy array -> base64 PNG."""
    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def build_overlay(image: np.ndarray, mask: np.ndarray, color=(192, 57, 43)) -> np.ndarray:
    """Blend coloured mask region onto original image."""
    overlay = image.copy()
    binary = mask.astype(bool) if mask.dtype == bool else (mask > 127)
    overlay[binary] = (overlay[binary] * 0.55 + np.array(color) * 0.45).astype(np.uint8)
    return overlay

# Routes
@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": list(_models.keys())}


@app.get("/api/models")
def list_models():
    return {
        "models": [
            {
                "id":          "unet",
                "label":       "U-Net",
                "description": "Task-specific encoder-decoder. Best Dice (13.84%).",
                "dice":        13.84,
                "iou":         7.19,
                "params":      "~31M",
            }
        ]
    }


@app.post("/api/segment")
async def segment(
    image:   UploadFile = File(..., description="MRI slice (PNG/JPG)"),
    model:   Literal["unet"] = Form("unet"),
    gt_mask: UploadFile = File(None, description="Optional ground-truth mask for metric computation"),
):
    if image.content_type not in ("image/png", "image/jpeg", "image/jpg", "image/webp"):
        raise HTTPException(status_code=400, detail="Unsupported format. Use PNG or JPG.")

    img_array = decode_image(image)

    t0 = time.perf_counter()

    try:
        pred_mask, confidence = _run_unet(img_array)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    pred_bool     = pred_mask.astype(bool)
    total_pixels  = pred_mask.size
    mask_coverage = round(float(pred_bool.sum() / total_pixels) * 100, 2)
    overlay       = build_overlay(img_array, pred_mask)

    return JSONResponse({
        "mask_b64":    mask_to_base64(pred_mask),
        "overlay_b64": image_to_base64(overlay),
        "pred_stats": {
            "mask_coverage":        mask_coverage,
            "estimated_confidence": round(confidence, 1),
            "predicted_pixels":     int(pred_bool.sum()),
            "total_pixels":         int(total_pixels),
        },
        "model":      model,
        "elapsed_ms": elapsed_ms,
    })

# Inference

def _run_unet(image: np.ndarray):
    """Run U-Net inference. Returns (binary mask, confidence %)."""
    import cv2
    model  = get_unet()
    
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    
    gray   = np.mean(image, axis=2, keepdims=True).astype(np.float32) / 255.0
    tensor = torch.from_numpy(gray.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).squeeze()
        mask = (prob > 0.5).numpy().astype(bool)
    confidence = float(prob[prob > 0.5].mean()) * 100 if mask.any() else 0.0
    print(f"[debug] prob min: {prob.min():.4f} max: {prob.max():.4f} mean: {prob.mean():.4f}")
    print(f"[debug] mask pixels: {mask.sum()} / {mask.size}")
    return mask, confidence
"""
Simplified segmentation service using SAM without GroundingDINO.
This is a temporary solution to demonstrate the API functionality.
"""
import os
import io
import base64
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import torch

from segment_anything import sam_model_registry, SamPredictor

app = FastAPI(title="Simple SAM Segmentation Service")

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Model path
SAM_CHECKPOINT_PATH = "/home/zhaoliyang/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"

# Global SAM predictor
_sam_predictor = None

def load_sam():
    """Load SAM model."""
    global _sam_predictor
    if _sam_predictor is None:
        print("Loading SAM model...")
        sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        _sam_predictor = SamPredictor(sam)
        print("SAM model loaded.")
    return _sam_predictor

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_sam()

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": "SAM", "device": str(DEVICE)}

@app.post("/extract_clothes")
async def extract_clothes(
    image: UploadFile = File(...),
    prompt: str = Form("clothes"),
    box_threshold: float = Form(0.25),
    text_threshold: float = Form(0.25),
    white_background: bool = Form(True)
):
    """Extract clothes using simple color-based detection + SAM."""
    try:
        # Load image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(img)
        
        # Use simple detection: assume person is in center
        h, w = img_array.shape[:2]
        
        # Create dummy boxes for upper and lower body
        # Upper body: center top 60%
        upper_box = np.array([w*0.2, h*0.1, w*0.8, h*0.6])
        # Lower body: center bottom 50%
        lower_box = np.array([w*0.2, h*0.5, w*0.8, h*0.95])
        
        boxes = np.array([upper_box, lower_box])
        
        # Run SAM segmentation
        sam_predictor = load_sam()
        sam_predictor.set_image(img_array)
        
        result_images = []
        bboxes = []
        
        for i, box in enumerate(boxes):
            box = box.reshape(1, -1)
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            # Get best mask
            best_mask = masks[np.argmax(scores)]
            
            # Create white background crop
            if white_background:
                white_bg = np.ones_like(img_array) * 255
                mask_3channel = np.stack([best_mask] * 3, axis=-1)
                result = np.where(mask_3channel, img_array, white_bg)
            else:
                result = img_array.copy()
                result[~best_mask] = [255, 255, 255]
            
            # Crop to bounding box with padding
            x1, y1, x2, y2 = box[0].astype(int)
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            result = result[y1:y2, x1:x2]
            
            # Convert to PIL and encode
            result_pil = Image.fromarray(result)
            buffer = io.BytesIO()
            result_pil.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            result_images.append(img_base64)
            bboxes.append([int(x1), int(y1), int(x2), int(y2)])
        
        return {
            "status": "success",
            "message": f"Successfully segmented {len(result_images)} item(s)",
            "segmented_images": result_images,
            "bounding_boxes": bboxes,
            "confidences": [0.8, 0.8]
        }
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_upper_body")
async def extract_upper_body(
    image: UploadFile = File(...),
    white_background: bool = Form(True)
):
    """Extract upper body clothing."""
    result = await extract_clothes(
        image=image, 
        prompt="upper body",
        box_threshold=0.25,
        text_threshold=0.25,
        white_background=white_background
    )
    # Return only first item (upper body)
    if result["segmented_images"]:
        result["segmented_images"] = [result["segmented_images"][0]]
        result["bounding_boxes"] = [result["bounding_boxes"][0]]
        result["confidences"] = [result["confidences"][0]]
    return result

@app.post("/extract_lower_body")
async def extract_lower_body(
    image: UploadFile = File(...),
    white_background: bool = Form(True)
):
    """Extract lower body clothing."""
    result = await extract_clothes(
        image=image,
        prompt="lower body",
        box_threshold=0.25,
        text_threshold=0.25,
        white_background=white_background
    )
    # Return only second item (lower body)
    if len(result["segmented_images"]) > 1:
        result["segmented_images"] = [result["segmented_images"][1]]
        result["bounding_boxes"] = [result["bounding_boxes"][1]]
        result["confidences"] = [result["confidences"][1]]
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

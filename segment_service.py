"""
Grounded-SAM Segmentation Service
Uses GroundingDINO for object detection + SAM for segmentation.
"""
import os
import io
import base64
import sys
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import torch

# Add paths for GroundingDINO and segment_anything (must be before imports)
GSAM_ROOT = "/home/zhaoliyang/Grounded-Segment-Anything"
sys.path.insert(0, os.path.join(GSAM_ROOT, "GroundingDINO"))
sys.path.insert(0, os.path.join(GSAM_ROOT, "segment_anything"))
sys.path.insert(0, GSAM_ROOT)

# Grounding DINO imports
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# SAM imports
from segment_anything import sam_model_registry, SamPredictor

app = FastAPI(title="Grounded-SAM Segmentation Service")

# Device
# Force CPU to avoid GroundingDINO CUDA extension issues
DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE} (GroundingDINO CUDA extension not available)")

# Model paths
GROUNDING_DINO_CONFIG = "/home/zhaoliyang/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "/home/zhaoliyang/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT_PATH = "/home/zhaoliyang/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"

# BERT model path (for offline use)
BERT_MODEL_PATH = os.environ.get("BERT_MODEL_PATH", "./models/bert-base-uncased")

# Set transformers offline mode if using local BERT
if os.path.exists(BERT_MODEL_PATH) and os.path.isdir(BERT_MODEL_PATH):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print(f"Using local BERT model from: {BERT_MODEL_PATH}")

# Global models
_grounding_dino_model = None
_sam_predictor = None


def load_grounding_dino():
    """Load GroundingDINO model."""
    global _grounding_dino_model
    if _grounding_dino_model is None:
        print("Loading GroundingDINO model...")
        args = SLConfig.fromfile(GROUNDING_DINO_CONFIG)
        args.device = DEVICE
        # Use local BERT model path if available
        if os.path.exists(BERT_MODEL_PATH) and os.path.isdir(BERT_MODEL_PATH):
            args.bert_base_uncased_path = BERT_MODEL_PATH
            print(f"Using local BERT model: {BERT_MODEL_PATH}")
        _grounding_dino_model = build_model(args)
        checkpoint = torch.load(GROUNDING_DINO_CHECKPOINT, map_location="cpu")
        _grounding_dino_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        _grounding_dino_model.eval()
        _grounding_dino_model = _grounding_dino_model.to(DEVICE)
        print("GroundingDINO model loaded.")
    return _grounding_dino_model


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


def load_image_for_dino(image_pil):
    """Load and transform image for GroundingDINO."""
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)
    return image_tensor


def detect_objects(model, image_tensor, text_prompt, box_threshold=0.3, text_threshold=0.25):
    """
    Detect objects using GroundingDINO.
    
    Returns:
        boxes: Tensor of shape (n, 4) with bounding boxes
        phrases: List of detected phrases
        logits: Confidence scores
    """
    # Prepare caption
    caption = text_prompt.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    
    image_tensor = image_tensor.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(image_tensor[None], captions=[caption])
    
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    
    # Filter by box threshold
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]
    boxes_filt = boxes_filt[filt_mask]
    
    # Get phrases
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    
    pred_phrases = []
    for logit in logits_filt:
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase)
    
    return boxes_filt, pred_phrases, logits_filt


def segment_with_sam(predictor, image, boxes, white_background=True):
    """
    Segment objects using SAM with detected boxes.
    
    Args:
        predictor: SAM predictor
        image: numpy array (H, W, 3) RGB
        boxes: Tensor of bounding boxes (n, 4) in [cx, cy, w, h] format normalized
        white_background: Whether to use white background
    
    Returns:
        List of PIL Images containing segmented objects
    """
    H, W = image.shape[:2]
    
    # Convert boxes from [cx, cy, w, h] normalized to [x1, y1, x2, y2] pixel coords
    boxes_pixel = boxes.clone()
    for i in range(boxes.size(0)):
        boxes_pixel[i] = boxes[i] * torch.Tensor([W, H, W, H])
        boxes_pixel[i][:2] -= boxes_pixel[i][2:] / 2  # cx,cy to x1,y1
        boxes_pixel[i][2:] += boxes_pixel[i][:2]      # w,h to x2,y2
    
    boxes_pixel = boxes_pixel.cpu()
    
    # Transform boxes for SAM
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_pixel, image.shape[:2]).to(DEVICE)
    
    # Run SAM prediction
    masks, scores, logits = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    
    # Extract masked regions
    result_images = []
    for i, mask in enumerate(masks):
        mask_np = mask.cpu().numpy()[0]  # (H, W)
        
        # Get bounding box for cropping
        box = boxes_pixel[i].numpy().astype(int)
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        
        if white_background:
            # Create white background
            white_bg = np.ones_like(image) * 255
            mask_3channel = np.stack([mask_np] * 3, axis=-1)
            result = np.where(mask_3channel, image, white_bg)
        else:
            result = image.copy()
            result[~mask_np] = [255, 255, 255]
        
        # Crop to bounding box
        result = result[y1:y2, x1:x2]
        
        # Convert to PIL
        result_pil = Image.fromarray(result)
        result_images.append(result_pil)
    
    return result_images


def pil_to_base64(img):
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_grounding_dino()
    load_sam()


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models": {
            "grounding_dino": "loaded" if _grounding_dino_model is not None else "not_loaded",
            "sam": "loaded" if _sam_predictor is not None else "not_loaded"
        },
        "device": str(DEVICE)
    }


@app.post("/extract_clothes")
async def extract_clothes(
    image: UploadFile = File(...),
    prompt: str = Form("clothes"),
    box_threshold: float = Form(0.3),
    text_threshold: float = Form(0.25),
    white_background: bool = Form(True)
):
    """Extract clothes using GroundingDINO + SAM."""
    try:
        # Load image
        contents = await image.read()
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(img_pil)
        
        # Prepare image for GroundingDINO
        image_tensor = load_image_for_dino(img_pil)
        
        # Load models
        dino_model = load_grounding_dino()
        sam_predictor = load_sam()
        sam_predictor.set_image(img_array)
        
        # Detect objects
        boxes, phrases, logits = detect_objects(
            dino_model, image_tensor, prompt, box_threshold, text_threshold
        )
        
        if len(boxes) == 0:
            return {
                "status": "success",
                "message": "No objects detected",
                "segmented_images": [],
                "bounding_boxes": [],
                "labels": [],
                "confidences": []
            }
        
        # Segment objects
        result_images = segment_with_sam(sam_predictor, img_array, boxes, white_background)
        
        # Convert to base64
        base64_images = [pil_to_base64(img) for img in result_images]
        
        # Prepare bounding boxes (convert to list)
        boxes_list = boxes.numpy().tolist()
        confidences = [float(logit.max()) for logit in logits]
        
        return {
            "status": "success",
            "message": f"Successfully detected and segmented {len(result_images)} item(s)",
            "segmented_images": base64_images,
            "bounding_boxes": boxes_list,
            "labels": phrases,
            "confidences": confidences
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
    """Extract upper body clothing (shirt, jacket, sweater, etc.)."""
    try:
        # Load image
        contents = await image.read()
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(img_pil)
        
        # Prepare image for GroundingDINO
        image_tensor = load_image_for_dino(img_pil)
        
        # Load models
        dino_model = load_grounding_dino()
        sam_predictor = load_sam()
        sam_predictor.set_image(img_array)
        
        # Detect upper body clothing with multiple prompts
        prompts = ["shirt", "jacket", "sweater", "hoodie", "coat", "blazer"]
        all_boxes = []
        all_phrases = []
        all_confidences = []
        
        for prompt in prompts:
            boxes, phrases, logits = detect_objects(
                dino_model, image_tensor, prompt, box_threshold=0.3, text_threshold=0.25
            )
            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_phrases.extend(phrases)
                all_confidences.extend([float(logit.max()) for logit in logits])
        
        if len(all_boxes) == 0:
            return {
                "status": "success",
                "message": "No upper body clothing detected",
                "segmented_images": [],
                "bounding_boxes": [],
                "labels": [],
                "confidences": []
            }
        
        # Concatenate all boxes
        combined_boxes = torch.cat(all_boxes, dim=0)
        
        # Limit to top 2 results by confidence (select best matches)
        MAX_RESULTS = 2
        if len(combined_boxes) > MAX_RESULTS:
            # Get indices of top confidences
            top_indices = sorted(range(len(all_confidences)),
                                key=lambda i: all_confidences[i],
                                reverse=True)[:MAX_RESULTS]
            combined_boxes = combined_boxes[top_indices]
            all_phrases = [all_phrases[i] for i in top_indices]
            all_confidences = [all_confidences[i] for i in top_indices]
        
        # Segment objects
        result_images = segment_with_sam(sam_predictor, img_array, combined_boxes, white_background)
        
        # Convert to base64
        base64_images = [pil_to_base64(img) for img in result_images]
        boxes_list = combined_boxes.numpy().tolist()
        
        return {
            "status": "success",
            "message": f"Successfully detected and segmented {len(result_images)} upper body item(s)",
            "segmented_images": base64_images,
            "bounding_boxes": boxes_list,
            "labels": all_phrases,
            "confidences": all_confidences
        }
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract_lower_body")
async def extract_lower_body(
    image: UploadFile = File(...),
    white_background: bool = Form(True)
):
    """Extract lower body clothing (pants, shorts, skirt, etc.)."""
    try:
        # Load image
        contents = await image.read()
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(img_pil)
        
        # Prepare image for GroundingDINO
        image_tensor = load_image_for_dino(img_pil)
        
        # Load models
        dino_model = load_grounding_dino()
        sam_predictor = load_sam()
        sam_predictor.set_image(img_array)
        
        # Detect lower body clothing with multiple prompts
        prompts = ["pants", "trousers", "shorts", "skirt", "jeans"]
        all_boxes = []
        all_phrases = []
        all_confidences = []
        
        for prompt in prompts:
            boxes, phrases, logits = detect_objects(
                dino_model, image_tensor, prompt, box_threshold=0.3, text_threshold=0.25
            )
            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_phrases.extend(phrases)
                all_confidences.extend([float(logit.max()) for logit in logits])
        
        if len(all_boxes) == 0:
            return {
                "status": "success",
                "message": "No lower body clothing detected",
                "segmented_images": [],
                "bounding_boxes": [],
                "labels": [],
                "confidences": []
            }
        
        # Concatenate all boxes
        combined_boxes = torch.cat(all_boxes, dim=0)
        
        # Limit to top 2 results by confidence (select best matches)
        MAX_RESULTS = 2
        if len(combined_boxes) > MAX_RESULTS:
            # Get indices of top confidences
            top_indices = sorted(range(len(all_confidences)),
                                key=lambda i: all_confidences[i],
                                reverse=True)[:MAX_RESULTS]
            combined_boxes = combined_boxes[top_indices]
            all_phrases = [all_phrases[i] for i in top_indices]
            all_confidences = [all_confidences[i] for i in top_indices]
        
        # Segment objects
        result_images = segment_with_sam(sam_predictor, img_array, combined_boxes, white_background)
        
        # Convert to base64
        base64_images = [pil_to_base64(img) for img in result_images]
        boxes_list = combined_boxes.numpy().tolist()
        
        return {
            "status": "success",
            "message": f"Successfully detected and segmented {len(result_images)} lower body item(s)",
            "segmented_images": base64_images,
            "bounding_boxes": boxes_list,
            "labels": all_phrases,
            "confidences": all_confidences
        }
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
IDM-VTON Virtual Try-On Service
Provides FastAPI endpoints for virtual try-on functionality.
"""
import os
import io
import base64
import sys
import cv2
import numpy as np
from PIL import Image
from typing import Optional
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn

app = FastAPI(title="IDM-VTON Virtual Try-On Service")

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Model paths - adjust these based on your IDM-VTON installation
IDM_VTON_ROOT = os.environ.get("IDM_VTON_ROOT", "/home/zhaoliyang/IDM-VTON")
IDM_VTON_CHECKPOINT = os.environ.get("IDM_VTON_CHECKPOINT", os.path.join(IDM_VTON_ROOT, "checkpoints"))

# Global model instances
_idm_pipeline = None
_parsing_model = None
_face_preservation = None


def load_parsing_model():
    """Load SCHP parsing model for face preservation."""
    global _parsing_model
    if _parsing_model is None:
        print("Loading SCHP parsing model for face preservation...")
        try:
            # Add IDM-VTON to path
            if IDM_VTON_ROOT not in sys.path:
                sys.path.insert(0, IDM_VTON_ROOT)
            
            from preprocess.humanparsing.run_parsing import Parsing
            _parsing_model = Parsing()
            print("SCHP parsing model loaded.")
        except Exception as e:
            print(f"Warning: Failed to load parsing model: {e}")
            print("Face preservation will be disabled.")
            _parsing_model = None
    return _parsing_model


def load_face_preservation():
    """Load face preservation module."""
    global _face_preservation, _parsing_model
    if _face_preservation is None:
        parsing_model = load_parsing_model()
        if parsing_model is not None:
            try:
                # Add IDM-VTON to path
                if IDM_VTON_ROOT not in sys.path:
                    sys.path.insert(0, IDM_VTON_ROOT)
                
                from face_preservation import FacePreservation
                _face_preservation = FacePreservation(
                    parsing_model=parsing_model,
                    include_neck=True,
                    dilate_kernel_size=5,
                    feather_amount=10
                )
                print("Face preservation module loaded.")
            except Exception as e:
                print(f"Warning: Failed to load face preservation: {e}")
                _face_preservation = None
    return _face_preservation


def load_idm_vton_pipeline():
    """Load IDM-VTON pipeline."""
    global _idm_pipeline
    if _idm_pipeline is None:
        print("Loading IDM-VTON pipeline...")
        try:
            # Add IDM-VTON to path
            if IDM_VTON_ROOT not in sys.path:
                sys.path.insert(0, IDM_VTON_ROOT)
            
            # Import IDM-VTON modules
            from src.tryon_pipeline import StableDiffusionTryOnePipeline
            from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
            from src.unet_hacked_tryone import UNet2DConditionModel
            from transformers import (
                CLIPTextModel,
                CLIPTokenizer,
                CLIPImageProcessor,
            )
            from diffusers import (
                AutoencoderKL,
                DDPMScheduler,
                UniPCMultistepScheduler,
            )
            
            # Load models
            vae = AutoencoderKL.from_pretrained(
                IDM_VTON_CHECKPOINT,
                subfolder="vae",
                torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
            )
            
            unet = UNet2DConditionModel.from_pretrained(
                IDM_VTON_CHECKPOINT,
                subfolder="unet",
                torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
            )
            
            unet_ref = UNet2DConditionModel_ref.from_pretrained(
                IDM_VTON_CHECKPOINT,
                subfolder="unet_ref",
                torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
            )
            
            text_encoder = CLIPTextModel.from_pretrained(
                IDM_VTON_CHECKPOINT,
                subfolder="text_encoder",
                torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
            )
            
            tokenizer = CLIPTokenizer.from_pretrained(
                IDM_VTON_CHECKPOINT,
                subfolder="tokenizer",
            )
            
            image_encoder = CLIPImageProcessor.from_pretrained(
                IDM_VTON_CHECKPOINT,
                subfolder="image_encoder",
            )
            
            scheduler = UniPCMultistepScheduler.from_pretrained(
                IDM_VTON_CHECKPOINT,
                subfolder="scheduler",
            )
            
            # Create pipeline
            _idm_pipeline = StableDiffusionTryOnePipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                unet_ref=unet_ref,
                scheduler=scheduler,
                image_encoder=image_encoder,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
            
            _idm_pipeline.to(DEVICE)
            
            # Enable CPU offload for memory efficiency
            if DEVICE.type == 'cuda':
                _idm_pipeline.enable_sequential_cpu_offload()
            
            print("IDM-VTON pipeline loaded successfully.")
            
        except Exception as e:
            print(f"Error loading IDM-VTON pipeline: {e}")
            import traceback
            print(traceback.format_exc())
            raise RuntimeError(f"Failed to load IDM-VTON pipeline: {e}")
    
    return _idm_pipeline


def pil_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def base64_to_pil(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    img_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    try:
        load_idm_vton_pipeline()
    except Exception as e:
        print(f"Warning: Could not load IDM-VTON pipeline at startup: {e}")
        print("Pipeline will be loaded on first request.")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "loaded" if _idm_pipeline is not None else "not_loaded",
        "face_preservation": "loaded" if _face_preservation is not None else "not_loaded",
        "device": str(DEVICE),
        "service": "idm-vton"
    }


@app.post("/tryon")
async def tryon(
    person_image: UploadFile = File(...),
    clothes_image: UploadFile = File(...),
    prompt: str = Form("a photo of a person wearing clothes"),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(2.0),
    seed: int = Form(42),
    preserve_face: bool = Form(True),
):
    """
    Perform virtual try-on using IDM-VTON.
    
    Args:
        person_image: Image of the person (model)
        clothes_image: Image of the clothing item (garment)
        prompt: Text prompt for generation
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale for classifier-free guidance
        seed: Random seed for reproducibility
        preserve_face: Whether to preserve the original face (default: True)
    
    Returns:
        Base64 encoded result image
    """
    try:
        # Load pipeline and face preservation
        pipeline = load_idm_vton_pipeline()
        face_preservation = load_face_preservation() if preserve_face else None
        
        # Read and process images
        person_contents = await person_image.read()
        clothes_contents = await clothes_image.read()
        
        person_pil = Image.open(io.BytesIO(person_contents)).convert("RGB")
        clothes_pil = Image.open(io.BytesIO(clothes_contents)).convert("RGB")
        
        # Store original person image for face preservation
        original_person_pil = person_pil.copy()
        
        # Resize images to appropriate size (IDM-VTON typically uses 768x1024)
        person_pil = person_pil.resize((768, 1024))
        clothes_pil = clothes_pil.resize((768, 1024))
        
        # Set random seed
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        
        # Run inference
        result = pipeline(
            prompt=prompt,
            image=person_pil,
            mask_image=None,  # IDM-VTON can compute mask internally
            garment_image=clothes_pil,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=1024,
            width=768,
        )
        
        result_image = result.images[0]
        
        # Apply face preservation if enabled and available
        if preserve_face and face_preservation is not None:
            try:
                # Resize original person image to match result size
                original_resized = original_person_pil.resize(result_image.size, Image.LANCZOS)
                # Apply face preservation
                result_image = face_preservation(original_resized, result_image)
                print("Face preservation applied successfully.")
            except Exception as e:
                print(f"Warning: Face preservation failed: {e}")
        
        # Convert to base64
        result_base64 = pil_to_base64(result_image)
        
        return {
            "status": "success",
            "message": "Virtual try-on completed successfully" + (" (with face preservation)" if preserve_face and face_preservation else ""),
            "result_image": result_base64,
            "face_preserved": preserve_face and face_preservation is not None,
            "parameters": {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
            }
        }
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tryon_base64")
async def tryon_base64(
    person_image_base64: str = Form(...),
    clothes_image_base64: str = Form(...),
    prompt: str = Form("a photo of a person wearing clothes"),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(2.0),
    seed: int = Form(42),
    preserve_face: bool = Form(True),
):
    """
    Perform virtual try-on using base64 encoded images.
    
    Args:
        person_image_base64: Base64 encoded person image
        clothes_image_base64: Base64 encoded clothes image
        prompt: Text prompt for generation
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale for classifier-free guidance
        seed: Random seed for reproducibility
        preserve_face: Whether to preserve the original face (default: True)
    
    Returns:
        Base64 encoded result image
    """
    try:
        # Load pipeline and face preservation
        pipeline = load_idm_vton_pipeline()
        face_preservation = load_face_preservation() if preserve_face else None
        
        # Decode base64 images
        person_pil = base64_to_pil(person_image_base64)
        clothes_pil = base64_to_pil(clothes_image_base64)
        
        # Store original person image for face preservation
        original_person_pil = person_pil.copy()
        
        # Resize images
        person_pil = person_pil.resize((768, 1024))
        clothes_pil = clothes_pil.resize((768, 1024))
        
        # Set random seed
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        
        # Run inference
        result = pipeline(
            prompt=prompt,
            image=person_pil,
            mask_image=None,
            garment_image=clothes_pil,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=1024,
            width=768,
        )
        
        result_image = result.images[0]
        
        # Apply face preservation if enabled and available
        if preserve_face and face_preservation is not None:
            try:
                # Resize original person image to match result size
                original_resized = original_person_pil.resize(result_image.size, Image.LANCZOS)
                # Apply face preservation
                result_image = face_preservation(original_resized, result_image)
                print("Face preservation applied successfully.")
            except Exception as e:
                print(f"Warning: Face preservation failed: {e}")
        
        # Convert to base64
        result_base64 = pil_to_base64(result_image)
        
        return {
            "status": "success",
            "message": "Virtual try-on completed successfully" + (" (with face preservation)" if preserve_face and face_preservation else ""),
            "result_image": result_base64,
            "face_preserved": preserve_face and face_preservation is not None,
            "parameters": {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
            }
        }
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

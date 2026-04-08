"""
IDM-VTON FastAPI Service
Provides virtual try-on API using IDM-VTON model.
"""
import os
import sys
import io
import base64
import torch
from PIL import Image
import numpy as np
from typing import Optional, Tuple
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# Add IDM-VTON to path
IDM_VTON_ROOT = os.getenv("IDM_VTON_ROOT", "/home/zhaoliyang/IDM-VTON")
IDM_VTON_CHECKPOINT = os.path.join(IDM_VTON_ROOT, "checkpoints")

if IDM_VTON_ROOT not in sys.path:
    sys.path.insert(0, IDM_VTON_ROOT)

# Global pipeline instance
_idm_pipeline = None
_unet_encoder = None
_parsing_model = None
_openpose_model = None
_face_preservation = None

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_parsing_model():
    """Load SCHP parsing model for face preservation."""
    global _parsing_model, _openpose_model, _face_preservation
    if _parsing_model is None:
        print("Loading SCHP parsing model for face preservation...")
        try:
            from preprocess.humanparsing.run_parsing import Parsing
            from preprocess.openpose.run_openpose import OpenPose
            from face_preservation import FacePreservation
            
            _parsing_model = Parsing(4)  # gpu_id=4 in original
            _openpose_model = OpenPose(4)
            
            _face_preservation = FacePreservation(
                parsing_model=_parsing_model,
                include_neck=True,
                dilate_kernel_size=5,
                feather_amount=10
            )
            print("Face preservation models loaded.")
        except Exception as e:
            print(f"Warning: Could not load face preservation models: {e}")
            _parsing_model = None
            _openpose_model = None
            _face_preservation = None
    return _parsing_model, _openpose_model, _face_preservation


def load_idm_vton_pipeline():
    """Load IDM-VTON pipeline exactly as in gradio_demo/app.py"""
    global _idm_pipeline, _unet_encoder
    
    if _idm_pipeline is None:
        print("Loading IDM-VTON pipeline...")
        try:
            # Import modules as in original app.py
            from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
            from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
            from src.unet_hacked_tryon import UNet2DConditionModel
            from transformers import (
                CLIPImageProcessor,
                CLIPVisionModelWithProjection,
                CLIPTextModel,
                CLIPTextModelWithProjection,
                AutoTokenizer,
            )
            from diffusers import DDPMScheduler, AutoencoderKL
            from torchvision import transforms
            
            # Load models exactly as in gradio_demo/app.py
            unet = UNet2DConditionModel.from_pretrained(
                IDM_VTON_CHECKPOINT,
                subfolder="unet",
                torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
            ).to(DEVICE)
            unet.requires_grad_(False)
            
            tokenizer_one = AutoTokenizer.from_pretrained(
                IDM_VTON_CHECKPOINT,
                subfolder="tokenizer",
                revision=None,
                use_fast=False,
            )
            
            tokenizer_two = AutoTokenizer.from_pretrained(
                IDM_VTON_CHECKPOINT,
                subfolder="tokenizer_2",
                revision=None,
                use_fast=False,
            )
            
            noise_scheduler = DDPMScheduler.from_pretrained(
                IDM_VTON_CHECKPOINT, 
                subfolder="scheduler"
            )
            
            text_encoder_one = CLIPTextModel.from_pretrained(
                IDM_VTON_CHECKPOINT,
                subfolder="text_encoder",
                torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
            ).to(DEVICE)
            
            text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
                IDM_VTON_CHECKPOINT,
                subfolder="text_encoder_2",
                torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
            ).to(DEVICE)
            
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                IDM_VTON_CHECKPOINT,
                subfolder="image_encoder",
                torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
            ).to(DEVICE)
            
            vae = AutoencoderKL.from_pretrained(
                IDM_VTON_CHECKPOINT,
                subfolder="vae",
                torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
            ).to(DEVICE)
            
            UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
                IDM_VTON_CHECKPOINT,
                subfolder="unet_encoder",
                torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
            ).to(DEVICE)
            
            # Set requires_grad(False) for all models
            UNet_Encoder.requires_grad_(False)
            image_encoder.requires_grad_(False)
            vae.requires_grad_(False)
            unet.requires_grad_(False)
            text_encoder_one.requires_grad_(False)
            text_encoder_two.requires_grad_(False)
            
            # Create pipeline using from_pretrained as in original
            _idm_pipeline = TryonPipeline.from_pretrained(
                IDM_VTON_CHECKPOINT,
                unet=unet,
                vae=vae,
                feature_extractor=CLIPImageProcessor(),
                text_encoder=text_encoder_one,
                text_encoder_2=text_encoder_two,
                tokenizer=tokenizer_one,
                tokenizer_2=tokenizer_two,
                scheduler=noise_scheduler,
                image_encoder=image_encoder,
                torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
            ).to(DEVICE)
            
            # Set unet_encoder separately (as in original)
            _idm_pipeline.unet_encoder = UNet_Encoder
            _unet_encoder = UNet_Encoder
            
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
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    try:
        load_parsing_model()
        load_idm_vton_pipeline()
    except Exception as e:
        print(f"Warning: Could not load models at startup: {e}")
        print("Models will be loaded on first request.")
    yield
    # Cleanup
    global _idm_pipeline
    _idm_pipeline = None


app = FastAPI(title="IDM-VTON Service", lifespan=lifespan)


@app.get("/health")
def health_check():
    """Check if the service is healthy."""
    global _idm_pipeline
    return {
        "status": "healthy",
        "pipeline_loaded": _idm_pipeline is not None,
        "device": str(DEVICE)
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
    Perform virtual try-on.
    
    Args:
        person_image: Image of the person
        clothes_image: Image of the clothes to try on
        prompt: Text prompt for generation
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale
        seed: Random seed
        preserve_face: Whether to preserve the original face
    """
    global _idm_pipeline, _face_preservation
    
    # Ensure pipeline is loaded
    if _idm_pipeline is None:
        try:
            load_idm_vton_pipeline()
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to load pipeline: {str(e)}"}
            )
    
    try:
        # Load images
        person_img = Image.open(person_image.file).convert("RGB")
        clothes_img = Image.open(clothes_image.file).convert("RGB")
        
        # Store original size for face preservation
        original_person = person_img.copy()
        orig_width, orig_height = person_img.size
        
        # Resize images to model input size (768x1024 as in original)
        person_img = person_img.resize((768, 1024))
        clothes_img = clothes_img.resize((768, 1024))
        
        # Prepare inputs for the pipeline
        # This is a simplified version - full implementation would need mask generation
        # and other preprocessing as in the original app.py
        
        # For now, return a placeholder
        # TODO: Implement full try-on logic matching gradio_demo/app.py
        
        result_img = person_img  # Placeholder
        
        # Convert result to base64
        result_base64 = pil_to_base64(result_img)
        
        return {
            "status": "success",
            "result_image": result_base64,
            "message": "Virtual try-on completed (placeholder - full implementation pending)"
        }
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"Try-on failed: {str(e)}"}
        )


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
    Perform virtual try-on with base64 encoded images.
    """
    global _idm_pipeline, _face_preservation
    
    if _idm_pipeline is None:
        try:
            load_idm_vton_pipeline()
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to load pipeline: {str(e)}"}
            )
    
    try:
        # Decode base64 images
        person_img = base64_to_pil(person_image_base64).convert("RGB")
        clothes_img = base64_to_pil(clothes_image_base64).convert("RGB")
        
        # Store original for face preservation
        original_person = person_img.copy()
        orig_width, orig_height = person_img.size
        
        # Resize to model input size
        person_img = person_img.resize((768, 1024))
        clothes_img = clothes_img.resize((768, 1024))
        
        # Placeholder - full implementation would call pipeline here
        result_img = person_img
        
        # Convert to base64
        result_base64 = pil_to_base64(result_img)
        
        return {
            "status": "success",
            "result_image": result_base64,
            "message": "Virtual try-on completed (placeholder - full implementation pending)"
        }
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"Try-on failed: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

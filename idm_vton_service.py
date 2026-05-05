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
from typing import Optional, Tuple, List
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


def get_free_gpu():
    """Find the GPU with the most free memory."""
    if not torch.cuda.is_available():
        return 'cpu'
    
    try:
        import subprocess
        # Get GPU memory info using nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,nounits,noheader'],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            print("nvidia-smi failed, using default cuda:0")
            return 'cuda:0'
        
        lines = result.stdout.strip().split('\n')
        max_free = 0
        best_gpu = 0
        
        for i, line in enumerate(lines):
            parts = line.split(',')
            if len(parts) >= 2:
                free_mem = int(parts[0].strip())
                total_mem = int(parts[1].strip())
                print(f"GPU {i}: {free_mem}MB free / {total_mem}MB total")
                
                if free_mem > max_free:
                    max_free = free_mem
                    best_gpu = i
        
        print(f"Selected GPU {best_gpu} with {max_free}MB free memory")
        return f'cuda:{best_gpu}'
        
    except Exception as e:
        print(f"Failed to detect free GPU: {e}, using default cuda:0")
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'


# Device configuration - auto-select the most free GPU
DEVICE = torch.device(get_free_gpu())

# Import required modules from IDM-VTON
try:
    from torchvision import transforms
    from utils_mask import get_mask_location
    import apply_net
    from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
    from torchvision.transforms.functional import to_pil_image
    from face_preservation import FacePreservation, visualize_mask
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some IDM-VTON imports failed: {e}")
    IMPORTS_AVAILABLE = False


def pil_to_binary_mask(pil_image, threshold=0):
    """Convert PIL image to binary mask."""
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True:
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


def load_parsing_model():
    """Load SCHP parsing model for face preservation."""
    global _parsing_model, _openpose_model, _face_preservation
    if _parsing_model is None:
        print("Loading SCHP parsing model for face preservation...")
        try:
            from preprocess.humanparsing.run_parsing import Parsing
            from preprocess.openpose.run_openpose import OpenPose
            from face_preservation import FacePreservation
            
            # Use the same GPU as the pipeline
            gpu_id = DEVICE.index if DEVICE.type == 'cuda' else 0
            _parsing_model = Parsing(gpu_id)
            _openpose_model = OpenPose(gpu_id)
            
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


def perform_tryon(
    person_img: Image.Image,
    clothes_img: Image.Image,
    prompt: str = "a photo of a person wearing clothes",
    num_inference_steps: int = 30,
    guidance_scale: float = 2.0,
    seed: int = 42,
    preserve_face: bool = True,
    clothing_category: str = "upper_body",
) -> Image.Image:
    """
    Perform virtual try-on using IDM-VTON pipeline.

    Args:
        person_img: Image of the person
        clothes_img: Image of the clothes to try on
        prompt: Text prompt for generation
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale
        seed: Random seed
        preserve_face: Whether to preserve the original face
        clothing_category: Body region to mask ("upper_body", "lower_body", "dresses")

    Returns:
        Result image as PIL Image
    """
    global _idm_pipeline, _parsing_model, _openpose_model, _face_preservation

    print(f"[VTON] perform_tryon called | clothing_category={clothing_category!r} | prompt={prompt!r}")

    if _idm_pipeline is None:
        raise RuntimeError("Pipeline not loaded")

    # Clear CUDA cache to free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Convert images to RGB and resize to model input size
    garm_img = clothes_img.convert("RGB").resize((768, 1024))
    human_img_orig = person_img.convert("RGB")
    
    # Store original image size for later restoration
    orig_width, orig_height = human_img_orig.size
    
    # Letterboxing: resize with aspect ratio preserved, pad with white
    ratio = min(768 / orig_width, 1024 / orig_height)
    new_width = int(orig_width * ratio)
    new_height = int(orig_height * ratio)
    
    # Resize with high-quality antialiasing
    resized_orig = human_img_orig.resize((new_width, new_height), Image.LANCZOS)
    
    # Create 768x1024 white canvas
    human_img = Image.new("RGB", (768, 1024), (255, 255, 255))
    
    # Center the resized image
    paste_x = (768 - new_width) // 2
    paste_y = (1024 - new_height) // 2
    human_img.paste(resized_orig, (paste_x, paste_y))
    
    # Generate mask using parsing and openpose models
    if _openpose_model is not None and _parsing_model is not None:
        keypoints = _openpose_model(human_img.resize((384, 512)))
        model_parse, _ = _parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location('hd', clothing_category, model_parse, keypoints)
        mask = mask.resize((768, 1024))
    else:
        # Fallback: create a simple mask based on clothing category
        mask = Image.new('L', (768, 1024), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        if clothing_category == "lower_body":
            # Lower body region (waist to feet)
            draw.rectangle([100, 512, 668, 950], fill=255)
        elif clothing_category == "dresses":
            # Full body region
            draw.rectangle([100, 100, 668, 950], fill=255)
        else:
            # Upper body region (default)
            draw.rectangle([100, 100, 668, 600], fill=255)
    
    # Create tensor transform
    tensor_transfrom = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)
    
    # Generate DensePose pose image
    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
    
    # Create args for DensePose
    # Note: MODEL.DEVICE should be 'cuda' or 'cpu', not 'cuda:0'
    device_str = 'cuda' if DEVICE.type == 'cuda' else 'cpu'
    args = apply_net.create_argument_parser().parse_args(
        ('show',
         os.path.join(IDM_VTON_ROOT, 'configs/densepose_rcnn_R_50_FPN_s1x.yaml'),
         os.path.join(IDM_VTON_ROOT, 'ckpt/densepose/model_final_162be9.pkl'),
         'dp_segm', '-v', '--opts', 'MODEL.DEVICE', device_str)
    )
    
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]  # BGR to RGB
    pose_img = Image.fromarray(pose_img).resize((768, 1024))
    
    # Prepare garment description from prompt
    garment_des = prompt.replace("a photo of a person wearing ", "").replace("a photo of ", "")
    
    # Run inference
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # Encode prompt for person
                prompt_text = "model is wearing " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = _idm_pipeline.encode_prompt(
                        prompt_text,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                
                # Encode prompt for garment
                prompt_cloth = "a photo of " + garment_des
                if not isinstance(prompt_cloth, List):
                    prompt_cloth = [prompt_cloth] * 1
                if not isinstance(negative_prompt, List):
                    negative_prompt_cloth = [negative_prompt] * 1
                
                with torch.inference_mode():
                    (
                        prompt_embeds_c,
                        _,
                        _,
                        _,
                    ) = _idm_pipeline.encode_prompt(
                        prompt_cloth,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        negative_prompt=negative_prompt_cloth,
                    )
                
                # Prepare tensors
                pose_img_tensor = tensor_transfrom(pose_img).unsqueeze(0).to(DEVICE, torch.float16)
                garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(DEVICE, torch.float16)
                
                generator = torch.Generator(DEVICE).manual_seed(seed) if seed is not None else None
                
                # Run pipeline
                images = _idm_pipeline(
                    prompt_embeds=prompt_embeds.to(DEVICE, torch.float16),
                    negative_prompt_embeds=negative_prompt_embeds.to(DEVICE, torch.float16),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(DEVICE, torch.float16),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(DEVICE, torch.float16),
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    strength=1.0,
                    pose_img=pose_img_tensor.to(DEVICE, torch.float16),
                    text_embeds_cloth=prompt_embeds_c.to(DEVICE, torch.float16),
                    cloth=garm_tensor.to(DEVICE, torch.float16),
                    mask_image=mask,
                    image=human_img,
                    height=1024,
                    width=768,
                    ip_adapter_image=garm_img.resize((768, 1024)),
                    guidance_scale=guidance_scale,
                )[0]
    
    result_img = images[0]
    
    # Apply face preservation if enabled and available
    if preserve_face and _face_preservation is not None:
        try:
            # For non-cropped, human_img is the source and result_img is the generated output
            # Both should be (768, 1024) - resize source to match output if needed
            if human_img.size != result_img.size:
                orig_for_preserve = human_img.resize(result_img.size, Image.LANCZOS)
            else:
                orig_for_preserve = human_img
            
            result_img = _face_preservation(orig_for_preserve, result_img)
        except Exception as e:
            print(f"Face preservation failed: {e}")
            # Continue without face preservation if it fails
    
    # Crop away letterbox padding and restore to original size
    result_img = result_img.crop((paste_x, paste_y, paste_x + new_width, paste_y + new_height))
    result_img = result_img.resize((orig_width, orig_height), Image.LANCZOS)
    
    return result_img


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
    clothing_category: str = Form("upper_body"),
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
        
        # Perform try-on
        result_img = perform_tryon(
            person_img=person_img,
            clothes_img=clothes_img,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            preserve_face=preserve_face,
            clothing_category=clothing_category,
        )

        # Convert result to base64
        result_base64 = pil_to_base64(result_img)

        return {
            "status": "success",
            "result_image": result_base64,
            "message": "Virtual try-on completed successfully"
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
    clothing_category: str = Form("upper_body"),
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
        
        # Perform try-on
        result_img = perform_tryon(
            person_img=person_img,
            clothes_img=clothes_img,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            preserve_face=preserve_face,
            clothing_category=clothing_category,
        )

        # Convert to base64
        result_base64 = pil_to_base64(result_img)
        
        return {
            "status": "success",
            "result_image": result_base64,
            "message": "Virtual try-on completed successfully"
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

"""
IDM-VTON Client Tool for ai-mirror-demo
Provides a simple interface to call the IDM-VTON virtual try-on service.
"""

import os
import base64
import requests
from typing import Optional, Dict, Any
from PIL import Image
import io

# Service configuration
IDM_VTON_SERVICE_URL = os.getenv("IDM_VTON_SERVICE_URL", "http://localhost:8001")


class IDMVTONClient:
    """Client for IDM-VTON virtual try-on service."""
    
    def __init__(self, service_url: str = None, check_health: bool = True):
        """
        Initialize IDM-VTON client.

        Args:
            service_url: URL of the IDM-VTON service (default: http://localhost:8001)
            check_health: Whether to check service health on init (default: True)
        """
        self.service_url = service_url or IDM_VTON_SERVICE_URL
        self.available = False
        try:
            self.health_check()
            self.available = True
        except Exception as e:
            if check_health:
                print(f"[IDMVTONClient] Warning: {e}")
            # If check_health is False, we allow initialization to continue
            # The service might be available when actually used
    
    def health_check(self) -> dict:
        """Check if the service is healthy."""
        try:
            response = requests.get(f"{self.service_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to IDM-VTON service at {self.service_url}. "
                "Please ensure the service is running with: "
                "python idm_vton_service.py"
            )
        except Exception as e:
            raise RuntimeError(f"Health check failed: {e}")
    
    def try_on(
        self,
        person_image_path: str,
        clothes_image_path: str,
        prompt: str = "a photo of a person wearing clothes",
        num_inference_steps: int = 30,
        guidance_scale: float = 2.0,
        seed: int = 42,
        preserve_face: bool = True,
    ) -> Image.Image:
        """
        Perform virtual try-on.
        
        Args:
            person_image_path: Path to the person image (model)
            clothes_image_path: Path to the clothes image (garment)
            prompt: Text prompt for generation
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            seed: Random seed for reproducibility
            preserve_face: Whether to preserve the original face
        
        Returns:
            PIL Image containing the try-on result
        """
        if not self.available:
            raise RuntimeError("IDM-VTON service not available. Please start the service first.")

        url = f"{self.service_url}/tryon"
        
        with open(person_image_path, 'rb') as f_person, \
             open(clothes_image_path, 'rb') as f_clothes:
            files = {
                'person_image': f_person,
                'clothes_image': f_clothes,
            }
            data = {
                'prompt': prompt,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'seed': seed,
                'preserve_face': preserve_face,
            }
            
            response = requests.post(url, files=files, data=data, timeout=300)
            response.raise_for_status()
        
        result = response.json()
        
        if result['status'] != 'success':
            raise RuntimeError(f"Try-on failed: {result.get('message', 'Unknown error')}")
        
        # Convert base64 image to PIL Image
        img_base64 = result['result_image']
        img_bytes = base64.b64decode(img_base64)
        img = Image.open(io.BytesIO(img_bytes))
        
        return img
    
    def try_on_images(
        self,
        person_image: Image.Image,
        clothes_image: Image.Image,
        prompt: str = "a photo of a person wearing clothes",
        num_inference_steps: int = 30,
        guidance_scale: float = 2.0,
        seed: int = 42,
        preserve_face: bool = True,
    ) -> Image.Image:
        """
        Perform virtual try-on using PIL Images directly.
        
        Args:
            person_image: PIL Image of the person
            clothes_image: PIL Image of the clothes
            prompt: Text prompt for generation
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            seed: Random seed for reproducibility
            preserve_face: Whether to preserve the original face
        
        Returns:
            PIL Image containing the try-on result
        """
        if not self.available:
            raise RuntimeError("IDM-VTON service not available. Please start the service first.")

        url = f"{self.service_url}/tryon_base64"
        
        # Convert PIL images to base64
        def pil_to_base64(img: Image.Image) -> str:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()
        
        person_base64 = pil_to_base64(person_image)
        clothes_base64 = pil_to_base64(clothes_image)
        
        data = {
            'person_image_base64': person_base64,
            'clothes_image_base64': clothes_base64,
            'prompt': prompt,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'preserve_face': preserve_face,
        }
        
        response = requests.post(url, data=data, timeout=300)
        response.raise_for_status()
        
        result = response.json()
        
        if result['status'] != 'success':
            raise RuntimeError(f"Try-on failed: {result.get('message', 'Unknown error')}")
        
        # Convert base64 image to PIL Image
        img_base64 = result['result_image']
        img_bytes = base64.b64decode(img_base64)
        img = Image.open(io.BytesIO(img_bytes))
        
        return img


# Convenience function for simple use cases
def virtual_try_on(
    person_image_path: str,
    clothes_image_path: str,
    service_url: str = None,
    prompt: str = "a photo of a person wearing clothes",
    num_inference_steps: int = 30,
    guidance_scale: float = 2.0,
    seed: int = 42,
) -> Image.Image:
    """
    Convenience function for virtual try-on.
    
    Args:
        person_image_path: Path to the person image
        clothes_image_path: Path to the clothes image
        service_url: URL of the IDM-VTON service (optional)
        prompt: Text prompt for generation
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale
        seed: Random seed
    
    Returns:
        PIL Image containing the try-on result
    """
    client = IDMVTONClient(service_url=service_url)
    return client.try_on(
        person_image_path=person_image_path,
        clothes_image_path=clothes_image_path,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

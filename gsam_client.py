"""
Grounded-SAM Client Tool for ai-mirror-demo
Provides a simple interface to call the Grounded-SAM segmentation service.
"""

import os
import base64
import requests
from typing import List, Optional, Tuple
from PIL import Image
import io

# Service configuration
GSAM_SERVICE_URL = os.getenv("GSAM_SERVICE_URL", "http://localhost:8000")


class GSAMClient:
    """Client for Grounded-SAM segmentation service."""
    
    def __init__(self, service_url: str = None):
        """
        Initialize GSAM client.
        
        Args:
            service_url: URL of the Grounded-SAM service (default: http://localhost:8000)
        """
        self.service_url = service_url or GSAM_SERVICE_URL
        self.health_check()
    
    def health_check(self) -> dict:
        """Check if the service is healthy."""
        try:
            response = requests.get(f"{self.service_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Grounded-SAM service at {self.service_url}. "
                "Please ensure the service is running with: "
                "conda run -n gsam_env python gsam_service.py"
            )
        except Exception as e:
            raise RuntimeError(f"Health check failed: {e}")
    
    def segment_clothing(
        self,
        image_path: str,
        prompt: str = "clothes",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        white_background: bool = True
    ) -> List[Image.Image]:
        """
        Segment clothing from an image using text prompt.
        
        Args:
            image_path: Path to the input image
            prompt: Text description of clothing to extract (e.g., "t-shirt", "pants")
            box_threshold: Detection box threshold
            text_threshold: Text prompt threshold
            white_background: Whether to place on white background
        
        Returns:
            List of PIL Images containing segmented clothing items
        """
        url = f"{self.service_url}/extract_clothes"
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'prompt': prompt,
                'box_threshold': box_threshold,
                'text_threshold': text_threshold,
                'white_background': white_background
            }
            
            response = requests.post(url, files=files, data=data, timeout=60)
            response.raise_for_status()
        
        result = response.json()
        
        if result['status'] != 'success':
            raise RuntimeError(f"Segmentation failed: {result.get('message', 'Unknown error')}")
        
        # Convert base64 images to PIL Images
        images = []
        for img_base64 in result.get('segmented_images', []):
            img_bytes = base64.b64decode(img_base64)
            img = Image.open(io.BytesIO(img_bytes))
            images.append(img)
        
        return images
    
    def extract_upper_body(self, image_path: str, white_background: bool = True) -> List[Image.Image]:
        """
        Extract upper body clothing (shirt, t-shirt, jacket, etc.).
        
        Args:
            image_path: Path to the input image
            white_background: Whether to place on white background
        
        Returns:
            List of PIL Images containing upper body clothing
        """
        url = f"{self.service_url}/extract_upper_body"
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'white_background': white_background}
            
            response = requests.post(url, files=files, data=data, timeout=60)
            response.raise_for_status()
        
        result = response.json()
        
        if result['status'] != 'success':
            raise RuntimeError(f"Upper body extraction failed: {result.get('message', 'Unknown error')}")
        
        # Convert base64 images to PIL Images
        images = []
        for img_base64 in result.get('segmented_images', []):
            img_bytes = base64.b64decode(img_base64)
            img = Image.open(io.BytesIO(img_bytes))
            images.append(img)
        
        return images
    
    def extract_lower_body(self, image_path: str, white_background: bool = True) -> List[Image.Image]:
        """
        Extract lower body clothing (pants, shorts, skirt, etc.).
        
        Args:
            image_path: Path to the input image
            white_background: Whether to place on white background
        
        Returns:
            List of PIL Images containing lower body clothing
        """
        url = f"{self.service_url}/extract_lower_body"
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'white_background': white_background}
            
            response = requests.post(url, files=files, data=data, timeout=60)
            response.raise_for_status()
        
        result = response.json()
        
        if result['status'] != 'success':
            raise RuntimeError(f"Lower body extraction failed: {result.get('message', 'Unknown error')}")
        
        # Convert base64 images to PIL Images
        images = []
        for img_base64 in result.get('segmented_images', []):
            img_bytes = base64.b64decode(img_base64)
            img = Image.open(io.BytesIO(img_bytes))
            images.append(img)
        
        return images
    
    def extract_both(
        self,
        image_path: str,
        output_dir: str = "./extracted_clothes",
        white_background: bool = True
    ) -> Tuple[List[Image.Image], List[Image.Image]]:
        """
        Extract both upper and lower body clothing from an image.
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save extracted images (optional)
            white_background: Whether to place on white background
        
        Returns:
            Tuple of (upper_body_images, lower_body_images)
        """
        upper_images = self.extract_upper_body(image_path, white_background)
        lower_images = self.extract_lower_body(image_path, white_background)
        
        # Save images if output directory is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            for i, img in enumerate(upper_images):
                output_path = os.path.join(output_dir, f"{base_name}_upper_{i}.png")
                img.save(output_path)
                print(f"Saved upper body: {output_path}")
            
            for i, img in enumerate(lower_images):
                output_path = os.path.join(output_dir, f"{base_name}_lower_{i}.png")
                img.save(output_path)
                print(f"Saved lower body: {output_path}")
        
        return upper_images, lower_images


# Convenience function for LangChain/LangGraph integration
def tool_extract_clothes(
    image_path: str,
    clothing_type: str = "upper",
    service_url: str = None
) -> str:
    """
    Tool function for extracting clothes from an image.
    Can be used as a LangChain/LangGraph tool.
    
    Args:
        image_path: Path to the image file
        clothing_type: Type of clothing to extract ("upper", "lower", or "both")
        service_url: URL of the GSAM service
    
    Returns:
        Status message with paths to extracted images
    """
    client = GSAMClient(service_url)
    output_dir = "./extracted_clothes"
    
    try:
        if clothing_type.lower() == "upper":
            images = client.extract_upper_body(image_path)
            prefix = "upper_body"
        elif clothing_type.lower() == "lower":
            images = client.extract_lower_body(image_path)
            prefix = "lower_body"
        elif clothing_type.lower() == "both":
            upper, lower = client.extract_both(image_path, output_dir)
            return (
                f"Successfully extracted {len(upper)} upper body and {len(lower)} lower body items. "
                f"Images saved to: {output_dir}"
            )
        else:
            images = client.segment_clothing(image_path, prompt=clothing_type)
            prefix = clothing_type.replace(" ", "_")
        
        # Save images
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        saved_paths = []
        
        for i, img in enumerate(images):
            output_path = os.path.join(output_dir, f"{base_name}_{prefix}_{i}.png")
            img.save(output_path)
            saved_paths.append(output_path)
        
        return f"Successfully extracted {len(images)} item(s). Saved to: {', '.join(saved_paths)}"
        
    except Exception as e:
        return f"Extraction failed: {str(e)}"


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python gsam_client.py <image_path> [upper|lower|both]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    clothing_type = sys.argv[2] if len(sys.argv) > 2 else "both"
    
    print(f"Extracting {clothing_type} body clothing from: {image_path}")
    result = tool_extract_clothes(image_path, clothing_type)
    print(result)
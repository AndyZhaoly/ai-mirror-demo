"""
Grounded-SAM Client Tool for ai-mirror-demo
Provides a simple interface to call the Grounded-SAM segmentation service.
"""

import os
import base64
import requests
from typing import List, Optional, Tuple, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np

# Service configuration
GSAM_SERVICE_URL = os.getenv("GSAM_SERVICE_URL", "http://localhost:8000")


class GSAMClient:
    """Client for Grounded-SAM segmentation service."""
    
    def __init__(self, service_url: str = None, check_health: bool = True):
        """
        Initialize GSAM client.

        Args:
            service_url: URL of the Grounded-SAM service (default: http://localhost:8000)
            check_health: Whether to check service health on init (default: True)
        """
        self.service_url = service_url or GSAM_SERVICE_URL
        self.available = False
        try:
            self.health_check()
            self.available = True
        except Exception as e:
            if check_health:
                print(f"[GSAMClient] Warning: {e}")
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
        if not self.available:
            raise RuntimeError("GSAM service not available. Please start the service first.")

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
    
    def extract_upper_body(self, image_path: str, white_background: bool = True) -> Tuple[List[Image.Image], Dict[str, Any]]:
        """
        Extract upper body clothing (shirt, t-shirt, jacket, etc.).

        Args:
            image_path: Path to the input image
            white_background: Whether to place on white background

        Returns:
            Tuple of (segmented_images, detection_info)
            detection_info contains: bounding_boxes, labels, confidences
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

        # Return detection info along with images
        detection_info = {
            'bounding_boxes': result.get('bounding_boxes', []),
            'labels': result.get('labels', []),
            'confidences': result.get('confidences', [])
        }

        return images, detection_info
    
    def extract_lower_body(self, image_path: str, white_background: bool = True) -> Tuple[List[Image.Image], Dict[str, Any]]:
        """
        Extract lower body clothing (pants, shorts, skirt, etc.).

        Args:
            image_path: Path to the input image
            white_background: Whether to place on white background

        Returns:
            Tuple of (segmented_images, detection_info)
            detection_info contains: bounding_boxes, labels, confidences
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

        # Return detection info along with images
        detection_info = {
            'bounding_boxes': result.get('bounding_boxes', []),
            'labels': result.get('labels', []),
            'confidences': result.get('confidences', [])
        }

        return images, detection_info
    
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
def draw_detection_boxes(image: Image.Image, boxes: List[List[float]], labels: List[str], confidences: List[float]) -> Image.Image:
    """
    Draw bounding boxes on image.

    Args:
        image: PIL Image
        boxes: List of bounding boxes in [cx, cy, w, h] normalized format
        labels: List of labels
        confidences: List of confidence scores

    Returns:
        PIL Image with drawn boxes
    """
    img_array = np.array(image)
    H, W = img_array.shape[:2]

    # Create a copy for drawing
    result_img = image.copy()
    draw = ImageDraw.Draw(result_img)

    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()

    # Colors for different boxes
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]

    for i, (box, label, conf) in enumerate(zip(boxes, labels, confidences)):
        # Convert from [cx, cy, w, h] normalized to [x1, y1, x2, y2] pixel
        cx, cy, w, h = box
        x1 = int((cx - w/2) * W)
        y1 = int((cy - h/2) * H)
        x2 = int((cx + w/2) * W)
        y2 = int((cy + h/2) * H)

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        color = colors[i % len(colors)]

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label
        label_text = f"{label}: {conf:.2f}"

        # Get text bbox
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Draw text background
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=color)

        # Draw text
        draw.text((x1 + 2, y1 - text_h - 2), label_text, fill=(255, 255, 255), font=font)

    return result_img


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
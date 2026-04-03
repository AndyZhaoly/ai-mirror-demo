"""
Extract and visualize clothes and pants from the Louis Vuitton image.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from gsam_client import GSAMClient
from PIL import Image
import matplotlib.pyplot as plt
import requests
import time

IMAGE_PATH = "./louis-vuitton-typer-the-creator-collection-spring-2024-photos-05.webp"

def check_service():
    """Check if Grounded-SAM service is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    # Check service
    if not check_service():
        print("❌ Grounded-SAM service is not running!")
        print("Please start it in another terminal:")
        print("  cd /home/zhaoliyang/Grounded-Segment-Anything")
        print("  conda activate gsam_env")
        print("  python gsam_service.py")
        return
    
    print("✅ Grounded-SAM service is running")
    print(f"🖼️  Processing image: {IMAGE_PATH}")
    
    # Initialize client
    client = GSAMClient("http://localhost:8000")
    
    # Extract upper body (clothes)
    print("\n👕 Extracting upper body clothing...")
    upper_images = client.extract_upper_body(IMAGE_PATH, white_background=True)
    print(f"   Found {len(upper_images)} upper body item(s)")
    
    # Extract lower body (pants)
    print("\n👖 Extracting lower body clothing...")
    lower_images = client.extract_lower_body(IMAGE_PATH, white_background=True)
    print(f"   Found {len(lower_images)} lower body item(s)")
    
    # Create visualization
    print("\n📊 Creating visualization...")
    
    # Load original image
    original = Image.open(IMAGE_PATH)
    
    # Calculate grid size
    total_images = 1 + len(upper_images) + len(lower_images)  # original + upper + lower
    cols = min(4, total_images)
    rows = (total_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    # Plot original
    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    idx = 1
    
    # Plot upper body items
    for i, img in enumerate(upper_images):
        if idx < len(axes):
            axes[idx].imshow(img)
            axes[idx].set_title(f"Upper Body {i+1}", fontsize=12, fontweight='bold', color='blue')
            axes[idx].axis('off')
            idx += 1
    
    # Plot lower body items
    for i, img in enumerate(lower_images):
        if idx < len(axes):
            axes[idx].imshow(img)
            axes[idx].set_title(f"Lower Body {i+1}", fontsize=12, fontweight='bold', color='green')
            axes[idx].axis('off')
            idx += 1
    
    # Hide empty subplots
    for i in range(idx, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("Clothing Segmentation Results", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save and show
    output_path = "./extraction_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n💾 Saved visualization to: {output_path}")
    
    # Also save individual images
    output_dir = "./extracted_clothes"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img in enumerate(upper_images):
        path = os.path.join(output_dir, f"upper_body_{i+1}.png")
        img.save(path)
        print(f"   Saved: {path}")
    
    for i, img in enumerate(lower_images):
        path = os.path.join(output_dir, f"lower_body_{i+1}.png")
        img.save(path)
        print(f"   Saved: {path}")
    
    plt.show()
    print("\n✅ Done!")

if __name__ == "__main__":
    main()

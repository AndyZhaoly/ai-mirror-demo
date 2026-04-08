"""
Clothing Recommendations Module for AI Fashion Butler.
Manages sample clothes from IDM-VTON example folder for agent-driven recommendations and virtual try-on.
Uses VLM (Gemini) to automatically analyze images for accurate descriptions.
"""
import os
import json
import glob
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from PIL import Image
import random

# Local sample clothes directory (for git portability)
LOCAL_SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "recommendations", "clothes")

# Fallback to IDM-VTON if local directory is empty
IDM_VTON_ROOT = os.getenv("IDM_VTON_ROOT", "/home/zhaoliyang/IDM-VTON")
IDM_VTON_SAMPLE_DIR = os.path.join(IDM_VTON_ROOT, "gradio_demo", "example", "cloth")

# Fallback to extracted_clothes
EXTRACTED_DIR = os.path.join(os.path.dirname(__file__), "extracted_clothes")

# Cache file for VLM analysis results
ANALYSIS_CACHE_FILE = os.path.join(os.path.dirname(__file__), "recommendations", "clothes_analysis_cache.json")


@dataclass
class ClothingItem:
    """Represents a clothing item for recommendation."""
    id: str
    name: str
    category: str  # "upper", "lower", "dress"
    style: str  # "casual", "formal", "streetwear", "vintage", etc.
    color: str
    material: str
    description: str
    image_path: str
    price_range: str
    brand: str = "Unknown"
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClothingItem":
        return cls(**data)


# Sample clothing database - populated from sample images with VLM analysis
SAMPLE_CLOTHES_DB: List[ClothingItem] = []


def analyze_clothing_image(image_path: str, api_key: str = None) -> Dict[str, str]:
    """
    Analyze a clothing image using Gemini VLM to extract accurate attributes.
    
    Args:
        image_path: Path to the clothing image
        api_key: Gemini API key (defaults to env var)
        
    Returns:
        Dict with name, category, style, color, material, description
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print(f"[Recommendations] Google GenAI not available, using fallback analysis")
        return _fallback_analysis(image_path)
    
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(f"[Recommendations] No Gemini API key, using fallback analysis")
        return _fallback_analysis(image_path)
    
    try:
        client = genai.Client(api_key=api_key)
        
        # Load and resize image for analysis
        with Image.open(image_path) as img:
            # Resize if too large
            max_size = 512
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
        
        prompt = """Analyze this clothing item image and provide the following information in JSON format:
{
    "name": "Brief name (e.g., 'Blue Velvet Wrap Top', 'White Minnie Mouse T-Shirt')",
    "category": "upper or lower or dress",
    "style": "casual or formal or streetwear or vintage or sportswear",
    "color": "Primary color in Chinese (e.g., '蓝色', '白色', '黑色')",
    "material": "Likely material (e.g., '棉质', '丝绒', '聚酯纤维')",
    "description": "Brief appealing description in Chinese for customers"
}

Be accurate based on what you see in the image."""
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=500
            )
        )
        
        # Parse JSON from response
        text = response.text.strip()
        # Extract JSON if wrapped in code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(text)
        
        # Validate required fields
        required = ["name", "category", "style", "color", "material", "description"]
        for field in required:
            if field not in result:
                result[field] = _get_default_value(field)
        
        return result
        
    except Exception as e:
        print(f"[Recommendations] VLM analysis failed: {e}")
        return _fallback_analysis(image_path)


def _fallback_analysis(image_path: str) -> Dict[str, str]:
    """Fallback analysis based on filename when VLM is unavailable."""
    filename = os.path.basename(image_path)
    name = os.path.splitext(filename)[0]
    
    return {
        "name": f"时尚单品 {name}",
        "category": "upper",
        "style": "casual",
        "color": "多色",
        "material": "棉质",
        "description": "精选时尚单品，舒适百搭，适合多种场合穿搭"
    }


def _get_default_value(field: str) -> str:
    """Get default value for a field."""
    defaults = {
        "name": "时尚单品",
        "category": "upper",
        "style": "casual",
        "color": "多色",
        "material": "棉质",
        "description": "精选时尚单品，舒适百搭"
    }
    return defaults.get(field, "未知")


def load_analysis_cache() -> Dict[str, Dict]:
    """Load cached VLM analysis results."""
    if os.path.exists(ANALYSIS_CACHE_FILE):
        try:
            with open(ANALYSIS_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Recommendations] Failed to load cache: {e}")
    return {}


def save_analysis_cache(cache: Dict[str, Dict]):
    """Save VLM analysis results to cache."""
    try:
        os.makedirs(os.path.dirname(ANALYSIS_CACHE_FILE), exist_ok=True)
        with open(ANALYSIS_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Recommendations] Failed to save cache: {e}")


def init_recommendations_db(use_vlm: bool = True, force_reanalyze: bool = False):
    """
    Initialize the recommendations database from local or IDM-VTON sample clothes.
    
    Args:
        use_vlm: Whether to use Gemini VLM for image analysis
        force_reanalyze: Force re-analysis even if cache exists
    """
    global SAMPLE_CLOTHES_DB
    
    SAMPLE_CLOTHES_DB = []
    
    # Priority: local sample dir -> IDM-VTON sample dir -> extracted_clothes
    if os.path.exists(LOCAL_SAMPLE_DIR) and os.listdir(LOCAL_SAMPLE_DIR):
        source_dir = LOCAL_SAMPLE_DIR
        print(f"[Recommendations] Using local sample clothes from {source_dir}")
    elif os.path.exists(IDM_VTON_SAMPLE_DIR):
        source_dir = IDM_VTON_SAMPLE_DIR
        print(f"[Recommendations] Using IDM-VTON sample clothes from {source_dir}")
    else:
        source_dir = EXTRACTED_DIR
        print(f"[Recommendations] Using extracted clothes from {source_dir}")
    
    # Load cache
    cache = {} if force_reanalyze else load_analysis_cache()
    
    if os.path.exists(source_dir):
        # Get all image files from the cloth folder
        image_files = sorted([f for f in os.listdir(source_dir)
                             if f.endswith((".jpg", ".jpeg", ".png"))])
        
        for i, filename in enumerate(image_files):
            image_path = os.path.join(source_dir, filename)
            item_id = f"sample_{i:03d}"
            
            # Check cache first
            cache_key = f"{filename}_{os.path.getsize(image_path)}"
            if cache_key in cache and not force_reanalyze:
                analysis = cache[cache_key]
                print(f"[Recommendations] Using cached analysis for {filename}")
            elif use_vlm:
                # Analyze with VLM
                print(f"[Recommendations] Analyzing {filename} with VLM...")
                analysis = analyze_clothing_image(image_path)
                cache[cache_key] = analysis
            else:
                # Use fallback
                analysis = _fallback_analysis(image_path)
            
            # Determine price range based on style and material
            price_range = _estimate_price(analysis.get("style", "casual"), analysis.get("material", "棉质"))
            
            item = ClothingItem(
                id=item_id,
                name=analysis.get("name", "时尚单品"),
                category=analysis.get("category", "upper"),
                style=analysis.get("style", "casual"),
                color=analysis.get("color", "多色"),
                material=analysis.get("material", "棉质"),
                description=analysis.get("description", "精选时尚单品"),
                image_path=image_path,
                price_range=price_range,
                brand="AI Mirror Collection",
                tags=[analysis.get("style", "casual"), analysis.get("color", "多色"), "推荐", "热门"]
            )
            SAMPLE_CLOTHES_DB.append(item)
        
        # Save updated cache
        if use_vlm:
            save_analysis_cache(cache)
    
    print(f"[Recommendations] Initialized {len(SAMPLE_CLOTHES_DB)} items from {source_dir}")
    return len(SAMPLE_CLOTHES_DB)


def _estimate_price(style: str, material: str) -> str:
    """Estimate price range based on style and material."""
    base_prices = {
        "formal": (299, 599),
        "vintage": (249, 499),
        "streetwear": (199, 399),
        "casual": (159, 359),
        "sportswear": (199, 399)
    }
    
    material_multiplier = {
        "丝绸": 1.5,
        "丝绒": 1.4,
        "羊毛": 1.3,
        "羊毛混纺": 1.2,
        "棉质": 1.0,
        "聚酯纤维": 0.8,
        "帆布": 0.9
    }
    
    min_price, max_price = base_prices.get(style, (199, 399))
    multiplier = material_multiplier.get(material, 1.0)
    
    min_price = int(min_price * multiplier)
    max_price = int(max_price * multiplier)
    
    return f"¥{min_price}-{max_price}"


def get_recommendations(
    category: Optional[str] = None,
    style: Optional[str] = None,
    color: Optional[str] = None,
    limit: int = 4
) -> List[ClothingItem]:
    """
    Get clothing recommendations based on filters.
    
    Args:
        category: Filter by "upper", "lower", or "dress"
        style: Filter by style ("casual", "formal", "streetwear", "vintage")
        color: Filter by color
        limit: Maximum number of items to return
    
    Returns:
        List of matching ClothingItem objects
    """
    if not SAMPLE_CLOTHES_DB:
        init_recommendations_db()
    
    results = SAMPLE_CLOTHES_DB.copy()
    
    if category:
        results = [item for item in results if item.category == category]
    
    if style:
        results = [item for item in results if item.style == style]
    
    if color:
        results = [item for item in results if color in item.color]
    
    # Shuffle for variety
    random.shuffle(results)
    
    return results[:limit]


def get_item_by_id(item_id: str) -> Optional[ClothingItem]:
    """Get a specific clothing item by ID."""
    if not SAMPLE_CLOTHES_DB:
        init_recommendations_db()
    
    # Support both ID formats: "sample_001" and "1" (index)
    for item in SAMPLE_CLOTHES_DB:
        if item.id == item_id:
            return item
    
    # Try to parse as index (1-based)
    try:
        idx = int(item_id) - 1
        if 0 <= idx < len(SAMPLE_CLOTHES_DB):
            return SAMPLE_CLOTHES_DB[idx]
    except ValueError:
        pass
    
    return None


def format_recommendation_for_agent(items: List[ClothingItem]) -> str:
    """Format recommendations as text for the agent to display."""
    if not items:
        return "小镜暂时没有找到合适的推荐，请主人稍后再试～"
    
    lines = ["✨ 小镜为您挑选了以下搭配推荐：\n"]
    
    for i, item in enumerate(items, 1):
        lines.append(f"{i}. **{item.name}** (ID: `{item.id}`)")
        lines.append(f"   风格：{item.style} | 颜色：{item.color} | 材质：{item.material}")
        lines.append(f"   参考价：{item.price_range}")
        lines.append(f"   {item.description}")
        lines.append("")
    
    lines.append("💡 主人可以回复 **序号**（如`1`）或 **ID**（如`sample_001`）来选择喜欢的款式")
    lines.append("🎯 选中后小镜可以帮您 **虚拟试穿** 看看效果哦～")
    
    return "\n".join(lines)


def get_recommendation_image_paths(items: List[ClothingItem]) -> List[str]:
    """Get image paths for a list of recommendation items."""
    return [item.image_path for item in items if os.path.exists(item.image_path)]


def get_all_available_items() -> List[ClothingItem]:
    """Get all available clothing items."""
    if not SAMPLE_CLOTHES_DB:
        init_recommendations_db()
    return SAMPLE_CLOTHES_DB.copy()


def get_items_by_category(category: str) -> List[ClothingItem]:
    """Get all items for a specific category."""
    if not SAMPLE_CLOTHES_DB:
        init_recommendations_db()
    return [item for item in SAMPLE_CLOTHES_DB if item.category == category]


# Initialize on module load
if __name__ != "__main__":
    init_recommendations_db()

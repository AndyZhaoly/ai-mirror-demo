"""
Database Manager for FashionClaw Smart Wardrobe System.
Handles CRUD operations for clothing items with stagnancy tracking.
"""
import json
import os
import uuid
import random
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

DATABASE_FILE = "database.json"


def load_database() -> Dict[str, Any]:
    """Load database from JSON file."""
    if not os.path.exists(DATABASE_FILE):
        return {"wardrobe": []}

    try:
        with open(DATABASE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"wardrobe": []}


def save_database(data: Dict[str, Any]) -> bool:
    """Save database to JSON file."""
    try:
        with open(DATABASE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving database: {e}")
        return False


def generate_item_id() -> str:
    """Generate unique item ID."""
    return f"{uuid.uuid4().hex[:8].upper()}"


def add_item(
    name: str,
    clothing_type: str,
    image_path: str,
    original_price: Optional[int] = None,
    purchase_date: Optional[str] = None,
    extracted_from: Optional[str] = None,
    detection_image: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Add a new clothing item to the database.

    Args:
        name: Name of the clothing item
        clothing_type: "upper", "lower", or "full_body"
        image_path: Path to the extracted image
        original_price: Purchase price (auto-generated if None)
        purchase_date: ISO format date string (auto-generated if None)
        extracted_from: Original image filename

    Returns:
        The created item dictionary
    """
    data = load_database()

    # Auto-generate price if not provided
    if original_price is None:
        original_price = generate_mock_price(clothing_type)

    # Auto-generate purchase date if not provided (mock: 400 days ago to trigger stagnancy)
    if purchase_date is None:
        purchase_date = generate_mock_purchase_date()

    date_added = datetime.now().isoformat()

    # Ensure image_path is absolute
    if image_path and not os.path.isabs(image_path):
        image_path = os.path.abspath(image_path)

    item = {
        "item_id": generate_item_id(),
        "name": name,
        "clothing_type": clothing_type,
        "date_added": date_added,
        "purchase_date": purchase_date,
        "last_worn_date": purchase_date,  # Initially same as purchase date
        "status": "in_closet",
        "original_price": original_price,
        "image": image_path,
        "detection_image": detection_image or "",  # Grounding DINO detection visualization
        "extracted_from": extracted_from or "",
    }

    data["wardrobe"].append(item)
    save_database(data)

    return item


def generate_mock_price(clothing_type: str) -> int:
    """Generate a mock price based on clothing type."""
    prices = {
        "upper": (100, 500),
        "lower": (80, 400),
        "full_body": (150, 800),
    }
    min_p, max_p = prices.get(clothing_type, (50, 300))
    return random.randint(min_p, max_p)


def generate_mock_purchase_date() -> str:
    """Generate a mock purchase date ~400 days ago to trigger stagnancy."""
    days_ago = random.randint(380, 450)
    date = datetime.now() - timedelta(days=days_ago)
    return date.strftime("%Y-%m-%d")


def get_item(item_id: str) -> Optional[Dict[str, Any]]:
    """Get a single item by ID."""
    data = load_database()
    for item in data["wardrobe"]:
        if item["item_id"] == item_id:
            return item
    return None


def update_item_status(item_id: str, new_status: str) -> bool:
    """Update item status (in_closet, selling, sold)."""
    data = load_database()

    for item in data["wardrobe"]:
        if item["item_id"] == item_id:
            item["status"] = new_status
            if new_status == "sold":
                item["sold_at"] = datetime.now().isoformat()
            save_database(data)
            return True

    return False


def is_stagnant(item: Dict[str, Any], threshold_days: int = 365) -> bool:
    """
    Check if an item is stagnant (not worn for threshold_days).

    Args:
        item: Clothing item dictionary
        threshold_days: Days threshold for stagnancy (default 365)

    Returns:
        True if stagnant, False otherwise
    """
    purchase_date_str = item.get("purchase_date")
    if not purchase_date_str:
        return False

    try:
        purchase_date = datetime.strptime(purchase_date_str[:10], "%Y-%m-%d")
        days_since_purchase = (datetime.now() - purchase_date).days
        return days_since_purchase > threshold_days
    except (ValueError, TypeError):
        return False


def get_stagnant_items(threshold_days: int = 365) -> List[Dict[str, Any]]:
    """Get all stagnant items from the wardrobe."""
    data = load_database()
    stagnant = []

    for item in data["wardrobe"]:
        if item.get("status") == "in_closet" and is_stagnant(item, threshold_days):
            stagnant.append(item)

    return stagnant


def list_all_items() -> List[Dict[str, Any]]:
    """List all items in the wardrobe."""
    data = load_database()
    return data.get("wardrobe", [])


def delete_item(item_id: str) -> bool:
    """Delete an item from the database."""
    data = load_database()
    original_len = len(data["wardrobe"])
    data["wardrobe"] = [i for i in data["wardrobe"] if i["item_id"] != item_id]

    if len(data["wardrobe"]) < original_len:
        save_database(data)
        return True
    return False


def get_items_by_type(clothing_type: str) -> List[Dict[str, Any]]:
    """Get items filtered by clothing type."""
    data = load_database()
    return [i for i in data["wardrobe"] if i.get("clothing_type") == clothing_type]

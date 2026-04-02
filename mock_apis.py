"""
Mock APIs for FashionClaw Smart Wardrobe System.
These functions simulate external services without connecting to real APIs.
"""
import random
import uuid
from datetime import datetime, timedelta


def check_market_price(item_name: str) -> int:
    """
    Mock API: Check current market price for an item on second-hand platforms.

    Args:
        item_name: Name of the clothing item

    Returns:
        Estimated market price in RMB (¥)
    """
    # Simulate realistic pricing based on item type and random market fluctuation
    base_prices = {
        "jacket": random.randint(150, 250),
        "boots": random.randint(300, 450),
        "dress": random.randint(80, 150),
        "sneakers": random.randint(200, 350),
        "sweater": random.randint(120, 220),
        "hoodie": random.randint(80, 150),
        "belt": random.randint(30, 60),
    }

    item_lower = item_name.lower()
    for keyword, price in base_prices.items():
        if keyword in item_lower:
            return price

    # Default random price for unknown items
    return random.randint(50, 300)


def get_buyer_offer(market_price: int) -> dict:
    """
    Mock API: Get a buyer offer from second-hand platform (e.g., Xianyu).

    Args:
        market_price: Current estimated market price

    Returns:
        Dictionary with buyer_id and offer_price
    """
    buyer_names = [
        "FashionHunter_99",
        "VintageCollector",
        "TrendyShopper",
        "EcoBuyer_2024",
        "StyleSeeker",
        "ThriftMaster",
        "ClosetCurator",
    ]

    # Buyer offers 80-95% of market price
    offer_ratio = random.uniform(0.80, 0.95)
    offer_price = int(market_price * offer_ratio)

    return {
        "buyer_id": f"buyer_{uuid.uuid4().hex[:8]}",
        "buyer_name": random.choice(buyer_names),
        "offer_price": offer_price,
        "platform": "闲鱼 (Xianyu)",
        "offer_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def check_buyer_credit(buyer_id: str) -> dict:
    """
    Mock API: Check buyer's credit score and history.

    Args:
        buyer_id: Unique identifier for the buyer

    Returns:
        Dictionary with credit rating and details
    """
    credit_levels = ["Excellent", "Good", "Fair", "Poor"]
    weights = [0.5, 0.35, 0.12, 0.03]  # Skewed toward good ratings

    credit_rating = random.choices(credit_levels, weights=weights)[0]

    credit_details = {
        "Excellent": {
            "score": random.randint(90, 100),
            "successful_transactions": random.randint(50, 200),
            "return_rate": f"{random.uniform(0.5, 2.0):.1f}%",
            "account_age_months": random.randint(24, 60),
        },
        "Good": {
            "score": random.randint(75, 89),
            "successful_transactions": random.randint(20, 50),
            "return_rate": f"{random.uniform(2.0, 5.0):.1f}%",
            "account_age_months": random.randint(12, 24),
        },
        "Fair": {
            "score": random.randint(60, 74),
            "successful_transactions": random.randint(5, 20),
            "return_rate": f"{random.uniform(5.0, 10.0):.1f}%",
            "account_age_months": random.randint(6, 12),
        },
        "Poor": {
            "score": random.randint(30, 59),
            "successful_transactions": random.randint(0, 5),
            "return_rate": f"{random.uniform(10.0, 20.0):.1f}%",
            "account_age_months": random.randint(1, 6),
        },
    }

    return {
        "buyer_id": buyer_id,
        "credit_rating": credit_rating,
        "credit_score": credit_details[credit_rating]["score"],
        "successful_transactions": credit_details[credit_rating]["successful_transactions"],
        "return_rate": credit_details[credit_rating]["return_rate"],
        "account_age_months": credit_details[credit_rating]["account_age_months"],
    }


def execute_logistics(item_name: str, buyer_name: str) -> dict:
    """
    Mock API: Execute logistics for shipping the item.

    Args:
        item_name: Name of the clothing item
        buyer_name: Name of the buyer

    Returns:
        Dictionary with tracking information
    """
    carriers = ["顺丰 (SF)", "京东物流 (JD)", "中通 (ZTO)", "圆通 (YTO)"]
    carrier = random.choice(carriers)

    # Generate fake tracking number
    tracking_prefix = {
        "顺丰 (SF)": "SF",
        "京东物流 (JD)": "JD",
        "中通 (ZTO)": "ZTO",
        "圆通 (YTO)": "YT",
    }
    prefix = tracking_prefix[carrier]
    tracking_number = f"{prefix}{random.randint(100000000, 999999999)}"

    # Estimate delivery
    delivery_days = random.randint(2, 5)
    estimated_delivery = (datetime.now() + timedelta(days=delivery_days)).strftime("%Y-%m-%d")

    return {
        "tracking_number": tracking_number,
        "carrier": carrier,
        "status": "已揽收 (Picked Up)",
        "estimated_delivery": estimated_delivery,
        "sender": "FashionClaw 智能衣橱系统",
        "recipient": buyer_name,
        "item": item_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def update_item_status(item_id: str, new_status: str) -> bool:
    """
    Mock API: Update item status in the database.

    Args:
        item_id: ID of the clothing item
        new_status: New status ("in_closet", "selling", or "sold")

    Returns:
        True if update successful, False otherwise
    """
    import json

    try:
        with open("database.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data["wardrobe"]:
            if item["item_id"] == item_id:
                item["status"] = new_status
                if new_status == "sold":
                    item["sold_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                break

        with open("database.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        print(f"Error updating database: {e}")
        return False

"""
Pricing Tool for FashionClaw
使用 Gemini 3.1 Pro 直接识别品牌、型号和价格
"""
import os
import json
import time
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 优先使用 Gemini，回退到 Kimi
USE_GEMINI = True
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBxPIrLAefGJXAykrW-scM9ygkWCu4-vJQ")


class PricingTool:
    """
    Pricing Tool - 使用 Gemini 3.1 Pro 直接识别品牌和价格
    """

    def __init__(self):
        self.gemini_analyzer = None
        self._init_analyzer()

    def _init_analyzer(self):
        """初始化 Gemini 分析器"""
        if USE_GEMINI and GEMINI_API_KEY:
            try:
                from tools.gemini_analyzer import GeminiClothingAnalyzer
                self.gemini_analyzer = GeminiClothingAnalyzer(api_key=GEMINI_API_KEY)
                print("[PricingTool] Gemini analyzer initialized")
            except Exception as e:
                print(f"[PricingTool] Failed to init Gemini: {e}")
                self.gemini_analyzer = None

    def analyze_clothing(self, image_path: str, max_retries: int = 2) -> Optional[Dict[str, Any]]:
        """
        使用 Gemini 分析服装图片

        Args:
            image_path: 图片路径
            max_retries: 重试次数

        Returns:
            分析结果字典
        """
        print(f"[PricingTool] Analyzing with Gemini: {image_path}")

        if not os.path.exists(image_path):
            print(f"[PricingTool] Image not found: {image_path}")
            return None

        # 使用 Gemini 分析
        if self.gemini_analyzer:
            try:
                result = self.gemini_analyzer.analyze(image_path, max_retries=max_retries)
                if result and result.get("success"):
                    # 转换为统一格式
                    return self._convert_gemini_result(result)
            except Exception as e:
                print(f"[PricingTool] Gemini analysis failed: {e}")

        # 回退：使用 Google Lens 搜索
        print("[PricingTool] Falling back to Google Lens...")
        return self._fallback_to_search(image_path)

    def _convert_gemini_result(self, gemini_result: Dict) -> Dict[str, Any]:
        """将 Gemini 结果转换为统一格式"""
        return {
            "success": True,
            "source": "gemini",
            "model_used": gemini_result.get("model_used", ""),
            "item_details": {
                "brand": gemini_result.get("brand", "Unknown"),
                "model_name": gemini_result.get("model_name", ""),
                "product_code": gemini_result.get("product_code", ""),
                "category": gemini_result.get("category", ""),
                "material": gemini_result.get("material", ""),
                "color": gemini_result.get("color", ""),
                "pattern": gemini_result.get("pattern", ""),
                "condition": gemini_result.get("condition", "良好"),
                "design_features": gemini_result.get("design_features", []),
                "confidence": gemini_result.get("confidence", "中")
            },
            "official_price": gemini_result.get("official_price", {}),
            "resale_estimate": gemini_result.get("resale_estimate", {}),
            "description": gemini_result.get("description", ""),
            "full_analysis": gemini_result
        }

    def _fallback_to_search(self, image_path: str) -> Optional[Dict[str, Any]]:
        """回退到 Google Lens 搜索"""
        try:
            from tools.bing_visual_search import search_clothing_on_google
            result = search_clothing_on_google(image_path)

            if result and result.get("success"):
                return {
                    "success": True,
                    "source": "google_lens",
                    "item_details": {
                        "brand": result.get("brand", "Unknown"),
                        "category": "",
                        "confidence": "中"
                    },
                    "pricing": {
                        "suggested_price": result.get("price", "未知"),
                        "source": "Google Lens"
                    }
                }
        except Exception as e:
            print(f"[PricingTool] Search fallback failed: {e}")

        return None

    def query_market_price(self, brand: str, category: str = "") -> Optional[Dict[str, Any]]:
        """
        查询当前市场价格（可选，用于验证）

        Args:
            brand: 品牌名称
            category: 商品类别

        Returns:
            市场价格数据
        """
        # 这里可以接入实时价格 API（如得物、StockX）
        # 目前返回基于品牌和类别的估算
        print(f"[PricingTool] Querying market price for: {brand} {category}")

        # 简单的价格估算逻辑
        brand_tiers = {
            "luxury": ["Louis Vuitton", "Chanel", "Hermes", "Prada", "Gucci", "Dior", "Balenciaga"],
            "premium": ["Nike", "Adidas", "Zara", "H&M", "Uniqlo", "CK", "Tommy Hilfiger"],
            "fast_fashion": ["Shein", "Forever 21", "Mango"]
        }

        # 判断品牌层级
        tier = "unknown"
        for t, brands in brand_tiers.items():
            if any(b.lower() in brand.lower() for b in brands):
                tier = t
                break

        # 根据层级估算价格范围
        tier_prices = {
            "luxury": {"min": 5000, "max": 30000, "avg": 15000},
            "premium": {"min": 200, "max": 2000, "avg": 800},
            "fast_fashion": {"min": 50, "max": 300, "avg": 150},
            "unknown": {"min": 100, "max": 1000, "avg": 500}
        }

        prices = tier_prices.get(tier, tier_prices["unknown"])

        return {
            "tier": tier,
            "price_range": f"¥{prices['min']} - ¥{prices['max']}",
            "estimated_price": prices['avg'],
            "source": "brand_tier_estimate"
        }

    def run(self, image_path: str) -> Dict[str, Any]:
        """
        执行完整的定价分析流程

        Args:
            image_path: 图片路径

        Returns:
            完整的定价报告
        """
        print(f"\n{'='*50}")
        print(f"[PricingTool] Starting analysis for: {image_path}")
        print(f"{'='*50}\n")

        # Step 1: Gemini 分析
        start_time = time.time()
        analysis = self.analyze_clothing(image_path)
        analysis_time = time.time() - start_time

        if not analysis:
            return {
                "success": False,
                "error": "Failed to analyze image",
                "item_details": {},
                "pricing": {}
            }

        print(f"[PricingTool] Analysis completed in {analysis_time:.2f}s")

        # Step 2: 如果 Gemini 没给出价格，补充市场估价
        item_details = analysis.get("item_details", {})
        brand = item_details.get("brand", "Unknown")

        official_price = analysis.get("official_price", {})
        resale_estimate = analysis.get("resale_estimate", {})

        # 整合定价信息
        pricing = {
            "brand": brand,
            "model": item_details.get("model_name", ""),
            "official_price": official_price,
            "resale_estimate": resale_estimate,
            "suggested_price": resale_estimate.get("max_price", 0) if resale_estimate else 0,
            "confidence": item_details.get("confidence", "低"),
            "source": analysis.get("source", "unknown")
        }

        # Step 3: 生成报告
        result = {
            "success": True,
            "analysis_source": analysis.get("source", "unknown"),
            "model_used": analysis.get("model_used", ""),
            "item_details": item_details,
            "pricing": pricing,
            "description": analysis.get("description", ""),
            "processing_time": analysis_time
        }

        print(f"\n[PricingTool] Result:")
        print(f"  Brand: {brand}")
        print(f"  Model: {item_details.get('model_name', 'N/A')}")
        print(f"  Official: {official_price.get('amount', 'N/A')} {official_price.get('currency', '')}")
        print(f"  Resale: {resale_estimate.get('min_price', 'N/A')} - {resale_estimate.get('max_price', 'N/A')} CNY")
        print(f"{'='*50}\n")

        return result


# Global instance for easy import
pricing_tool = PricingTool()


def generate_pricing_report(result: Dict[str, Any]) -> str:
    """生成格式化的定价报告"""
    if not result.get("success"):
        return f"❌ 分析失败: {result.get('error', 'Unknown error')}"

    item = result.get("item_details", {})
    pricing = result.get("pricing", {})
    official = pricing.get("official_price", {})
    resale = pricing.get("resale_estimate", {})

    report = f"""📊 **Gemini 智能分析报告**

👕 **商品信息**
• 品牌: **{item.get('brand', 'Unknown')}**
• 型号: {item.get('model_name', '未识别')}
• 货号: {item.get('product_code', 'N/A')}
• 类别: {item.get('category', 'N/A')}
• 材质: {item.get('material', 'N/A')}
• 成色: {item.get('condition', 'N/A')}

💰 **价格参考**
• 官方指导价: {official.get('amount', 'N/A')} {official.get('currency', '')}
• 二手建议价: ¥{resale.get('min_price', 0)} - ¥{resale.get('max_price', 0)}
• 置信度: {pricing.get('confidence', '低')}

📝 **商品描述**
{result.get('description', '无')[:200]}...

⚡ 分析来源: {result.get('analysis_source', 'unknown')} ({result.get('model_used', '')})
⏱️ 处理时间: {result.get('processing_time', 0):.2f}s
"""
    return report


if __name__ == "__main__":
    # 测试
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            result = pricing_tool.run(image_path)
            print(generate_pricing_report(result))
        else:
            print(f"文件不存在: {image_path}")
    else:
        print("用法: python pricing_tool.py <image_path>")

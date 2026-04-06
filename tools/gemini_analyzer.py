"""
Gemini 3.1 Pro 服装分析模块
直接识别品牌、型号、价格，无需图搜
"""
import os
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path

import google.generativeai as genai

# API 配置
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBxPIrLAefGJXAykrW-scM9ygkWCu4-vJQ")
DEFAULT_MODEL = "gemini-3.1-pro-preview"  # 最新版本
FALLBACK_MODEL = "gemini-2.5-pro"  # 备用

# 系统 Prompt - 让 Gemini 返回结构化数据
SYSTEM_PROMPT = """你是专业的奢侈品和时尚商品鉴定专家，擅长通过图片识别服装品牌、型号、材质和价格。

分析要求：
1. 仔细观察品牌标识、Logo、洗水标、五金件等细节
2. 识别具体的商品编号（如有）
3. 判断面料材质（棉、羊毛、尼龙、皮革等）
4. 估算官方指导价（基于品牌定位和材质）
5. 给出二手市场建议售价（考虑成色和品牌保值度）

输出必须严格遵循以下 JSON 格式：
{
    "brand": "品牌名称（如 Louis Vuitton, Nike, Zara）",
    "model_name": "官方商品名称",
    "product_code": "商品编号或货号，没有则填 null",
    "category": "服装类别（如夹克、T恤、连衣裙）",
    "material": "面料材质描述",
    "design_features": ["设计特征1", "设计特征2", "设计特征3"],
    "color": "主色调",
    "pattern": "图案描述（如纯色、印花、格纹）",
    "condition": "成色评估（全新/几乎全新/良好/有明显穿着痕迹）",
    "official_price": {
        "currency": "货币代码（EUR/USD/CNY）",
        "amount": 数字,
        "price_range": "价格区间描述"
    },
    "resale_estimate": {
        "currency": "CNY",
        "min_price": 数字,
        "max_price": 数字,
        "confidence": "高/中/低"
    },
    "description": "详细的商品描述",
    "confidence": "整体识别置信度（高/中/低）"
}

注意：
- 如果无法确定品牌，brand 填 "Unknown"
- 如果无法确定具体型号，model_name 填 "未识别"
- official_price 基于你训练数据的知识截止点估算
- resale_estimate 基于：品牌保值度 × 成色 × 市场热度"""


class GeminiClothingAnalyzer:
    """使用 Gemini 3.1 Pro 分析服装图片"""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model or DEFAULT_MODEL
        self.model = None
        self._init_client()

    def _init_client(self):
        """初始化 Gemini 客户端"""
        if not self.api_key:
            print("[Gemini] Warning: No API key provided")
            return

        try:
            genai.configure(api_key=self.api_key)

            # 尝试使用指定模型，失败则回退
            try:
                self.model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=SYSTEM_PROMPT
                )
                print(f"[Gemini] Initialized with model: {self.model_name}")
            except Exception as e:
                print(f"[Gemini] {self.model_name} not available, trying fallback: {e}")
                self.model_name = FALLBACK_MODEL
                self.model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=SYSTEM_PROMPT
                )
                print(f"[Gemini] Initialized with fallback model: {self.model_name}")

        except Exception as e:
            print(f"[Gemini] Failed to initialize: {e}")
            self.model = None

    def analyze(self, image_path: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        分析服装图片

        Args:
            image_path: 图片本地路径
            max_retries: 最大重试次数

        Returns:
            结构化分析结果
        """
        if not self.model:
            return {"error": "Gemini client not initialized", "success": False}

        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}", "success": False}

        # 上传图片
        try:
            print(f"[Gemini] Uploading image: {image_path}")
            file_obj = genai.upload_file(image_path, mime_type="image/jpeg")
            print(f"[Gemini] File uploaded: {file_obj.name}")
        except Exception as e:
            print(f"[Gemini] Upload failed: {e}")
            return {"error": f"Upload failed: {e}", "success": False}

        # 调用 API
        for attempt in range(max_retries):
            try:
                print(f"[Gemini] Analyzing (attempt {attempt + 1}/{max_retries})...")

                response = self.model.generate_content(
                    [file_obj, "请分析这件衣服，返回 JSON 格式的详细信息。"],
                    generation_config={
                        "temperature": 0.2,
                        "max_output_tokens": 2048,
                        "response_mime_type": "application/json"
                    }
                )

                # 解析 JSON 响应
                try:
                    result = json.loads(response.text)
                    result["success"] = True
                    result["model_used"] = self.model_name
                    result["image_path"] = image_path

                    print(f"[Gemini] Analysis complete: {result.get('brand')} {result.get('model_name')}")
                    return result

                except json.JSONDecodeError as e:
                    print(f"[Gemini] JSON parse error: {e}")
                    print(f"[Gemini] Raw response: {response.text[:500]}")
                    # 尝试提取 JSON
                    result = self._extract_json_from_text(response.text)
                    if result:
                        result["success"] = True
                        result["model_used"] = self.model_name
                        return result
                    raise Exception(f"JSON parse error: {e}")

            except Exception as e:
                print(f"[Gemini] Analysis failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    return {"error": str(e), "success": False, "model_used": self.model_name}

        return {"error": "Max retries exceeded", "success": False}

    def _extract_json_from_text(self, text: str) -> Optional[Dict]:
        """从文本中提取 JSON"""
        import re

        # 尝试找 JSON 块
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass

        # 尝试直接找 JSON 对象
        json_match = re.search(r'(\{[\s\S]*"brand"[\s\S]*\})', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass

        return None

    def quick_identify(self, image_path: str) -> Dict[str, Any]:
        """
        快速识别（只返回关键信息）

        Returns:
            {brand, model_name, official_price, resale_estimate, confidence}
        """
        result = self.analyze(image_path)

        if not result.get("success"):
            return result

        # 提取关键信息
        return {
            "success": True,
            "brand": result.get("brand", "Unknown"),
            "model_name": result.get("model_name", "未识别"),
            "category": result.get("category", ""),
            "official_price": result.get("official_price", {}),
            "resale_estimate": result.get("resale_estimate", {}),
            "confidence": result.get("confidence", "低"),
            "model_used": result.get("model_used", ""),
            "full_result": result  # 保留完整结果
        }


# 便捷函数
def analyze_clothing(image_path: str, api_key: str = None) -> Dict[str, Any]:
    """快速分析函数"""
    analyzer = GeminiClothingAnalyzer(api_key=api_key)
    return analyzer.analyze(image_path)


def quick_identify(image_path: str, api_key: str = None) -> Dict[str, Any]:
    """快速识别函数"""
    analyzer = GeminiClothingAnalyzer(api_key=api_key)
    return analyzer.quick_identify(image_path)


if __name__ == "__main__":
    # 测试
    import sys
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        if os.path.exists(test_image):
            print(f"\n测试图片: {test_image}\n")
            result = analyze_clothing(test_image)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"文件不存在: {test_image}")
    else:
        print("用法: python gemini_analyzer.py <image_path>")

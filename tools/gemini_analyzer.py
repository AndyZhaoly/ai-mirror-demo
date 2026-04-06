"""
Gemini 服装分析模块 (使用新版 google-genai API)
直接识别品牌、型号、价格，无需图搜
"""
import os
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path

from google import genai
from google.genai import types

# API 配置
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_MODEL = "models/gemini-3.1-pro-preview"  # Gemini 3.1 Pro

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
    """使用 Gemini 分析服装图片"""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model or DEFAULT_MODEL
        self.client = None
        self._init_client()

    def _init_client(self):
        """初始化 Gemini 客户端"""
        if not self.api_key:
            print("[Gemini] Warning: No API key provided")
            return

        try:
            self.client = genai.Client(api_key=self.api_key)
            print(f"[Gemini] Initialized with model: {self.model_name}")
        except Exception as e:
            print(f"[Gemini] Failed to initialize: {e}")
            self.client = None

    def analyze(self, image_path: str, max_retries: int = 2, use_search: bool = False) -> Dict[str, Any]:
        """
        分析服装图片

        Args:
            image_path: 图片本地路径
            max_retries: 最大重试次数
            use_search: 是否启用 Google Search 获取实时价格参考

        Returns:
            结构化分析结果（包含 reference_sources 当 use_search=True 时）
        """
        if not self.client:
            return {"error": "Gemini client not initialized", "success": False}

        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}", "success": False}

        # 读取图片
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        except Exception as e:
            return {"error": f"Failed to read image: {e}", "success": False}

        # 准备工具配置
        tools = None
        if use_search:
            tools = [types.Tool(google_search=types.GoogleSearch())]
            print("[Gemini] Google Search enabled")

        # 准备请求内容
        # 当启用搜索时，使用明确的搜索指令来触发 Google Search
        if use_search:
            prompt_text = (
                "分析这件衣服并执行 Google 搜索获取二手市场价格参考。\n\n"
                "请搜索以下内容：\n"
                "1. 类似款式的二手市场价格\n"
                "2. 品牌服装的二手交易平台价格参考\n\n"
                "然后以JSON格式返回分析结果：\n"
                '{"brand": "品牌名", "model_name": "型号", "product_code": "货号或null", '
                '"category": "类别", "material": "材质", "color": "颜色", "pattern": "图案", '
                '"condition": "成色", "official_price": {"currency": "CNY", "amount": 价格}, '
                '"resale_estimate": {"currency": "CNY", "min_price": 数字, "max_price": 数字, "confidence": "高/中/低"}, '
                '"description": "描述", "confidence": "高/中/低"}'
            )
            system_instruction = "你是专业的服装鉴定专家。必须使用 Google 搜索获取实时的二手市场价格信息。"
        else:
            prompt_text = "请分析这件衣服，返回 JSON 格式的详细信息。"
            system_instruction = SYSTEM_PROMPT

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    types.Part.from_text(text=prompt_text),
                ]
            )
        ]

        # 配置生成参数
        # 注意：当启用 Google Search 时，不能使用 response_mime_type="application/json"
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
            max_output_tokens=8192,  # 增加 token 限制避免截断
            tools=tools
        )

        # 调用 API
        for attempt in range(max_retries):
            try:
                print(f"[Gemini] Analyzing (attempt {attempt + 1}/{max_retries})...")

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config
                )

                # 解析 JSON 响应
                try:
                    result = json.loads(response.text)
                    result["success"] = True
                    result["model_used"] = self.model_name
                    result["image_path"] = image_path

                    # 提取 Grounding 参考链接（如果启用了搜索）
                    if use_search:
                        sources = self._extract_grounding_sources(response)
                        if sources:
                            result["reference_sources"] = sources

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
                        result["image_path"] = image_path
                        # 即使 JSON 解析失败也要提取 grounding 数据
                        if use_search:
                            sources = self._extract_grounding_sources(response)
                            if sources:
                                result["reference_sources"] = sources
                        return result
                    raise Exception(f"JSON parse error: {e}")

            except Exception as e:
                print(f"[Gemini] Analysis failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    return {"error": str(e), "success": False, "model_used": self.model_name}

        return {"error": "Max retries exceeded", "success": False}

    def _extract_grounding_sources(self, response) -> list:
        """从响应中提取 grounding 参考来源"""
        sources = []
        try:
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]

                if candidate.grounding_metadata:
                    grounding = candidate.grounding_metadata

                    # 方法1: 提取 grounding_chunks 中的来源
                    if grounding.grounding_chunks:
                        for chunk in grounding.grounding_chunks:
                            if chunk.web:
                                sources.append({
                                    "title": chunk.web.title or "",
                                    "url": chunk.web.uri or "",
                                    "domain": getattr(chunk.web, 'domain', '')
                                })

                    # 方法2: 从 search_entry_point 提取搜索链接
                    if not sources and grounding.search_entry_point:
                        import re
                        html = grounding.search_entry_point.rendered_content or ""
                        links = re.findall(r'href="([^"]+)"[^>]*>([^<]+)</a>', html)
                        for url, text in links:
                            if 'grounding-api-redirect' in url:
                                sources.append({
                                    "title": text.replace('&#39;', "'").strip(),
                                    "url": url,
                                    "domain": "Google Search"
                                })

                    # 方法3: 从 grounding_supports 提取
                    if not sources and grounding.grounding_supports:
                        for support in grounding.grounding_supports[:3]:
                            if support.segment:
                                sources.append({
                                    "title": support.segment.text[:50] + "...",
                                    "url": "",
                                    "domain": "Google Search Grounding"
                                })

            if sources:
                print(f"[Gemini] Found {len(sources)} reference sources")
        except Exception as e:
            print(f"[Gemini] Failed to extract grounding metadata: {e}")

        return sources

    def _extract_json_from_text(self, text: str) -> Optional[Dict]:
        """从文本中提取 JSON，支持截断响应的修复"""
        import re

        # 尝试找 JSON 块
        json_match = re.search(r'```json\s*(\{.*?)\s*```', text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'```json\s*(\{.*)', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            # 尝试修复截断的 JSON
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # 尝试补全截断的 JSON
                fixed = self._fix_truncated_json(json_str)
                if fixed:
                    try:
                        return json.loads(fixed)
                    except:
                        pass

        # 尝试直接找 JSON 对象
        json_match = re.search(r'(\{[\s\S]*"brand"[\s\S]*)', text)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                fixed = self._fix_truncated_json(json_str)
                if fixed:
                    try:
                        return json.loads(fixed)
                    except:
                        pass

        return None

    def _fix_truncated_json(self, json_str: str) -> Optional[str]:
        """尝试修复截断的 JSON"""
        # 移除尾部的不完整内容
        lines = json_str.split('\n')
        fixed_lines = []
        open_braces = 0
        open_brackets = 0
        in_string = False
        escape_next = False

        for line in lines:
            for char in line:
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if char == '{':
                        open_braces += 1
                    elif char == '}':
                        open_braces -= 1
                    elif char == '[':
                        open_brackets += 1
                    elif char == ']':
                        open_brackets -= 1

            fixed_lines.append(line)

            # 如果所有括号都闭合了，可以停止了
            if open_braces == 0 and open_brackets == 0 and not in_string:
                break

        fixed = '\n'.join(fixed_lines)

        # 补全未闭合的括号
        while open_braces > 0:
            fixed += '}'
            open_braces -= 1
        while open_brackets > 0:
            fixed += ']'
            open_brackets -= 1

        return fixed

    def quick_identify(self, image_path: str, use_search: bool = True) -> Dict[str, Any]:
        """
        快速识别（只返回关键信息）

        Args:
            image_path: 图片路径
            use_search: 是否启用 Google Search 获取实时价格（默认启用）

        Returns:
            {brand, model_name, official_price, resale_estimate, confidence, reference_sources}
        """
        result = self.analyze(image_path, use_search=use_search)

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
            "reference_sources": result.get("reference_sources", []),
            "full_result": result
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
            result = analyze_clothing(test_image, use_search=True)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"文件不存在: {test_image}")
    else:
        print("用法: python gemini_analyzer.py <image_path>")

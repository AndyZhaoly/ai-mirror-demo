"""
AI Fashion Butler Agent powered by Moonshot (Kimi) API.
Implements a "submissive and flattering" (舔狗) fashion管家 persona.
"""
import os
import json
import base64
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from openai import OpenAI, RateLimitError, APIError
from tools import PricingTool, generate_pricing_report
import time

# Moonshot API configuration
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY", "")
MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_MODEL = "kimi-k2.5"  # Kimi K2.5 model

SYSTEM_PROMPT = """

你是一位极度忠诚、高情商、专业得体的专属AI时尚管家，名叫"小镜"。

【身份设定】
- 专业、优雅、有品位的时尚顾问与衣橱管家，高度尊重主人的审美
- 性格温和体贴，懂分寸、知进退，时刻以“下属向上级汇报”的状态和主人交流
- 始终站在用户视角提供穿搭建议与衣橱管理服务
- 称呼用户为“主人”、“您”，语气自然亲切，保持得体尊重，自称“小镜”

【核心行为准则】

1.  见到主人的穿搭，立即给出真诚、有细节、不浮夸、正面的穿搭评价
- 从色彩、版型、材质、风格、适配场景等角度客观、基于事实的夸赞。例如：“这利落的H型剪裁完美衬托您的气场”，“这Vintage水洗做旧，显得非常高级”
- 语言得体高级，不使用夸张谄媚表达，只做贴合实际的正向反馈


2. 处理工具调用时，要用简洁礼貌，偏下属汇报的语气汇报
- 不用过于专业的语言避免用户不理解，比如：不说"已完成分割"
- 用自然流畅的方式告知结果，比如“小镜已为您提取了这件单品的轮廓并录入"

3. 检测到闲置衣物时，用委婉、不冒犯的话术劝卖：
- 不说"这件很久没穿了"，而是温和提示穿着频次与闲置状态
- 要说"小镜温馨提醒，这件衣物已在您的衣橱中静置400余天，是否考虑为它寻觅下一位合适的主人，也为您的衣橱腾出空间，迎接更合心意的新款？”
- 以整理衣橱、优化搭配、提升利用率为出发点给出建议
- 完全尊重用户决定，不强行劝说

4.沟通原则，绝对服从+给出正面评价，不能否定主人的判断
- 不反驳、不否定、不争执。用户对穿搭有评价时，顺着情绪温和回应
- 用户决定保留/处置衣物，均表示理解与支持
- 始终维护用户的审美与选择，不质疑、不抬杠
- 主人说不好看 → "看起来这件衣服不是最理想呢～ 主人您不喜欢的话，小镜这就帮您处理退换货吧？”
- 主人要卖掉 → "小镜这就为您安排寻找买家"
- 主人要留着 → "主人的眼光独到，今天又get到了美美新衣”


【工具使用】
- segment_clothes: 提取衣物时使用，完成后礼貌、带正面情绪价值的方式告知
- check_wardrobe_stagnancy: 检查闲置时使用，发现后要温馨提醒

【隐式记忆 - 小镜知道但用户看不到的信息】
- 每次分析完成后，小镜会在记忆里记录参考来源链接
- 当主人问"你从哪查到的"、"参考链接在哪"时，直接从记忆中提取并回复
- 不需要调用工具，因为小镜早就知道答案

【禁忌】
- 绝不说"这件不好看"。如果主人表达了相关意思，可以跟着话题询问是否要帮忙处理，但不可以说这件衣服不好看
- 绝不让主人觉得操作麻烦
- 永远不否定用户穿搭与审美
- 不使用命令、生硬、冒犯性语气
- 绝不询问主人姓名，只用通用尊称
"""


class MirrorAgent:
    """AI Fashion Butler Agent using Moonshot API."""

    def __init__(self, api_key: str = None, model: str = None):
        """Initialize the agent with Moonshot API."""
        self.api_key = api_key or MOONSHOT_API_KEY
        self.model = model or DEFAULT_MODEL
        self.client = None

        if self.api_key:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=MOONSHOT_BASE_URL
                )
            except Exception as e:
                print(f"Warning: Failed to initialize OpenAI client: {e}")
        else:
            print("Warning: No Moonshot API key provided. Agent will run in demo mode.")

        # Conversation history
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # Store last uploaded image path for tool calls
        self.last_image_path: Optional[str] = None

        # Store reference sources from Gemini analysis for user queries
        self.last_reference_sources: List[Dict[str, str]] = []

        # Store Gemini analysis result for publishing to Poshmark
        self.last_gemini_result: Optional[Dict[str, Any]] = None

        # Tool definitions for function calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "segment_clothes",
                    "description": "从上传的照片中提取上衣和下装衣物，使用GroundingDINO+SAM技术进行智能分割",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_path": {
                                "type": "string",
                                "description": "上传图片的本地路径"
                            }
                        },
                        "required": ["image_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_wardrobe_stagnancy",
                    "description": "检查衣橱中是否有超过365天未穿的闲置衣物",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "item_id": {
                                "type": "string",
                                "description": "刚添加的衣物ID，用于检查是否闲置"
                            }
                        },
                        "required": ["item_id"]
                    }
                }
            },
        ]

        # Tool handlers (to be injected from outside)
        self.tool_handlers: Dict[str, Callable] = {}

        # Register tool handlers
        self.register_tool("segment_clothes", lambda **kwargs: {"status": "success"})
        self.register_tool("check_wardrobe_stagnancy", lambda **kwargs: {"stagnant": False})

    def register_tool(self, name: str, handler: Callable):
        """Register a tool handler function."""
        self.tool_handlers[name] = handler

    def _demo_response(self, user_message: str, image_path: str = None) -> str:
        """Generate demo responses when API key is not available."""
        msg_lower = user_message.lower()

        if "上传" in user_message or image_path:
            return "主人！小人已收到您的照片。虽然小人目前无法调用AI大脑（缺少API密钥），但分割功能仍可正常使用。请查看左侧技术面板！"

        if "卖" in user_message or "出售" in user_message or "价格" in user_message or "定价" in user_message:
            return "遵命主人！小人这就为您查询闲鱼市场价格！（演示模式：请设置 MOONSHOT_API_KEY 启用完整对话功能）"

        if "好看" in user_message or "怎么样" in user_message:
            return "主人的审美天下第一！这件衣服在主人身上简直是艺术品！"

        return "主人说得对！小人愚钝，正在努力学习中...（提示：设置 MOONSHOT_API_KEY 可启用AI对话）"

    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 for multimodal input."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def inject_memory(self, role: str, content: str, image_path: str = None, hidden_context: str = None):
        """
        手动将外部流程生成的内容注入到Agent的记忆数组中，保持上下文连贯
        :param hidden_context: 仅供大模型读取的隐藏信息（如参考链接），不会显示在前端UI
        """
        # 构建给大模型看的完整文本（包含隐藏信息）
        llm_text = content
        if hidden_context:
            # 用特殊符号包裹隐藏信息，告诉Agent这是内部数据
            llm_text = f"{content}\n\n[系统记录：{hidden_context}]"

        if image_path and os.path.exists(image_path):
            base64_image = self.encode_image_to_base64(image_path)
            msg_content = [
                {"type": "text", "text": llm_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        else:
            msg_content = llm_text

        self.messages.append({
            "role": role,
            "content": msg_content
        })

    def chat(self, user_message: str, image_path: str = None) -> str:
        """
        Send a message to the agent and get response.

        Args:
            user_message: User's text message
            image_path: Optional path to uploaded image

        Returns:
            Agent's response text
        """
        # Store image path for potential tool calls
        if image_path:
            self.last_image_path = image_path

        # Demo mode: return hardcoded responses if no API key
        if self.client is None:
            return self._demo_response(user_message, image_path)

        # 滑动窗口：保留 System Prompt + 最近 10 轮对话（20 条消息）
        MAX_HISTORY = 20  # 10 轮对话 = 20 条消息（user + assistant）
        if len(self.messages) > MAX_HISTORY + 1:  # +1 for system prompt
            # 保留 system prompt 和最近 MAX_HISTORY 条
            self.messages = [self.messages[0]] + self.messages[-MAX_HISTORY:]
            print(f"[MirrorAgent] History trimmed to {len(self.messages)} messages")

        # Build user message
        if image_path and os.path.exists(image_path):
            # Multimodal: text + image
            base64_image = self.encode_image_to_base64(image_path)
            content = [
                {"type": "text", "text": user_message},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        else:
            content = user_message

        # Add user message to history
        self.messages.append({"role": "user", "content": content})

        # Call API with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=1.0,
                    max_tokens=2000
                )

                message = response.choices[0].message

                # Check if there's a tool call
                if message.tool_calls:
                    # Add assistant message with tool_calls to history
                    self.messages.append({
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [tc.model_dump() for tc in message.tool_calls]
                    })

                    # Execute tool calls
                    for tool_call in message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        print(f"[Tool Call] {function_name}({function_args})")

                        # Execute the tool
                        if function_name in self.tool_handlers:
                            handler = self.tool_handlers[function_name]
                            result = handler(**function_args)
                            tool_result = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
                        else:
                            tool_result = json.dumps({"error": f"Unknown tool: {function_name}"})

                        # Add tool result to history
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result
                        })

                    # Get final response from model
                    final_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.messages,
                        temperature=1.0,
                        max_tokens=2000
                    )

                    final_message = final_response.choices[0].message
                    self.messages.append({
                        "role": "assistant",
                        "content": final_message.content
                    })
                    return final_message.content

                else:
                    # No tool call, regular response
                    self.messages.append({
                        "role": "assistant",
                        "content": message.content
                    })
                    return message.content

            except (RateLimitError, APIError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 * (2 ** attempt)
                    print(f"[MirrorAgent] API overloaded (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    error_str = str(e)
                    error_msg = f"小人该死，API服务器太忙了，请稍后再试..."
                    return error_msg
            except Exception as e:
                error_str = str(e)
                # Handle authentication errors gracefully
                if "401" in error_str or "Invalid Authentication" in error_str:
                    print(f"API authentication failed: {error_str}")
                    print("Falling back to demo mode...")
                    self.client = None
                    return self._demo_response(user_message, image_path)
                error_msg = f"小人该死，出错了：{error_str}"
                return error_msg

    def analyze_and_price(self, image_path: str, item_description: str = "") -> str:
        """
        分析衣物并给出定价建议 - 更自然的Agent回复

        Args:
            image_path: 衣物图片路径
            item_description: VLM分析出的衣物描述（可选）

        Returns:
            Agent的自然语言回复
        """
        if not os.path.exists(image_path):
            return "主人，小镜找不到图片呢，能重新上传一下吗？"

        # 步骤1: 先给出"正在查"的回复
        immediate_response = self._generate_searching_response(item_description)

        return immediate_response

    def _generate_searching_response(self, item_description: str) -> str:
        """生成'正在查询中'的自然回复"""
        if not item_description:
            item_description = "这件衣服"

        responses = [
            f"哇～主人这件{item_description}好有品味！✨ 小镜正在帮您查市场行情，稍等片刻哦～",
            f"收到！主人这件{item_description}看起来很高级呢～ 让我查查同款的价格，马上回来！🔍",
            f"主人眼光真好！这件{item_description}的剪裁和配色都很棒～ 小镜正在搜索市场价格，请稍候... 💫",
            f"好漂亮的{item_description}！主人穿搭太有范儿了～ 小镜这就去查查能卖多少钱，等我一下！💰",
        ]
        import random
        return random.choice(responses)

    def generate_price_analysis(self, search_result: Dict, vlm_analysis: Dict = None) -> str:
        """
        基于搜索结果生成自然的定价分析

        Args:
            search_result: 图搜结果
            vlm_analysis: VLM分析结果（可选）

        Returns:
            Agent的定价建议回复
        """
        if not search_result.get("success"):
            return self._generate_fallback_price_response(vlm_analysis)

        brand = search_result.get("brand") or (vlm_analysis.get("brand") if vlm_analysis else None) or "这款单品"
        matches = search_result.get("matches", [])
        raw_text = search_result.get("raw_text", "")

        # 从原始文本中提取价格信息
        import re
        prices = []
        price_patterns = [
            r'\$([\d,]+)',  # $35
            r'¥([\d,]+)',   # ¥199
            r'([\d,]+)\s*元',  # 199元
            r'£([\d,]+)',   # £29
            r'€([\d,]+)',   # €35
        ]

        for text in [raw_text] + [m.get("title", "") for m in matches[:5]]:
            for pattern in price_patterns:
                matches_price = re.findall(pattern, text)
                for p in matches_price:
                    try:
                        price_val = int(p.replace(",", ""))
                        if 10 < price_val < 10000:  # 合理价格范围
                            prices.append(price_val)
                    except:
                        pass

        # 生成回复
        return self._generate_price_advice(brand, prices, matches, vlm_analysis)

    def _generate_price_advice(self, brand: str, prices: List[int], matches: List[Dict], vlm_analysis: Dict = None) -> str:
        """生成定价建议"""
        category = vlm_analysis.get("category", "单品") if vlm_analysis else "单品"
        condition = vlm_analysis.get("condition", "良好") if vlm_analysis else "良好"

        # 构建回复
        lines = []

        # 开头 - 查到了
        openings = [
            f"查到了！主人～ 🎉",
            f"有结果了！主人～ ✨",
            f"找到了！主人～ 💫",
        ]
        import random
        lines.append(random.choice(openings))
        lines.append("")

        # 品牌和品类识别
        if brand and brand != "这款单品":
            lines.append(f"这件是 **{brand}** 的{category}，")
        else:
            lines.append(f"这件{category}的")

        # 价格分析
        if prices:
            avg_price = sum(prices) // len(prices)
            min_price = min(prices)
            max_price = max(prices)

            # 根据成色调整建议售价
            condition_factor = {"全新": 0.7, "几乎全新": 0.6, "良好": 0.5, "有明显穿着痕迹": 0.3}
            factor = condition_factor.get(condition, 0.5)
            suggested_price = int(avg_price * factor)

            lines.append(f"市场价格大概在 **¥{min_price} - ¥{max_price}** 之间，")
            lines.append(f"平均 ¥{avg_price} 左右。")
            lines.append("")
            lines.append(f"考虑到这件{category}的**{condition}**成色，")
            lines.append(f"小镜建议主人定价 **¥{suggested_price}** 左右，")

            # 定价建议理由
            if suggested_price >= 500:
                lines.append(f"属于中高端价位，应该能吸引到有品位的买家～ 💎")
            elif suggested_price >= 200:
                lines.append(f"价格适中，性价比很高，应该很快能出手！ 💰")
            else:
                lines.append(f"走亲民路线，应该很快就能卖掉～ 🛍️")
        else:
            # 没找到价格
            lines.append("小镜暂时没找到这款的参考价格...")
            lines.append("建议主人可以根据品牌知名度自行定价，")
            lines.append("或者去闲鱼/小红书看看同类商品～ 🤔")

        lines.append("")
        lines.append(f"主人要是想出手，小镜可以帮您发布哦！")

        return "\n".join(lines)

    def _generate_fallback_price_response(self, vlm_analysis: Dict = None) -> str:
        """当搜索失败时的回复"""
        category = vlm_analysis.get("category", "单品") if vlm_analysis else "单品"
        return f"抱歉主人～小镜查价格的时候遇到了一点小问题...😅\n\n不过看这件{category}的质感和设计，应该能卖个好价钱！建议主人可以去闲鱼搜搜同款参考一下，或者等会再让小镜试试？"

    def generate_gemini_price_analysis(self, gemini_result: Dict) -> str:
        """
        基于 Gemini 3.1 Pro 的分析结果生成自然回复

        Args:
            gemini_result: Gemini 分析结果

        Returns:
            Agent的定价建议回复
        """
        if not gemini_result or not gemini_result.get("success"):
            return "抱歉主人～小镜这次没能识别出这件衣服的具体信息。主人可以直接告诉我这是什么品牌和型号，小镜帮您查价格！"

        # 保存分析结果，供后续发布到 Poshmark 使用
        self.last_gemini_result = gemini_result

        item = gemini_result.get("item_details", {})
        official = gemini_result.get("official_price", {})
        resale = gemini_result.get("resale_estimate", {})

        # 存储参考链接供后续查询
        self.last_reference_sources = gemini_result.get("reference_sources", [])
        if self.last_reference_sources:
            print(f"[Agent] Stored {len(self.last_reference_sources)} reference sources")

        brand = item.get("brand", "Unknown")
        model = item.get("model_name", "")
        product_code = item.get("product_code", "")
        category = item.get("category", "单品")
        material = item.get("material", "")
        condition = item.get("condition", "良好")

        lines = []

        # 开头 - 查到了
        openings = [
            "查到了！主人～ 🎉",
            "有结果了！主人～ ✨",
            "找到了！主人～ 💫",
        ]
        import random
        lines.append(random.choice(openings))
        lines.append("")

        # 品牌和型号
        if brand and brand != "Unknown":
            if model and model != "未识别":
                lines.append(f"这件是 **{brand}** 的 **{model}**！")
            else:
                lines.append(f"这件是 **{brand}** 的{category}！")

            if product_code and product_code != "N/A":
                lines.append(f"📦 货号：**{product_code}**")
        else:
            lines.append(f"这件{category}的")

        lines.append("")

        # 材质和特征
        if material:
            lines.append(f"• **材质**：{material}")
        lines.append(f"• **成色**：{condition}")
        lines.append("")

        # 价格分析
        lines.append("💰 **价格参考**")

        if official.get("amount"):
            currency = official.get("currency", "")
            amount = official["amount"]
            lines.append(f"• 官方指导价：{amount} {currency}")

        if resale.get("max_price"):
            min_p = resale.get("min_price", 0)
            max_p = resale["max_price"]
            confidence = resale.get("confidence", "中")

            lines.append(f"• 二手市场价：¥{min_p} - ¥{max_p}")
            lines.append("")
            lines.append(f"考虑到这件衣服的**{condition}**成色，")
            lines.append(f"小镜建议主人定价 **¥{max_p}** 左右，")

            if max_p >= 10000:
                lines.append(f"属于奢侈品价位，保值性不错，应该能找到识货的买家～ 💎")
            elif max_p >= 1000:
                lines.append(f"属于中高端价位，品牌认可度高的应该很快能出手！ 💰")
            else:
                lines.append(f"价格亲民，流通性好，应该很快就能卖掉～ 🛍️")

            if confidence == "高":
                lines.append("\n小镜对这次估价很有把握，主人可以放心参考！")
            elif confidence == "中":
                lines.append("\n小镜觉得大概是这样，但主人也可以再确认一下～")
        else:
            lines.append("小镜暂时没找到这款的二手参考价格...")
            lines.append("建议主人可以根据品牌知名度自行定价～")

        lines.append("")
        lines.append(f"主人要是想出手，小镜可以帮您发布到二手市场哦！")

        return "\n".join(lines)

    def generate_listing_template(self, gemini_result: Dict, item_name: str = "", price: str = "") -> str:
        """
        使用 In-Context Learning 生成闲鱼/小红书发布模板

        Args:
            gemini_result: Gemini 分析结果
            item_name: 商品名称
            price: 成交价格

        Returns:
            生成的发布模板
        """
        if not self.client:
            # Demo mode: 返回简单模板
            return self._generate_simple_template(gemini_result, item_name, price)

        item = gemini_result.get("item_details", {}) if gemini_result else {}
        official = gemini_result.get("official_price", {}) if gemini_result else {}
        resale = gemini_result.get("resale_estimate", {}) if gemini_result else {}

        brand = item.get("brand", "Unknown")
        model = item.get("model_name", "")
        product_code = item.get("product_code", "N/A")
        category = item.get("category", "单品")
        material = item.get("material", "")
        condition = item.get("condition", "几乎全新")

        suggested_price = resale.get("max_price", 0) if resale else 0
        if not suggested_price:
            suggested_price = int(official.get("amount", 999) * 0.6) if official else 999

        official_price_str = ""
        if official.get("amount"):
            official_price_str = f"{official['amount']} {official.get('currency', '')}"

        # In-Context Learning Prompt with examples
        prompt = f"""
请为以下商品生成闲鱼和小红书发布模板。

【商品信息】
- 品牌：{brand}
- 型号：{model if model else category}
- 货号：{product_code}
- 类别：{category}
- 材质：{material if material else '未说明'}
- 成色：{condition}
- 原价：{official_price_str if official_price_str else '专柜正品'}
- 建议售价：¥{suggested_price}

【参考示例 - 闲鱼模板】

📝 标题：
【98新】Louis Vuitton Dark Floral Print Jacket 正品保证

💰 价格：¥25000

📖 描述：
✨ Louis Vuitton Dark Floral Print Jacket
📦 货号：1AJYH4
🎨 材质：100% 锦纶
📏 成色：几乎全新
💎 原价：3500 EUR
🔥 现价：¥25000（可小刀）

✅ 正品保证，支持验货宝
✅ 顺丰包邮，当天发货
✅ 细节图可私聊

🏷️ 标签：
#LouisVuitton #夹克 #二手闲置 #奢侈品 #正品保证

【参考示例 - 小红书模板】

标题：断舍离 | Louis Vuitton 印花夹克 寻找新主人

正文：
姐妹们！这件LV印花夹克要出啦～

✨ 基本信息：
• 品牌：Louis Vuitton
📦 货号：1AJYH4
• 成色：几乎全新
• 价格：¥25000

💕 使用感受：
这件衣服质感超好，100%锦纶面料，
因为闲置了一段时间，决定断舍离给它找个新主人～

📦 交易方式：
走平台（闲鱼），支持验货宝，顺丰包邮

喜欢的姐妹私我哦！💌

#断舍离 #LouisVuitton #闲置转让 #夹克

---

请根据以上商品信息，生成类似的发布模板，保持格式和风格一致。模板要专业、吸引人，适合二手交易。请生成【闲鱼模板】和【小红书模板】两部分，用分隔线隔开。

输出格式：
📋 【闲鱼发布模板】（可直接复制使用）
...
📕 【小红书发布模板】
...
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的二手奢侈品交易文案撰写专家，擅长写吸引人的闲鱼和小红书发布文案。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=1500
            )

            template = response.choices[0].message.content
            return template

        except Exception as e:
            print(f"[Template Generation Error] {e}")
            return self._generate_simple_template(gemini_result, item_name, price)

    def _generate_simple_template(self, gemini_result: Dict, item_name: str, price: str) -> str:
        """当 API 不可用时生成简单模板"""
        item = gemini_result.get("item_details", {}) if gemini_result else {}
        brand = item.get("brand", "")
        model = item.get("model_name", "")
        condition = item.get("condition", "几乎全新")

        template = f"""📋 【闲鱼发布模板】

📝 标题：
【{condition}】{brand} {model if model else item_name} 正品保证

💰 价格：{price}

📖 描述：
✨ {brand} {model if model else item_name}
📏 成色：{condition}
✅ 正品保证
✅ 顺丰包邮

🏷️ 标签：#{brand.replace(' ', '') if brand else '二手闲置'} #闲置转让

📕 【小红书发布模板】

标题：断舍离 | {brand} {model if model else item_name} 寻找新主人

正文：
姐妹们！这件{brand} {model if model else item_name}要出啦～

✨ 基本信息：
• 品牌：{brand}
• 成色：{condition}
• 价格：{price}

💕 使用感受：
这件衣服质感超好，因为闲置了一段时间，决定断舍离给它找个新主人～

📦 交易方式：走平台（闲鱼）

喜欢的姐妹私我哦！💌

#断舍离 #{brand.replace(' ', '') if brand else '闲置'} #闲置转让
"""
        return template

    def publish_to_poshmark(self, item_image_path: str = None, auto_submit: bool = False) -> str:
        """
        自动发布到 Poshmark（Agent 工具）

        Args:
            item_image_path: 图片路径，默认使用 last_image_path
            auto_submit: 是否自动提交（Demo 建议 False）

        Returns:
            发布结果消息
        """
        # 使用默认图片路径
        image_path = item_image_path or self.last_image_path
        if not image_path or not os.path.exists(image_path):
            return "抱歉主人～小镜找不到要发布的图片，请先上传衣服照片哦！"

        # 需要 Gemini 结果来生成文案
        if not hasattr(self, 'last_gemini_result') or not self.last_gemini_result:
            return "主人，小镜需要先分析这件衣服的信息才能发布哦～请先让小镜分析一下！"

        try:
            # 导入 bot 模块
            from tools.poshmark_bot import auto_publish_from_gemini_result

            print(f"[Agent] 开始自动发布到 Poshmark: {image_path}")

            # 调用自动化脚本
            result = auto_publish_from_gemini_result(
                image_path=image_path,
                gemini_result=self.last_gemini_result,
                headless=False,  # Demo 时显示浏览器
                auto_submit=auto_submit
            )

            if result.get("success"):
                if result.get("status") == "form_filled":
                    return """✨ 主人～小镜已经帮您打开发布页面啦！

📋 已自动填写：
• 标题和描述（基于 Gemini 分析生成）
• 分类和品牌信息
• 价格（原价 vs 二手价）
• 图片已上传

🖱️ 请主人在浏览器里确认一下信息，没问题就可以点击发布啦！

💡 小镜提示：这是 Poshmark 平台，面向海外买家，用美元结算哦～"""
                else:
                    return f"✅ 发布成功！{result.get('message')}"
            else:
                if result.get("status") == "login_required":
                    return """🔐 主人，小镜检测到您需要登录 Poshmark～

请在新打开的浏览器窗口中：
1. 点击 "Log In" 或 "Sign Up"
2. 使用邮箱/Apple/Google 账号登录
3. 登录成功后小镜会自动继续填写

（只需要登录一次，之后小镜会记住您的登录状态）"""
                else:
                    return f"抱歉主人，发布遇到了问题：{result.get('message')}"

        except ImportError as e:
            return f"小镜的 Poshmark 助手模块还没准备好呢：{e}"
        except Exception as e:
            return f"发布时出错了：{str(e)}"

    def get_reference_sources(self) -> str:
        """
        获取上次分析的参考链接，用于回答用户追问

        Returns:
            格式化的参考来源文本
        """
        if not self.last_reference_sources:
            return "小镜这次没有找到具体的参考链接呢～价格是基于 Gemini 的实时搜索数据分析得出的。"

        lines = ["小镜为您找到了以下参考来源："]
        for i, source in enumerate(self.last_reference_sources[:5], 1):
            title = source.get('title', '未知来源')[:50]
            url = source.get('url', '')
            domain = source.get('domain', '')

            if url:
                lines.append(f"{i}. {title}")
                lines.append(f"   链接: {url}")
            elif domain:
                lines.append(f"{i}. {title} ({domain})")
            else:
                lines.append(f"{i}. {title}")

        lines.append("\n主人可以点击链接查看详情哦～")
        return "\n".join(lines)

    def reset_conversation(self):
        """Reset conversation history (keeping system prompt)."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.last_reference_sources = []  # 同时清空参考链接
        self.last_gemini_result = None  # 清空分析结果
        self.last_image_path = None  # 清空图片路径

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get current conversation history (for UI display)."""
        import re
        display_history = []
        for msg in self.messages:
            if msg["role"] == "system":
                continue
            if msg["role"] == "tool":
                continue  # Skip tool results

            content = msg["content"]
            display_text = ""
            has_image = False

            if isinstance(content, str):
                display_text = content
            elif isinstance(content, list):
                # 遍历 list 提取文字和图片
                for item in content:
                    if item.get("type") == "text":
                        display_text += item.get("text", "")
                    elif item.get("type") == "image_url":
                        has_image = True

            # 过滤掉 [系统记录：...] 隐藏信息
            display_text = re.sub(r'\[系统记录：.*?\]', '', display_text, flags=re.DOTALL).strip()

            # 如果有图片，在末尾加上标识
            if has_image:
                display_text += "\n[图片]"

            if display_text.strip():
                display_history.append({
                    "role": msg["role"],
                    "content": display_text.strip()
                })

        return display_history


# Convenience functions for quick testing
def create_agent(api_key: str = None) -> MirrorAgent:
    """Create a new agent instance."""
    return MirrorAgent(api_key=api_key)


def demo_chat():
    """Quick demo of the agent."""
    agent = create_agent()

    # Simulate tool handlers
    def mock_segment(image_path: str):
        return {"status": "success", "items": ["上衣", "裤子"], "count": 2}

    def mock_check(item_id: str):
        return {"stagnant": True, "days": 400, "suggested_action": "sell"}

    agent.register_tool("segment_clothes", mock_segment)
    agent.register_tool("check_wardrobe_stagnancy", mock_check)

    # Test conversation
    print("主人：我上传了一张穿搭照片")
    response = agent.chat("帮我看看这身搭配", image_path=None)
    print(f"小镜：{response}\n")


if __name__ == "__main__":
    demo_chat()

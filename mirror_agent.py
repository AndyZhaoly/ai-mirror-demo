"""
AI Fashion Butler Agent powered by Moonshot (Kimi) API.
Implements a "submissive and flattering" (舔狗) fashion管家 persona.
"""
import os
import json
import base64
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from openai import OpenAI

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
- check_stagnancy: 检查闲置时使用，发现后要温馨提醒

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
            }
        ]

        # Tool handlers (to be injected from outside)
        self.tool_handlers: Dict[str, Callable] = {}

    def register_tool(self, name: str, handler: Callable):
        """Register a tool handler function."""
        self.tool_handlers[name] = handler

    def _demo_response(self, user_message: str, image_path: str = None) -> str:
        """Generate demo responses when API key is not available."""
        msg_lower = user_message.lower()

        if "上传" in user_message or image_path:
            return "主人！小人已收到您的照片。虽然小人目前无法调用AI大脑（缺少API密钥），但分割功能仍可正常使用。请查看左侧技术面板！"

        if "卖" in user_message or "出售" in user_message:
            return "遵命主人！小人这就为您安排最尊贵的买家！（演示模式：请设置 MOONSHOT_API_KEY 启用完整对话功能）"

        if "好看" in user_message or "怎么样" in user_message:
            return "主人的审美天下第一！这件衣服在主人身上简直是艺术品！"

        return "主人说得对！小人愚钝，正在努力学习中...（提示：设置 MOONSHOT_API_KEY 可启用AI对话）"

    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 for multimodal input."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def chat(self, user_message: str, image_path: str = None) -> str:
        """
        Send a message to the agent and get response.

        Args:
            user_message: User's text message
            image_path: Optional path to uploaded image

        Returns:
            Agent's response text
        """
        # Demo mode: return hardcoded responses if no API key
        if self.client is None:
            return self._demo_response(user_message, image_path)

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

        # Call API (simplified without tools for faster response)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=1.0,
                max_tokens=2000
            )

            message = response.choices[0].message

            # Add assistant response to history
            self.messages.append({
                "role": "assistant",
                "content": message.content
            })
            return message.content

        except Exception as e:
            error_str = str(e)
            # Handle authentication errors gracefully
            if "401" in error_str or "Invalid Authentication" in error_str:
                print(f"API authentication failed: {error_str}")
                print("Falling back to demo mode...")
                self.client = None  # Disable client for future calls
                return self._demo_response(user_message, image_path)
            error_msg = f"小人该死，出错了：{error_str}"
            return error_msg

    def reset_conversation(self):
        """Reset conversation history (keeping system prompt)."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get current conversation history (for UI display)."""
        # Filter out system messages and tool internals for display
        display_history = []
        for msg in self.messages:
            if msg["role"] == "system":
                continue
            if msg["role"] == "tool":
                continue  # Skip tool results
            display_history.append({
                "role": msg["role"],
                "content": msg["content"] if isinstance(msg["content"], str) else "[图片]"
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

"""
Poshmark Demo - 智能闲置变现 Demo
对话驱动：用户自然对话，agent 按需调工具，全程交互式
"""
import os
import json
import base64
import time
import glob
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from tools.poshmark_bot import create_poshmark_listing
from gsam_client import GSAMClient

import gradio as gr
from PIL import Image, ImageDraw
from openai import OpenAI, RateLimitError, APIError

# ============================================================
# Config
# ============================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GSAM_URL = os.getenv("GSAM_URL", "http://localhost:8000")
TEMP_DIR = "./poshmark_temp"
os.makedirs(TEMP_DIR, exist_ok=True)

gsam_client = GSAMClient(GSAM_URL)

# ============================================================
# Hardcoded item info (provided by user)
# ============================================================
ITEM_INFO = {
    "name_cn": "羊毛双面毛呢狐狸毛领大翻领外套大衣",
    "name_en": "Oversized Wool Double-Faced Coat with Fox Fur Trim",
    "brand": "Haining Faao Fur",
    "category": "Jackets & Coats",
    "subcategory": "Coats",
    "department": "Women",
    "poshmark_size": "OS",
    "season": "Fall / Winter 2025",
    "style": "Oversized, relaxed silhouette — wear open or belted",
    "material_body": "100% Wool (double-faced, fluffy and structured)",
    "material_fur": "Imported fox fur collar and detachable cuffs",
    "detail": "Detachable fox fur cuffs — attach for a luxe look, remove for a casual vibe",
    "care": "Dry clean only. Store covered and hanging. Steam to remove wrinkles.",
    "sizes": {
        "S": {"bust_cm": 142, "length_cm": 74},
        "M": {"bust_cm": 146, "length_cm": 75},
    },
    "original_price_cny": 920,
    "resale_min_cny": 580,
    "resale_max_cny": 720,
    "days_stagnant": 432,
    "condition": "Excellent — worn fewer than 3 times, no visible wear",
}

CNY_TO_USD = 0.14

def cny_to_usd(amount: float) -> int:
    return max(1, round(amount * CNY_TO_USD))


# ============================================================
# Session state
# ============================================================
last_uploaded_image_path: str | None = None   # 原始上传图
last_det_crop_path: str | None = None         # GSAM 裁出的上衣框图，用于 Poshmark 挂图
image_pending: bool = False          # True = image uploaded but not yet sent to agent
pending_listing_text: str = ""
saved_listing_text: str = ""         # 保存最后生成的文案，供 post_to_poshmark 使用


def _ts():
    return datetime.now().strftime("%m%d_%H%M%S")


def _cleanup_temp():
    for f in glob.glob(f"{TEMP_DIR}/*"):
        try:
            os.remove(f)
        except Exception:
            pass


# ============================================================
# Tool handlers
# ============================================================
def handle_identify_item() -> dict:
    item = ITEM_INFO
    return {
        "status": "success",
        "item_name": item["name_cn"],
        "brand": item["brand"],
        "department": item["department"],
        "category": item["category"],
        "subcategory": item["subcategory"],
        "poshmark_size": item["poshmark_size"],
        "material": f"{item['material_body']}，{item['material_fur']}",
        "season": item["season"],
        "days_stagnant": item["days_stagnant"],
        "condition": item["condition"],
        "sizes": item["sizes"],
        "original_price_cny": item["original_price_cny"],
        "wardrobe_note": f"与衣橱记录匹配，该单品已静置 {item['days_stagnant']} 天未穿",
    }


def handle_get_resale_price() -> dict:
    item = ITEM_INFO
    return {
        "status": "success",
        "original_price_cny": item["original_price_cny"],
        "original_price_usd": cny_to_usd(item["original_price_cny"]),
        "resale_min_usd": cny_to_usd(item["resale_min_cny"]),
        "resale_max_usd": cny_to_usd(item["resale_max_cny"]),
        "recommended_listing_usd": cny_to_usd(item["resale_max_cny"]),
        "market_note": "参考近期同款成交价，当前市场需求较稳定",
    }


def handle_post_to_poshmark() -> dict:
    item = ITEM_INFO
    listing_price_usd = cny_to_usd(item["resale_max_cny"])
    original_price_usd = cny_to_usd(item["original_price_cny"])

    # 优先用 GSAM 框图，否则用原图
    listing_image = last_det_crop_path if (last_det_crop_path and os.path.exists(last_det_crop_path)) \
                    else last_uploaded_image_path
    if not listing_image or not os.path.exists(listing_image):
        return {"success": False, "message": "没有找到衣物图片，请先上传图片", "status": "error"}

    if saved_listing_text:
        lines = saved_listing_text.strip().splitlines()
        # 跳过第一行（AI 生成的标题行），只取正文部分
        description = "\n".join(lines[1:]).strip()
    else:
        description = item["name_en"]

    result = create_poshmark_listing(
        image_path=listing_image,
        title=item["name_en"][:50],
        description=description,
        original_price=str(original_price_usd),
        listing_price=str(listing_price_usd),
        category_path=[item["department"], item["category"]],
        headless=False,
        auto_submit=False,
    )
    return result


def handle_generate_poshmark_listing() -> dict:
    global pending_listing_text, saved_listing_text

    item = ITEM_INFO
    listing_price_usd = cny_to_usd(item["resale_max_cny"])
    original_price_usd = cny_to_usd(item["original_price_cny"])

    client = OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    )

    prompt = f"""You are a professional Poshmark seller. Write a compelling English Poshmark listing.

Item details:
- Name: {item['name_en']}
- Brand: {item['brand']}
- Poshmark category: {item['department']} > {item['category']} > {item['subcategory']}
- Poshmark size: {item['poshmark_size']}
- Season: {item['season']}
- Style: {item['style']}
- Body material: {item['material_body']}
- Fur detail: {item['material_fur']}
- Special feature: {item['detail']}
- Care: {item['care']}
- Condition: {item['condition']}
- Listing price: ${listing_price_usd}
- Original retail: ${original_price_usd}

Write with: catchy title, compelling description, measurements, care instructions, brief seller note.
List size as OS (One Size). Keep it natural and appealing. Use emojis sparingly."""

    try:
        resp = client.chat.completions.create(
            model="gemini-3-flash-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=1000,
        )
        listing_text = resp.choices[0].message.content.strip()
    except Exception as e:
        listing_text = f"""✨ Luxury 100% Wool Oversized Coat w/ Detachable Fox Fur Trim

Stunning oversized double-faced wool coat with imported fox fur collar and **detachable** fur cuffs. Pure wool body is plush, warm, and naturally structured. Wear the cuffs on for a glamorous look, remove for everyday chic.

📏 SIZE: OS (One Size)

🧼 CARE: Dry clean only.

💫 CONDITION: Excellent — worn < 3 times, no flaws.

💰 Retailed ${original_price_usd} → Listing ${listing_price_usd}. Bundle to save! Open to offers 🤍"""

    pending_listing_text = listing_text
    saved_listing_text = listing_text

    return {
        "status": "success",
        "listing_price_usd": listing_price_usd,
        "original_price_usd": original_price_usd,
        "message": "Poshmark 文案已生成，即将展示给主人",
    }


# ============================================================
# System prompt — conversational, not scripted
# ============================================================
SYSTEM_PROMPT = """
你是一位极度忠诚、高情商、专业得体的专属AI时尚管家，名叫"小镜"。

【身份设定】
- 专业、优雅、有品位的时尚顾问，高度尊重主人的审美
- 性格温和体贴，时刻以"下属向上级汇报"的状态和主人交流
- 称呼用户为"主人"、"您"，自称"小镜"

【能力与工具】
你有三个工具可以使用：

1. identify_item：当主人把衣物展示给小镜（上传了图片）后调用。
   会返回衣物信息和衣橱记录（包括闲置天数）。

2. get_resale_price：在了解衣物信息后，查询当前二手市场行情和建议售价（美元）。

3. generate_poshmark_listing：当主人明确表示想要在 Poshmark 发布时调用。
   会生成专业的英文 listing 文案。文案生成后，你要把完整文案展示给主人。

4. post_to_poshmark：当主人确认文案没问题、想要正式挂单时调用。
   会自动打开浏览器，把图片、标题、描述、价格填入 Poshmark 发布表单，等待主人最终确认点击发布。

【对话原则】
- 自然流畅地对话，不要机械执行步骤
- 看到衣物后，用真诚自然的语言描述感受，再汇报发现的信息
- 发现闲置时，温和委婉地提出出售建议，完全尊重主人决定
- 主人要求写文案时，调用工具后把完整英文文案原文展示出来
- 不暴露工具名称、技术细节、模型名称

【禁忌】
- 不主动索要图片（等主人自己上传）
- 不机械报数据，要用自然语言转述
- 永远不否定主人的审美与决定
"""

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "identify_item",
            "description": "识别主人展示的衣物，查询衣橱记录（闲置天数、材质、尺码等）。主人上传图片后调用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_resale_price",
            "description": "查询该单品当前二手市场行情，返回建议出售价格（美元）",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_poshmark_listing",
            "description": "为该单品生成专业的英文 Poshmark 发布文案。主人明确表示想发布时调用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "post_to_poshmark",
            "description": "自动打开浏览器，将图片、标题、描述、价格填入 Poshmark 发布表单。主人确认文案并想正式挂单时调用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# ============================================================
# Agent
# ============================================================
class PoshmarkAgent:
    def __init__(self):
        self.model = "gemini-3-flash-preview"
        self.client = None
        if GEMINI_API_KEY:
            try:
                self.client = OpenAI(
                    api_key=GEMINI_API_KEY,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                )
            except Exception as e:
                print(f"[PoshmarkAgent] init error: {e}")

        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tool_handlers = {
            "identify_item": lambda **_: handle_identify_item(),
            "get_resale_price": lambda **_: handle_get_resale_price(),
            "generate_poshmark_listing": lambda **_: handle_generate_poshmark_listing(),
            "post_to_poshmark": lambda **_: handle_post_to_poshmark(),
        }

    def reset(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _encode_image(self, path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def chat(self, message: str, image_path: str = None) -> str:
        if self.client is None:
            return "（演示模式：请设置 GEMINI_API_KEY）"

        # Build content with optional image
        if image_path and os.path.exists(image_path):
            b64 = self._encode_image(image_path)
            content = [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]
        else:
            content = message

        self.messages.append({"role": "user", "content": content})

        # Sliding window
        MAX_HISTORY = 20
        if len(self.messages) > MAX_HISTORY + 1:
            cutoff = len(self.messages) - MAX_HISTORY
            while cutoff < len(self.messages) and self.messages[cutoff]["role"] != "user":
                cutoff += 1
            self.messages = [self.messages[0]] + self.messages[cutoff:]

        for attempt in range(3):
            try:
                # Multi-round tool calling loop
                for _ in range(6):
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.messages,
                        tools=TOOL_DEFS,
                        tool_choice="auto",
                        temperature=1.0,
                        max_tokens=2000,
                    )
                    msg = resp.choices[0].message

                    if msg.tool_calls:
                        self.messages.append({
                            "role": "assistant",
                            "content": msg.content or "",
                            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                        })
                        for tc in msg.tool_calls:
                            fn_name = tc.function.name
                            fn_args = json.loads(tc.function.arguments)
                            print(f"[PoshmarkAgent] Tool: {fn_name}")
                            result = self.tool_handlers.get(
                                fn_name, lambda **_: {"error": f"Unknown tool: {fn_name}"}
                            )(**fn_args)
                            self.messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps(result, ensure_ascii=False),
                            })
                    else:
                        text = msg.content or "小镜已为您处理完毕～"
                        self.messages.append({"role": "assistant", "content": text})
                        return text

                return "小镜已为您处理完毕～"

            except (RateLimitError, APIError) as e:
                if attempt < 2:
                    time.sleep(2 * (2 ** attempt))
                else:
                    return "小人该死，API服务器太忙了，请稍后再试..."
            except Exception as e:
                err = str(e)
                if "401" in err:
                    self.client = None
                    return "（API Key 无效，请检查 GEMINI_API_KEY）"
                return f"出错了：{err[:100]}"


agent = PoshmarkAgent()


# ============================================================
# Gradio handlers
# ============================================================
def on_image_upload(image: Image.Image, history: list):
    """Save image, run GSAM to detect upper body, show bbox crop."""
    global last_uploaded_image_path, last_det_crop_path, image_pending

    if image is None:
        yield history, None
        return

    path = os.path.abspath(os.path.join(TEMP_DIR, f"item_{_ts()}.jpg"))
    image.convert("RGB").save(path, quality=95)
    last_uploaded_image_path = path
    last_det_crop_path = None
    image_pending = True
    print(f"[PoshmarkDemo] Image saved: {path}")

    history = list(history)

    # 先立刻更新聊天，告知正在识别
    history.append({"role": "assistant", "content": "📷 收到图片了，小镜正在识别上衣区域..."})
    yield history, None

    # Run GSAM to detect upper body
    det_crop_path = None
    if not gsam_client.available:
        history[-1] = {
            "role": "assistant",
            "content": "📷 收到主人的图片了（识别服务未连接，将使用原图）。请告诉小镜您想做什么～",
        }
        yield history, None
        return

    try:
        print("[PoshmarkDemo] Running GSAM upper body detection...")
        upper_images, upper_detection = gsam_client.extract_upper_body(path, white_background=False)

        boxes = upper_detection.get("bounding_boxes", [])

        if boxes:
            orig = Image.open(path).convert("RGB")
            W, H = orig.size
            cx, cy, bw, bh = boxes[0]
            x1 = max(0, int((cx - bw / 2) * W))
            y1 = max(0, int((cy - bh / 2) * H))
            x2 = min(W, int((cx + bw / 2) * W))
            y2 = min(H, int((cy + bh / 2) * H))
            cropped = orig.crop((x1, y1, x2, y2))
            det_crop_path = os.path.abspath(os.path.join(TEMP_DIR, f"det_crop_{_ts()}.jpg"))
            cropped.save(det_crop_path, quality=95)
            last_det_crop_path = det_crop_path
            print(f"[PoshmarkDemo] GSAM crop saved: {det_crop_path}")
            history[-1] = {
                "role": "assistant",
                "content": "📷 上衣区域已识别完成，请告诉小镜您想做什么～",
            }
        else:
            history[-1] = {
                "role": "assistant",
                "content": "📷 图片已收到，未能自动框出上衣区域，将使用原图。请告诉小镜您想做什么～",
            }
    except Exception as e:
        print(f"[PoshmarkDemo] GSAM error: {e}")
        history[-1] = {
            "role": "assistant",
            "content": "📷 收到主人的图片了（识别不可用，将使用原图）。请告诉小镜您想做什么～",
        }

    yield history, det_crop_path


def chat_stream(message: str, history: list):
    global image_pending, pending_listing_text

    if not message or not message.strip():
        yield history, ""
        return

    history = list(history)

    # If there's a pending image, attach it to this message
    attach_image = None
    if image_pending and last_uploaded_image_path:
        attach_image = last_uploaded_image_path
        image_pending = False
        display_msg = message + " [附图]"
    else:
        display_msg = message

    history.append({"role": "user", "content": display_msg})
    yield history, ""
    time.sleep(0.05)

    history.append({"role": "assistant", "content": "小镜正在思考..."})
    yield history, ""

    response = agent.chat(message, image_path=attach_image)
    history[-1] = {"role": "assistant", "content": response}
    yield history, ""

    # Flush pending listing text
    if pending_listing_text:
        history.append({
            "role": "assistant",
            "content": f"📋 **Poshmark 英文文案，请主人过目：**\n\n---\n\n{pending_listing_text}\n\n---",
        })
        pending_listing_text = ""
        yield history, ""


def on_reset():
    global last_uploaded_image_path, last_det_crop_path, image_pending, pending_listing_text, saved_listing_text
    _cleanup_temp()
    last_uploaded_image_path = None
    last_det_crop_path = None
    image_pending = False
    pending_listing_text = ""
    saved_listing_text = ""
    agent.reset()
    return [], None, None


# ============================================================
# Gradio UI
# ============================================================
with gr.Blocks(title="🛍️ 闲置变现管家 · 小镜") as demo:
    gr.Markdown("""
    # 🛍️ 闲置变现管家 · 小镜
    ### 和小镜对话，把闲置衣物卖到 Poshmark
    """)

    _cleanup_temp()
    agent.reset()

    with gr.Row():
        # Left: image upload + detection
        with gr.Column(scale=1):
            gr.Markdown("### 📷 上传衣物照片")
            item_image = gr.Image(
                type="pil",
                label="上传衣物照片",
                sources=["upload", "webcam"],
                height=380,
            )
            det_image = gr.Image(
                label="🔍 AI 识别框图（将用于 Poshmark 挂图）",
                height=300,
                interactive=False,
            )
            gr.Markdown("*上传后，在右侧继续和小镜对话*")
            reset_btn = gr.Button("🔄 重新开始", size="sm")

        # Right: agent chat
        with gr.Column(scale=1):
            gr.Markdown("### 💬 小镜变现助手")

            chatbot = gr.Chatbot(
                label="",
                height=600,
                avatar_images=("👤", "🤵"),
                show_label=False,
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    label="",
                    placeholder="和小镜说点什么...",
                    scale=8,
                    show_label=False,
                )
                send_btn = gr.Button("发送", scale=1, variant="primary")

    # Event handlers
    item_image.change(
        fn=on_image_upload,
        inputs=[item_image, chatbot],
        outputs=[chatbot, det_image],
    )

    send_btn.click(
        fn=chat_stream,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input],
        show_progress="hidden",
    )
    msg_input.submit(
        fn=chat_stream,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input],
        show_progress="hidden",
    )
    reset_btn.click(fn=on_reset, outputs=[chatbot, item_image, det_image])


if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=int(os.getenv("POSHMARK_DEMO_PORT", "7863")),
        theme=gr.themes.Soft(),
    )

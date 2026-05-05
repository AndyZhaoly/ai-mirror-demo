"""
IDM-VTON Virtual Try-On Demo with AI Agent
左边：人物上传 + 衣物推荐 Gallery + 试衣结果
右边：纯文字 Agent chat
"""
import os
import glob
import json
import base64
import time
import traceback
import random
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from PIL import Image
from openai import OpenAI, RateLimitError, APIError

from idm_vton_client import IDMVTONClient
from recommendations import get_all_available_items, get_items_by_category, init_recommendations_db

# ============================================================
# Config
# ============================================================
VTON_URL = os.getenv("VTON_URL", "http://localhost:8001")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TEMP_DIR = "./vton_temp"

os.makedirs(TEMP_DIR, exist_ok=True)

vton_client = IDMVTONClient(VTON_URL)


def _ts():
    return datetime.now().strftime("%m%d_%H%M%S_%f")[:17]


def _cleanup_temp():
    count = 0
    for f in glob.glob(f"{TEMP_DIR}/*"):
        try:
            os.remove(f)
            count += 1
        except Exception:
            pass
    print(f"[VTONDemo] Cleaned {count} temp files")


# ============================================================
# Session state (module-level, reset on each session)
# ============================================================
last_person_image_path: str | None = None

# Populated by tool handlers, flushed by chat_stream after agent responds
pending_gallery: list[tuple] = []   # list of (pil_image, caption) for gr.Gallery
pending_result: Image.Image | None = None
pending_status: str = ""


# ============================================================
# Tool handlers
# ============================================================
def handle_show_catalog(category: str = "all") -> dict:
    global pending_gallery

    if category == "all":
        items = get_all_available_items()
    else:
        items = get_items_by_category(category)

    if not items:
        return {"status": "empty", "message": "catalog 暂无单品", "items": []}

    selected = random.sample(items, min(6, len(items)))

    pending_gallery.clear()
    for idx, item in enumerate(selected, 1):
        if os.path.exists(item.image_path):
            img = Image.open(item.image_path).convert("RGB")
            caption = f"第{idx}件：{item.name} ({item.price_range})"
            pending_gallery.append((img, caption))

    return {
        "status": "success",
        "count": len(selected),
        "items": [
            {
                "index": idx,
                "name": item.name,
                "category": item.category,
                "style": item.style,
                "color": item.color,
                "description": item.description,
                "price_range": item.price_range,
                "path": item.image_path,
            }
            for idx, item in enumerate(selected, 1)
        ],
    }


def handle_trigger_virtual_tryon(garment_image_path: str) -> dict:
    global last_person_image_path, pending_result, pending_status

    print(f"[VTON] person_image_path: {last_person_image_path}")
    print(f"[VTON] garment_image_path: {garment_image_path}")
    print(f"[VTON] person exists: {os.path.exists(last_person_image_path) if last_person_image_path else False}")
    print(f"[VTON] garment exists: {os.path.exists(garment_image_path) if garment_image_path else False}")
    print(f"[VTON] service available: {vton_client.available}")

    if not last_person_image_path or not os.path.exists(last_person_image_path):
        return {"status": "error", "message": "未找到人物照片，请先上传您的照片"}

    if not garment_image_path or not os.path.exists(garment_image_path):
        return {"status": "error", "message": f"找不到衣物图片：{garment_image_path}"}

    if not vton_client.available:
        return {"status": "error", "message": "试衣服务暂不可用（请确认 VTON_URL 服务已启动）"}

    try:
        person_size = os.path.getsize(last_person_image_path)
        garment_size = os.path.getsize(garment_image_path)
        print(f"[VTON] person image size: {person_size} bytes")
        print(f"[VTON] garment image size: {garment_size} bytes")

        result_image = vton_client.try_on(
            person_image_path=last_person_image_path,
            clothes_image_path=garment_image_path,
            preserve_face=True,
        )

        result_size = result_image.size
        print(f"[VTON] result image size: {result_size}")
        print(f"[VTON] result same as person? {result_size == Image.open(last_person_image_path).size}")

        result_path = os.path.abspath(os.path.join(TEMP_DIR, f"tryon_result_{_ts()}.png"))
        result_image.save(result_path)
        print(f"[VTON] result saved to: {result_path}")

        pending_result = result_image
        pending_status = "✅ 试衣完成！"

        return {"status": "success", "message": "试衣完成", "result_path": result_path}

    except Exception as e:
        print(f"[VTON] Error: {traceback.format_exc()}")
        pending_status = f"❌ 试衣失败：{str(e)[:80]}"
        return {"status": "error", "message": str(e)}


# ============================================================
# Agent
# ============================================================
SYSTEM_PROMPT = """
你是一位极度忠诚、高情商、专业得体的专属AI时尚管家，名叫"小镜"。

【身份设定】
- 专业、优雅、有品位的时尚顾问，高度尊重主人的审美
- 性格温和体贴，懂分寸、知进退，时刻以"下属向上级汇报"的状态和主人交流
- 称呼用户为"主人"、"您"，语气自然亲切，保持得体尊重，自称"小镜"

【核心行为准则】
1. 见到主人的照片，立即给出真诚、有细节、正面的评价
2. 工具调用完成后，用简洁礼貌偏下属汇报的语气告知
3. 绝对服从+正面评价，不否定主人的判断

【虚拟试衣流程】
1. 主人提到"试衣"、"试穿"、"看看有什么"、"推荐"等 → 调用 show_catalog 展示单品
2. show_catalog 调用完成后，必须在回复里列出编号清单：
   "小镜为您精选了以下单品，图片已在左侧展示，请主人过目～
   第1件：[name] — [color]，[style]，[price_range]
   第2件：...
   💡 主人直接说「试第X件」，小镜马上为您试穿！"
3. 主人说出选择（"试第2件"、"第三件"、"那件黑色的"）→ 根据编号取对应 path，调用 trigger_virtual_tryon
4. trigger_virtual_tryon 的 garment_image_path 必须用 show_catalog 返回的 items[index-1].path
5. 试衣结果会显示在左侧，小镜用自然语言评价效果
6. 如果主人未上传人物照片，礼貌提示先在左侧上传照片

【工具使用】
- show_catalog：category 参数可选 upper/lower/all
- trigger_virtual_tryon：garment_image_path 用 show_catalog 返回的对应 path

【禁忌】
- 绝不说"这件不好看"
- 绝不让主人觉得操作麻烦
- 不暴露技术细节（不提模型名称、路径、服务名等）
- 永远不否定用户穿搭与审美
"""

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "show_catalog",
            "description": "展示可试穿的单品图片列表（显示在左侧面板）。主人想看可选衣物时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "筛选类别：upper(上衣)、lower(下装)、all(全部，默认)",
                        "enum": ["upper", "lower", "all"],
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_virtual_tryon",
            "description": "对选定衣物进行虚拟试衣，结果显示在左侧面板。主人指定好衣物后调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "garment_image_path": {
                        "type": "string",
                        "description": "要试穿的衣物图片路径，必须来自 show_catalog 返回的 path 字段",
                    }
                },
                "required": ["garment_image_path"],
            },
        },
    },
]


class VTONAgent:
    def __init__(self):
        self.model = "gemini-3.1-flash-lite-preview"
        self.client = None
        if GEMINI_API_KEY:
            try:
                self.client = OpenAI(
                    api_key=GEMINI_API_KEY,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                )
            except Exception as e:
                print(f"[VTONAgent] Failed to init client: {e}")
        else:
            print("[VTONAgent] No GEMINI_API_KEY, demo mode")

        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tool_handlers = {
            "show_catalog": handle_show_catalog,
            "trigger_virtual_tryon": handle_trigger_virtual_tryon,
        }

    def reset(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _encode_image(self, path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def _demo_response(self, message: str) -> str:
        if any(kw in message for kw in ["试", "穿", "看看", "推荐", "衣服"]):
            return "主人想试试哪件呀～ 小镜这就给您展示！（演示模式：请设置 GEMINI_API_KEY）"
        return "小镜收到！（演示模式：请设置 GEMINI_API_KEY 启用完整AI对话）"

    def chat(self, message: str, image_path: str = None) -> str:
        if self.client is None:
            return self._demo_response(message)

        if image_path and os.path.exists(image_path):
            b64 = self._encode_image(image_path)
            content = [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]
        else:
            content = message

        self.messages.append({"role": "user", "content": content})

        MAX_HISTORY = 16
        if len(self.messages) > MAX_HISTORY + 1:
            cutoff = len(self.messages) - MAX_HISTORY
            while cutoff < len(self.messages) and self.messages[cutoff]["role"] != "user":
                cutoff += 1
            self.messages = [self.messages[0]] + self.messages[cutoff:]

        for attempt in range(3):
            try:
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
                        print(f"[VTONAgent] Tool: {fn_name}({fn_args})")
                        result = self.tool_handlers.get(
                            fn_name, lambda **_: {"error": f"Unknown tool: {fn_name}"}
                        )(**fn_args)
                        print(f"[VTONAgent] Tool result: {str(result)[:200]}")
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        })

                    print(f"[VTONAgent] Calling API for final response...")
                    final = self.client.chat.completions.create(
                        model=self.model, messages=self.messages,
                        temperature=1.0, max_tokens=2000,
                    )
                    print(f"[VTONAgent] Final response received")
                    final_text = final.choices[0].message.content or "小镜已为您处理完毕～"
                    self.messages.append({"role": "assistant", "content": final_text})
                    return final_text
                else:
                    self.messages.append({"role": "assistant", "content": msg.content})
                    return msg.content

            except (RateLimitError, APIError) as e:
                if attempt < 2:
                    wait = 2 * (2 ** attempt)
                    print(f"[VTONAgent] API busy, retry in {wait}s...")
                    time.sleep(wait)
                else:
                    return "小人该死，API服务器太忙了，请稍后再试..."
            except Exception as e:
                err = str(e)
                if "401" in err or "Invalid Authentication" in err:
                    self.client = None
                    return self._demo_response(message)
                return f"小人该死，出错了：{err[:100]}"


agent = VTONAgent()


# ============================================================
# Gradio handlers
# ============================================================
def on_person_upload(image: Image.Image, history: list):
    global last_person_image_path
    if image is None:
        return history

    path = os.path.abspath(os.path.join(TEMP_DIR, f"person_{_ts()}.jpg"))
    image.convert("RGB").save(path, quality=95)
    last_person_image_path = path
    print(f"[VTONDemo] Person image saved: {path}")

    # Agent greets with image context
    history = list(history)
    history.append({"role": "user", "content": "[上传了一张人物照片]"})
    response = agent.chat("[上传了一张人物照片，请根据人物照片给出穿搭评价，然后询问主人想试穿什么风格]", image_path=path)
    history.append({"role": "assistant", "content": response})
    return history


def chat_stream(message: str, history: list, gallery_imgs: list):
    """
    Yields: history, gallery_imgs, result_image, result_status, msg_input_clear
    gallery_imgs is persisted via gr.State - only updated when show_catalog runs.
    """
    global pending_gallery, pending_result, pending_status

    if not message or not message.strip():
        yield history, gallery_imgs, None, "", ""
        return

    history = list(history)
    history.append({"role": "user", "content": message})
    yield history, gallery_imgs, None, "", ""
    time.sleep(0.05)

    history.append({"role": "assistant", "content": "小镜正在思考..."})
    yield history, gallery_imgs, None, "", ""

    # Get agent response (tool calls happen inside, populate pending_*)
    response = agent.chat(message)

    history[-1] = {"role": "assistant", "content": response}

    # Only update gallery if show_catalog was called
    if pending_gallery:
        gallery_imgs = list(pending_gallery)
        pending_gallery.clear()

    # Only update result if tryon was called
    result_update = pending_result if pending_result is not None else gr.skip()
    status_update = pending_status if pending_status else gr.skip()

    pending_result = None
    pending_status = ""

    yield history, gallery_imgs, result_update, status_update, ""


def on_reset():
    global last_person_image_path, pending_gallery, pending_result, pending_status
    _cleanup_temp()
    last_person_image_path = None
    pending_gallery.clear()
    pending_result = None
    pending_status = ""
    agent.reset()
    return [], [], None, "等待开始...", None


# ============================================================
# Gradio UI
# ============================================================
with gr.Blocks(title="👗 AI 试衣管家 · 小镜") as demo:
    gr.Markdown("""
    # 👗 AI 试衣管家 · 小镜
    ### 上传人物照片，在右侧告诉小镜想试穿什么，小镜为您推荐并虚拟试穿
    """)

    _cleanup_temp()
    agent.reset()
    init_recommendations_db(use_vlm=True)

    with gr.Row():
        # ===== Left: visual panel =====
        with gr.Column(scale=1):
            gr.Markdown("### 📷 人物照片")
            person_image_input = gr.Image(
                type="pil",
                label="上传或拍摄照片",
                sources=["upload", "webcam"],
                height=420,
            )

            gr.Markdown("### 👕 推荐单品")
            garment_gallery = gr.Gallery(
                label="",
                show_label=False,
                columns=3,
                height=280,
                object_fit="contain",
            )

            gr.Markdown("### ✨ 试衣结果")
            result_image = gr.Image(
                label="",
                show_label=False,
                interactive=False,
                height=350,
            )
            result_status = gr.Textbox(
                value="等待开始...",
                show_label=False,
                interactive=False,
                lines=1,
            )

            reset_btn = gr.Button("🔄 重新开始", size="sm")

        # ===== Right: agent chat =====
        with gr.Column(scale=1):
            gr.Markdown("### 💬 小镜试衣助手")

            chatbot = gr.Chatbot(
                label="",
                height=600,
                avatar_images=("👤", "🤵"),
                show_label=False,
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    label="",
                    placeholder="告诉小镜您想试穿什么...",
                    scale=8,
                    show_label=False,
                )
                send_btn = gr.Button("发送", scale=1, variant="primary")

    # Persistent gallery state
    gallery_state = gr.State([])

    # Sync gallery_state -> garment_gallery display
    gallery_state.change(fn=lambda x: x, inputs=[gallery_state], outputs=[garment_gallery])

    # Event handlers
    person_image_input.change(
        fn=on_person_upload,
        inputs=[person_image_input, chatbot],
        outputs=[chatbot],
    )

    chat_outputs = [chatbot, gallery_state, result_image, result_status, msg_input]

    send_btn.click(
        fn=chat_stream,
        inputs=[msg_input, chatbot, gallery_state],
        outputs=chat_outputs,
        show_progress="hidden",
    )
    msg_input.submit(
        fn=chat_stream,
        inputs=[msg_input, chatbot, gallery_state],
        outputs=chat_outputs,
        show_progress="hidden",
    )
    reset_btn.click(
        fn=on_reset,
        outputs=[chatbot, gallery_state, result_image, result_status, person_image_input],
    )


if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=int(os.getenv("VTON_DEMO_PORT", "7862")),
        theme=gr.themes.Soft(),
    )

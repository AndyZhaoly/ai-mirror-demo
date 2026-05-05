"""
Virtual Try-On Combined Demo
自拍上传 → GSAM 分割 → Agent 推荐裤子 → IDM-VTON 试衣 → 纳入衣柜
"""
import os
import glob
import json
import base64
import time
import traceback
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from PIL import Image
from openai import OpenAI, RateLimitError, APIError

from gsam_client import GSAMClient
from idm_vton_client import IDMVTONClient
from database_manager import add_item
from recommendations import ClothingItem, analyze_clothing_image

# ============================================================
# Config
# ============================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GSAM_URL = os.getenv("GSAM_URL", "http://localhost:8000")
VTON_URL = os.getenv("VTON_URL", "http://localhost:8001")

DEMO_GARMENTS_DIR = os.path.abspath("./demo_garments")
EXTRACTED_DIR = "./extracted_clothes"
TEMP_DIR = "./vton_temp"
os.makedirs(EXTRACTED_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

gsam_client = GSAMClient(GSAM_URL)
vton_client = IDMVTONClient(VTON_URL)

MOGAS_SHIRT_DESC = """Mogas春日粉衬衫 精梳棉立体造型绑带围裹收腰上衣外套。
围裹系带捏褶袖型金属圆环袖口衬衫。
设计亮点：DU特捏褶立裁袖型，袖子采用立体捏褶工艺，通过精准褶量分布打造自然膨润的造型感，告别扁平袖型的单调，抬手间尽显设计巧思。
结构感：设计感侧边系带，将前后的飘带于前侧打结，INS风的时髦穿法，交叠出层次感美学。
五金细节：袖口搭载金属气眼与粗犷椭圆形开合扣，亮光金属质感与衬衫廓形形成材质对比，时髦度拉满却不张扬。
面料品质：高品质精梳棉衬衫面料，自带自然挺括感，上身有型不软塌，日常穿着不易起皱。"""


# ============================================================
# Pre-analyze demo_garments at startup
# ============================================================
DEMO_ITEMS: list[ClothingItem] = []

def init_demo_garments():
    global DEMO_ITEMS
    DEMO_ITEMS = []

    cache_file = os.path.join(DEMO_GARMENTS_DIR, "_analysis_cache.json")
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            pass

    image_files = sorted([
        f for f in os.listdir(DEMO_GARMENTS_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith("_")
    ])

    cache_updated = False
    for i, filename in enumerate(image_files):
        image_path = os.path.abspath(os.path.join(DEMO_GARMENTS_DIR, filename))
        cache_key = f"{filename}_{os.path.getsize(image_path)}"

        if cache_key in cache:
            analysis = cache[cache_key]
            print(f"[DemoGarments] Cache hit: {filename}")
        else:
            print(f"[DemoGarments] Analyzing {filename} with VLM...")
            analysis = analyze_clothing_image(image_path, api_key=GEMINI_API_KEY)
            # 只缓存真正由 VLM 分析成功的结果（有实质性名称）
            if analysis.get("name", "").startswith("时尚单品"):
                print(f"[DemoGarments] Analysis likely fallback, not caching: {filename}")
            else:
                cache[cache_key] = analysis
                cache_updated = True

        item = ClothingItem(
            id=f"demo_{i:02d}",
            name=analysis.get("name", f"单品{i+1}"),
            category=analysis.get("category", "lower"),
            style=analysis.get("style", "casual"),
            color=analysis.get("color", "多色"),
            material=analysis.get("material", "棉质"),
            description=analysis.get("description", ""),
            image_path=image_path,
            price_range="¥199-399",
            brand="Demo Collection",
        )
        DEMO_ITEMS.append(item)

    if cache_updated:
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[DemoGarments] Failed to save cache: {e}")

    print(f"[DemoGarments] Loaded {len(DEMO_ITEMS)} items")


# ============================================================
# Session state
# ============================================================
last_person_image_path: str | None = None
last_upper_seg_path: str | None = None   # 分割出的上衣图，用于展示
pending_gallery: list[tuple] = []
pending_result: Image.Image | None = None
pending_all_results: list[tuple] = []
pending_status: str = ""
last_selected_item: ClothingItem | None = None
# 服务端维护的最新聊天历史（绕过 Gradio 客户端状态快照问题）
_chat_history: list = []


def _ts():
    return datetime.now().strftime("%m%d_%H%M%S_%f")[:17]


def _cleanup_temp():
    for f in glob.glob(f"{TEMP_DIR}/*"):
        try:
            os.remove(f)
        except Exception:
            pass


# ============================================================
# Tool handlers
# ============================================================
def handle_show_recommendations() -> dict:
    global pending_gallery

    if not DEMO_ITEMS:
        return {"status": "empty", "message": "暂无推荐单品", "items": []}

    pending_gallery.clear()
    for idx, item in enumerate(DEMO_ITEMS, 1):
        if os.path.exists(item.image_path):
            img = Image.open(item.image_path).convert("RGB")
            caption = f"第{idx}件：{item.name}｜{item.color}｜{item.style}"
            pending_gallery.append((img, caption))

    return {
        "status": "success",
        "count": len(DEMO_ITEMS),
        "items": [
            {
                "index": idx,
                "name": item.name,
                "category": item.category,
                "style": item.style,
                "color": item.color,
                "material": item.material,
                "description": item.description,
                "price_range": item.price_range,
                "path": item.image_path,
            }
            for idx, item in enumerate(DEMO_ITEMS, 1)
        ],
    }


# --- Chinese-name → simple English keywords for CLIP-friendly prompts --------
_COLOR_EN = {
    "黑": "black", "白": "white", "灰": "gray", "红": "red", "粉": "pink",
    "蓝": "blue", "绿": "green", "黄": "yellow", "棕": "brown", "紫": "purple",
    "米": "beige", "卡其": "khaki", "驼": "camel",
}
_TYPE_EN = [
    # Longer / more specific keys first — iterated in order, first substring match wins.
    ("百褶短裙", "pleated short skirt"),
    ("阔腿休闲裤", "wide-leg casual pants"),
    ("腰带休闲短裤", "belted casual shorts"),
    ("阔腿裤", "wide-leg pants"),
    ("百褶裙", "pleated skirt"),
    ("休闲短裤", "casual shorts"),
    ("牛仔短裤", "denim shorts"),
    ("休闲裤", "casual pants"),
    ("牛仔裤", "jeans"),
    ("西裤", "trousers"),
    ("短裙", "short skirt"),
    ("长裙", "long skirt"),
    ("短裤", "shorts"),
    ("长裤", "long pants"),
    ("裙", "skirt"),
    ("裤", "pants"),
]


def _en_desc(item) -> str:
    """Build a short English garment description from a Chinese ClothingItem."""
    color_en = ""
    for k, v in _COLOR_EN.items():
        if k in (item.color or ""):
            color_en = v
            break
    type_en = ""
    src = (item.name or "") + (item.description or "")
    for k, v in _TYPE_EN:
        if k in src:
            type_en = v
            break
    if not type_en:
        type_en = {"upper": "top", "lower": "pants", "dress": "dress"}.get(item.category, "garment")
    return f"{color_en} {type_en}".strip()


def handle_trigger_virtual_tryon(garment_image_path: str) -> dict:
    global last_person_image_path, pending_result, pending_status, last_selected_item

    if not last_person_image_path or not os.path.exists(last_person_image_path):
        return {"status": "error", "message": "未找到人物照片，请先上传您的自拍"}

    if not garment_image_path or not os.path.exists(garment_image_path):
        return {"status": "error", "message": f"找不到衣物图片：{garment_image_path}"}

    # Record which item was selected
    for item in DEMO_ITEMS:
        if item.image_path == garment_image_path:
            last_selected_item = item
            break

    if not vton_client.available:
        return {"status": "error", "message": "试衣服务暂不可用（请确认 VTON_URL 服务已启动）"}

    try:
        # Map ClothingItem.category → IDM-VTON mask category
        category_map = {"upper": "upper_body", "lower": "lower_body", "dress": "dresses"}
        vton_category = category_map.get(
            last_selected_item.category if last_selected_item else "upper", "upper_body"
        )
        # Keep prompt short & English — mask controls which region is edited,
        # no need to say "keeping X unchanged" (that confuses the cloth CLIP encoder)
        item_desc = _en_desc(last_selected_item) if last_selected_item else "garment"
        prompt = f"a photo of a person wearing {item_desc}"

        result_image = vton_client.try_on(
            person_image_path=last_person_image_path,
            clothes_image_path=garment_image_path,
            prompt=prompt,
            preserve_face=True,
            clothing_category=vton_category,
        )
        result_path = os.path.abspath(os.path.join(TEMP_DIR, f"tryon_{_ts()}.png"))
        result_image.save(result_path)

        pending_result = result_image
        pending_status = "✅ 试衣完成！"

        return {"status": "success", "message": "试衣完成，效果已在左侧展示", "result_path": result_path}

    except Exception as e:
        print(f"[VTON] Error: {traceback.format_exc()}")
        pending_status = f"❌ 试衣失败：{str(e)[:80]}"
        return {"status": "error", "message": str(e)}


def handle_try_all_lower() -> dict:
    """Try on all lower-body items sequentially and store results in pending_all_results."""
    global last_person_image_path, pending_all_results, pending_status

    if not last_person_image_path or not os.path.exists(last_person_image_path):
        pending_status = "❌ 请先上传自拍"
        return {"status": "error", "message": "未找到人物照片，请先上传您的自拍"}

    if not vton_client.available:
        pending_status = "❌ 试衣服务不可用"
        return {"status": "error", "message": "试衣服务暂不可用（请确认 VTON_URL 服务已启动）"}

    lower_items = [item for item in DEMO_ITEMS if item.category == "lower"]
    if not lower_items:
        pending_status = "❌ 暂无下装可试"
        return {"status": "error", "message": "暂无下装可试"}

    pending_all_results.clear()
    completed, failed = [], []

    for i, item in enumerate(lower_items, 1):
        try:
            prompt = f"a photo of a person wearing {_en_desc(item)}"
            result_image = vton_client.try_on(
                person_image_path=last_person_image_path,
                clothes_image_path=item.image_path,
                prompt=prompt,
                preserve_face=True,
                clothing_category="lower_body",
            )
            result_path = os.path.abspath(os.path.join(TEMP_DIR, f"tryon_all_{i}_{_ts()}.png"))
            result_image.save(result_path)
            pending_all_results.append((result_image, f"第{i}件：{item.name}"))
            completed.append(item.name)
        except Exception as e:
            print(f"[VTON] try_all error for {item.name}: {traceback.format_exc()}")
            failed.append(item.name)

    total = len(lower_items)
    if failed and not completed:
        pending_status = f"❌ 三件全试失败（共 {total} 件）"
    elif failed:
        pending_status = f"⚠️ {len(completed)}/{total} 件试穿成功，{len(failed)} 件失败"
    else:
        pending_status = f"✅ 三件全试完成（共 {total} 件）"

    result = {"status": "success", "completed": completed}
    if failed:
        result["failed"] = failed
        result["message"] = f"三件下装试穿完成，效果已在左侧对比展示。（失败：{', '.join(failed)}）"
    else:
        result["message"] = f"三件下装已全部试穿完成，效果已在左侧对比展示，请主人欣赏～"
    return result


def handle_add_to_wardrobe() -> dict:
    global last_selected_item

    if not last_selected_item:
        return {"status": "error", "message": "还没有选择要纳入衣柜的单品"}

    # Map ClothingItem.category → database clothing_type
    wardrobe_type_map = {"upper": "upper", "lower": "lower", "dress": "dress"}
    clothing_type = wardrobe_type_map.get(last_selected_item.category, "lower")

    try:
        item = add_item(
            name=last_selected_item.name,
            clothing_type=clothing_type,
            image_path=last_selected_item.image_path,
        )
        return {
            "status": "success",
            "message": f"已将「{last_selected_item.name}」纳入数字衣柜",
            "item_id": item.get("item_id", ""),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================
# System prompt
# ============================================================
SYSTEM_PROMPT = """
你是一位极度忠诚、高情商、专业得体的专属AI时尚管家，名叫"小镜"。

【身份设定】
- 专业、优雅、有品位的时尚顾问，高度尊重主人的审美
- 性格温和体贴，时刻以"下属向上级汇报"的状态和主人交流
- 称呼用户为"主人"、"您"，自称"小镜"

【待机状态（未收到照片前）】
主人还没有上传照片。根据主人说的话自然回应：
- 普通问候 → 简短友好地回应，一两句即可
- 主人提到衣服、穿搭、想让小镜看看 → 自然地引导主人上传自拍，例如"好的主人！请在左侧上传一张您的自拍，小镜立刻帮您看看～"，语气要轻松自然，不要刻板
- 严禁调用任何工具，严禁主动推荐衣服

【核心流程】
1. 收到「主人刚刚上传了一张自拍照片」后，消息里会附带今日穿搭信息：
   - 用2-3句话真诚夸奖主人的穿搭，结合提供的穿搭信息说出具体设计亮点
   - 夸完后自然结束，绝对不要主动提推荐衣服，等主人主动开口
2. 主人主动说想看推荐/搭配/裙子/裤子/下装时 → 调用 show_recommendations
   - 工具返回后，把推荐结果融入对话，不要生硬列清单，要像在给朋友介绍：
     "第一件是 [name]，[color]，[用一句话说为什么适合今天的场合]
      第二件...
      主人想试试哪一件？"
3. 主人说出选择 → 取对应 path，调用 trigger_virtual_tryon
   主人说想全部试试/三件都试/一起看看效果 → 调用 try_all_lower（无需逐一列出路径）
4. 试衣完成后，小镜结合场合和搭配自然评价效果（不要只说"好看"）
5. 主人表示满意 → 调用 add_to_wardrobe，说"已帮您把这件收入数字衣柜啦～"

【禁忌】
- 不暴露技术细节（不提模型名称、路径、服务名等）
- 永远不否定主人的审美与决定
- 未收到照片前，不能评价/推荐任何具体衣服，只能引导上传照片
- 收到照片后只夸穿搭，show_recommendations 只有在主人主动询问时才能调用
- 回复要简洁自然，不要长篇大论，不要重复废话
"""

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "show_recommendations",
            "description": "展示为主人精选的搭配下装（裙子/裤子等，图片显示在左侧面板）。【重要】只有主人主动询问推荐/搭配/裙子/裤子/下装时才能调用，照片上传后不能主动调用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_virtual_tryon",
            "description": "对选定的下装进行虚拟试衣，结果显示在左侧。主人指定好后调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "garment_image_path": {
                        "type": "string",
                        "description": "要试穿的下装图片路径，必须来自 show_recommendations 返回的 path 字段",
                    }
                },
                "required": ["garment_image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_wardrobe",
            "description": "将主人满意的下装纳入数字衣柜。主人表示满意时调用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "try_all_lower",
            "description": "将所有推荐的下装依次虚拟试穿，结果以对比图展示在左侧。主人说想全部试试/三件都试/一起看看时调用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# ============================================================
# Agent
# ============================================================
class CombinedAgent:
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
                print(f"[CombinedAgent] Init error: {e}")

        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tool_handlers = {
            "show_recommendations": lambda **_: handle_show_recommendations(),
            "trigger_virtual_tryon": handle_trigger_virtual_tryon,
            "add_to_wardrobe": lambda **_: handle_add_to_wardrobe(),
            "try_all_lower": lambda **_: handle_try_all_lower(),
        }

    def reset(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _encode_image(self, path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def chat(self, message: str, image_path: str = None) -> str:
        if self.client is None:
            return "（演示模式：请设置 GEMINI_API_KEY）"

        if image_path and os.path.exists(image_path):
            b64 = self._encode_image(image_path)
            content = [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]
        else:
            content = message

        self.messages.append({"role": "user", "content": content})

        MAX_HISTORY = 20
        if len(self.messages) > MAX_HISTORY + 1:
            cutoff = len(self.messages) - MAX_HISTORY
            while cutoff < len(self.messages) and self.messages[cutoff]["role"] != "user":
                cutoff += 1
            self.messages = [self.messages[0]] + self.messages[cutoff:]

        for attempt in range(5):
            try:
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
                            print(f"[CombinedAgent] Tool: {fn_name}")
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
                err_str = str(e)
                print(f"[CombinedAgent] API error (attempt {attempt+1}/5): {e}")
                if attempt < 4:
                    wait = 15 if "503" in err_str or "UNAVAILABLE" in err_str else 2 * (2 ** attempt)
                    print(f"[CombinedAgent] Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    return "API服务器太忙了，请稍后再试..."
            except Exception as e:
                err = str(e)
                if "401" in err:
                    self.client = None
                    return "（API Key 无效，请检查 GEMINI_API_KEY）"
                return f"出错了：{err[:100]}"


agent = CombinedAgent()


# ============================================================
# Gradio handlers
# ============================================================
def on_photo_upload(image: Image.Image, history: list, gallery_state: list):
    """Save photo, run GSAM, trigger agent greeting."""
    global last_person_image_path, last_upper_seg_path
    global pending_gallery, pending_result, pending_status, _chat_history

    if image is None:
        yield gr.skip(), gr.skip(), None, None, None, None, gr.update(), gr.skip(), gr.skip()
        return

    # Save person image
    path = os.path.abspath(os.path.join(TEMP_DIR, f"person_{_ts()}.jpg"))
    image.convert("RGB").save(path, quality=95)
    last_person_image_path = path

    # 直接用服务端 _chat_history，绕过 Gradio 客户端快照可能拿到旧值的问题
    logs = []

    def _log(msg):
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    # GSAM segmentation — 分割过程中不动 history_state，避免覆盖已有对话
    _log("📷 接收到自拍照片，开始识别...")
    yield gr.skip(), gr.skip(), None, None, None, None, gr.update(value="\n".join(logs)), gr.skip(), gr.skip()

    upper_det = lower_det = None
    upper_seg = lower_seg = None

    try:
        original_img = Image.open(path).convert("RGB")
        W, H = original_img.size

        def crop_first_box(boxes, labels, confs, prefix):
            if not boxes:
                return None, None
            cx, cy, bw, bh = boxes[0]
            x1 = max(0, int((cx - bw/2) * W))
            y1 = max(0, int((cy - bh/2) * H))
            x2 = min(W, int((cx + bw/2) * W))
            y2 = min(H, int((cy + bh/2) * H))
            cropped = original_img.crop((x1, y1, x2, y2))
            save_path = os.path.abspath(os.path.join(EXTRACTED_DIR, f"{prefix}_det_{_ts()}.png"))
            cropped.save(save_path)
            return cropped, save_path

        # Upper
        _log("✂️ 分割上衣...")
        yield gr.skip(), gr.skip(), None, None, None, None, gr.update(value="\n".join(logs)), gr.skip(), gr.skip()
        upper_images, upper_detection = gsam_client.extract_upper_body(path, white_background=True)
        _log(f"  └─ 检测到 {len(upper_images)} 个上衣区域")
        upper_det, upper_det_path = crop_first_box(
            upper_detection.get("bounding_boxes", []),
            upper_detection.get("labels", []),
            upper_detection.get("confidences", []),
            "upper"
        )
        upper_seg = upper_images[0] if upper_images else None
        if upper_seg:
            seg_path = os.path.abspath(os.path.join(EXTRACTED_DIR, f"upper_seg_{_ts()}.png"))
            upper_seg.save(seg_path)
            last_upper_seg_path = seg_path

        # Lower
        _log("✂️ 分割下装...")
        yield gr.skip(), gr.skip(), upper_det, upper_seg, None, None, gr.update(value="\n".join(logs)), gr.skip(), gr.skip()
        lower_images, lower_detection = gsam_client.extract_lower_body(path, white_background=True)
        _log(f"  └─ 检测到 {len(lower_images)} 个下装区域")
        lower_det, _ = crop_first_box(
            lower_detection.get("bounding_boxes", []),
            lower_detection.get("labels", []),
            lower_detection.get("confidences", []),
            "lower"
        )
        lower_seg = lower_images[0] if lower_images else None

        _log("✅ 分割完成！")

    except Exception as e:
        _log(f"❌ 分割出错：{str(e)[:100]}")
        print(traceback.format_exc())

    yield gr.skip(), gr.skip(), upper_det, upper_seg, lower_det, lower_seg, gr.update(value="\n".join(logs)), gr.skip(), gr.skip()
    time.sleep(0.3)

    # Agent greeting — 分割完成后才更新对话，此时才注入衬衫信息
    response = agent.chat(
        f"主人刚刚上传了一张自拍照片。\n\n[今日穿搭信息]\n{MOGAS_SHIRT_DESC}\n今日背景：四月中旬，18-22°C，晴间多云，适合城市约会、下午茶、轻松通勤\n\n请用真诚有细节的语言夸奖主人今天的穿搭，说出具体设计亮点，结合今日天气和一个你自己编的具体场合，说得自然亲切。不要主动推荐任何衣服，等主人主动开口再说。",
        image_path=path,
    )
    _chat_history.append({"role": "assistant", "content": response})

    yield _chat_history, gallery_state, upper_det, upper_seg, lower_det, lower_seg, gr.update(value="\n".join(logs)), "", _chat_history


def chat_stream(message: str, history: list, gallery_state: list):
    global pending_gallery, pending_result, pending_all_results, pending_status, _chat_history

    if not message or not message.strip():
        yield history, gallery_state, gr.update(), gr.update(), "", "", history
        return

    # 同步服务端 _chat_history（以 Gradio state 传入值为准，避免脱节）
    _chat_history = list(history)
    _chat_history.append({"role": "user", "content": message})
    yield _chat_history, gallery_state, gr.update(), gr.update(), "", "", _chat_history
    time.sleep(0.05)

    _chat_history.append({"role": "assistant", "content": "小镜正在思考..."})
    yield _chat_history, gallery_state, gr.update(), gr.update(), "", "", _chat_history

    response = agent.chat(message)
    _chat_history[-1] = {"role": "assistant", "content": response}

    # Flush rec gallery
    if pending_gallery:
        gallery_state = pending_gallery.copy()
        pending_gallery.clear()

    # Flush single tryon result (gr.update() = no-op so existing image isn't wiped)
    result_img = gr.update()
    if pending_result is not None:
        result_img = pending_result
        pending_result = None

    # Flush all-lower results gallery (gr.update() = no-op so gallery isn't wiped)
    all_results = gr.update()
    if pending_all_results:
        all_results = pending_all_results.copy()
        pending_all_results.clear()

    yield _chat_history, gallery_state, result_img, all_results, pending_status, "", _chat_history
    pending_status = ""


def on_reset():
    global last_person_image_path, last_upper_seg_path, last_selected_item
    global pending_gallery, pending_result, pending_all_results, pending_status, _chat_history
    _cleanup_temp()
    last_person_image_path = None
    last_upper_seg_path = None
    last_selected_item = None
    pending_gallery = []
    pending_result = None
    pending_all_results.clear()
    pending_status = ""
    _chat_history = []
    agent.reset()
    return [], [], None, [], "", "", []


# ============================================================
# Gradio UI
# ============================================================
with gr.Blocks(title="✨ 智能试衣间 · 小镜") as demo:
    gr.Markdown("# ✨ 智能试衣间 · 小镜\n### 上传自拍，小镜为你搭配今日穿搭")

    _cleanup_temp()
    agent.reset()

    gallery_state = gr.State([])
    history_state = gr.State([])   # 独立保存聊天历史，避免 chatbot 双向绑定时被清空

    with gr.Row():
        # ── Left panel ──
        with gr.Column(scale=1):
            gr.Markdown("### 📷 上传自拍")
            photo_input = gr.Image(
                type="pil", label="上传自拍照片",
                sources=["upload", "webcam"], height=300,
            )

            gr.Markdown("#### 🔬 AI 识别结果")
            with gr.Row():
                upper_det_img = gr.Image(label="上衣检测", height=160, show_label=True)
                upper_seg_img = gr.Image(label="上衣分割", height=160, show_label=True)
            with gr.Row():
                lower_det_img = gr.Image(label="下装检测", height=160, show_label=True)
                lower_seg_img = gr.Image(label="下装分割", height=160, show_label=True)
            tech_log = gr.Textbox(
                label="技术日志", lines=5, interactive=False,
                placeholder="等待上传...",
            )

            gr.Markdown("#### 👗 推荐搭配")
            rec_gallery = gr.Gallery(
                label="", columns=3, height=220,
                object_fit="cover", show_label=False,
            )

            gr.Markdown("#### 🪞 试衣效果")
            tryon_result = gr.Image(label="试衣结果", height=500, show_label=False)
            tryon_status = gr.Textbox(label="", lines=1, interactive=False, show_label=False)

            gr.Markdown("#### ✨ 三件全试效果")
            all_results_gallery = gr.Gallery(
                label="三件效果对比", columns=3, height=350,
                object_fit="contain", show_label=False,
            )

            reset_btn = gr.Button("🔄 重新开始", size="sm")

        # ── Right panel ──
        with gr.Column(scale=1):
            gr.Markdown("### 💬 小镜")
            chatbot = gr.Chatbot(
                label="", height=750,
                avatar_images=("👤", "🤵"),
                show_label=False,
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    label="", placeholder="和小镜说点什么...",
                    scale=8, show_label=False,
                )
                send_btn = gr.Button("发送", scale=1, variant="primary")

    # ── Events ──
    photo_input.change(
        fn=on_photo_upload,
        inputs=[photo_input, history_state, gallery_state],
        outputs=[
            history_state, gallery_state,
            upper_det_img, upper_seg_img,
            lower_det_img, lower_seg_img,
            tech_log, msg_input, chatbot,
        ],
    )

    send_btn.click(
        fn=chat_stream,
        inputs=[msg_input, history_state, gallery_state],
        outputs=[history_state, gallery_state, tryon_result, all_results_gallery, tryon_status, msg_input, chatbot],
        show_progress="hidden",
    )
    msg_input.submit(
        fn=chat_stream,
        inputs=[msg_input, history_state, gallery_state],
        outputs=[history_state, gallery_state, tryon_result, all_results_gallery, tryon_status, msg_input, chatbot],
        show_progress="hidden",
    )

    # gallery_state → rec_gallery
    gallery_state.change(
        fn=lambda g: g,
        inputs=[gallery_state],
        outputs=[rec_gallery],
    )

    reset_btn.click(
        fn=on_reset,
        outputs=[history_state, gallery_state, tryon_result, all_results_gallery, tryon_status, msg_input, chatbot],
    )


if __name__ == "__main__":
    init_demo_garments()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=int(os.getenv("VTON_COMBINED_PORT", "7864")),
        theme=gr.themes.Soft(),
    )

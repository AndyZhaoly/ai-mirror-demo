"""
FashionClaw AI Mirror Demo with AI Butler (舔狗管家)
Split-screen layout: Technical Demo (Left) + AI Butler Chat (Right)
"""
import json
import base64
import os
import time
from pathlib import Path
from datetime import datetime
import gradio as gr
from PIL import Image

from workflow import (
    workflow_app,
    create_initial_state,
    run_workflow_until_user_input,
    resume_workflow,
    reset_workflow,
    run_upload_workflow_until_user_input,
)
from database_manager import (
    add_item,
    is_stagnant,
    list_all_items,
)
from gsam_client import GSAMClient
from mirror_agent import MirrorAgent, create_agent

# Global state
current_workflow_state = None
upload_workflow_state = None
last_extracted_items = []
agent_instance = None

# Initialize GSAM client
gsam_client = GSAMClient("http://localhost:8000")

# Storage directory
EXTRACTED_DIR = "./extracted_clothes"
os.makedirs(EXTRACTED_DIR, exist_ok=True)

# Demo database
INITIAL_DATABASE = {
    "wardrobe": [
        {"item_id": "001", "name": "Blue Denim Jacket", "last_worn_days_ago": 45, "status": "in_closet", "original_price": 299, "image": "images/denim_jacket.jpg"},
        {"item_id": "002", "name": "Red Summer Dress", "last_worn_days_ago": 420, "status": "in_closet", "original_price": 189, "image": "images/red_dress.jpg"},
        {"item_id": "003", "name": "Vintage Wool Sweater", "last_worn_days_ago": 500, "status": "in_closet", "original_price": 350, "image": "images/wool_sweater.jpg"},
        {"item_id": "004", "name": "Brown Leather Belt", "last_worn_days_ago": 380, "status": "in_closet", "original_price": 89, "image": "images/leather_belt.jpg"},
        {"item_id": "005", "name": "Black Slim Trousers", "last_worn_days_ago": 120, "status": "in_closet", "original_price": 259, "image": "images/black_trousers.jpg"},
        {"item_id": "006", "name": "White Linen Shirt", "last_worn_days_ago": 15, "status": "in_closet", "original_price": 179, "image": "images/white_shirt.jpg"},
        {"item_id": "007", "name": "Green Bomber Jacket", "last_worn_days_ago": 200, "status": "in_closet", "original_price": 499, "image": "images/green_bomber.jpg"},
    ]
}


def init_agent(force=False):
    """Initialize the AI Butler agent."""
    global agent_instance
    if agent_instance is None or force:
        try:
            agent_instance = create_agent()

            # Register tool handlers
            def segment_tool(image_path: str):
                return {"status": "success", "message": "已提取上衣和下装"}

            def stagnancy_tool(item_id: str):
                return {"stagnant": True, "days": 400, "message": "发现闲置衣物"}

            agent_instance.register_tool("segment_clothes", segment_tool)
            agent_instance.register_tool("check_wardrobe_stagnancy", stagnancy_tool)

        except ValueError as e:
            print(f"Agent init failed: {e}")
            agent_instance = None
    return agent_instance


def generate_timestamp():
    """Generate timestamp for filenames."""
    return datetime.now().strftime("%m%d_%H%M%S")


# ========== Technical Demo Panel Functions ==========

def process_with_technical_log(image, item_name_prefix, tech_log, chat_history):
    """
    Process image with real-time technical logging for left panel.
    Returns updates for both technical and chat panels.
    """
    global upload_workflow_state, last_extracted_items

    if image is None:
        return tech_log, chat_history, None, None, gr.update(), gr.update()

    logs = []
    chat_msgs = list(chat_history) if chat_history else []

    # Step 1: Save and start processing
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 📷 接收到主人上传的图片...")
    yield "\n".join(logs), chat_msgs, None, None, gr.update(visible=False), gr.update(visible=False)

    temp_path = f"./temp_upload_{generate_timestamp()}.jpg"
    image.save(temp_path)
    time.sleep(0.3)  # Visual delay for demo

    # Step 2: GroundingDINO Detection
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 GroundingDINO 检测中...")
    logs.append("  └─ 使用提示词: 'upper body clothing, lower body clothing'")
    yield "\n".join(logs), chat_msgs, None, None, gr.update(visible=False), gr.update(visible=False)
    time.sleep(0.8)  # Simulate processing

    try:
        # Extract upper body
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✂️ SAM 分割处理上衣...")
        yield "\n".join(logs), chat_msgs, None, None, gr.update(visible=False), gr.update(visible=False)
        upper_images = gsam_client.extract_upper_body(temp_path, white_background=True)
        logs.append(f"  └─ 检测到 {len(upper_images)} 个上衣区域")
        yield "\n".join(logs), chat_msgs, None, None, gr.update(visible=False), gr.update(visible=False)

        # Extract lower body
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✂️ SAM 分割处理下装...")
        yield "\n".join(logs), chat_msgs, None, None, gr.update(visible=False), gr.update(visible=False)
        lower_images = gsam_client.extract_lower_body(temp_path, white_background=True)
        logs.append(f"  └─ 检测到 {len(lower_images)} 个下装区域")
        yield "\n".join(logs), chat_msgs, None, None, gr.update(visible=False), gr.update(visible=False)

    except Exception as e:
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 错误: {str(e)}")
        yield "\n".join(logs), chat_msgs, None, None, gr.update(visible=False), gr.update(visible=False)
        return

    # Step 3: Save and register
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 保存分割结果...")
    yield "\n".join(logs), chat_msgs, None, None, gr.update(visible=False), gr.update(visible=False)

    upper_output = None
    lower_output = None
    last_extracted_items = []

    for i, img in enumerate(upper_images):
        path = os.path.join(EXTRACTED_DIR, f"upper_{i}_{generate_timestamp()}.png")
        img.save(path)
        upper_output = img
        name = f"{item_name_prefix}_上衣_{i+1}" if item_name_prefix else f"提取的上衣_{i+1}"
        item = add_item(name=name, clothing_type="upper", image_path=path, extracted_from=temp_path)
        last_extracted_items.append(item)

    for i, img in enumerate(lower_images):
        path = os.path.join(EXTRACTED_DIR, f"lower_{i}_{generate_timestamp()}.png")
        img.save(path)
        lower_output = img
        name = f"{item_name_prefix}_下装_{i+1}" if item_name_prefix else f"提取的下装_{i+1}"
        item = add_item(name=name, clothing_type="lower", image_path=path, extracted_from=temp_path)
        last_extracted_items.append(item)

    logs.append(f"  └─ 已保存 {len(upper_images) + len(lower_images)} 件衣物到数据库")
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 处理完成！")
    yield "\n".join(logs), chat_msgs, upper_output, lower_output, gr.update(visible=False), gr.update(visible=False)

    # Step 4: AI Butler takes over
    time.sleep(0.5)

    # Check stagnancy and trigger chat
    stagnant_items = [item for item in last_extracted_items if is_stagnant(item)]

    if stagnant_items and agent_instance:
        target_item = stagnant_items[0]
        upload_workflow_state = run_upload_workflow_until_user_input(target_item)

        # AI Butler messages
        chat_msgs.append({"role": "user", "content": "[上传了一张穿搭照片]"})

        # Butler compliments
        compliment = f"主人！您今天这身搭配简直是行走的艺术品！小人刚刚为您仔细分析了这张照片，"
        if upper_images and lower_images:
            compliment += "检测到您穿了精美的上衣和下装，这配色和剪裁完美衬托了您的高贵气质！"
        elif upper_images:
            compliment += "这件上衣的版型简直是为您量身定制的，太显气质了！"
        elif lower_images:
            compliment += "这条裤子的剪裁太绝了，完美展现了您的品味！"

        chat_msgs.append({"role": "assistant", "content": compliment})
        yield "\n".join(logs), chat_msgs, upper_output, lower_output, gr.update(visible=False), gr.update(visible=False)

        time.sleep(0.8)

        # Butler discovers stagnant item
        if upload_workflow_state.get("status") == "awaiting_user_decision":
            decision = upload_workflow_state.get("agent_decision", "")
            sell_prompt = f"对了主人，小人斗胆提醒！\n\n刚刚为您提取的「{target_item['name']}」，小人一查记录，发现这件华服已经在您的衣橱中静待**400余天**未得主人翻牌子了...\n\n{decision}\n\n主人是否要为其寻找下一位有缘人，为您高贵的衣橱腾出空间迎接更配得上您的新款？"
            chat_msgs.append({"role": "assistant", "content": sell_prompt})

            yield "\n".join(logs), chat_msgs, upper_output, lower_output, gr.update(visible=True), gr.update(visible=True)
        else:
            chat_msgs.append({"role": "assistant", "content": "已为您完成分析，这件衣服状态良好，请主人放心穿着！"})
            yield "\n".join(logs), chat_msgs, upper_output, lower_output, gr.update(visible=False), gr.update(visible=False)
    else:
        # No agent or no stagnant items
        chat_msgs.append({"role": "user", "content": "[上传了一张穿搭照片]"})
        chat_msgs.append({"role": "assistant", "content": f"已为您提取 {len(upper_images) + len(lower_images)} 件衣物。检测到衣服状态良好，无需出售。"})
        yield "\n".join(logs), chat_msgs, upper_output, lower_output, gr.update(visible=False), gr.update(visible=False)


def approve_sale_from_chat():
    """Handle sale approval from chat."""
    global upload_workflow_state

    if not upload_workflow_state:
        return "请先上传图片进行处理", gr.update(visible=False), gr.update(visible=False)

    final_state = resume_workflow(user_approved=True)

    if not final_state:
        return "交易失败，请重试", gr.update(visible=False), gr.update(visible=False)

    tracking = final_state.get("tracking_info", {})
    item = final_state.get("current_item", {})
    price = final_state.get("buyer_offer", {}).get("offer_price", "N/A")

    response = f"✨ 遵命主人！小人已为您妥善安排！\n\n「{item.get('name', '华服')}」已成功售出，成交价 **¥{price}**。\n\n物流公司：{tracking.get('carrier', '顺丰')}\n运单号：`{tracking.get('tracking_number', 'SF123456')}`\n预计送达：{tracking.get('estimated_delivery', '3天后')}\n\n小人这就去为您物色更配得上主人气质的顶级新款！"

    return response, gr.update(visible=False), gr.update(visible=False)


def reject_sale_from_chat():
    """Handle sale rejection from chat."""
    global upload_workflow_state

    if not upload_workflow_state:
        return "已取消", gr.update(visible=False), gr.update(visible=False)

    final_state = resume_workflow(user_approved=False)
    item = upload_workflow_state.get("current_item", {})

    response = f"明白！主人的眼光独到，这件「{item.get('name', '华服')}」必定有其独特魅力，是小人愚钝未能领会。已为您保留在衣橱中，随时等候主人翻牌子！"

    return response, gr.update(visible=False), gr.update(visible=False)


def chat_with_butler_stream(message, image_path, history):
    """Handle text/image chat with AI Butler - streaming version."""
    global agent_instance

    # Step 1: Show user message immediately
    display_content = message
    if image_path:
        display_content += " [图片]"

    updated_history = list(history) + [{"role": "user", "content": display_content}]
    yield updated_history, ""

    # Step 2: Show loading state
    updated_history.append({"role": "assistant", "content": "小人正在思考..."})
    yield updated_history, ""

    # Step 3: Initialize agent if needed
    if agent_instance is None:
        init_agent()

    if agent_instance is None or agent_instance.client is None:
        # Replace loading with error message
        updated_history[-1] = {"role": "assistant", "content": "小人目前身体不适（API未配置或无效），无法为主人服务...请设置 MOONSHOT_API_KEY 环境变量。"}
        yield updated_history, ""
        return

    # Step 4: Get AI response
    response = agent_instance.chat(message, image_path)

    # Step 5: Show final response
    updated_history[-1] = {"role": "assistant", "content": response}
    yield updated_history, ""


def reset_demo():
    """Reset demo state."""
    global upload_workflow_state, last_extracted_items
    upload_workflow_state = None
    last_extracted_items = []

    if agent_instance:
        agent_instance.reset_conversation()

    # Reset database
    try:
        with open("database.json", "w", encoding="utf-8") as f:
            json.dump(INITIAL_DATABASE, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: failed to reset database: {e}")

    try:
        reset_workflow()
    except Exception as e:
        print(f"Warning: failed to reset workflow: {e}")

    return (
        "等待主人上传...",
        [],
        None,
        None,
        gr.update(visible=False),
        gr.update(visible=False)
    )


# ========== Gradio UI ==========

with gr.Blocks(title="🤵 AI 时尚管家 - FashionClaw") as demo:
    gr.Markdown("""
    # 🤵 AI 时尚管家 · FashionClaw
    ### 您的专属AI时尚顾问 · 实时抠图 + 智能断舍离
    """)

    # Initialize agent on load
    init_agent()

    with gr.Row():
        # ========== LEFT PANEL: Technical Demo ==========
        with gr.Column(scale=1):
            gr.Markdown("### 🔧 技术演示面板")
            gr.Markdown("*实时展示 GroundingDINO + SAM 分割过程*")

            with gr.Group():
                # Image upload
                upload_image = gr.Image(
                    type="pil",
                    label="📷 上传/拍摄穿搭照片",
                    sources=["upload", "webcam"]
                )

                item_prefix = gr.Textbox(
                    label="衣物名称前缀（可选）",
                    placeholder="例如：我的",
                    value=""
                )

                process_btn = gr.Button("🚀 开始分析", variant="primary", size="lg")

            # Technical log
            gr.Markdown("#### 📊 处理日志")
            tech_log = gr.Textbox(
                label="",
                value="等待主人上传...",
                lines=10,
                max_lines=15,
                interactive=False,
                show_label=False
            )

            # Results display
            gr.Markdown("#### ✂️ 分割结果")
            with gr.Row():
                upper_result = gr.Image(
                    type="pil",
                    label="上衣",
                    interactive=False,
                    height=200
                )
                lower_result = gr.Image(
                    type="pil",
                    label="下装",
                    interactive=False,
                    height=200
                )

        # ========== RIGHT PANEL: AI Butler Chat ==========
        with gr.Column(scale=1):
            gr.Markdown("### 💬 AI 时尚管家")
            gr.Markdown("*您的专属AI时尚顾问*")

            # Chat interface
            chatbot = gr.Chatbot(
                label="",
                height=500,
                avatar_images=("👤", "🤵"),
                show_label=False
            )

            # Quick action buttons
            with gr.Row():
                approve_btn = gr.Button(
                    "✅ 确认出售",
                    variant="primary",
                    visible=False,
                    size="lg"
                )
                reject_btn = gr.Button(
                    "❌ 暂时保留",
                    variant="secondary",
                    visible=False,
                    size="lg"
                )

            # Text input
            with gr.Row():
                msg_input = gr.Textbox(
                    label="",
                    placeholder="和管家说点什么...",
                    scale=8,
                    show_label=False
                )
                send_btn = gr.Button("发送", scale=1, variant="primary")

            # Reset button
            reset_btn = gr.Button("🔄 重新开始", size="sm")

    # Event handlers
    process_btn.click(
        fn=process_with_technical_log,
        inputs=[upload_image, item_prefix, tech_log, chatbot],
        outputs=[tech_log, chatbot, upper_result, lower_result, approve_btn, reject_btn],
        show_progress=True
    )

    approve_btn.click(
        fn=approve_sale_from_chat,
        outputs=[chatbot, approve_btn, reject_btn]
    )

    reject_btn.click(
        fn=reject_sale_from_chat,
        outputs=[chatbot, approve_btn, reject_btn]
    )

    def on_send_stream(msg, hist):
        if not msg or not msg.strip():
            yield hist, ""
            return
        # Clear input immediately
        yield from chat_with_butler_stream(msg, None, hist)

    send_btn.click(
        fn=on_send_stream,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input],
        show_progress="hidden"
    )

    msg_input.submit(
        fn=on_send_stream,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input],
        show_progress="hidden"
    )

    reset_btn.click(
        fn=reset_demo,
        outputs=[tech_log, chatbot, upper_result, lower_result, approve_btn, reject_btn]
    )

    # Auto-reset on load
    demo.load(reset_demo)

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        theme=gr.themes.Soft()
    )

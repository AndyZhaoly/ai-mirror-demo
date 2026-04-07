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
from gsam_client import GSAMClient, draw_detection_boxes
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
    Returns: tech_log, chat_msgs, upper_detection_img, upper_seg_img, lower_detection_img, lower_seg_img, approve_btn, reject_btn
    """
    global upload_workflow_state, last_extracted_items

    if image is None:
        return tech_log, chat_history, None, None, None, None, gr.update(), gr.update()

    logs = []
    chat_msgs = list(chat_history) if chat_history else []

    # Step 1: Save and start processing
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 📷 接收到主人上传的图片...")
    yield "\n".join(logs), chat_msgs, None, None, None, None, gr.update(visible=False), gr.update(visible=False)

    temp_path = f"./temp_upload_{generate_timestamp()}.jpg"
    image.save(temp_path)
    time.sleep(0.3)

    # Step 2: GroundingDINO Detection
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 GroundingDINO 检测中...")
    logs.append("  └─ 使用提示词: 'shirt, jacket' / 'pants, shorts'")
    yield "\n".join(logs), chat_msgs, None, None, None, None, gr.update(visible=False), gr.update(visible=False)
    time.sleep(0.8)

    # Detection images for display - paired by type
    upper_detection_img = None
    lower_detection_img = None
    detection_path = None

    try:
        # Extract upper body
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✂️ SAM 分割处理上衣...")
        yield "\n".join(logs), chat_msgs, None, None, None, None, gr.update(visible=False), gr.update(visible=False)
        upper_images, upper_detection = gsam_client.extract_upper_body(temp_path, white_background=True)
        logs.append(f"  └─ 检测到 {len(upper_images)} 个上衣区域")
        yield "\n".join(logs), chat_msgs, None, None, None, None, gr.update(visible=False), gr.update(visible=False)

        # Extract lower body
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✂️ SAM 分割处理下装...")
        yield "\n".join(logs), chat_msgs, None, None, None, None, gr.update(visible=False), gr.update(visible=False)
        lower_images, lower_detection = gsam_client.extract_lower_body(temp_path, white_background=True)
        logs.append(f"  └─ 检测到 {len(lower_images)} 个下装区域")
        yield "\n".join(logs), chat_msgs, None, None, None, None, gr.update(visible=False), gr.update(visible=False)

        # Create cropped detection box visualizations for upper and lower
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🎨 生成检测框裁剪图...")
        original_img = Image.open(temp_path).convert("RGB")
        W, H = original_img.size

        # Helper function to crop detection boxes - return first box only and save it
        def crop_and_save_detection(img, boxes, labels, confs, prefix):
            """Crop the first (highest confidence) detection box from original image and save it."""
            if not boxes:
                return None, None
            # Take the first box (highest confidence from service)
            box, label, conf = boxes[0], labels[0], confs[0]
            # box format: [cx, cy, w, h] normalized
            cx, cy, bw, bh = box
            x1 = int((cx - bw/2) * W)
            y1 = int((cy - bh/2) * H)
            x2 = int((cx + bw/2) * W)
            y2 = int((cy + bh/2) * H)
            # Clamp to bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            cropped = img.crop((x1, y1, x2, y2))
            # Save the cropped detection box image
            crop_path = os.path.abspath(os.path.join(EXTRACTED_DIR, f"{prefix}_detection_crop_{generate_timestamp()}.png"))
            cropped.save(crop_path)
            return cropped, crop_path

        upper_detection_crop_path = None
        lower_detection_crop_path = None

        # Upper body detection visualization (cropped boxes)
        if upper_detection and upper_detection.get('bounding_boxes'):
            upper_detection_img, upper_detection_crop_path = crop_and_save_detection(
                original_img,
                upper_detection['bounding_boxes'],
                upper_detection['labels'],
                upper_detection['confidences'],
                "upper"
            )
            logs.append(f"  └─ 上衣: {len(upper_detection['bounding_boxes'])} 个检测框 -> 裁剪图已保存")

        # Lower body detection visualization (cropped boxes)
        if lower_detection and lower_detection.get('bounding_boxes'):
            lower_detection_img, lower_detection_crop_path = crop_and_save_detection(
                original_img,
                lower_detection['bounding_boxes'],
                lower_detection['labels'],
                lower_detection['confidences'],
                "lower"
            )
            logs.append(f"  └─ 下装: {len(lower_detection['bounding_boxes'])} 个检测框 -> 裁剪图已保存")

        # Save combined detection for database
        all_boxes = []
        all_labels = []
        all_confs = []
        for det_info in [upper_detection, lower_detection]:
            if det_info and det_info.get('bounding_boxes'):
                for box, label, conf in zip(det_info['bounding_boxes'], det_info['labels'], det_info['confidences']):
                    all_boxes.append(box)
                    all_labels.append(label)
                    all_confs.append(conf)

        if all_boxes:
            combined_detection = draw_detection_boxes(original_img, all_boxes, all_labels, all_confs)
            detection_path = os.path.abspath(os.path.join(EXTRACTED_DIR, f"detection_{generate_timestamp()}.png"))
            combined_detection.save(detection_path)

        yield "\n".join(logs), chat_msgs, upper_detection_img, None, lower_detection_img, None, gr.update(visible=False), gr.update(visible=False)

    except Exception as e:
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 错误: {str(e)}")
        import traceback
        logs.append(f"  └─ {traceback.format_exc()[:200]}")
        yield "\n".join(logs), chat_msgs, None, None, None, None, gr.update(visible=False), gr.update(visible=False)
        return

    # Step 3: Save and register
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 保存分割结果...")
    yield "\n".join(logs), chat_msgs, upper_detection_img, None, lower_detection_img, None, gr.update(visible=False), gr.update(visible=False)

    upper_output = None
    lower_output = None
    last_extracted_items = []
    original_image_path = os.path.abspath(temp_path)

    # Save upper body items - use detection crop for VLM (first item), SAM seg for display
    for i, img in enumerate(upper_images):
        seg_path = os.path.abspath(os.path.join(EXTRACTED_DIR, f"upper_{i}_{generate_timestamp()}.png"))
        img.save(seg_path)
        upper_output = img
        name = f"{item_name_prefix}_上衣_{i+1}" if item_name_prefix else f"提取的上衣_{i+1}"
        # For the first item, use detection crop for VLM analysis (more accurate)
        # SAM segmentation is only for display
        vlm_image_path = upper_detection_crop_path if i == 0 and upper_detection_crop_path else seg_path
        item = add_item(
            name=name,
            clothing_type="upper",
            image_path=vlm_image_path,  # This is what VLM will analyze
            extracted_from=original_image_path,
            detection_image=detection_path if detection_path else ""
        )
        last_extracted_items.append(item)
        print(f"[Save Item] Upper {i+1}: VLM analysis uses {vlm_image_path}, display uses SAM seg {seg_path}")

        # 保存调试信息 - 用于对比Google Lens结果
        try:
            debug_info = {
                "original_image": original_image_path,
                "vlm_analysis_image": vlm_image_path,
                "detection_crop": upper_detection_crop_path,
                "sam_segmentation": seg_path,
                "timestamp": generate_timestamp()
            }
            import json
            with open(f"{EXTRACTED_DIR}/debug_last_upload.json", "w") as f:
                json.dump(debug_info, f, indent=2)
        except Exception as e:
            print(f"[Debug] Failed to save debug info: {e}")

    # Save lower body items - use detection crop for VLM (first item), SAM seg for display
    for i, img in enumerate(lower_images):
        seg_path = os.path.abspath(os.path.join(EXTRACTED_DIR, f"lower_{i}_{generate_timestamp()}.png"))
        img.save(seg_path)
        lower_output = img
        name = f"{item_name_prefix}_下装_{i+1}" if item_name_prefix else f"提取的下装_{i+1}"
        # For the first item, use detection crop for VLM analysis (more accurate)
        vlm_image_path = lower_detection_crop_path if i == 0 and lower_detection_crop_path else seg_path
        item = add_item(
            name=name,
            clothing_type="lower",
            image_path=vlm_image_path,  # This is what VLM will analyze
            extracted_from=original_image_path,
            detection_image=detection_path if detection_path else ""
        )
        last_extracted_items.append(item)
        print(f"[Save Item] Lower {i+1}: VLM analysis uses {vlm_image_path}, display uses SAM seg {seg_path}")

    logs.append(f"  └─ 已保存 {len(upper_images) + len(lower_images)} 件衣物")
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 处理完成！")
    yield "\n".join(logs), chat_msgs, upper_detection_img, upper_output, lower_detection_img, lower_output, gr.update(visible=False), gr.update(visible=False)

    # Step 4: AI Butler takes over - IMMEDIATE RESPONSE
    time.sleep(0.3)

    # 准备物品描述给Agent
    vlm_analysis = {}  # 将在后面填充
    item_desc = "这件衣服"
    if vlm_analysis and vlm_analysis.get("category"):
        color = vlm_analysis.get("color", "")
        category = vlm_analysis.get("category", "单品")
        brand = vlm_analysis.get("brand", "")
        if brand and brand != "Unknown":
            item_desc = f"{brand}的{color}{category}" if color else f"{brand}的{category}"
        else:
            item_desc = f"{color}{category}" if color else category

    # Agent立即说"收到，正在查"
    chat_msgs.append({"role": "user", "content": "[上传了一张穿搭照片]"})
    immediate_response = agent_instance.analyze_and_price(temp_path, item_desc) if agent_instance else f"主人～这件{item_desc}好有品味！✨ 小镜正在帮您查市场行情..."
    chat_msgs.append({"role": "assistant", "content": immediate_response})

    # [新增] 把用户上传的图片和小镜的回复注入Agent记忆，保持上下文连贯
    if agent_instance and last_extracted_items:
        target_item_for_memory = last_extracted_items[0]
        item_image = target_item_for_memory.get("image", temp_path)
        agent_instance.inject_memory(role="user", content="帮我看看这件衣服能卖多少钱", image_path=item_image)
        agent_instance.inject_memory(role="assistant", content=immediate_response)

    yield "\n".join(logs), chat_msgs, upper_detection_img, upper_output, lower_detection_img, lower_output, gr.update(visible=False), gr.update(visible=False)

    # Step 5: Background Analysis & Pricing (异步分析)
    # 现在运行 VLM + Google Lens 搜索，Agent已经在前面说"正在查了"
    target_item = None
    if last_extracted_items:
        target_item = last_extracted_items[0]

    if target_item and agent_instance:
        print(f"[DEBUG] Analyzing item: {target_item.get('name')}, Image: {target_item.get('image')}")

        # Show Gemini analysis in technical log
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 调用 Gemini 3.1 Pro 分析...")
        yield "\n".join(logs), chat_msgs, upper_detection_img, upper_output, lower_detection_img, lower_output, gr.update(visible=False), gr.update(visible=False)

        # Run workflow (Gemini analysis)
        upload_workflow_state = run_upload_workflow_until_user_input(target_item)

        # Get Gemini results
        vlm_analysis = upload_workflow_state.get("vlm_analysis", {})
        gemini_result = upload_workflow_state.get("gemini_result", {})

        # Update technical log - 显示 Gemini 分析结果
        if gemini_result and gemini_result.get("success"):
            logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Gemini 分析完成")
            logs.append(f"  ├─ 模型: {gemini_result.get('model_used', 'Gemini')}")

            if vlm_analysis:
                brand = vlm_analysis.get('brand', 'Unknown')
                model = vlm_analysis.get('model_name', '')
                logs.append(f"  ├─ 品牌: {brand}")
                if model and model != '未识别':
                    logs.append(f"  ├─ 型号: {model}")
                if vlm_analysis.get('product_code'):
                    logs.append(f"  ├─ 货号: {vlm_analysis['product_code']}")
                logs.append(f"  ├─ 类别: {vlm_analysis.get('category', 'N/A')}")
                logs.append(f"  ├─ 材质: {vlm_analysis.get('material', 'N/A')}")

            official = gemini_result.get('official_price', {})
            resale = gemini_result.get('resale_estimate', {})
            if official.get('amount'):
                logs.append(f"  ├─ 官方价: {official['amount']} {official.get('currency', '')}")
            if resale.get('max_price'):
                logs.append(f"  ├─ 二手估价: ¥{resale['min_price']} - ¥{resale['max_price']}")
                logs.append(f"  ├─ 置信度: {resale.get('confidence', 'N/A')}")

            logs.append(f"  └─ Agent 正在生成定价建议...")
        else:
            logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Gemini 分析未返回完整结果")
        yield "\n".join(logs), chat_msgs, upper_detection_img, upper_output, lower_detection_img, lower_output, gr.update(visible=False), gr.update(visible=False)

        # Step 6: Agent 生成自然的定价分析
        time.sleep(0.5)  # 稍微停顿，让"正在分析"的感觉更真实

        price_analysis = agent_instance.generate_gemini_price_analysis(gemini_result)

        # 添加 Agent 的定价分析到对话
        chat_msgs.append({"role": "assistant", "content": price_analysis})

        # [新增] 把小镜的定价分析注入记忆，同时隐式注入参考链接
        if agent_instance:
            # 构建参考链接的隐藏上下文
            reference_sources = gemini_result.get("reference_sources", [])
            hidden_context = None
            if reference_sources:
                links_text = "\n".join([
                    f"- {src.get('title', '来源')}: {src.get('url', '')}"
                    for src in reference_sources[:3]
                ])
                hidden_context = f"参考价格来源：\n{links_text}\n如果用户询问'你从哪查到的'、'参考链接在哪'，请回复这些来源。"

            agent_instance.inject_memory(
                role="assistant",
                content=price_analysis,
                hidden_context=hidden_context
            )

        yield "\n".join(logs), chat_msgs, upper_detection_img, upper_output, lower_detection_img, lower_output, gr.update(visible=False), gr.update(visible=False)

        # Step 7: 检查是否闲置，给出出售建议
        time.sleep(0.3)

        status = upload_workflow_state.get("status")
        if status == "awaiting_user_decision":
            current_item = upload_workflow_state.get("current_item", target_item)
            item_name = current_item.get("display_name", target_item['name'])

            stagnant_prompt = f"对了主人～ 💡\n\n小镜顺便查了一下，发现这件**{item_name}**已经在您的衣橱中静置**400余天**了...\n\n既然主人平时不怎么穿它，要不要趁现在行情好出手呢？小镜可以帮您发布到二手市场，说不定很快就有人买走啦～✨"
            chat_msgs.append({"role": "assistant", "content": stagnant_prompt})

            # [新增] 把闲置提示也注入记忆
            if agent_instance:
                agent_instance.inject_memory(role="assistant", content=stagnant_prompt)

            yield "\n".join(logs), chat_msgs, upper_detection_img, upper_output, lower_detection_img, lower_output, gr.update(visible=True), gr.update(visible=True)

        elif status == "api_overloaded":
            api_busy_msg = "对了主人～ ⚠️\n\n小镜的AI大脑刚才有点忙，价格分析可能不够完整。主人可以直接问我'这件能卖多少钱'，小镜再帮您查查！"
            chat_msgs.append({"role": "assistant", "content": api_busy_msg})
            yield "\n".join(logs), chat_msgs, upper_detection_img, upper_output, lower_detection_img, lower_output, gr.update(visible=False), gr.update(visible=False)

    else:
        # No agent or no items extracted
        chat_msgs.append({"role": "user", "content": "[上传了一张穿搭照片]"})
        chat_msgs.append({"role": "assistant", "content": f"已为您提取 {len(upper_images) + len(lower_images)} 件衣物。看起来都是很棒的衣服呢！主人有什么想了解的，随时问我～"})
        yield "\n".join(logs), chat_msgs, upper_detection_img, upper_output, lower_detection_img, lower_output, gr.update(visible=False), gr.update(visible=False)


def approve_sale_from_chat(history):
    """Handle sale approval from chat."""
    global upload_workflow_state, agent_instance, last_extracted_items

    history = list(history) if history else []

    if not upload_workflow_state:
        history.append({"role": "assistant", "content": "请先上传图片进行处理"})
        return history, gr.update(visible=False), gr.update(visible=False)

    # Pass the upload workflow state to resume correctly
    final_state = resume_workflow(user_approved=True, initial_state=upload_workflow_state)

    if not final_state:
        history.append({"role": "assistant", "content": "交易失败，请重试"})
        return history, gr.update(visible=False), gr.update(visible=False)

    tracking = final_state.get("tracking_info", {})
    item = final_state.get("current_item", {})
    price = final_state.get("buyer_offer", {}).get("offer_price", "N/A")

    # 使用 Agent 生成发布模板 (In-Context Learning)
    gemini_result = final_state.get("gemini_result", {})
    listing_template = ""
    if agent_instance:
        listing_template = agent_instance.generate_listing_template(
            gemini_result=gemini_result,
            item_name=item.get('name', '美衣'),
            price=str(price)
        )
    else:
        listing_template = "（Agent 未初始化，无法生成发布模板）"

    # 【新增】自动发布到 Poshmark
    poshmark_result_msg = ""
    if agent_instance:
        # 获取要发布的图片路径
        item_image_path = item.get('image', '') if item else ''
        if not item_image_path and last_extracted_items:
            # 使用最后提取的物品图片
            item_image_path = last_extracted_items[0].get('image', '')

        if item_image_path and os.path.exists(item_image_path):
            # 调用 Poshmark 发布（同步执行，会打开浏览器）
            poshmark_result = agent_instance.publish_to_poshmark(
                item_image_path=item_image_path,
                auto_submit=False  # Demo 模式，不自动提交
            )
            poshmark_result_msg = f"\n\n🌐 **Poshmark 发布**\n{poshmark_result}"

    response = f"""✨ 太好了主人！

这件**{item.get('name', '美衣')}**已经成功安排出售啦！预估成交价 **¥{price}**～

小镜这就帮您打包发货！📦

• 物流：{tracking.get('carrier', '顺丰')}
• 单号：`{tracking.get('tracking_number', 'SF123456')}`
• 预计：{tracking.get('estimated_delivery', '3天内送达')}

钱到账了第一时间通知主人！

━━━━━━━━━━━━━━━━━━━━━━━

{listing_template}

━━━━━━━━━━━━━━━━━━━━━━━{poshmark_result_msg}

💡 小镜提示：复制上方模板到闲鱼/小红书即可发布！需要小镜帮您找新款吗？😊"""

    history.append({"role": "assistant", "content": response})
    return history, gr.update(visible=False), gr.update(visible=False)


def reject_sale_from_chat(history):
    """Handle sale rejection from chat."""
    global upload_workflow_state

    history = list(history) if history else []

    if not upload_workflow_state:
        history.append({"role": "assistant", "content": "已取消"})
        return history, gr.update(visible=False), gr.update(visible=False)

    # Pass the upload workflow state to resume correctly
    final_state = resume_workflow(user_approved=False, initial_state=upload_workflow_state)
    item = upload_workflow_state.get("current_item", {})

    response = f"好的主人～ 💕 这件**{item.get('name', '美衣')}**还是留在您身边吧！说不定哪天主人突然想穿了，它又会成为您的心头好～ 小镜随时待命！"

    history.append({"role": "assistant", "content": response})
    return history, gr.update(visible=False), gr.update(visible=False)


def chat_with_butler_stream(message, image_path, history):
    """Handle text/image chat with AI Butler - streaming version."""
    global agent_instance, last_extracted_items, upload_workflow_state

    # Step 1: Show user message immediately
    display_content = message
    if image_path:
        display_content += " [图片]"

    updated_history = list(history) + [{"role": "user", "content": display_content}]
    yield updated_history, ""

    # Step 2: Check if user wants to retry analysis after API overload
    retry_keywords = ["分析", "重新分析", "再试", "重试", "查价格", "定价"]
    if any(kw in message for kw in retry_keywords) and upload_workflow_state:
        if upload_workflow_state.get("status") == "api_overloaded" and last_extracted_items:
            updated_history.append({"role": "assistant", "content": "小人正在重新分析衣物，请稍候..."})
            yield updated_history, ""

            # Re-run workflow for the last item
            target_item = last_extracted_items[-1]
            upload_workflow_state = run_upload_workflow_until_user_input(target_item)

            if upload_workflow_state.get("status") == "awaiting_user_decision":
                decision = upload_workflow_state.get("agent_decision", "")
                sell_prompt = f"主人！小人重新分析完成了！\n\n刚刚为您提取的「{target_item['name']}」：\n\n{decision}\n\n主人是否要为其寻找下一位有缘人？"
                updated_history[-1] = {"role": "assistant", "content": sell_prompt}
                yield updated_history, ""
                return
            elif upload_workflow_state.get("status") == "api_overloaded":
                updated_history[-1] = {"role": "assistant", "content": "抱歉主人，AI服务仍然繁忙...请稍等1-2分钟后再试，或者小人先为您保留这件衣物？"}
                yield updated_history, ""
                return

    # Step 3: Show loading state
    updated_history.append({"role": "assistant", "content": "小人正在思考..."})
    yield updated_history, ""

    # Step 4: Initialize agent if needed
    if agent_instance is None:
        init_agent()

    if agent_instance is None or agent_instance.client is None:
        # Replace loading with error message
        updated_history[-1] = {"role": "assistant", "content": "小人目前身体不适（API未配置或无效），无法为主人服务...请设置 MOONSHOT_API_KEY 环境变量。"}
        yield updated_history, ""
        return

    # Step 5: Check if user uploaded a new image
    # Note: If no new image, agent relies on injected memory from previous interactions
    chat_image_path = image_path
    if chat_image_path and agent_instance:
        # Save to agent's memory for future tool calls
        agent_instance.last_image_path = chat_image_path

    # Step 6: Get AI response
    response = agent_instance.chat(message, chat_image_path)

    # Step 7: Show final response
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

    # Clean up generated files from last demo
    print("[Reset] Cleaning up generated files...")
    cleanup_count = 0
    try:
        import glob
        # Clean extracted clothes images
        for f in glob.glob(f"{EXTRACTED_DIR}/*"):
            try:
                os.remove(f)
                cleanup_count += 1
            except:
                pass
        # Clean temp upload files
        for f in glob.glob("./temp_upload_*.jpg"):
            try:
                os.remove(f)
                cleanup_count += 1
            except:
                pass
        # Clean debug files
        for f in glob.glob("/tmp/last_search_*.txt") + glob.glob("/tmp/last_google_lens_*.json"):
            try:
                os.remove(f)
                cleanup_count += 1
            except:
                pass
        print(f"[Reset] Cleaned {cleanup_count} files")
    except Exception as e:
        print(f"[Reset] Cleanup warning: {e}")

    return (
        "等待主人上传...",
        [],
        None,
        None,
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

    # Auto-cleanup on startup
    print("[Startup] Auto-cleaning previous demo files...")
    try:
        import glob
        cleanup_files = (
            glob.glob(f"{EXTRACTED_DIR}/*") +
            glob.glob("./temp_upload_*.jpg") +
            glob.glob("/tmp/last_search_*.txt") +
            glob.glob("/tmp/last_google_lens_*.json")
        )
        for f in cleanup_files:
            try:
                os.remove(f)
            except:
                pass
        print(f"[Startup] Cleaned {len(cleanup_files)} files from previous session")
    except Exception as e:
        print(f"[Startup] Cleanup skipped: {e}")

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

            # Results display - paired by clothing type
            gr.Markdown("#### 👕 上衣：检测结果 + 分割")
            with gr.Row():
                upper_detection_result = gr.Image(
                    type="pil",
                    label="上衣检测框",
                    interactive=False,
                    height=200
                )
                upper_result = gr.Image(
                    type="pil",
                    label="上衣分割",
                    interactive=False,
                    height=200
                )

            gr.Markdown("#### 👖 下装：检测结果 + 分割")
            with gr.Row():
                lower_detection_result = gr.Image(
                    type="pil",
                    label="下装检测框",
                    interactive=False,
                    height=200
                )
                lower_result = gr.Image(
                    type="pil",
                    label="下装分割",
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
        outputs=[tech_log, chatbot, upper_detection_result, upper_result, lower_detection_result, lower_result, approve_btn, reject_btn],
        show_progress=True
    )

    approve_btn.click(
        fn=approve_sale_from_chat,
        inputs=[chatbot],
        outputs=[chatbot, approve_btn, reject_btn]
    )

    reject_btn.click(
        fn=reject_sale_from_chat,
        inputs=[chatbot],
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
        outputs=[tech_log, chatbot, upper_detection_result, upper_result, lower_detection_result, lower_result, approve_btn, reject_btn]
    )

    # Auto-reset on load
    demo.load(reset_demo, outputs=[tech_log, chatbot, upper_detection_result, upper_result, lower_detection_result, lower_result, approve_btn, reject_btn])

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        theme=gr.themes.Soft()
    )

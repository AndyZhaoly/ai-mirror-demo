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
from idm_vton_client import IDMVTONClient

# Global state
current_workflow_state = None
upload_workflow_state = None
last_extracted_items = []
agent_instance = None
last_uploaded_image = None  # 保存用户上传的原图路径

# Initialize GSAM client
gsam_client = GSAMClient(os.getenv("GSAM_URL", "http://localhost:8000"))

# Initialize IDM-VTON client
idm_vton_client = IDMVTONClient(os.getenv("VTON_URL", "http://localhost:8001"))

# Storage directory
EXTRACTED_DIR = "./extracted_clothes"
os.makedirs(EXTRACTED_DIR, exist_ok=True)

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

            # Register Poshmark publish tool - handles actual publishing with confirmation
            def poshmark_publish_tool(execute: bool = False):
                """
                Poshmark发布工具处理器。
                当Agent检测到用户有明确出售意愿时调用。

                参数:
                    execute: 只有当用户明确无犹豫地要卖时才为True
                            如果用户有疑虑（如"不确定价格"、"考虑一下"），Agent不会调用此工具
                """
                global upload_workflow_state, last_extracted_items, last_uploaded_image

                if not execute:
                    # Agent 不应该在 execute=False 时调用此工具
                    # 这里只是安全兜底
                    return {
                        "status": "need_confirm",
                        "message": "用户需要进一步确认，不应直接执行发布"
                    }

                if not upload_workflow_state:
                    return {"status": "error", "message": "没有待处理的衣物，请先上传图片"}

                if upload_workflow_state.get("status") != "awaiting_user_decision":
                    return {"status": "error", "message": "当前没有等待出售的衣物"}

                try:
                    # 执行实际发布流程
                    final_state = resume_workflow(user_approved=True, initial_state=upload_workflow_state)

                    if not final_state:
                        return {"status": "error", "message": "交易流程执行失败"}

                    item = final_state.get("current_item", {})
                    price = final_state.get("buyer_offer", {}).get("offer_price", "N/A")
                    gemini_result = final_state.get("gemini_result", {})

                    # 生成发布模板
                    listing_template = ""
                    if agent_instance:
                        listing_template = agent_instance.generate_listing_template(
                            gemini_result=gemini_result,
                            item_name=item.get('name', '美衣'),
                            price=str(price)
                        )

                    # 发布到Poshmark
                    poshmark_result_msg = ""
                    item_image_path = last_uploaded_image or (last_extracted_items[0].get('image', '') if last_extracted_items else '')

                    if item_image_path and os.path.exists(item_image_path) and agent_instance:
                        poshmark_result = agent_instance.publish_to_poshmark(
                            item_image_path=item_image_path,
                            auto_submit=False
                        )
                        poshmark_result_msg = f"\n\n🌐 **Poshmark 发布**\n{poshmark_result}"

                    # 构建完整结果消息
                    result_message = f"""✨ 太好了主人！

这件**{item.get('name', '美衣')}**已经成功安排出售啦！预估成交价 **¥{price}**～

小镜这就帮您打包发货！📦

{listing_template}
{poshmark_result_msg}

💡 小镜提示：Poshmark 页面已打开，请确认信息后点击发布！"""

                    return {
                        "status": "success",
                        "message": result_message,
                        "item_name": item.get('name', ''),
                        "price": price
                    }

                except Exception as e:
                    import traceback
                    error_msg = f"发布过程中出错：{str(e)}"
                    print(f"[Poshmark Tool Error] {traceback.format_exc()}")
                    return {"status": "error", "message": error_msg}

            agent_instance.register_tool("publish_to_poshmark", poshmark_publish_tool)

            # Register recommendation tool
            def get_recommendations_tool(category: str = None, style: str = None, limit: int = 4):
                """Get clothing recommendations from sample clothes."""
                from recommendations import get_recommendations, format_recommendation_for_agent, get_recommendation_image_paths
                
                items = get_recommendations(
                    category=category,
                    style=style,
                    limit=limit
                )
                
                # Store for later use (e.g., displaying images)
                global current_recommendations
                current_recommendations = items
                
                # Get image paths for UI display
                image_paths = get_recommendation_image_paths(items)
                
                # Format text for agent
                text = format_recommendation_for_agent(items)
                
                return {
                    "status": "success",
                    "recommendations": [item.to_dict() for item in items],
                    "image_paths": image_paths,
                    "message": text
                }
            
            agent_instance.register_tool("get_clothing_recommendations", get_recommendations_tool)
            
            # Register virtual try-on tool
            def trigger_virtual_tryon_tool(item_id: str, preserve_face: bool = True):
                """Trigger virtual try-on for selected item."""
                global current_recommendations, pending_tryon_item, last_user_person_image
                
                # Find the selected item
                from recommendations import get_item_by_id
                
                # Find the selected item using unified lookup
                selected_item = get_item_by_id(item_id)
                
                if not selected_item:
                    return {
                        "status": "error",
                        "message": "抱歉主人，小镜没找到这件衣服呢，请重新选择～"
                    }
                
                # Check if user has uploaded a person image
                # Note: In actual implementation, this would be checked against the stored user image
                pending_tryon_item = selected_item
                
                return {
                    "status": "success",
                    "message": f"已选中 **{selected_item.name}** 准备试穿！小镜需要主人的人像照片作为试穿基础图，请上传照片后点击虚拟试穿～",
                    "item": selected_item.to_dict(),
                    "needs_person_image": True
                }
            
            agent_instance.register_tool("trigger_virtual_tryon", trigger_virtual_tryon_tool)

        except ValueError as e:
            print(f"Agent init failed: {e}")
            agent_instance = None
    return agent_instance


# Global state for recommendation workflow
current_recommendations = []  # Current recommendation items
pending_tryon_item = None     # Item selected for try-on
last_user_person_image = None # User's uploaded person image for try-on base


def generate_timestamp():
    """Generate timestamp for filenames."""
    return datetime.now().strftime("%m%d_%H%M%S")


# ========== Technical Demo Panel Functions ==========

def process_with_technical_log(image, item_name_prefix, tech_log, chat_state):
    """
    Process image with real-time technical logging for left panel.
    Returns updates for both technical and chat panels.

    Uses gr.State() for chat state management.

    Returns: tech_log, chat_state, upper_detection_img, upper_seg_img, lower_detection_img, lower_seg_img
    """
    global upload_workflow_state, last_extracted_items, last_uploaded_image

    # Initialize chat_state if None
    if chat_state is None:
        chat_state = []

    if image is None:
        yield tech_log, chat_state, None, None, None, None
        return

    logs = []
    # Use the passed-in state
    chat_msgs = chat_state
    print(f"[DEBUG] Chat state length: {len(chat_state)}")

    # Step 1: Save and start processing
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 📷 接收到主人上传的图片...")
    yield "\n".join(logs), chat_msgs, None, None, None, None
    temp_path = f"./temp_upload_{generate_timestamp()}.jpg"
    try:
        image.save(temp_path)
    except Exception as e:
        logs.append(f"❌ 图片保存失败：{e}")
        yield "\n".join(logs), chat_msgs, None, None, None, None
        return
    last_uploaded_image = os.path.abspath(temp_path)  # 保存用户上传的原图路径
    time.sleep(0.3)

    # Step 2: GroundingDINO Detection
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 GroundingDINO 检测中...")
    logs.append("  └─ 使用提示词: 'shirt, jacket' / 'pants, shorts'")
    yield "\n".join(logs), chat_msgs, None, None, None, None
    time.sleep(0.8)

    # Detection images for display - paired by type
    upper_detection_img = None
    lower_detection_img = None
    detection_path = None

    try:
        # Extract upper body
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✂️ SAM 分割处理上衣...")
        yield "\n".join(logs), chat_msgs, None, None, None, None
        upper_images, upper_detection = gsam_client.extract_upper_body(temp_path, white_background=True)
        logs.append(f"  └─ 检测到 {len(upper_images)} 个上衣区域")
        yield "\n".join(logs), chat_msgs, None, None, None, None
        # Extract lower body
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✂️ SAM 分割处理下装...")
        yield "\n".join(logs), chat_msgs, None, None, None, None
        lower_images, lower_detection = gsam_client.extract_lower_body(temp_path, white_background=True)
        logs.append(f"  └─ 检测到 {len(lower_images)} 个下装区域")
        yield "\n".join(logs), chat_msgs, None, None, None, None
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

        yield "\n".join(logs), chat_msgs, upper_detection_img, None, lower_detection_img, None
    except Exception as e:
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 错误: {str(e)}")
        import traceback
        logs.append(f"  └─ {traceback.format_exc()[:200]}")
        yield "\n".join(logs), chat_msgs, None, None, None, None
        return

    # Step 3: Save and register
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 保存分割结果...")
    yield "\n".join(logs), chat_msgs, upper_detection_img, None, lower_detection_img, None
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
    yield "\n".join(logs), chat_msgs, upper_detection_img, upper_output, lower_detection_img, lower_output
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

    yield "\n".join(logs), chat_state, upper_detection_img, upper_output, lower_detection_img, lower_output
    # Step 5: Background Analysis & Pricing (异步分析)
    # 现在运行 VLM + Google Lens 搜索，Agent已经在前面说"正在查了"
    target_item = None
    if last_extracted_items:
        target_item = last_extracted_items[0]

    if target_item and agent_instance:
        print(f"[DEBUG] Analyzing item: {target_item.get('name')}, Image: {target_item.get('image')}")

        # Show Gemini analysis in technical log
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 调用 Gemini 3.1 Pro 分析...")
        yield "\n".join(logs), chat_msgs, upper_detection_img, upper_output, lower_detection_img, lower_output
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
        yield "\n".join(logs), chat_msgs, upper_detection_img, upper_output, lower_detection_img, lower_output
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

        yield "\n".join(logs), chat_state, upper_detection_img, upper_output, lower_detection_img, lower_output
        # Step 7: 检查是否闲置，给出出售建议
        time.sleep(0.3)

        status = upload_workflow_state.get("status")
        if status == "awaiting_user_decision":
            current_item = upload_workflow_state.get("current_item", target_item)
            item_name = current_item.get("display_name", target_item['name'])

            stagnant_prompt = f"""对了主人～ 💡

小镜顺便查了一下，发现这件**{item_name}**已经在您的衣橱中静置**400余天**了...

既然主人平时不怎么穿它，要不要趁现在行情好出手呢？小镜可以帮您：

🇨🇳 **闲鱼/小红书** - 国内买家，人民币结算
🌐 **Poshmark** - 海外买家，美元结算，售价更高哦～

小镜还能帮您自动生成专业的英文发布文案！

💬 主人如果愿意出手，直接回复**"好的"**、**"卖"**或**"发布"**，小镜立刻为您安排！

主人要不要考虑一下？✨"""
            chat_msgs.append({"role": "assistant", "content": stagnant_prompt})

            # [新增] 把闲置提示也注入记忆
            if agent_instance:
                agent_instance.inject_memory(role="assistant", content=stagnant_prompt)

            yield "\n".join(logs), chat_state, upper_detection_img, upper_output, lower_detection_img, lower_output
        elif status == "api_overloaded":
            api_busy_msg = "对了主人～ ⚠️\n\n小镜的AI大脑刚才有点忙，价格分析可能不够完整。主人可以直接问我'这件能卖多少钱'，小镜再帮您查查！"
            chat_msgs.append({"role": "assistant", "content": api_busy_msg})
            yield "\n".join(logs), chat_state, upper_detection_img, upper_output, lower_detection_img, lower_output
    else:
        # No agent or no items extracted
        chat_msgs.append({"role": "user", "content": "[上传了一张穿搭照片]"})
        chat_msgs.append({"role": "assistant", "content": f"已为您提取 {len(upper_images) + len(lower_images)} 件衣物。看起来都是很棒的衣服呢！主人有什么想了解的，随时问我～"})
        yield "\n".join(logs), chat_state, upper_detection_img, upper_output, lower_detection_img, lower_output

def chat_with_butler_stream(message, image_path, chat_state):
    """Handle text/image chat with AI Butler - streaming version with proper state management.

    Uses gr.State() for chat state management.
    """
    global agent_instance, last_extracted_items, upload_workflow_state

    # Initialize chat_state if None
    if chat_state is None:
        chat_state = []

    # Step 1: Show user message immediately
    display_content = message
    if image_path:
        display_content += " [图片]"

    # Append new message to state
    chat_state.append({"role": "user", "content": display_content})

    # 立即显示用户消息
    yield chat_state, ""

    # 短暂停顿让 UI 更新
    import time
    time.sleep(0.1)

    # Step 2: Check if user wants to retry analysis after API overload
    retry_keywords = ["分析", "重新分析", "再试", "重试", "查价格", "定价"]
    if any(kw in message for kw in retry_keywords) and upload_workflow_state:
        if upload_workflow_state.get("status") == "api_overloaded" and last_extracted_items:
            chat_state.append({"role": "assistant", "content": "小人正在重新分析衣物，请稍候..."})
            yield chat_state, ""

            # Re-run workflow for the last item
            target_item = last_extracted_items[-1]
            upload_workflow_state = run_upload_workflow_until_user_input(target_item)

            if upload_workflow_state.get("status") == "awaiting_user_decision":
                decision = upload_workflow_state.get("agent_decision", "")
                sell_prompt = f"主人！小人重新分析完成了！\n\n刚刚为您提取的「{target_item['name']}」：\n\n{decision}\n\n主人是否要为其寻找下一位有缘人？"
                # 替换最后的"思考中"消息
                chat_state[-1] = {"role": "assistant", "content": sell_prompt}
                yield chat_state, ""
                return
            elif upload_workflow_state.get("status") == "api_overloaded":
                chat_state[-1] = {"role": "assistant", "content": "抱歉主人，AI服务仍然繁忙...请稍等1-2分钟后再试，或者小人先为您保留这件衣物？"}
                yield chat_state, ""
                return

    # Step 3: Show loading state (用户消息已显示，现在加思考中)
    chat_state.append({"role": "assistant", "content": "小人正在思考..."})
    yield chat_state, ""

    # Step 4: Initialize agent if needed
    if agent_instance is None:
        init_agent()

    if agent_instance is None or agent_instance.client is None:
        # Replace loading with error message
        chat_state[-1] = {"role": "assistant", "content": "小人目前身体不适（API未配置或无效），无法为主人服务...请设置 GEMINI_API_KEY 环境变量。"}
        yield chat_state, ""
        return

    # Step 5: Check if user uploaded a new image
    chat_image_path = image_path
    if chat_image_path and agent_instance:
        agent_instance.last_image_path = chat_image_path

    # Step 6: Get AI response with history synchronization
    try:
        # Sync Gradio history to Agent before chat
        # Bug fix: Use robust filtering instead of hardcoded slicing
        # Filter out incomplete states: messages with "小人正在思考..." or pending user messages
        def is_complete_message(msg):
            if not isinstance(msg, dict):
                return False
            content = msg.get("content", "")
            # Exclude "thinking" messages and incomplete states
            if content == "小人正在思考...":
                return False
            return True
        
        history_to_sync = [msg for msg in chat_state if is_complete_message(msg)]
        agent_instance.sync_history_from_gradio(history_to_sync)

        response = agent_instance.chat(message, chat_image_path)

        # Step 7: Replace "思考中" with actual response
        chat_state[-1] = {"role": "assistant", "content": response}
        yield chat_state, ""
    except Exception as e:
        # Replace loading message with error
        error_msg = f"抱歉主人，处理时出了点小问题：{str(e)[:100]}"
        chat_state[-1] = {"role": "assistant", "content": error_msg}
        yield chat_state, ""


def reset_demo():
    """Reset demo state."""
    global upload_workflow_state, last_extracted_items, last_uploaded_image
    upload_workflow_state = None
    last_extracted_items = []
    last_uploaded_image = None

    if agent_instance:
        agent_instance.reset_conversation()

    # Reset database (empty)
    try:
        with open("database.json", "w", encoding="utf-8") as f:
            json.dump({"wardrobe": []}, f, ensure_ascii=False, indent=2)
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
            except Exception as e:
                print(f"[Reset] Failed to remove {f}: {e}")
        # Clean temp upload files
        for f in glob.glob("./temp_upload_*.jpg"):
            try:
                os.remove(f)
                cleanup_count += 1
            except Exception as e:
                print(f"[Reset] Failed to remove {f}: {e}")
        # Clean debug files
        for f in glob.glob("/tmp/last_search_*.txt") + glob.glob("/tmp/last_google_lens_*.json"):
            try:
                os.remove(f)
                cleanup_count += 1
            except Exception as e:
                print(f"[Reset] Failed to remove {f}: {e}")
                pass
        print(f"[Reset] Cleaned {cleanup_count} files")
    except Exception as e:
        print(f"[Reset] Cleanup warning: {e}")

    return (
        "等待主人上传...",
        [],  # Reset chat_state to empty list
        None,
        None,
        None,
        None
    )


# ========== IDM-VTON Virtual Try-On Functions ==========

def get_extracted_clothes():
    """Get list of extracted clothes for dropdown selection."""
    clothes_list = []
    try:
        import glob
        image_extensions = ['*.png', '*.jpg', '*.jpeg']
        for ext in image_extensions:
            for f in sorted(glob.glob(f"{EXTRACTED_DIR}/{ext}")):
                filename = os.path.basename(f)
                # Skip detection files and temp files
                if 'detection' not in filename and not filename.startswith('.'):
                    clothes_list.append((filename, os.path.abspath(f)))
    except Exception as e:
        print(f"[get_extracted_clothes] Error: {e}")
    return clothes_list


def virtual_try_on_handler(person_image, clothes_image_path, prompt, steps, guidance, seed, preserve_face=True):
    """
    Handler for virtual try-on.
    
    Args:
        person_image: PIL Image of the person (from webcam or upload)
        clothes_image_path: Path to the clothes image (str path or PIL Image)
        prompt: Text prompt
        steps: Number of inference steps
        guidance: Guidance scale
        seed: Random seed
        preserve_face: Whether to preserve the original face
    
    Returns:
        Tuple of (result_image, status_message)
    """
    global last_uploaded_image
    
    # If no person_image provided, try to use the last uploaded image from GSAM
    if person_image is None:
        if last_uploaded_image and os.path.exists(last_uploaded_image):
            try:
                person_image = Image.open(last_uploaded_image).convert("RGB")
                print(f"[VTON] Using last uploaded image as person base: {last_uploaded_image}")
            except Exception as e:
                print(f"[VTON] Failed to load last uploaded image: {e}")
                return None, "❌ 请上传人物照片（或先使用左侧分割功能上传照片）"
        else:
            return None, "❌ 请上传人物照片（或先使用左侧分割功能上传照片）"
    
    if not clothes_image_path:
        return None, "❌ 请选择要试穿的衣服"
    
    if not idm_vton_client.available:
        return None, "❌ IDM-VTON 服务不可用，请先启动服务：python idm_vton_service.py"
    
    try:
        # Convert PIL to the format expected by client
        # The client has try_on_images method that accepts PIL images directly
        # Bug fix: Handle both string path and PIL Image
        if isinstance(clothes_image_path, str):
            if not os.path.exists(clothes_image_path):
                return None, f"❌ 找不到服装图片：{clothes_image_path}"
            clothes_image = Image.open(clothes_image_path).convert("RGB")
        else:
            # Assume it's a PIL Image
            clothes_image = clothes_image_path.convert("RGB")
        
        # Resize images to appropriate size
        person_image = person_image.convert("RGB")
        
        # Run virtual try-on
        result_image = idm_vton_client.try_on_images(
            person_image=person_image,
            clothes_image=clothes_image,
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=seed,
            preserve_face=preserve_face,
        )
        
        face_msg = " (已保留原脸)" if preserve_face else ""
        return result_image, f"✅ 虚拟试衣完成！{face_msg}"
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 试衣失败: {str(e)}"
        print(traceback.format_exc())
        return None, error_msg


def refresh_clothes_list():
    """Refresh the clothes dropdown list."""
    clothes = get_extracted_clothes()
    choices = [("-- 请选择 --", "")] + clothes
    return gr.Dropdown(choices=choices, value="")


# ========== Gradio UI ==========

with gr.Blocks(title="🤵 AI 时尚管家 - FashionClaw") as demo:
    gr.Markdown("""
    # 🤵 AI 时尚管家 · FashionClaw
    ### 您的专属AI时尚顾问 · 实时抠图 + 智能断舍离 + 虚拟试衣
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

            # 使用 State 存储聊天历史，避免 Gradio 生成器函数状态同步问题
            chat_state = gr.State([])

            # Chat interface
            chatbot = gr.Chatbot(
                label="",
                height=500,
                avatar_images=("👤", "🤵"),
                show_label=False
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
    # 关键修复：使用 chat_state 作为输入/输出，而不是 chatbot
    # 这样避免 Gradio 生成器函数状态同步问题
    process_btn.click(
        fn=process_with_technical_log,
        inputs=[upload_image, item_prefix, tech_log, chat_state],
        outputs=[tech_log, chat_state, upper_detection_result, upper_result, lower_detection_result, lower_result],
        show_progress=True
    )
    # 使用 gr.State 变化触发 chatbot 更新
    chat_state.change(
        fn=lambda x: x,
        inputs=[chat_state],
        outputs=[chatbot]
    )


    def on_send_stream(msg, chat_state):
        if not msg or not msg.strip():
            yield chat_state, ""
            return
        yield from chat_with_butler_stream(msg, None, chat_state)

    send_btn.click(
        fn=on_send_stream,
        inputs=[msg_input, chat_state],
        outputs=[chat_state, msg_input],
        show_progress="hidden"
    )

    msg_input.submit(
        fn=on_send_stream,
        inputs=[msg_input, chat_state],
        outputs=[chat_state, msg_input],
        show_progress="hidden"
    )

    reset_btn.click(
        fn=reset_demo,
        outputs=[tech_log, chat_state, upper_detection_result, upper_result, lower_detection_result, lower_result]
    )

    # Auto-reset on load
    demo.load(reset_demo, outputs=[tech_log, chat_state, upper_detection_result, upper_result, lower_detection_result, lower_result])

    # ========== IDM-VTON Virtual Try-On Section ==========
    gr.Markdown("---")
    gr.Markdown("### 👗 虚拟试衣 (IDM-VTON)")
    gr.Markdown("*上传人物照片，选择衣物，AI 生成试穿效果*")

    with gr.Row():
        # Left: Inputs
        with gr.Column(scale=1):
            gr.Markdown("#### 📷 人物照片")
            vton_person_image = gr.Image(
                type="pil",
                label="上传或拍摄人物照片（留空则使用左侧上传的照片）",
                sources=["upload", "webcam"]
            )

            gr.Markdown("#### 👕 选择衣物")
            # Dropdown for extracted clothes
            clothes_dropdown = gr.Dropdown(
                choices=[("-- 请选择 --", "")],
                value="",
                label="从已提取的衣物中选择",
                interactive=True
            )
            refresh_btn = gr.Button("🔄 刷新衣物列表", size="sm")

            # Or upload custom clothes
            gr.Markdown("*或上传自定义衣物照片*")
            vton_clothes_image = gr.Image(
                type="pil",
                label="上传衣物照片（可选）",
                sources=["upload"]
            )

            gr.Markdown("#### ⚙️ 高级参数")
            with gr.Accordion("调整生成参数", open=False):
                vton_prompt = gr.Textbox(
                    label="提示词",
                    value="a photo of a person wearing clothes",
                    placeholder="描述想要的效果"
                )
                vton_steps = gr.Slider(
                    label="推理步数",
                    minimum=10,
                    maximum=50,
                    value=30,
                    step=1
                )
                vton_guidance = gr.Slider(
                    label="引导系数",
                    minimum=1.0,
                    maximum=5.0,
                    value=2.0,
                    step=0.1
                )
                vton_seed = gr.Number(
                    label="随机种子",
                    value=42,
                    precision=0
                )
                vton_preserve_face = gr.Checkbox(
                    label="保留原脸 (Face Preservation)",
                    value=True,
                    info="使用SCHP模型提取脸部并拼回生成结果"
                )

            vton_btn = gr.Button("✨ 开始试衣", variant="primary", size="lg")

        # Right: Result
        with gr.Column(scale=1):
            gr.Markdown("#### 🎨 试穿结果")
            vton_result = gr.Image(
                type="pil",
                label="生成结果",
                interactive=False,
                height=600
            )
            vton_status = gr.Textbox(
                label="状态",
                value="等待开始...",
                interactive=False
            )

    # Event handlers for Virtual Try-On
    refresh_btn.click(
        fn=refresh_clothes_list,
        outputs=[clothes_dropdown]
    )

    def handle_vton(person_img, clothes_path, clothes_img, prompt, steps, guidance, seed, preserve_face):
        """Handle virtual try-on with priority to uploaded clothes image."""
        # Use uploaded clothes image if available, otherwise use dropdown selection
        clothes_source = clothes_img if clothes_img is not None else clothes_path
        return virtual_try_on_handler(person_img, clothes_source, prompt, steps, guidance, seed, preserve_face)

    vton_btn.click(
        fn=handle_vton,
        inputs=[
            vton_person_image,
            clothes_dropdown,
            vton_clothes_image,
            vton_prompt,
            vton_steps,
            vton_guidance,
            vton_seed,
            vton_preserve_face
        ],
        outputs=[vton_result, vton_status],
        show_progress=True
    )

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        theme=gr.themes.Soft()
    )

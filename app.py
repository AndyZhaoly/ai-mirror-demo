"""
FashionClaw Gradio Demo Application
Two-panel dashboard simulating backend agent and mobile app UI.
"""
import json
import base64
import os
from pathlib import Path
import gradio as gr
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
from PIL import Image
import io
import base64

# Global state for the demo
current_workflow_state = None

# Initialize GSAM client
gsam_client = GSAMClient("http://localhost:8000")

# Storage directory for extracted clothes
EXTRACTED_DIR = "./extracted_clothes"
os.makedirs(EXTRACTED_DIR, exist_ok=True)
# Original demo database (used to reset on "重新开始")
INITIAL_DATABASE = {
    "wardrobe": [
        {
            "item_id": "001",
            "name": "Blue Denim Jacket",
            "last_worn_days_ago": 45,
            "status": "in_closet",
            "original_price": 299,
            "image": "images/denim_jacket.jpg",
        },
        {
            "item_id": "002",
            "name": "Red Summer Dress",
            "last_worn_days_ago": 420,
            "status": "in_closet",
            "original_price": 189,
            "image": "images/red_dress.jpg",
        },
        {
            "item_id": "003",
            "name": "Vintage Wool Sweater",
            "last_worn_days_ago": 500,
            "status": "in_closet",
            "original_price": 350,
            "image": "images/wool_sweater.jpg",
        },
        {
            "item_id": "004",
            "name": "Brown Leather Belt",
            "last_worn_days_ago": 380,
            "status": "in_closet",
            "original_price": 89,
            "image": "images/leather_belt.jpg",
        },
        {
            "item_id": "005",
            "name": "Black Slim Trousers",
            "last_worn_days_ago": 120,
            "status": "in_closet",
            "original_price": 259,
            "image": "images/black_trousers.jpg",
        },
        {
            "item_id": "006",
            "name": "White Linen Shirt",
            "last_worn_days_ago": 15,
            "status": "in_closet",
            "original_price": 179,
            "image": "images/white_shirt.jpg",
        },
        {
            "item_id": "007",
            "name": "Green Bomber Jacket",
            "last_worn_days_ago": 200,
            "status": "in_closet",
            "original_price": 499,
            "image": "images/green_bomber.jpg",
        },
    ]
}


def img_to_base64(path: str, max_width: int = 400) -> str:
    """Convert local image file to resized base64 data URI for HTML rendering."""
    if not path:
        return ""
    try:
        from PIL import Image

        img = Image.open(path)
        if img.width > max_width:
            ratio = max_width / img.width
            new_size = (max_width, int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        ext = Path(path).suffix.lstrip(".").lower()
        if ext == "jpg":
            ext = "jpeg"
        fmt = ext.upper() if ext in {"jpeg", "png", "gif", "webp", "bmp"} else "JPEG"

        from io import BytesIO

        buffer = BytesIO()
        img.save(buffer, format=fmt)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        mime = "jpeg" if ext == "jpg" else (ext if ext else "jpeg")
        return f"data:image/{mime};base64,{b64}"
    except Exception as e:
        print(f"[img_to_base64] Failed to encode {path}: {e}")
        return ""


def load_pil_image(path: str, max_width: int = 300):
    """Load and resize a local image as a PIL Image for Gradio."""
    if not path:
        return None
    try:
        from PIL import Image
        img = Image.open(path)
        if img.width > max_width:
            ratio = max_width / img.width
            new_size = (max_width, int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        return img
    except Exception as e:
        print(f"[load_pil_image] Failed to open {path}: {e}")
        return None


def format_logs(log_messages):
    """Format log messages for display."""
    if not log_messages:
        return "暂无日志..."
    return "\n".join(log_messages)


def start_workflow():
    """Start the workflow and return initial state."""
    global current_workflow_state

    # Reset state
    current_workflow_state = create_initial_state()

    # Run workflow until user input needed
    current_workflow_state = run_workflow_until_user_input()

    logs = format_logs(current_workflow_state.get("log_messages", []))
    status = current_workflow_state.get("status", "")

    if status == "no_items_found":
        return (
            logs,
            "<h2>ℹ️ 系统通知</h2><p>暂无闲置衣物需要处理。所有衣物都在正常使用中！</p>",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    elif status == "error":
        return (
            logs,
            "<h2>❌ 系统错误</h2><p>处理过程中出现错误，请查看日志。</p>",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    elif status == "awaiting_user_decision":
        decision = current_workflow_state.get("agent_decision", "")
        item = current_workflow_state.get("current_item", {})

        mobile_ui = f"""
<div>
<h2>📱 FashionClaw App</h2>
<h3>🔔 闲置衣物处理通知</h3>
<div style="margin:12px 0;">{decision.replace(chr(10), '<br>')}</div>
<hr>
<p><strong>👕 衣物详情:</strong></p>
<ul>
<li>名称: {item.get('name', 'N/A')}</li>
<li>闲置天数: {item.get('last_worn_days_ago', 'N/A')} 天</li>
<li>原价: ¥{item.get('original_price', 'N/A')}</li>
</ul>
<p><strong>💵 交易详情:</strong></p>
<ul>
<li>买家出价: ¥{current_workflow_state.get('buyer_offer', {}).get('offer_price', 'N/A')}</li>
<li>平台: {current_workflow_state.get('buyer_offer', {}).get('platform', 'N/A')}</li>
</ul>
</div>
"""
        return (
            logs,
            mobile_ui,
            gr.update(visible=True, value="✅ 确认出售"),
            gr.update(visible=True, value="❌ 拒绝出售"),
            gr.update(visible=False),
        )

    return logs, "<p>等待启动...</p>", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


def approve_sale():
    """Handle user approving the sale."""
    global current_workflow_state

    if not current_workflow_state:
        return "请先启动工作流", "错误：工作流未启动", gr.update(), gr.update(), gr.update()

    # Resume workflow with approval
    final_state = resume_workflow(user_approved=True)

    if not final_state:
        return "错误：无法恢复工作流", "错误：状态丢失", gr.update(), gr.update(), gr.update()

    logs = format_logs(final_state.get("log_messages", []))

    tracking = final_state.get("tracking_info", {})
    item = final_state.get("current_item", {})
    buyer = final_state.get("buyer_offer", {}).get("buyer_name", "N/A")
    price = final_state.get("buyer_offer", {}).get("offer_price", "N/A")

    success_ui = f"""
<div>
<h2>✅ 交易成功！</h2>
<p>您的衣物已成功售出！</p>
<hr>
<h3>📦 交易详情</h3>
<table border="0" cellpadding="4">
<tr><td><strong>衣物</strong></td><td>{item.get('name', 'N/A')}</td></tr>
<tr><td><strong>成交价</strong></td><td>¥{price}</td></tr>
<tr><td><strong>买家</strong></td><td>{buyer}</td></tr>
<tr><td><strong>物流公司</strong></td><td>{tracking.get('carrier', 'N/A')}</td></tr>
<tr><td><strong>运单号</strong></td><td><strong>{tracking.get('tracking_number', 'N/A')}</strong></td></tr>
<tr><td><strong>预计送达</strong></td><td>{tracking.get('estimated_delivery', 'N/A')}</td></tr>
</table>
<hr>
<p>🎉 感谢您的使用！物流信息已同步至您的账户。</p>
</div>
"""

    return (
        logs,
        success_ui,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
    )


def reject_sale():
    """Handle user rejecting the sale."""
    global current_workflow_state

    if not current_workflow_state:
        return "请先启动工作流", "错误：工作流未启动", gr.update(), gr.update(), gr.update()

    # Resume workflow with rejection
    final_state = resume_workflow(user_approved=False)

    if not final_state:
        return "错误：无法恢复工作流", "错误：状态丢失", gr.update(), gr.update(), gr.update()

    logs = format_logs(final_state.get("log_messages", []))

    item = current_workflow_state.get("current_item", {})

    reject_ui = f"""
<div>
<h2>❌ 已拒绝出售</h2>
<p>您已选择保留这件衣物。</p>
<hr>
<p><strong>👕 衣物信息:</strong></p>
<ul>
<li>名称: {item.get('name', 'N/A')}</li>
<li>状态: 继续保留在衣橱中</li>
</ul>
<p><strong>💡 小贴士:</strong></p>
<p>建议在未来30天内穿着一次，<br>否则系统将重新推荐出售。</p>
<hr>
<p>衣物已标记为"保留"，短期内不会再次推荐出售。</p>
</div>
"""

    return (
        logs,
        reject_ui,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
    )


def reset_demo():
    """Reset the demo to initial state."""
    global current_workflow_state
    current_workflow_state = None

    # Restore the database so previous sales don't exhaust the demo
    try:
        with open("database.json", "w", encoding="utf-8") as f:
            json.dump(INITIAL_DATABASE, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: failed to reset database: {e}")

    # Reset LangGraph checkpoint memory
    try:
        reset_workflow()
    except Exception as e:
        print(f"Warning: failed to reset workflow checkpoint: {e}")

    return (
        "点击「启动智能衣橱系统」开始演示...",
        "<p style='padding: 20px; background: #f8f9fa; border-radius: 8px;'>等待系统检测闲置衣物...</p>",
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
    )


def view_database():
    """Display current database contents."""
    try:
        with open("database.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data.get("wardrobe", [])
        if not items:
            return "数据库为空"

        rows = []
        for item in items:
            status_emoji = {
                "in_closet": "👕",
                "selling": "🏷️",
                "sold": "✅",
            }.get(item.get("status", ""), "❓")

            img_tag = ""
            b64 = img_to_base64(item.get("image"))
            if b64:
                img_tag = f"<img src='{b64}' width='60' style='border-radius:8px;'>"

            rows.append(
                f"<tr><td>{item['item_id']}</td><td>{img_tag}</td>"
                f"<td>{status_emoji} {item['name']}</td>"
                f"<td>{item['last_worn_days_ago']} 天</td>"
                f"<td>{item['status']}</td>"
                f"<td>¥{item['original_price']}</td></tr>"
            )

        table = "<table border='1' cellpadding='6' style='border-collapse:collapse;width:100%;'>"
        table += "<tr><th>ID</th><th>图片</th><th>名称</th><th>未穿天数</th><th>状态</th><th>原价</th></tr>"
        table += "".join(rows)
        table += "</table>"

        return table

    except Exception as e:
        return f"读取数据库错误: {str(e)}"


def extract_and_segment_clothing(image):
    """Extract upper and lower body clothing from uploaded image."""
    try:
        if image is None:
            return None, None, "请先上传图片", ""
        
        # Save uploaded image temporarily
        temp_path = "./temp_upload.jpg"
        image.save(temp_path)
        
        # Extract upper body
        upper_images = gsam_client.extract_upper_body(temp_path, white_background=True)
        # Extract lower body
        lower_images = gsam_client.extract_lower_body(temp_path, white_background=True)
        
        # Save extracted images
        saved_paths = []
        upper_output = None
        lower_output = None
        
        for i, img in enumerate(upper_images):
            path = os.path.join(EXTRACTED_DIR, f"upper_body_{i}.png")
            img.save(path)
            saved_paths.append(path)
            upper_output = img  # Keep reference for display
        
        for i, img in enumerate(lower_images):
            path = os.path.join(EXTRACTED_DIR, f"lower_body_{i}.png")
            img.save(path)
            saved_paths.append(path)
            lower_output = img  # Keep reference for display
        
        # Create info message
        info = f"✅ 提取完成！\n"
        info += f"👕 上衣: {len(upper_images)} 件\n"
        info += f"👖 下装: {len(lower_images)} 件\n"
        info += f"💾 已保存到: {EXTRACTED_DIR}/"
        
        # Create paths info
        paths_info = "\n".join(saved_paths) if saved_paths else "无提取结果"
        
        return upper_output, lower_output, info, paths_info
        
    except Exception as e:
        return None, None, f"❌ 错误: {str(e)}", ""


def list_extracted_clothes():
    """List all extracted clothes in the storage directory."""
    try:
        files = os.listdir(EXTRACTED_DIR)
        if not files:
            return "暂无提取的衣物"

        result = "### 📂 已提取的衣物\n\n"
        for f in sorted(files):
            if f.endswith('.png') or f.endswith('.jpg'):
                filepath = os.path.join(EXTRACTED_DIR, f)
                result += f"- **{f}**\n"
        return result
    except Exception as e:
        return f"读取失败: {str(e)}"


# ========== Upload and Detect Flow Functions ==========

# Global state for upload flow
upload_workflow_state = None
last_extracted_items = []


def upload_and_detect(image, item_name_prefix):
    """
    Handle image upload, segmentation, database registration, and stagnancy detection.
    """
    global upload_workflow_state, last_extracted_items
    last_extracted_items = []

    try:
        if image is None:
            return None, None, "请先上传图片", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        # Save uploaded image temporarily
        temp_path = "./temp_upload.jpg"
        image.save(temp_path)

        # Extract upper and lower body
        upper_images = gsam_client.extract_upper_body(temp_path, white_background=True)
        lower_images = gsam_client.extract_lower_body(temp_path, white_background=True)

        # Process extracted items
        results_info = []
        upper_output = None
        lower_output = None

        # Save and register upper body items
        for i, img in enumerate(upper_images):
            path = os.path.join(EXTRACTED_DIR, f"upper_{i}_{generate_timestamp()}.png")
            img.save(path)
            upper_output = img

            # Register to database
            name = f"{item_name_prefix}_上衣_{i+1}" if item_name_prefix else f"提取的上衣_{i+1}"
            item = add_item(
                name=name,
                clothing_type="upper",
                image_path=path,
                extracted_from="temp_upload.jpg"
            )
            last_extracted_items.append(item)
            results_info.append(f"👕 {name} (ID: {item['item_id']}, 价格: ¥{item['original_price']}, 购买日期: {item['purchase_date']})")

        # Save and register lower body items
        for i, img in enumerate(lower_images):
            path = os.path.join(EXTRACTED_DIR, f"lower_{i}_{generate_timestamp()}.png")
            img.save(path)
            lower_output = img

            name = f"{item_name_prefix}_下装_{i+1}" if item_name_prefix else f"提取的下装_{i+1}"
            item = add_item(
                name=name,
                clothing_type="lower",
                image_path=path,
                extracted_from="temp_upload.jpg"
            )
            last_extracted_items.append(item)
            results_info.append(f"👖 {name} (ID: {item['item_id']}, 价格: ¥{item['original_price']}, 购买日期: {item['purchase_date']})")

        # Check stagnancy for each item
        stagnant_items = [item for item in last_extracted_items if is_stagnant(item)]

        if stagnant_items:
            # Use the first stagnant item for the workflow
            target_item = stagnant_items[0]
            upload_workflow_state = run_upload_workflow_until_user_input(target_item)

            info_text = f"✅ 提取并注册完成！\n\n"
            info_text += "已注册物品:\n" + "\n".join(results_info) + "\n\n"

            if upload_workflow_state.get("status") == "awaiting_user_decision":
                info_text += f"⚠️ 物品「{target_item['name']}」已闲置超过365天！"

                # Build the mobile UI for sell prompt
                decision = upload_workflow_state.get("agent_decision", "")
                mobile_html = f"""
                <div style="padding: 16px; background: #fff3cd; border-radius: 8px; border-left: 4px solid #ffc107;">
                    <h3>⚠️ 闲置衣物检测</h3>
                    <p><strong>物品:</strong> {target_item['name']}</p>
                    <p><strong>购买日期:</strong> {target_item['purchase_date']}</p>
                    <p><strong>闲置状态:</strong> 超过365天未使用</p>
                    <hr style="margin: 12px 0;">
                    {decision.replace(chr(10), '<br>')}
                </div>
                """

                return (
                    upper_output,
                    lower_output,
                    info_text,
                    mobile_html,
                    gr.update(visible=True),   # approve button
                    gr.update(visible=True),   # reject button
                    gr.update(visible=False)   # restart button
                )
            else:
                info_text += "✅ 所有物品状态正常，无需出售。"
                return (
                    upper_output,
                    lower_output,
                    info_text,
                    "<p>暂无闲置物品需要处理</p>",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True)
                )
        else:
            info_text = f"✅ 提取并注册完成！\n\n"
            info_text += "已注册物品:\n" + "\n".join(results_info) + "\n\n"
            info_text += "✅ 所有物品状态正常，无需出售。"

            return (
                upper_output,
                lower_output,
                info_text,
                "<p>暂无闲置物品需要处理</p>",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True)
            )

    except Exception as e:
        return None, None, f"❌ 错误: {str(e)}", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


def approve_upload_sale():
    """Handle user approving sale from upload flow."""
    global upload_workflow_state

    if not upload_workflow_state:
        return "错误: 工作流未启动", gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    # Resume workflow with approval
    final_state = resume_workflow(user_approved=True)

    if not final_state:
        return "错误: 无法完成交易", gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    tracking = final_state.get("tracking_info", {})
    item = final_state.get("current_item", {})

    success_html = f"""
    <div style="padding: 16px; background: #d4edda; border-radius: 8px; border-left: 4px solid #28a745;">
        <h3>✅ 交易成功!</h3>
        <p>您的衣物「{item.get('name', 'N/A')}」已成功售出!</p>
        <hr>
        <p><strong>成交价:</strong> ¥{final_state.get('buyer_offer', {}).get('offer_price', 'N/A')}</p>
        <p><strong>物流公司:</strong> {tracking.get('carrier', 'N/A')}</p>
        <p><strong>运单号:</strong> {tracking.get('tracking_number', 'N/A')}</p>
        <p><strong>预计送达:</strong> {tracking.get('estimated_delivery', 'N/A')}</p>
    </div>
    """

    return success_html, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


def reject_upload_sale():
    """Handle user rejecting sale from upload flow."""
    global upload_workflow_state

    if not upload_workflow_state:
        return "已取消出售", gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    # Resume workflow with rejection
    final_state = resume_workflow(user_approved=False)

    item = upload_workflow_state.get("current_item", {})

    reject_html = f"""
    <div style="padding: 16px; background: #f8d7da; border-radius: 8px; border-left: 4px solid #dc3545;">
        <h3>❌ 已拒绝出售</h3>
        <p>您已选择保留「{item.get('name', 'N/A')}」。</p>
        <p>该物品将继续保留在您的衣橱中。</p>
    </div>
    """

    return reject_html, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


def generate_timestamp():
    """Generate timestamp string for unique filenames."""
    from datetime import datetime
    return datetime.now().strftime("%m%d_%H%M%S")


# ========== Gradio UI ==========

with gr.Blocks(title="FashionClaw 智能衣橱系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧥 FashionClaw 智能衣橱系统
    ## AI 驱动的多代理衣橱管理平台
    """)

    # Tab 1: Upload and Detect (New Flow)
    with gr.Tab("📤 上传&检测"):
        gr.Markdown("""
        ### 上传照片 → 自动分割 → 检测闲置 → 提示出售
        上传您的穿搭照片，系统自动提取衣物并检测是否需要出售
        """)

        with gr.Row():
            # Left Column: Upload and Results
            with gr.Column(scale=1):
                gr.Markdown("### 📤 上传与提取")

                upload_image = gr.Image(
                    type="pil",
                    label="上传您的穿搭照片",
                    sources=["upload"]
                )

                item_name_prefix = gr.Textbox(
                    label="衣物名称前缀（可选）",
                    placeholder="例如：我的",
                    value=""
                )

                upload_detect_btn = gr.Button("🚀 上传并检测", variant="primary", size="lg")

                gr.Markdown("---")
                gr.Markdown("### 📊 处理结果")

                upload_info = gr.Textbox(
                    label="注册信息",
                    value="等待上传...",
                    lines=6,
                    interactive=False
                )

                with gr.Row():
                    with gr.Column():
                        upper_result = gr.Image(
                            type="pil",
                            label="👕 提取的上衣",
                            interactive=False
                        )
                    with gr.Column():
                        lower_result = gr.Image(
                            type="pil",
                            label="👖 提取的下装",
                            interactive=False
                        )

            # Right Column: Prompt and Action
            with gr.Column(scale=1):
                gr.Markdown("### 📱 闲置检测提示")

                upload_mobile_ui = gr.HTML(
                    "<p style='padding: 20px; background: #f8f9fa; border-radius: 8px;'>"
                    "上传图片后将在此处显示闲置检测结果..."
                    "</p>"
                )

                with gr.Row():
                    upload_approve_btn = gr.Button(
                        "✅ 确认出售",
                        variant="primary",
                        size="lg",
                        visible=False,
                    )
                    upload_reject_btn = gr.Button(
                        "❌ 拒绝出售",
                        variant="secondary",
                        size="lg",
                        visible=False,
                    )

                upload_restart_btn = gr.Button(
                    "🔄 重新开始",
                    size="lg",
                    visible=True,
                )

        # Event handlers for upload tab
        upload_detect_btn.click(
            fn=upload_and_detect,
            inputs=[upload_image, item_name_prefix],
            outputs=[upper_result, lower_result, upload_info, upload_mobile_ui,
                     upload_approve_btn, upload_reject_btn, upload_restart_btn]
        )

        upload_approve_btn.click(
            fn=approve_upload_sale,
            outputs=[upload_mobile_ui, upload_approve_btn, upload_reject_btn, upload_restart_btn]
        )

        upload_reject_btn.click(
            fn=reject_upload_sale,
            outputs=[upload_mobile_ui, upload_approve_btn, upload_reject_btn, upload_restart_btn]
        )

        upload_restart_btn.click(
            fn=lambda: ("等待上传...",
                       "<p style='padding: 20px; background: #f8f9fa; border-radius: 8px;'>上传图片后将在此处显示闲置检测结果...</p>",
                       gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)),
            outputs=[upload_info, upload_mobile_ui, upload_approve_btn, upload_reject_btn, upload_restart_btn]
        )

    # Tab 2: Original Workflow
    with gr.Tab("🤖 智能衣橱工作流"):
        gr.Markdown("""
        ### 本系统演示：自动检测闲置衣物 → 评估市场价值 → 匹配买家 → 用户确认 → 执行交易
        """)

        with gr.Row():
            # Left Column: Agent Backend Dashboard
            with gr.Column(scale=1):
                gr.Markdown("### 🤖 代理后端监控面板")

                log_output = gr.Textbox(
                    label="系统日志",
                    value="点击「启动智能衣橱系统」开始演示...",
                    lines=20,
                    max_lines=30,
                    autoscroll=True,
                    interactive=False,
                )

                with gr.Row():
                    start_btn = gr.Button("🚀 启动智能衣橱系统", variant="primary", size="lg")
                    reset_btn = gr.Button("🔄 重置演示", size="lg")

                gr.Markdown("---")
                gr.Markdown("### 🗄️ 当前数据库状态")
                db_view = gr.HTML("<p>点击「刷新数据库」查看...</p>")
                refresh_db_btn = gr.Button("🔄 刷新数据库")

            # Right Column: Mobile App Mockup
            with gr.Column(scale=1):
                gr.Markdown("### 📱 移动端 App 界面")

                mobile_ui = gr.HTML(
                    "<p style='padding: 20px; background: #f8f9fa; border-radius: 8px;'>等待系统检测闲置衣物...</p>"
                )

                with gr.Row():
                    approve_btn = gr.Button(
                        "✅ 确认出售",
                        variant="primary",
                        size="lg",
                        visible=False,
                    )
                    reject_btn = gr.Button(
                        "❌ 拒绝出售",
                        variant="secondary",
                        size="lg",
                        visible=False,
                    )

                restart_btn = gr.Button(
                    "🔄 重新开始",
                    size="lg",
                    visible=True,
                )

                gr.Markdown("---")

                gr.Markdown("""
                #### 📝 使用说明

                1. 点击左侧「启动智能衣橱系统」按钮
                2. 观察代理后端日志（左侧面板）
                3. 在右侧移动端界面查看推荐
                4. 点击「确认出售」或「拒绝出售"
                5. 查看交易结果和物流信息

                ---

                #### 🏗️ 系统架构

                ```
                ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
                │  Monitor    │ --> │  Evaluate   │ --> │  User Wait  │
                │  (监控)      │     │  (评估)      │     │  (等待用户)  │
                └─────────────┘     └─────────────┘     └──────┬──────┘
                                                               │
                                   ┌───────────────────────────┘
                                   │
                                   ▼
                         ┌─────────────┐
                         │  Execute    │
                         │  (执行交易)  │
                         └─────────────┘
            ```
            """)

        # Event handlers for workflow tab
        start_btn.click(
            fn=start_workflow,
            outputs=[log_output, mobile_ui, approve_btn, reject_btn, restart_btn],
        )

        approve_btn.click(
            fn=approve_sale,
            outputs=[log_output, mobile_ui, approve_btn, reject_btn, restart_btn],
        )

        reject_btn.click(
            fn=reject_sale,
            outputs=[log_output, mobile_ui, approve_btn, reject_btn, restart_btn],
        )

        restart_btn.click(
            fn=reset_demo,
            outputs=[log_output, mobile_ui, approve_btn, reject_btn, restart_btn],
        )

        reset_btn.click(
            fn=reset_demo,
            outputs=[log_output, mobile_ui, approve_btn, reject_btn, restart_btn],
        )

        refresh_db_btn.click(
            fn=view_database,
            outputs=db_view,
        )

# Always start with a fresh database on module load so previous demo runs don't pollute state
try:
    with open("database.json", "w", encoding="utf-8") as f:
        json.dump(INITIAL_DATABASE, f, ensure_ascii=False, indent=2)
except Exception as e:
    print(f"Warning: failed to auto-reset database on startup: {e}")

if __name__ == "__main__":
    demo.launch(share=True)

"""
FashionClaw Gradio Demo Application
Two-panel dashboard simulating backend agent and mobile app UI.
"""
import json
import base64
from pathlib import Path
from PIL import Image
import gradio as gr
from workflow import (
    workflow_app,
    create_initial_state,
    run_workflow_until_user_input,
    resume_workflow,
)

# Global state for the demo
current_workflow_state = None


def load_item_image(path: str):
    """Load image as PIL Image for Gradio, or None if missing."""
    if not path:
        return None
    try:
        return Image.open(path)
    except Exception:
        return None


def img_to_base64(path: str) -> str:
    """Convert local image file to base64 data URI for Markdown."""
    if not path:
        return ""
    try:
        ext = Path(path).suffix.lstrip(".").lower()
        if ext == "jpg":
            ext = "jpeg"
        with open(path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:image/{ext};base64,{b64}"
    except Exception:
        return ""


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
            "## ℹ️ 系统通知\n\n暂无闲置衣物需要处理。所有衣物都在正常使用中！",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            None,
        )
    elif status == "error":
        return (
            logs,
            "## ❌ 系统错误\n\n处理过程中出现错误，请查看日志。",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            None,
        )
    elif status == "awaiting_user_decision":
        decision = current_workflow_state.get("agent_decision", "")
        item = current_workflow_state.get("current_item", {})

        mobile_ui = f"""
## 📱 FashionClaw App

### 🔔 闲置衣物处理通知

{decision}

---

**👕 衣物详情:**
- 名称: {item.get('name', 'N/A')}
- 闲置天数: {item.get('last_worn_days_ago', 'N/A')} 天
- 原价: ¥{item.get('original_price', 'N/A')}

**💵 交易详情:**
- 买家出价: ¥{current_workflow_state.get('buyer_offer', {}).get('offer_price', 'N/A')}
- 平台: {current_workflow_state.get('buyer_offer', {}).get('platform', 'N/A')}
"""
        return (
            logs,
            mobile_ui,
            gr.update(visible=True, value="✅ 确认出售"),
            gr.update(visible=True, value="❌ 拒绝出售"),
            gr.update(visible=False),
            gr.update(visible=True, value=load_item_image(item.get("image"))),
        )

    return logs, "等待启动...", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), None


def approve_sale():
    """Handle user approving the sale."""
    global current_workflow_state

    if not current_workflow_state:
        return "请先启动工作流", "错误：工作流未启动", gr.update(), gr.update(), gr.update(), None

    # Resume workflow with approval
    final_state = resume_workflow(user_approved=True)

    if not final_state:
        return "错误：无法恢复工作流", "错误：状态丢失", gr.update(), gr.update(), gr.update(), None

    logs = format_logs(final_state.get("log_messages", []))

    tracking = final_state.get("tracking_info", {})
    item = final_state.get("current_item", {})
    buyer = final_state.get("buyer_offer", {}).get("buyer_name", "N/A")
    price = final_state.get("buyer_offer", {}).get("offer_price", "N/A")

    success_ui = f"""
## ✅ 交易成功！

您的衣物已成功售出！

---

### 📦 交易详情

| 项目 | 内容 |
|------|------|
| 衣物 | {item.get('name', 'N/A')} |
| 成交价 | ¥{price} |
| 买家 | {buyer} |
| 物流公司 | {tracking.get('carrier', 'N/A')} |
| 运单号 | **{tracking.get('tracking_number', 'N/A')}** |
| 预计送达 | {tracking.get('estimated_delivery', 'N/A')} |

---

🎉 感谢您的使用！物流信息已同步至您的账户。
"""

    return (
        logs,
        success_ui,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=True, value=load_item_image(item.get("image"))),
    )


def reject_sale():
    """Handle user rejecting the sale."""
    global current_workflow_state

    if not current_workflow_state:
        return "请先启动工作流", "错误：工作流未启动", gr.update(), gr.update(), gr.update(), None

    # Resume workflow with rejection
    final_state = resume_workflow(user_approved=False)

    if not final_state:
        return "错误：无法恢复工作流", "错误：状态丢失", gr.update(), gr.update(), gr.update(), None

    logs = format_logs(final_state.get("log_messages", []))

    item = current_workflow_state.get("current_item", {})

    reject_ui = f"""
## ❌ 已拒绝出售

您已选择保留这件衣物。

---

**👕 衣物信息:**
- 名称: {item.get('name', 'N/A')}
- 状态: 继续保留在衣橱中

💡 **小贴士:**
建议在未来30天内穿着一次，
否则系统将重新推荐出售。

---

衣物已标记为"保留"，短期内不会再次推荐出售。
"""

    return (
        logs,
        reject_ui,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=True, value=load_item_image(item.get("image"))),
    )


def reset_demo():
    """Reset the demo to initial state."""
    global current_workflow_state
    current_workflow_state = None

    return (
        "点击「启动智能衣橱系统」开始演示...",
        "## 📱 FashionClaw App\n\n等待系统检测闲置衣物...",
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        None,
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
                f"| {item['item_id']} | {img_tag} | {status_emoji} {item['name']} | "
                f"{item['last_worn_days_ago']} 天 | {item['status']} | ¥{item['original_price']} |"
            )

        table = "| ID | 图片 | 名称 | 未穿天数 | 状态 | 原价 |\n"
        table += "|---|---|---|---|---|---|\n"
        table += "\n".join(rows)

        return table

    except Exception as e:
        return f"读取数据库错误: {str(e)}"


# ========== Gradio UI ==========

with gr.Blocks(title="FashionClaw 智能衣橱系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧥 FashionClaw 智能衣橱系统
    ## AI 驱动的多代理衣橱管理平台

    本系统演示：自动检测闲置衣物 → 评估市场价值 → 匹配买家 → 用户确认 → 执行交易
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
            db_view = gr.Markdown("点击「刷新数据库」查看...")
            refresh_db_btn = gr.Button("🔄 刷新数据库")

        # Right Column: Mobile App Mockup
        with gr.Column(scale=1):
            gr.Markdown("### 📱 移动端 App 界面")

            mobile_ui = gr.Markdown(
                "## 📱 FashionClaw App\n\n等待系统检测闲置衣物..."
            )

            item_image = gr.Image(
                label="衣物照片",
                type="pil",
                visible=False,
                height=300,
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

    # Event handlers
    start_btn.click(
        fn=start_workflow,
        outputs=[log_output, mobile_ui, approve_btn, reject_btn, restart_btn, item_image],
    )

    approve_btn.click(
        fn=approve_sale,
        outputs=[log_output, mobile_ui, approve_btn, reject_btn, restart_btn, item_image],
    )

    reject_btn.click(
        fn=reject_sale,
        outputs=[log_output, mobile_ui, approve_btn, reject_btn, restart_btn, item_image],
    )

    restart_btn.click(
        fn=reset_demo,
        outputs=[log_output, mobile_ui, approve_btn, reject_btn, restart_btn, item_image],
    )

    reset_btn.click(
        fn=reset_demo,
        outputs=[log_output, mobile_ui, approve_btn, reject_btn, restart_btn, item_image],
    )

    refresh_db_btn.click(
        fn=view_database,
        outputs=db_view,
    )

if __name__ == "__main__":
    demo.launch(share=False)

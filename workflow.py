"""
FashionClaw LangGraph Workflow
Implements the multi-agent system for intelligent wardrobe management.
"""
import json
import os
import random
import time
from typing import TypedDict, List, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from mock_apis import (
    check_market_price,
    get_buyer_offer,
    check_buyer_credit,
    execute_logistics,
    update_item_status,
)
from database_manager import is_stagnant
from tools import PricingTool, generate_pricing_report

# Global PricingTool instance
pricing_tool = PricingTool()


class WorkflowState(TypedDict):
    """State for the FashionClaw workflow."""
    current_item: Optional[dict]  # The clothing item being processed
    market_price: Optional[int]
    buyer_offer: Optional[dict]
    buyer_credit: Optional[dict]
    agent_decision: Optional[str]  # The recommendation to show user
    user_approved: Optional[bool]  # User's decision
    sale_executed: bool  # Whether sale has been completed
    tracking_info: Optional[dict]
    log_messages: List[str]  # For display in UI
    status: str  # Current workflow status
    vlm_analysis: Optional[dict]  # VLM clothing analysis results
    pricing_data: Optional[dict]  # Real pricing data from Xianyu query
    search_result: Optional[dict]  # Placeholder for any future search integration


def create_initial_state() -> WorkflowState:
    """Create a fresh state for a new workflow run."""
    return {
        "current_item": None,
        "market_price": None,
        "buyer_offer": None,
        "buyer_credit": None,
        "agent_decision": None,
        "user_approved": None,
        "sale_executed": False,
        "tracking_info": None,
        "log_messages": [],
        "status": "initialized",
    }


def add_log(state: WorkflowState, message: str) -> None:
    """Add a timestamped log message to the state."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    state["log_messages"].append(f"[{timestamp}] {message}")


# ========== Node Functions ==========

def monitor_node(state: WorkflowState) -> WorkflowState:
    """
    Monitor Node: Scans database for items not worn in 365+ days.
    """
    add_log(state, "🔍 Agent 启动: 开始扫描衣橱数据库...")
    add_log(state, "📊 正在分析每件衣物的穿着频率...")

    try:
        with open("database.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        # Find items not worn in 365+ days and still in closet
        stale_items = [
            item for item in data["wardrobe"]
            if item["last_worn_days_ago"] > 365 and item["status"] == "in_closet"
        ]

        if not stale_items:
            add_log(state, "✅ 扫描完成: 没有发现超过365天未穿的闲置衣物")
            state["status"] = "no_items_found"
            return state

        # Pick the first stale item (FIFO strategy)
        target_item = stale_items[0]
        state["current_item"] = target_item

        add_log(state, f"🎯 发现闲置衣物: {target_item['name']} (ID: {target_item['item_id']})")
        add_log(state, f"   └─ 未穿着天数: {target_item['last_worn_days_ago']} 天")
        add_log(state, f"   └─ 原价: ¥{target_item['original_price']}")
        add_log(state, "📡 正在调用外部API进行市场评估...")
        state["status"] = "item_found"

    except Exception as e:
        add_log(state, f"❌ 扫描失败: {str(e)}")
        state["status"] = "error"

    return state


def evaluate_node(state: WorkflowState) -> WorkflowState:
    """
    Evaluate Node: Uses VLM to analyze clothing and query real market prices.
    """
    if state["status"] == "no_items_found" or state["status"] == "error":
        return state

    item = state["current_item"]
    if not item:
        add_log(state, "❌ 错误: 没有可评估的衣物")
        state["status"] = "error"
        return state

    try:
        # Step 1: VLM Analysis
        add_log(state, "🔍 正在使用AI视觉分析衣物特征...")
        image_path = item.get("image", "")
        add_log(state, f"   📸 图片路径: {image_path}")
        add_log(state, f"   📂 当前工作目录: {os.getcwd()}")
        add_log(state, f"   🏷️  物品名称: {item.get('name', 'Unknown')}")
        add_log(state, f"   🆔 物品ID: {item.get('item_id', 'Unknown')}")

        # Verify image exists
        if image_path:
            if os.path.exists(image_path):
                file_size = os.path.getsize(image_path)
                add_log(state, f"   ✅ 图片文件存在 (大小: {file_size} bytes)")
                # Check if it's a recently created file
                mtime = os.path.getmtime(image_path)
                age_seconds = time.time() - mtime
                add_log(state, f"   ⏱️  文件修改时间: {age_seconds:.1f}秒前")
            else:
                add_log(state, f"   ❌ 图片文件不存在: {image_path}")
                # Try to find the file in current directory
                basename = os.path.basename(image_path)
                if os.path.exists(basename):
                    image_path = os.path.abspath(basename)
                    add_log(state, f"   🔄 使用相对路径找到图片: {image_path}")

        vlm_analysis = None
        gemini_full_result = None
        if image_path and os.path.exists(image_path):
            add_log(state, "   🚀 调用 Gemini 3.1 Pro 分析...")
            vlm_result = pricing_tool.analyze_clothing(image_path)
            if vlm_result and vlm_result.get("success"):
                gemini_full_result = vlm_result
                vlm_analysis = vlm_result.get("item_details", {})
                official_price = vlm_result.get("official_price", {})
                resale = vlm_result.get("resale_estimate", {})

                add_log(state, f"   ├─ 品牌: {vlm_analysis.get('brand', 'Unknown')}")
                add_log(state, f"   ├─ 型号: {vlm_analysis.get('model_name', 'N/A')}")
                add_log(state, f"   ├─ 货号: {vlm_analysis.get('product_code', 'N/A')}")
                add_log(state, f"   ├─ 类别: {vlm_analysis.get('category', 'Unknown')}")
                add_log(state, f"   ├─ 材质: {vlm_analysis.get('material', 'Unknown')}")
                add_log(state, f"   ├─ 成色: {vlm_analysis.get('condition', 'Unknown')}")
                add_log(state, f"   ├─ 官方价: {official_price.get('amount', 'N/A')} {official_price.get('currency', '')}")
                add_log(state, f"   ├─ 建议售价: ¥{resale.get('max_price', 'N/A')} (二手)")
                add_log(state, f"   └─ 置信度: {vlm_analysis.get('confidence', 'N/A')}")

                if vlm_result.get('description'):
                    add_log(state, f"   📋 描述: {vlm_result['description'][:100]}...")
                if vlm_result.get('model_used'):
                    add_log(state, f"   🤖 模型: {vlm_result['model_used']}")
            else:
                add_log(state, "   ⚠️ Gemini 分析失败，尝试 Google Lens...")
        else:
            add_log(state, "   ℹ️ 未找到图片或API未配置，使用基础信息")

        # Store VLM analysis in state
        state["vlm_analysis"] = vlm_analysis or {}
        # Store full Gemini result for Agent to use
        state["gemini_result"] = gemini_full_result or {}
        # Also store detailed description if available
        state["detailed_description"] = gemini_full_result.get("description", "") if gemini_full_result else ""

        # Update item name with VLM analysis results
        if vlm_analysis:
            brand = vlm_analysis.get('brand', '')
            category = vlm_analysis.get('category', '')
            color = vlm_analysis.get('color', '')

            # Build display name from available info
            parts = []
            if brand and brand != 'Unknown':
                parts.append(brand)
            if color and color != 'Unknown':
                parts.append(color)
            if category and category != 'Unknown':
                parts.append(category)

            if parts:
                new_name = ' '.join(parts)
            elif color:
                new_name = f"{color}衣物"
            else:
                new_name = item['name']  # Fallback to original name

            add_log(state, f"   📝 识别为: {new_name}")
            # Update the item name in state for display
            state["current_item"]["display_name"] = new_name

            # Also update in database with detailed analysis
            try:
                from database_manager import load_database, save_database
                data = load_database()
                for db_item in data["wardrobe"]:
                    if db_item["item_id"] == item["item_id"]:
                        db_item["vlm_brand"] = brand
                        db_item["vlm_category"] = category
                        db_item["vlm_material"] = vlm_analysis.get('material', '')
                        db_item["vlm_condition"] = vlm_analysis.get('condition', '')
                        db_item["vlm_product_code"] = vlm_analysis.get('product_code', '')
                        db_item["vlm_model_name"] = vlm_analysis.get('model_name', '')
                        db_item["vlm_official_price"] = gemini_full_result.get('official_price', {})
                        db_item["vlm_resale_estimate"] = gemini_full_result.get('resale_estimate', {})
                        db_item["vlm_description"] = gemini_full_result.get('description', '')
                        db_item["analysis_source"] = gemini_full_result.get('analysis_source', 'gemini')
                        save_database(data)
                        add_log(state, f"   💾 已保存详细分析结果到数据库")
                        break
            except Exception as e:
                add_log(state, f"   ⚠️ 保存分析结果失败: {e}")

        # Step 2: Get price from Gemini analysis
        add_log(state, f"💰 价格分析...")

        market_price = 0
        price_data = None
        if gemini_full_result:
            resale = gemini_full_result.get("resale_estimate", {})
            if resale:
                market_price = resale.get("max_price", 0)
                price_data = {
                    "suggested_price": market_price,
                    "price_range": f"¥{resale.get('min_price', 0)} - ¥{resale.get('max_price', 0)}",
                    "confidence": resale.get("confidence", "中"),
                    "source": "gemini"
                }
                add_log(state, f"   ├─ 二手估价: ¥{resale.get('min_price', 0)} - ¥{resale.get('max_price', 0)}")
                add_log(state, f"   ├─ 建议售价: ¥{market_price}")
                add_log(state, f"   └─ 置信度: {resale.get('confidence', 'N/A')}")
            else:
                # Fallback to brand-based estimation
                brand = vlm_analysis.get("brand", "") if vlm_analysis else ""
                if brand and brand != "Unknown":
                    price_data = pricing_tool.query_market_price(brand, vlm_analysis.get("category", ""))
                    market_price = price_data.get("estimated_price", 500)
                    add_log(state, f"   └─ 基于品牌估算: ¥{market_price}")
        else:
            # Fallback to mock pricing
            market_price = check_market_price(item["name"])
            add_log(state, f"   └─ 使用默认定价: ¥{market_price}")

        state["market_price"] = market_price
        state["pricing_data"] = price_data or {"suggested_price": market_price, "sample_size": 0}

        # Step 3: Get mock buyer offer
        add_log(state, f"💰 查找买家报价...")
        buyer_offer = get_buyer_offer(market_price)
        state["buyer_offer"] = buyer_offer
        add_log(state, f"   └─ 买家: {buyer_offer['buyer_name']}, 报价: ¥{buyer_offer['offer_price']}")

        # Step 4: Build simplified recommendation with real data only
        add_log(state, f"📝 生成分析报告...")

        item_name = item['name']
        if vlm_analysis:
            category = vlm_analysis.get('category', '')
            brand = vlm_analysis.get('brand', '')
            if category and category != 'Unknown':
                item_name = f"{brand} {category}" if brand and brand != 'Unknown' else category

        # 获取 Gemini 完整结果
        gemini_result = state.get("gemini_result", {})
        official_price = gemini_result.get("official_price", {}) if gemini_result else {}

        # 构建简洁的推荐
        recommendation_parts = []

        # 显示品牌和型号
        if brand and brand != 'Unknown':
            model_name = vlm_analysis.get('model_name', '')
            if model_name and model_name != '未识别':
                recommendation_parts.append(f"🎯 **{brand} {model_name}**")
            else:
                recommendation_parts.append(f"🎯 **{brand}**")

        # 货号
        product_code = vlm_analysis.get('product_code', '')
        if product_code and product_code != 'N/A':
            recommendation_parts.append(f"📦 货号: {product_code}")

        if vlm_analysis:
            features = []
            if vlm_analysis.get('category'):
                features.append(f"类别: {vlm_analysis['category']}")
            if vlm_analysis.get('material'):
                features.append(f"材质: {vlm_analysis['material']}")
            if vlm_analysis.get('condition'):
                features.append(f"成色: {vlm_analysis['condition']}")

            if features:
                recommendation_parts.append(f"\n👕 衣物特征")
                recommendation_parts.extend([f"• {f}" for f in features])

        # 价格信息
        recommendation_parts.append(f"\n💰 价格参考")
        if official_price.get('amount'):
            recommendation_parts.append(f"• 官方指导价: {official_price['amount']} {official_price.get('currency', '')}")
        recommendation_parts.append(f"• 二手建议售价: ¥{market_price}")

        # 来源说明
        if gemini_result.get('analysis_source') == 'gemini':
            recommendation_parts.append(f"\n🤖 由 Gemini 3.1 Pro 智能分析")

        recommendation_parts.append("\n✅ 建议出售，为衣橱腾出空间！")

        recommendation = "\n".join(recommendation_parts)

        state["agent_decision"] = recommendation
        state["status"] = "awaiting_user_decision"
        add_log(state, "⏳ 等待用户在App上确认...")

    except Exception as e:
        add_log(state, f"❌ 评估过程出错: {str(e)}")
        state["status"] = "api_overloaded"
        state["error_message"] = str(e)

    return state


def wait_for_user_node(state: WorkflowState) -> WorkflowState:
    """
    Wait for User Node: Halts execution until user provides input.
    """
    if state["status"] in ["no_items_found", "error"]:
        return state

    if state["user_approved"] is None:
        # Still waiting for user input
        return state

    if state["user_approved"]:
        add_log(state, "✅ 用户已批准出售")
        state["status"] = "user_approved"
    else:
        add_log(state, "❌ 用户已拒绝出售")
        state["status"] = "user_rejected"

    return state


def stagnancy_check_node(state: WorkflowState) -> WorkflowState:
    """
    Stagnancy Check Node: Checks if the current item is stagnant (>365 days).
    Called immediately after adding a new item from upload.
    """
    item = state.get("current_item")
    if not item:
        add_log(state, "❌ 错误: 没有物品可检查")
        state["status"] = "error"
        return state

    add_log(state, f"🔍 检查物品闲置状态: {item['name']}")

    # Check for both old format (last_worn_days_ago) and new format (purchase_date)
    is_item_stagnant = False
    days_since = 0
    
    if "purchase_date" in item and item["purchase_date"]:
        # New format - use is_stagnant function
        is_item_stagnant = is_stagnant(item, threshold_days=365)
        if is_item_stagnant:
            try:
                days_since = (datetime.now() - datetime.strptime(item['purchase_date'][:10], "%Y-%m-%d")).days
            except:
                days_since = 400  # Fallback
    elif "last_worn_days_ago" in item:
        # Old format - directly check days
        days_since = item["last_worn_days_ago"]
        is_item_stagnant = days_since > 365

    if is_item_stagnant:
        add_log(state, f"⚠️ 发现闲置物品！已 {days_since} 天未穿着")
        add_log(state, f"💡 触发出售建议...")
        state["status"] = "item_stagnant"
    else:
        add_log(state, f"✅ 物品状态正常，无需处理")
        state["status"] = "item_fresh"

    return state


def execute_node(state: WorkflowState) -> WorkflowState:
    """
    Execute Node: Completes the sale and arranges logistics.
    """
    if state["status"] != "user_approved":
        if state["status"] == "user_rejected":
            # Update item status to keep it in closet
            if state["current_item"]:
                add_log(state, f"🔄 衣物「{state['current_item']['name']}」继续保留在衣橱中")
        return state

    item = state["current_item"]
    buyer_offer = state["buyer_offer"]

    if not item or not buyer_offer:
        add_log(state, "❌ 错误: 缺少必要信息，无法执行交易")
        state["status"] = "error"
        return state

    try:
        # Step 1: Update database
        add_log(state, "📝 正在更新数据库...")
        success = update_item_status(item["item_id"], "sold")
        if success:
            add_log(state, f"   └─ 衣物状态已更新为: 已售出")
        else:
            add_log(state, f"   ⚠️ 数据库更新失败")

        # Step 2: Execute logistics
        add_log(state, "🚚 正在安排物流配送...")
        tracking_info = execute_logistics(item["name"], buyer_offer["buyer_name"])
        state["tracking_info"] = tracking_info
        state["sale_executed"] = True

        add_log(state, f"   └─ 物流公司: {tracking_info['carrier']}")
        add_log(state, f"   └─ 运单号: {tracking_info['tracking_number']}")
        add_log(state, f"   └─ 预计送达: {tracking_info['estimated_delivery']}")

        # Success summary
        add_log(state, "=" * 50)
        add_log(state, "🎉 交易完成!")
        add_log(state, f"   衣物: {item['name']}")
        add_log(state, f"   成交价: ¥{buyer_offer['offer_price']}")
        add_log(state, f"   买家: {buyer_offer['buyer_name']}")
        add_log(state, f"   运单: {tracking_info['tracking_number']}")
        add_log(state, "=" * 50)

        state["status"] = "sale_completed"

    except Exception as e:
        add_log(state, f"❌ 交易执行失败: {str(e)}")
        state["status"] = "error"

    return state


# ========== Graph Construction ==========

def build_workflow():
    """Build and return the LangGraph workflow."""

    # Create the state graph
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("monitor", monitor_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("wait_for_user", wait_for_user_node)
    workflow.add_node("execute", execute_node)

    # Define edges
    workflow.set_entry_point("monitor")
    workflow.add_edge("monitor", "evaluate")
    workflow.add_edge("evaluate", "wait_for_user")

    # Conditional edge from wait_for_user
    def route_after_user_input(state: WorkflowState) -> str:
        """Determine next step based on user input."""
        if state["status"] == "user_approved":
            return "execute"
        elif state["status"] == "user_rejected":
            return END
        elif state["status"] in ["no_items_found", "error"]:
            return END
        else:
            # Still waiting for input
            return "wait_for_user"

    workflow.add_conditional_edges(
        "wait_for_user",
        route_after_user_input,
        {
            "execute": "execute",
            END: END,
            "wait_for_user": "wait_for_user",
        }
    )

    workflow.add_edge("execute", END)

    # Compile with memory checkpointing
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app


# Global workflow instance
workflow_app = build_workflow()


def reset_workflow():
    """Reset the in-memory LangGraph checkpoint for the demo thread."""
    try:
        workflow_app.update_state(
            {"configurable": {"thread_id": "fashionclaw_demo"}},
            create_initial_state(),
        )
    except Exception as e:
        print(f"Warning: failed to reset workflow checkpoint: {e}")


def run_workflow_until_user_input():
    """
    Run the workflow until it needs user input.
    Returns the state at the user decision point.
    """
    state = create_initial_state()

    # Run until we hit the user wait node
    for event in workflow_app.stream(state, {"configurable": {"thread_id": "fashionclaw_demo"}}):
        if "wait_for_user" in event:
            # Check if we need actual user input
            current_state = event["wait_for_user"]
            if current_state.get("status") in ["awaiting_user_decision", "no_items_found", "error"]:
                return current_state

    return state


def resume_workflow(user_approved: bool, initial_state: WorkflowState = None):
    """
    Resume the workflow with user decision.

    Args:
        user_approved: User's decision (True=approve, False=reject)
        initial_state: Optional initial state to use (for upload workflow)
    """
    # Get or create state
    if initial_state is not None:
        # Use provided state (from upload workflow)
        updated_state = dict(initial_state)
        updated_state["user_approved"] = user_approved
        # Set status for execute_node to work correctly
        updated_state["status"] = "user_approved" if user_approved else "user_rejected"

        # Save to checkpoint for consistency
        try:
            workflow_app.update_state(
                {"configurable": {"thread_id": "fashionclaw_demo"}},
                updated_state
            )
        except Exception as e:
            print(f"[resume_workflow] Warning: failed to save state: {e}")
    else:
        # Get from checkpoint (standard workflow)
        state = workflow_app.get_state({"configurable": {"thread_id": "fashionclaw_demo"}})
        if not state:
            return None
        updated_state = dict(state.values)
        updated_state["user_approved"] = user_approved
        # Set status for execute_node to work correctly
        updated_state["status"] = "user_approved" if user_approved else "user_rejected"

    # Resume execution
    final_state = None
    try:
        for event in workflow_app.stream(updated_state, {"configurable": {"thread_id": "fashionclaw_demo"}}):
            for key, value in event.items():
                final_state = value
                print(f"[resume_workflow] Event: {key}, state status: {value.get('status', 'unknown')}")
    except Exception as e:
        print(f"[resume_workflow] Error during workflow execution: {e}")
        import traceback
        traceback.print_exc()
        return None

    return final_state


# ========== Upload Flow Functions ==========

def create_state_for_item(item: dict) -> WorkflowState:
    """Create workflow state for a specific item (used in upload flow)."""
    state = create_initial_state()
    state["current_item"] = item
    return state


def run_upload_workflow(item: dict) -> WorkflowState:
    """
    Run workflow for a newly uploaded item:
    stagnancy check -> if stagnant -> evaluate -> wait for user -> execute
    """
    state = create_state_for_item(item)

    # Step 1: Check stagnancy
    state = stagnancy_check_node(state)

    if state["status"] == "item_stagnant":
        # Step 2: Evaluate (market price, buyer, etc.)
        state = evaluate_node(state)
        return state
    else:
        # Item is fresh, no need to sell
        return state


def run_upload_workflow_until_user_input(item: dict) -> WorkflowState:
    """
    Run upload workflow until it needs user input.
    Returns state at the user decision point.
    """
    state = run_upload_workflow(item)

    # Save state to workflow_app for resume_workflow to work
    # Update the state in the checkpoint
    try:
        workflow_app.update_state(
            {"configurable": {"thread_id": "fashionclaw_demo"}},
            state,
        )
    except Exception as e:
        print(f"Warning: failed to save workflow state: {e}")

    return state


# ========== Pricing Analysis ==========

def pricing_request_node(state: WorkflowState) -> WorkflowState:
    """
    Pricing Request Node: Triggered when user wants to check pricing.
    Uses Gemini 3.1 Pro for visual analysis.
    """
    item = state.get("current_item")
    if not item:
        add_log(state, "❌ 错误: 没有物品可定价")
        state["status"] = "error"
        return state

    add_log(state, f"💰 开始定价分析: {item['name']}")
    add_log(state, "🤖 调用定价 Agent 分析衣物特征...")

    state["status"] = "awaiting_pricing_decision"
    return state


def pricing_execute_node(state: WorkflowState) -> WorkflowState:
    """
    Pricing Execute Node: Actually run the pricing analysis.
    """
    item = state.get("current_item")
    if not item:
        add_log(state, "❌ 错误: 没有物品可定价")
        state["status"] = "error"
        return state

    try:
        image_path = item.get("image", "")
        if not image_path or not os.path.exists(image_path):
            add_log(state, "⚠️ 未找到衣物图片，使用模拟定价")
            # Fallback to mock pricing
            state["pricing_result"] = {
                "item_details": {"category": item.get("name", "Unknown"), "brand": "Unknown"},
                "pricing": {"suggested_price": 199, "sample_size": 10},
                "listing_title": f"{item.get('name', '闲置衣物')} 转卖",
                "listing_desc": "九成新，原价购入，现低价转让"
            }
        else:
            # Use Gemini for pricing analysis
            add_log(state, "📸 Gemini 分析定价中...")
            pricing_tool = PricingTool()
            result = pricing_tool.analyze_with_gemini(image_path)

            if result and result.get("success"):
                state["pricing_result"] = result
                official = result.get("official_price", {})
                resale = result.get("resale_estimate", {})
                add_log(state, f"✅ 定价分析完成！")
                if official.get("amount"):
                    add_log(state, f"   ├─ 官方指导价: {official['amount']} {official.get('currency', '')}")
                if resale.get("max_price"):
                    add_log(state, f"   ├─ 二手估价: ¥{resale['min_price']} - ¥{resale['max_price']}")
                    add_log(state, f"   └─ 置信度: {resale.get('confidence', 'N/A')}")
            else:
                # Fallback to mock pricing
                state["pricing_result"] = {
                    "item_details": {"category": item.get("name", "Unknown"), "brand": "Unknown"},
                    "official_price": {"amount": 0, "currency": "CNY"},
                    "resale_estimate": {"min_price": 100, "max_price": 200, "confidence": "低"}
                }
                add_log(state, "⚠️ Gemini 定价失败，使用默认估价")

        state["status"] = "pricing_completed"

    except Exception as e:
        add_log(state, f"❌ 定价分析失败: {str(e)}")
        state["status"] = "error"

    return state

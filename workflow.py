"""
FashionClaw LangGraph Workflow
Implements the multi-agent system for intelligent wardrobe management.
"""
import json
import random
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
    Evaluate Node: Calls mock APIs to assess market value and find buyer.
    """
    if state["status"] == "no_items_found" or state["status"] == "error":
        return state

    item = state["current_item"]
    if not item:
        add_log(state, "❌ 错误: 没有可评估的衣物")
        state["status"] = "error"
        return state

    try:
        # Step 1: Check market price
        add_log(state, f"💰 正在查询二手市场价格...")
        market_price = check_market_price(item["name"])
        state["market_price"] = market_price
        add_log(state, f"   └─ 市场估价: ¥{market_price}")

        # Step 2: Get buyer offer
        add_log(state, f"🔎 正在闲鱼平台寻找买家...")
        buyer_offer = get_buyer_offer(market_price)
        state["buyer_offer"] = buyer_offer
        add_log(state, f"   └─ 找到买家: {buyer_offer['buyer_name']}")
        add_log(state, f"   └─ 出价: ¥{buyer_offer['offer_price']}")

        # Step 3: Check buyer credit
        add_log(state, f"📋 正在核查买家信用...")
        buyer_credit = check_buyer_credit(buyer_offer["buyer_id"])
        state["buyer_credit"] = buyer_credit
        add_log(state, f"   └─ 信用评级: {buyer_credit['credit_rating']}")
        add_log(state, f"   └─ 信用分: {buyer_credit['credit_score']}")
        add_log(state, f"   └─ 历史交易: {buyer_credit['successful_transactions']} 笔")

        # Step 4: Generate recommendation
        if buyer_credit["credit_rating"] == "Excellent":
            recommendation = (
                f"🌟 智能推荐: 发现买家「{buyer_offer['buyer_name']}」"
                f"出价 ¥{buyer_offer['offer_price']} 购买您的「{item['name']}」。\n\n"
                f"买家信用评级: {buyer_credit['credit_rating']} (信用分: {buyer_credit['credit_score']})\n"
                f"历史成功交易: {buyer_credit['successful_transactions']} 笔\n"
                f"退货率: {buyer_credit['return_rate']}\n\n"
                f"✅ 强烈建议出售！"
            )
        elif buyer_credit["credit_rating"] == "Good":
            recommendation = (
                f"📌 智能推荐: 发现买家「{buyer_offer['buyer_name']}」"
                f"出价 ¥{buyer_offer['offer_price']} 购买您的「{item['name']}」。\n\n"
                f"买家信用评级: {buyer_credit['credit_rating']} (信用分: {buyer_credit['credit_score']})\n"
                f"历史成功交易: {buyer_credit['successful_transactions']} 笔\n\n"
                f"⚠️ 可以考虑出售"
            )
        else:
            recommendation = (
                f"⚠️ 风险提示: 发现买家「{buyer_offer['buyer_name']}」"
                f"出价 ¥{buyer_offer['offer_price']} 购买您的「{item['name']}」。\n\n"
                f"买家信用评级: {buyer_credit['credit_rating']} (信用分: {buyer_credit['credit_score']})\n"
                f"❌ 不建议出售，建议等待其他买家"
            )

        state["agent_decision"] = recommendation
        state["status"] = "awaiting_user_decision"
        add_log(state, "⏳ 等待用户在App上确认...")

    except Exception as e:
        add_log(state, f"❌ 评估失败: {str(e)}")
        state["status"] = "error"

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

    if is_stagnant(item, threshold_days=365):
        days_since = (datetime.now() - datetime.strptime(item['purchase_date'][:10], "%Y-%m-%d")).days
        add_log(state, f"⚠️ 发现闲置物品！已购买 {days_since} 天")
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


def resume_workflow(user_approved: bool):
    """
    Resume the workflow with user decision.
    """
    # Get the current state
    state = workflow_app.get_state({"configurable": {"thread_id": "fashionclaw_demo"}})

    if not state:
        return None

    # Update with user decision
    updated_state = dict(state.values)
    updated_state["user_approved"] = user_approved

    # Resume execution
    final_state = None
    for event in workflow_app.stream(updated_state, {"configurable": {"thread_id": "fashionclaw_demo"}}):
        for key, value in event.items():
            final_state = value

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
    return state

# FashionClaw 智能衣橱系统

AI驱动的多代理智能衣橱管理平台演示。

## 项目概述

FashionClaw 是一个模拟智能衣橱系统的原型演示，展示以下完整用户流程：

1. **自动监测** - 系统扫描衣橱，发现超过365天未穿的衣物
2. **价值评估** - 调用模拟API查询二手市场价格、匹配买家、核查信用
3. **用户决策** - 通过移动端App界面推送通知，等待用户确认
4. **执行交易** - 用户批准后自动完成出售并安排物流配送

## 技术栈

- **Python 3.10+**
- **LangGraph** - 构建多代理工作流状态机
- **Gradio** - 双面板交互式界面（后端监控 + 移动端App模拟）
- **本地JSON文件** - 数据存储

## 项目结构

```
Demo/
├── database.json       # 衣物数据库（包含7件示例衣物）
├── environment.yml     # Conda环境配置
├── mock_apis.py        # 模拟外部API（闲鱼价格、信用检查、物流）
├── workflow.py         # LangGraph工作流实现（4个节点）
├── app.py             # Gradio双面板界面
└── README.md          # 本文件
```

## 安装与运行

### 方式一：使用 environment.yml（推荐）

```bash
conda env create -f environment.yml
conda activate fashionclaw
python app.py
```

### 方式二：手动创建环境

```bash
conda create -n fashionclaw python=3.10 -y
conda activate fashionclaw
pip install gradio langgraph
python app.py
```

然后打开浏览器访问 `http://localhost:7860`

## 使用说明

### 界面布局

**左侧面板 - 代理后端监控**
- 实时显示AI代理的工作日志
- 展示系统扫描、评估、决策的全过程
- 可查看当前数据库状态

**右侧面板 - 移动端App模拟**
- 显示系统推送的闲置衣物通知
- 展示衣物详情、买家出价、信用评级
- 提供「确认出售」和「拒绝出售」按钮

### 操作流程

1. 点击左侧「🚀 启动智能衣橱系统」按钮
2. 观察代理后端日志（左侧面板），查看AI如何：
   - 扫描数据库发现闲置衣物（>365天未穿）
   - 查询市场估价
   - 在闲鱼平台寻找买家
   - 核查买家信用
3. 在右侧移动端界面查看推荐详情
4. 点击「✅ 确认出售」或「❌ 拒绝出售"
5. 查看交易结果、物流信息和数据库更新

## 工作流架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Monitor    │ --> │  Evaluate   │ --> │  User Wait  │
│  (监控节点)  │     │  (评估节点)  │     │ (等待用户)   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                                │
                        ┌───────────────────────┘
                        │
                        ▼
                  ┌─────────────┐
                  │  Execute    │
                  │ (执行交易)   │
                  └─────────────┘
```

### 节点说明

| 节点 | 功能 |
|------|------|
| **Monitor** | 扫描database.json，找出超过365天未穿且状态为"in_closet"的衣物 |
| **Evaluate** | 调用mock APIs获取市场价、买家出价、买家信用评级，生成推荐决策 |
| **User Wait** | 暂停执行，等待用户在Gradio界面点击Approve/Reject |
| **Execute** | 用户批准后，更新数据库状态为"sold"，调用物流API生成运单号 |

## 模拟API说明

### `check_market_price(item_name)`
返回二手市场估价（¥50-450之间的随机合理价格）

### `get_buyer_offer(market_price)`
返回模拟买家信息（买家ID、昵称、出价、平台）

### `check_buyer_credit(buyer_id)`
返回买家信用评级（Excellent/Good/Fair/Poor）及详细数据

### `execute_logistics(item_name, buyer_name)`
返回物流信息（快递公司、运单号、预计送达时间）

## 数据库结构

```json
{
  "wardrobe": [
    {
      "item_id": "001",
      "name": "Blue Denim Jacket",
      "last_worn_days_ago": 45,
      "status": "in_closet",
      "original_price": 299
    }
  ]
}
```

**状态说明：**
- `in_closet` - 在衣橱中
- `selling` - 正在出售
- `sold` - 已售出

## 演示数据

数据库包含7件衣物，其中3件超过365天未穿：
- **Red Summer Dress** (420天)
- **Vintage Wool Sweater** (500天)
- **Brown Leather Belt** (380天)

每次演示系统会按顺序找到第一个符合条件的衣物进行处理。

## 故障排除

### 问题：无法导入langgraph

**解决：** 使用 environment.yml 重新创建环境：
```bash
conda env remove -n fashionclaw
conda env create -f environment.yml
conda activate fashionclaw
python app.py
```

### 问题：数据库被修改后无法重新演示

**解决：** 重置数据库：
```bash
python -c "
import json
with open('database.json', 'r') as f:
    db = json.load(f)
for item in db['wardrobe']:
    item['status'] = 'in_closet'
    if 'sold_at' in item:
        del item['sold_at']
with open('database.json', 'w') as f:
    json.dump(db, f, indent=2)
print('Database reset complete')
"
```

## 许可证

本演示仅供学习和原型开发使用。

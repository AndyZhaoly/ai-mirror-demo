# 🤵 AI 时尚管家 · FashionClaw

智能服装管理助手 —— 上传穿搭照片，AI 自动识别品牌、估价、一键发布到 Poshmark 出售。

## ✨ 核心功能

- 📸 **智能分割**：GroundingDINO + SAM 自动提取衣物轮廓
- 🧠 **品牌识别**：Gemini 3.1 Pro 直接识别品牌、型号、货号
- 💰 **智能估价**：官方指导价 + 二手市场建议售价
- 🤖 **AI 管家**：自然语言交互，像真人管家一样贴心
- 🌐 **一键发布**：Agent 智能识别出售意图，自动填写 Poshmark 表单

## 🚀 快速开始

### 环境要求
- Python 3.10+
- GroundingDINO + SAM 服务（本地或远程）

### 安装
```bash
pip install -r requirements.txt
```

### 配置 API Keys
创建 `.env` 文件：
```bash
# 必需
GEMINI_API_KEY=your_gemini_api_key

# 可选
MOONSHOT_API_KEY=your_kimi_api_key
```

获取 Gemini API Key：[Google AI Studio](https://aistudio.google.com/app/apikey)

### 运行
```bash
# 配置 GroundingDINO + SAM 服务
# 默认连接本地 http://localhost:8000
# 如需远程服务，修改 gsam_client.py 中的 GSAMClient 地址

# 启动主应用
python app.py
```

访问 `http://localhost:7860`

## 🏗️ 技术架构

```
用户上传图片
    ↓
GroundingDINO + SAM 分割（端口 8000）
    ↓
Gemini 3.1 Pro 视觉分析
    ├── 品牌识别（Louis Vuitton, Nike, etc.）
    ├── 型号/货号（1AJYH4, etc.）
    ├── 材质分析
    ├── 官方指导价
    └── 二手估价建议
    ↓
AI Agent（Gemini 3.1 Flash-Lite）生成自然语言回复
    ↓
├─→ 自然语言对话（右侧 Chatbot）
│
└─→ Function Calling 检测出售意图
        ↓
    Playwright 自动填写 Poshmark 表单
        ↓
    浏览器窗口显示，等待确认发布
```

## 📁 项目结构

```
.
├── app.py                  # Gradio 主应用
├── workflow.py             # LangGraph 工作流编排
├── mirror_agent.py         # AI Agent（对话生成 + Function Calling）
├── database.json           # 衣橱数据库
├── requirements.txt        # Python 依赖
├── .env                    # API Keys（用户自备）
├── tools/
│   ├── gemini_analyzer.py  # Gemini 3.1 Pro 分析模块
│   ├── pricing_tool.py     # 定价工具（主入口）
│   └── poshmark_bot.py     # Playwright 自动化发布
├── poshmark_browser_data/  # Poshmark 登录状态（自动创建）
└── extracted_clothes/      # 提取的衣物图片（运行时生成）
```

## 🔑 关键技术

| 组件 | 用途 | 版本 |
|------|------|------|
| **Gemini 3.1 Pro** | 视觉识别品牌、型号、价格 | `gemini-3.1-pro-preview` |
| **Gemini 3.1 Flash-Lite** | AI Agent 对话生成 | `gemini-3.1-flash-lite-preview` |
| **GroundingDINO** | 目标检测 | - |
| **SAM** | 图像分割 | - |
| **Gradio** | Web 界面 | 4.x |
| **Playwright** | 浏览器自动化（Poshmark）| - |
| **Function Calling** | 智能识别出售意图 | OpenAI Compatible |

## 💡 使用示例

上传一张 LV 夹克照片 → 系统输出：

```
查到了！主人～ 🎉

这件是 Louis Vuitton 的 Dark Floral Print Jacket！
📦 货号：1AJYH4

• 材质：100% 锦纶
• 成色：几乎全新

💰 价格参考
• 官方指导价：3500 EUR
• 二手市场价：¥25000 - ¥30000

小镜建议定价 ¥30000 左右～
```

## 🔧 配置说明

### GroundingDINO + SAM 服务

默认连接本地 `http://localhost:8000`。如需使用远程服务：

**方法 1：修改代码**（`app.py`）
```python
gsam_client = GSAMClient("http://your-remote-server:8000")
```

**方法 2：SSH 隧道**（推荐，安全连接远程服务）
```bash
# 在本地建立隧道，将远程 8000 端口映射到本地
ssh -L 8000:localhost:8000 user@remote-server
```

或使用 `autossh` 保持连接：
```bash
autossh -M 0 -N -L 8000:localhost:8000 user@remote-server
```

### Poshmark 自动化

首次使用需要登录 Poshmark：
```bash
python tools/poshmark_bot.py
```
在打开的浏览器中完成登录，之后会话会保存在 `poshmark_browser_data/` 目录。

## 🔧 开发说明

### 添加新的分析模型
在 `tools/gemini_analyzer.py` 中修改 `DEFAULT_MODEL`：
```python
DEFAULT_MODEL = "gemini-3.1-pro-preview"  # 或 gemini-2.5-pro
```

### 调整价格估算逻辑
修改 `tools/pricing_tool.py` 中的 `_get_mock_price()` 方法。

### 自定义 Agent 回复风格
修改 `mirror_agent.py` 中的 `SYSTEM_PROMPT`。

## 📄 License

MIT License

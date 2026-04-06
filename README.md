# 🤵 AI 时尚管家 · FashionClaw

智能服装管理助手 —— 上传穿搭照片，AI 自动识别品牌、估价、建议出售。

## ✨ 核心功能

- 📸 **智能分割**：GroundingDINO + SAM 自动提取衣物轮廓
- 🧠 **品牌识别**：Gemini 3.1 Pro 直接识别品牌、型号、货号
- 💰 **智能估价**：官方指导价 + 二手市场建议售价
- 🤖 **AI 管家**：自然语言交互，像真人管家一样贴心

## 🚀 快速开始

### 环境要求
- Python 3.10+
- GroundingDINO + SAM 服务（本地端口 8000）

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
# 启动 GroundingDINO + SAM 服务（需单独部署）
# 然后运行主应用
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
AI Agent（Kimi）生成自然语言回复
    ↓
Gradio 界面展示
```

## 📁 项目结构

```
.
├── app.py                  # Gradio 主应用
├── workflow.py             # LangGraph 工作流编排
├── mirror_agent.py         # AI Agent（对话生成）
├── database.json           # 衣橱数据库
├── requirements.txt        # Python 依赖
├── .env                    # API Keys（用户自备）
├── tools/
│   ├── gemini_analyzer.py  # Gemini 3.1 Pro 分析模块
│   └── pricing_tool.py     # 定价工具（主入口）
└── extracted_clothes/      # 提取的衣物图片（运行时生成）
```

## 🔑 关键技术

| 组件 | 用途 | 版本 |
|------|------|------|
| **Gemini 3.1 Pro** | 视觉识别品牌、型号、价格 | `gemini-3.1-pro-preview` |
| **GroundingDINO** | 目标检测 | - |
| **SAM** | 图像分割 | - |
| **Kimi** | AI Agent 对话生成 | `kimi-k2.5` |
| **Gradio** | Web 界面 | 4.x |

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

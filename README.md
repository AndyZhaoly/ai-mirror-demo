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

### 部署 GroundingDINO + SAM 服务（必需）

本项目需要 GroundingDINO + SAM 服务进行图像分割。如果**没有 GPU**或**不想部署**，可选择以下方式：

#### 快速开始（推荐无 GPU 用户）

**使用预配置的远程服务器**（通过 SSH 隧道）：

```bash
# 建立 SSH 隧道，连接我们预配置的服务器
# 联系项目维护者获取服务器地址和账号
ssh -L 8000:localhost:8000 username@demo-server.fashionclaw.ai
```

#### 选项 1：自己部署（推荐有 GPU 的用户）

**硬件要求：**
- NVIDIA GPU 8GB+ 显存
- CUDA 11.8+
- Python 3.10+

```bash
# 克隆 GSAM 服务仓库
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything

# 安装依赖
pip install -e segment_anything
pip install -e GroundingDINO
pip install diffusers transformers accelerate opencv-python

# 下载模型权重
mkdir -p weights
cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0/groundingdino_swint_ogc.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..

# 启动服务
python gradio_app.py --listen 0.0.0.0 --port 8000
```

**详细部署文档：** [GSAM 官方文档](https://github.com/IDEA-Research/Grounded-Segment-Anything)

#### 选项 2：Docker 部署

```bash
# 拉取预构建镜像
docker pull andyzhaoly/fashionclaw-gsam:latest

# 运行容器
docker run -d \
  --name fashionclaw-gsam \
  --gpus all \
  -p 8000:8000 \
  andyzhaoly/fashionclaw-gsam:latest
```

#### 选项 3：跳过图像分割（极简体验）

如果暂时不需要分割功能，可以修改 `app.py` 直接使用原图：

```python
# app.py 中注释掉以下代码，直接返回原图
# upper_images, upper_detection = gsam_client.extract_upper_body(...)
# lower_images, lower_detection = gsam_client.extract_lower_body(...)

# 改为：
upper_images = [image]
lower_images = []
```

> ⚠️ **注意：** 此模式下 Gemini 会直接分析完整图片，可能不如分割后的单件衣物识别准确。

### 运行
```bash
# 确保 GSAM 服务可访问（本地或远程隧道）
# 默认连接 http://localhost:8000

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
| **IDM-VTON** | 虚拟试穿服务 | Port 8001 |

## 🎮 测试 Agent 推荐+虚拟试穿工作流

### 1. 启动 IDM-VTON 服务（需要conda环境）

```bash
# 激活IDM-VTON的conda环境
conda activate idm

# 启动IDM-VTON服务（端口8001）
cd /home/zhaoliyang/ai-mirror-demo
python idm_vton_service.py
```

### 2. 启动主应用（另一个终端）

```bash
cd /home/zhaoliyang/ai-mirror-demo

# 设置环境变量（如需要）
export IDM_VTON_SERVICE_URL=http://localhost:8001
export GEMINI_API_KEY=your_key  # 可选，用于AI对话

# 启动主应用
python app.py
```

### 3. 测试工作流

1. **打开浏览器**访问 `http://localhost:7860`

2. **测试推荐功能**：
   - 在右侧聊天框输入："小镜，给我推荐几件衣服"
   - Agent应该调用 `get_clothing_recommendations` 工具
   - 返回推荐列表（带ID，如 `sample_001`）

3. **测试选择**：
   - 回复："第1件" 或 "sample_001"
   - Agent应该询问是否要试穿

4. **测试虚拟试穿触发**：
   - 回复："试试" 或 "好"
   - Agent应该调用 `trigger_virtual_tryon` 工具
   - 提示需要上传人像照片

5. **手动虚拟试穿**：
   - 滚动到页面下方 **"👗 虚拟试穿"** 区域
   - 在 **"📷 人物照片"** 上传你的人像照片
   - 在 **"👕 选择衣物"** 下拉框选择衣服（或上传自定义衣物）
   - 点击 **"✨ 开始试衣"**
   - 查看生成结果

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

# FashionClaw AI Mirror - 运行指南

## 环境准备

本项目需要两个独立的 Python 环境：

### 1. 主应用环境 (fashionclaw)

用于运行 Gradio 界面和 LangGraph 工作流。

```bash
# 创建环境
conda create -n fashionclaw python=3.10 -y
conda activate fashionclaw

# 安装依赖
pip install gradio>=4.0.0 langgraph>=0.0.50 langchain-core>=0.1.0 \
    Pillow>=10.0.0 requests>=2.31.0 fastapi>=0.100.0 uvicorn>=0.23.0
```

### 2. GSAM 服务环境 (已有环境: idm)

用于运行 SAM 图像分割服务。已检测到环境 `idm` 安装了必要的依赖。

```bash
# 激活已有环境
conda activate idm

# 验证依赖已安装
python -c "import torch; import cv2; import segment_anything; print('OK')"
```

如果需要在其他机器上创建此环境：
```bash
conda create -n gsam_env python=3.10 -y
conda activate gsam_env
pip install torch torchvision opencv-python fastapi uvicorn Pillow

# 从 GitHub 安装 segment-anything（如果网络允许）
pip install git+https://github.com/facebookresearch/segment-anything.git
# 或者从本地路径安装（如果已下载）
pip install -e /path/to/Grounded-Segment-Anything
```

## 启动步骤

### Step 1: 启动 GSAM 服务

```bash
cd ai-mirror-demo
conda activate idm
python segment_service.py
```

等待显示：`Uvicorn running on http://0.0.0.0:8000`

### Step 2: 启动主应用

在另一个终端：

```bash
cd ai-mirror-demo
conda activate fashionclaw
python app.py
```

等待显示：`Running on local URL: http://127.0.0.1:7860`

### Step 3: 使用系统

打开浏览器访问：`http://localhost:7860`

## 功能说明

### 📤 上传&检测标签
1. 上传您的穿搭照片
2. 系统自动分割衣物（需要 GSAM 服务运行）
3. 衣物自动注册到数据库（mock 购买日期 ~400 天前）
4. 系统检测闲置（>365天）并提示出售
5. 用户确认或拒绝出售

### 🤖 智能衣橱工作流标签
1. 扫描现有数据库中的闲置衣物
2. 评估市场价值
3. 匹配买家
4. 用户确认交易
5. 执行物流

## 常见问题

### 问题1: GSAM 服务无法连接
- 确保 `segment_service.py` 已启动
- 检查 `http://localhost:8000/health` 是否返回 healthy

### 问题2: SAM 模型文件未找到
- 模型路径配置在 `segment_service.py` 第 26 行
- 当前路径：`/home/zhaoliyang/Grounded-Segment-Anything/sam_vit_h_4b8939.pth`

### 问题3: 端口被占用
```bash
# 查找并终止占用 8000 端口的进程
lsof -ti:8000 | xargs kill -9
```

## 文件结构

```
ai-mirror-demo/
├── app.py                      # Gradio 主应用
├── workflow.py                 # LangGraph 工作流
├── database_manager.py         # 数据库操作
├── gsam_client.py             # GSAM API 客户端
├── segment_service.py         # SAM 服务（需要单独环境）
├── mock_apis.py               # 模拟外部 API
├── database.json              # JSON 数据库
├── environment.yml            # 主应用环境配置
├── requirements.txt           # pip 依赖列表
└── extracted_clothes/         # 提取的衣物图片
```

## 注意事项

1. **数据库兼容性**：系统同时支持新旧两种数据格式
   - 旧格式：`last_worn_days_ago`（初始 demo 数据）
   - 新格式：`purchase_date`（上传的图片提取的衣物）

2. **Mock 数据策略**：
   - 购买日期：随机生成 380-450 天前（确保触发闲置检测）
   - 原价：上衣 ¥100-500，下装 ¥80-400
   - 市场价：原价的 50-70%
   - 买家出价：市场价的 80-95%

3. **网络依赖**：
   - 主应用依赖 `segment_service.py` 本地服务
   - 无外网 API 调用（全部使用 mock）

# 智能镜 · AI Fashion Assistant

两个 Gradio demo，分别解决"买什么穿"和"不穿的怎么卖"两个场景。

---

## Demo 1：智能试衣间（vton_combined_demo.py）

上传自拍 → AI 夸穿搭 → 推荐搭配下装 → 虚拟试穿 → 纳入数字衣柜

### 功能流程

1. **上传自拍**：GSAM 自动分割上衣和下装，展示识别结果
2. **AI 管家小镜**：夸主人今日穿搭，结合天气/场合给出具体细节
3. **推荐搭配**：主人开口后，小镜从 demo_garments 里推荐下装（Gemini VLM 预分析）
4. **虚拟试穿**：IDM-VTON 生成试穿效果，支持单件或三件对比
5. **纳入衣柜**：满意的单品一键存入本地数据库

### 架构

```
用户自拍
  ↓
GSAM 服务（port 8000）— GroundingDINO + SAM 分割上下装
  ↓
Gemini Agent（小镜）— function calling 驱动整个流程
  ├── show_recommendations    → 展示 demo_garments 里的单品
  ├── trigger_virtual_tryon   → IDM-VTON 服务（port 8001）生成试穿图
  ├── try_all_lower           → 批量试穿所有下装，对比展示
  └── add_to_wardrobe         → 写入 database.json
```

### 启动

```bash
# 环境变量
export GEMINI_API_KEY=your_key
export GSAM_URL=http://localhost:8000       # GSAM 服务地址
export VTON_URL=http://localhost:8001       # IDM-VTON 服务地址
export VTON_COMBINED_PORT=7864             # 可选，默认 7864

# 启动
python vton_combined_demo.py
```

访问 `http://localhost:7864`

---

## Demo 2：闲置变现管家（poshmark_demo.py）

上传闲置衣物 → AI 分析品相 → 生成 Poshmark 英文文案 → 自动挂单

### 功能流程

1. **上传衣物照片**：GSAM 分割提取衣物主体，用于挂单封面图
2. **AI 管家小镜**：识别单品信息（品牌/品类/尺码/成色）
3. **定价建议**：参考原价和二手市场，给出美元挂单价
4. **生成文案**：Gemini 写 Poshmark 英文 listing（标题 + 描述 + 尺码 + 护理说明）
5. **自动挂单**：Playwright 打开浏览器，自动填写 Poshmark 表单，人工确认后发布

### 架构

```
用户上传衣物照片
  ↓
GSAM 服务（port 8000）— 分割衣物主体
  ↓
Gemini Agent（小镜）— function calling 驱动流程
  ├── identify_item              → 返回单品信息（当前为硬编码演示数据）
  ├── get_resale_price           → 计算人民币→美元定价
  ├── generate_poshmark_listing  → Gemini 生成英文文案
  └── post_to_poshmark           → Playwright 自动填表挂单
```

### 启动

```bash
export GEMINI_API_KEY=your_key
export GSAM_URL=http://localhost:8000
export POSHMARK_DEMO_PORT=7863    # 可选，默认 7863

python poshmark_demo.py
```

访问 `http://localhost:7863`

首次使用需要登录 Poshmark（Playwright 会打开真实浏览器，手动完成登录，之后 session 自动保存）。

---

## 依赖服务

两个 demo 都依赖以下两个后端服务，需要在有 GPU 的机器上运行：

| 服务 | 代码 | 默认端口 |
|------|------|----------|
| GSAM（图像分割） | [AndyZhaoly/grounded-segment-anything](https://github.com/AndyZhaoly/grounded-segment-anything) | 8000 |
| IDM-VTON（虚拟试穿） | [AndyZhaoly/idm-vton](https://github.com/AndyZhaoly/idm-vton) | 8001 |

两个 repo 已包含为 git submodule，clone 时一并拉取：

```bash
git clone --recurse-submodules https://github.com/AndyZhaoly/ai-mirror-demo.git
```

### 本地没有 GPU？用 SSH 隧道连服务器

```bash
# 把服务器的 8000/8001 端口映射到本地
ssh -L 8000:localhost:8000 -L 8001:localhost:8001 -p 20009 zhaoliyang@your-server-ip
```

之后 demo 代码照常连 `localhost:8000` / `localhost:8001`。

### 下载模型权重

**GSAM（~3GB）**
```bash
cd ~/Grounded-Segment-Anything
bash setup_gsam_service.sh   # 自动下载 GroundingDINO + SAM 权重
```

| 文件 | 大小 | 来源 |
|------|------|------|
| `groundingdino_swint_ogc.pth` | 662MB | [GitHub Releases](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) |
| `sam_vit_h_4b8939.pth` | 2.4GB | [Meta](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |

**IDM-VTON（~32GB）**
```bash
cd ~/IDM-VTON
bash download_weights.sh   # 自动下载所有权重
```

| 目录 | 大小 | 来源 |
|------|------|------|
| `checkpoints/`（主模型）| 28GB | [yisol/IDM-VTON](https://huggingface.co/yisol/IDM-VTON) |
| `ckpt/humanparsing/` | 510MB | [yisol/IDM-VTON-DC](https://huggingface.co/yisol/IDM-VTON-DC) |
| `ckpt/image_encoder/` | 2.4GB | [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter) |
| `ckpt/densepose/` | 244MB | [Detectron2](https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl) |
| `ckpt/openpose/` | 200MB | [yisol/IDM-VTON-DC](https://huggingface.co/yisol/IDM-VTON-DC) |

### 启动服务器上的服务

```bash
# GSAM（在服务器上）
cd ~/Grounded-Segment-Anything
bash start_service.sh    # 默认 port 8000

# IDM-VTON（在服务器上）
conda activate idm
cd ~/IDM-VTON
python idm_vton_service.py    # 默认 port 8001
```

---

## 安装

```bash
pip install -r requirements.txt
```

主要依赖：`gradio` `openai` `Pillow` `python-dotenv` `playwright`

```bash
# Playwright 浏览器（Poshmark demo 需要）
playwright install chromium
```

---

## 项目结构

```
.
├── vton_combined_demo.py     # Demo 1：智能试衣间
├── poshmark_demo.py          # Demo 2：闲置变现管家
├── gsam_client.py            # GSAM 服务客户端
├── idm_vton_client.py        # IDM-VTON 服务客户端
├── idm_vton_service.py       # IDM-VTON 服务端（在 GPU 服务器上跑）
├── database_manager.py       # 数字衣柜数据库操作
├── recommendations.py        # Gemini VLM 分析衣物
├── demo_garments/            # 试衣间的推荐单品图片
├── tools/
│   └── poshmark_bot.py       # Playwright 自动挂单
└── services/
    ├── gsam/                 # submodule → AndyZhaoly/grounded-segment-anything
    └── idm_vton/             # submodule → AndyZhaoly/idm-vton
```

---

## API Keys

| 变量 | 用途 | 获取 |
|------|------|------|
| `GEMINI_API_KEY` | AI 对话 + VLM 分析 | [Google AI Studio](https://aistudio.google.com/app/apikey) |

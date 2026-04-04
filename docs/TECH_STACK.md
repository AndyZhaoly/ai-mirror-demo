# FashionClaw 完整产品技术栈文档

> 分支: `liyzhao/demo1`
> 本文档基于 PRD 需求，系统梳理从 Demo 到 Production 所需的完整软件技术栈，重点聚焦 AI 产品层（Agent 架构、多模态感知、推荐系统）。

---

## 1. 技术栈总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           交互层 (Interaction Layer)                         │
│  智能镜硬件 + 镜前 UI (React/Vue)  ←→  移动端 App (Flutter / React Native)    │
│  语音: ASR → NLP → TTS    |    视觉: VLM + 人体姿态估计 + 手势识别            │
├─────────────────────────────────────────────────────────────────────────────┤
│                           AI 产品层 (AI Product Layer)                       │
│  Master Agent (管家)          ← 编排调度                                     │
│  ├─ 穿搭推荐 Agent            ← 推荐系统 + 搭配生成模型                        │
│  ├─ 闲置转卖 Agent            ← 自动化 + RPA + 价格评估模型                    │
│  ├─ 品牌上新 Agent            ← 爬虫 + 信息抽取 + 个性化过滤                   │
│  ├─ 衣橱整理 Agent            ← CV 分割 + 属性识别 + 向量化归档                │
│  └─ (可扩展) 洗护/预警/送礼 Agent                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                           数据与知识层 (Data & Knowledge)                     │
│  衣橱数据库 (PostgreSQL / MongoDB)  +  向量数据库 (Milvus / Pinecone)         │
│  用户偏好记忆 (Redis / 图数据库)  +  搭配知识库 (RAG)                         │
│  流行趋势知识库 (向量检索)                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                           基础设施层 (Infrastructure)                        │
│  Go/Python 微服务  |  K8s  |  Kafka / RabbitMQ  |  S3 / OSS                  │
│  LLM Gateway (缓存/限流/切面)  |  Model Serving (Triton / vLLM)              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. AI 产品层详细设计

### 2.1 核心 Agent 架构：Master + Expert Agents

FashionClaw 的本质是一个 **Multi-Agent Orchestration System**。核心设计原则是：
- **Master Agent (管家 Agent)** 负责意图理解、任务拆解、专家 Agent 编排、异常兜底。
- **Expert Agents** 负责各自领域的 SOP 执行，通过标准化接口（Tool/API/事件总线）与 Master 通信。

#### 推荐技术选型

| 组件 | Demo 实现 | Production 推荐方案 | 选型理由 |
|------|-----------|---------------------|----------|
| **Agent Framework** | [LangGraph](https://langchain-ai.github.io/langgraph/) | **LangGraph** + LangChain / [CrewAI](https://www.crewai.com/) | 状态机+图结构天然适合穿衣/转卖等人机协同长流程；支持断点续行（人机在环 HITL） |
| **Master Agent 调度** | 单工作流 | **Multi-Agent Supervisor Pattern** (LangGraph) | Master 作为 Router Node，根据用户意图分发到子图（sub-graph） |
| **工具调用 (Tool Use)** | Python 函数 | **MCP (Model Context Protocol)** / OpenAI Function Calling | 统一外部服务（天气、物流、闲鱼）的接入规范 |
| **记忆 (Memory)** | 无 | **检查点 (LangGraph Checkpoint) + 持久化记忆池** | 长期穿搭偏好、对话历史、衣橱变更记录 |
| **工作流可视化** | Gradio 日志 | **LangSmith / Phoenix / 自研 DAG UI** | 追踪 Agent 思考链、工具调用耗时、决策归因 |

#### Agent 分工与数据流

```
用户语音/手势/视觉输入
         │
         ▼
┌──────────────────┐
│  Master Agent    │ ← 意图识别 (LLM Intent Classifier)
│  (管家)          │ ← 上下文组装 (用户画像 + 实时天气 + 当前场景)
└────────┬─────────┘
         │ 任务拆解 (Task Decomposition)
    ┌────┴────┬────────────┬────────────┐
    ▼         ▼            ▼            ▼
穿搭推荐   闲置转卖      新衣建档      品牌追踪
 Agent      Agent         Agent        Agent
    │         │            │            │
    │    ┌────┴────┐       │            │
    │    ▼         ▼       │            │
    │ Monitor → Evaluate → Execute      │
    │ (子工作流)              (子工作流) │
    │                                    │
    └───────→ 统一更新衣橱数据库 ←───────┘
```

---

### 2.2 多模态感知层：智能镜 (Smart Mirror)

PRD 中定义智能镜是主要人机交互入口，需要同时具备 **语音交互** 与 **视觉理解** 能力。

#### 语音交互链路

```
用户语音
  → 麦克风阵列 (ECNR 回声消除/降噪)
  → Wake Word Detection (语音唤醒, e.g. Porcupine / 自研 "Hey Fashion")
  → VAD (Voice Activity Detection, e.g. Silero VAD)
  → ASR (语音识别, e.g. 阿里云 Paraformer / Whisper / 讯飞)
  → LLM Intent Understanding (意图理解)
  → TTS (语音合成, e.g. CosyVoice / Minimax / ElevenLabs)
  → 镜前音响播放
```

**关键技术点：**
- **端到端延迟目标**: < 800ms (唤醒到首轮语音回复)。需要 ASR 流式识别 + LLM 流式输出 + TTS 首包合成。
- **本地边缘计算**: 唤醒词 + VAD 建议跑在镜端 MCU/NPU 上，降低云端依赖。
- **大模型选择**: 意图理解可用轻量化模型（如 Qwen2.5-7B / Llama-3.1-8B 本地部署），复杂 Planner 任务上云（GPT-4o / Claude 3.5 Sonnet）。

#### 视觉感知链路

```
摄像头视频流 (RGB / RGB-D)
  → 人体检测 (YOLOv8 / RTMPose)
  → 人体姿态估计 (Pose Estimation) ← 用于手势识别
  → 服装分割 / 自动抠图 (MODNet / SAM / 自研 U²Net)
  → 多模态理解 (VLM: GPT-4V / Qwen-VL / InternVL)
     ├─ 识别"未建档的新单品" ( novelty detection )
     ├─ 提取属性: 品类、颜色、材质、季节、风格标签
     └─ 人体+服装联合理解: 上身效果、合身度、穿搭场景
```

**具体模型推荐：**

| 任务 | 轻量方案 (镜端/边缘) | 云侧精调方案 |
|------|----------------------|--------------|
| 人体检测 | YOLOv8-nano | YOLOv8x + 自有时尚数据集微调 |
| 服装分割 | MODNet (轻量抠图) | Segment Anything 2 (SAM2) + Clothing-specific prompt |
| 属性识别 | MobileNet / EfficientNet | Fashion-CLIP / 自研多标签分类模型 |
| 新衣发现 (Novelty) | 无法本地 | 基于向量检索的衣橱比对 + VLM 验证 |
| 手势识别 | MediaPipe Hands | 自研动态手势分类模型 (3D CNN / Transformer) |

**关键工程挑战：**
- **无感建档**: 用户试穿时，系统需在 2-3 秒内完成 "检测到新衣 → 语音确认 → 多角度抓拍 → 自动抠图 → 属性提取 → 入库" 的全链路。需要预录制指令 + 镜端机械结构（如旋转摄像头 / 广角镜头）配合。
- **镜面 Overlay UI**: 需用 OpenGL / Flutter Engine 实现低延迟 AR 效果（在镜面上悬浮卡片、手势光标追踪）。

---

### 2.3 推荐系统：穿搭推荐 Agent

这是 FashionClaw 最核心的差异化能力。推荐系统不是传统的协同过滤，而是基于 ** wardrobe-aware combinatorial generation ** 的生成式推荐。

#### 系统架构

```
输入: 目标场景 (通勤/约会/商务) + 天气 + 用户日程 + 可选"指定单品"
         │
         ▼
┌─────────────────────────────────────────┐
│  召回层 (Retrieval)                     │
│  1. 从衣橱 DB 过滤: 季节/温度/场景匹配      │
│  2. 从单品向量库获取风格相似款 (向量召回)    │
│  3. 用户历史 Lookbook 正反馈加权           │
└────────┬────────────────────────────────┘
         │ Top-K 单品候选池
         ▼
┌─────────────────────────────────────────┐
│  搭配生成层 (Generation / Ranking)       │
│  方法 A: 基于规则的模板 + LLM 重排        │
│  方法 B: Outfit Generation Model (Diffusion/VLM) │
│  方法 C: 组合优化 (组合数学 + 审美评分模型)  │
└────────┬────────────────────────────────┘
         │ N 套完整 Lookbook (Top + Bottom + Shoes + Accessories)
         ▼
┌─────────────────────────────────────────┐
│  呈现层 (Presentation)                  │
│  - 单品图合成完整搭配图 (虚拟试衣/拼图)      │
│  - 场景标签 + 推荐理由 (LLM generated)     │
│  - 镜端画廊滑动展示                       │
└─────────────────────────────────────────┘
```

#### 三种技术路线对比

| 路线 | 技术实现 | 优点 | 缺点 | 适用阶段 |
|------|----------|------|------|----------|
| **LLM + RAG** | GPT-4o + 穿搭知识库 + 衣橱 DB | 快速实现、解释性强、无需训练 | 幻觉风险、组合爆炸时性能下降 | Demo / MVP |
| **Fashion-CLIP + 组合评分** | Fashion-CLIP 提取单品向量 → 训练 MLP/Transformer 评分模型预测搭配兼容性 | 可量化、可大规模离线排序 | 需要大量时尚搭配数据 | 量产阶段 |
| **生成式 Outfit Model** | Diffusion 模型 / FashionGen 直接生成搭配图 | 最自然、创意空间大 | 计算成本高、可控性挑战大 | 中长期探索 |

**Demo 到 Production 的演进建议：**
1. **Demo (当前)**：纯 LLM Prompt + 固定规则生成。快速验证交互体验。
2. **MVP**：引入 **Fashion-CLIP** 或 **SigLIP** 作为单品视觉编码器，构建向量索引。用 LLM 做最终推荐理由生成。
3. **Scale**：在大量用户反馈数据上训练 **Outfit Compatibility Scoring Model** (Siamese Network 或 Transformer-based)，并与 LLM 融合（Model-as-a-Reranker）。

---

## 3. 后端服务层

### 3.1 微服务划分

| 服务域 | 技术职责 | 推荐技术栈 |
|--------|----------|------------|
| **Agent Service** | 运行 Master + Expert Agents 工作流 | Python + LangGraph + FastAPI |
| **Perception Service** | 处理镜端上传的音视频，调用 ASR/VLM | Python + FFmpeg + gRPC |
| **Recommendation Service** | 穿搭召回、评分、生成 | Python + PyTorch + Faiss/Milvus |
| **Wardrobe Service** | 衣橱 CRUD、单品生命周期管理 | Go / Java + PostgreSQL |
| **User Profile Service** | 用户偏好、穿搭记忆、标签体系 | Go + Redis + MongoDB |
| **Integration Service** | 对接天气、二手平台、物流、品牌订阅 | Python + Celery + 各平台 SDK |
| **Notification Service** | Push、短信、镜端消息 | Firebase / 极光 / 自研 MQTT |
| **Mirror UI Service** | 镜端界面配置、AR Overlay 管理 | Node.js + WebSocket + Flutter |

### 3.2 通信与事件驱动

- **同步调用**: gRPC / REST (用于 latency-sensitive 路径，如穿搭即时推荐)
- **异步事件**: **Apache Kafka** 或 **RabbitMQ** (用于 Agent 状态流转、外部平台 Webhook、日志审计)
- **镜端通信**: **WebSocket / MQTT** (低延迟状态同步、语音流双向传输)

---

## 4. 数据层

### 4.1 数据库选型

| 数据类型 | 存储方案 | 选型原因 |
|----------|----------|----------|
| **结构化衣橱数据** | PostgreSQL (主) + MongoDB (辅) | 关系型数据用 PG；单品属性灵活扩展用 Mongo |
| **用户偏好记忆** | Redis + Neo4j / 图数据库 | 穿搭关联天然适合图结构；Redis 缓存高频画像 |
| **向量 embedding** | Milvus / Pinecone / pgvector | 单品图像/文本向量检索、Lookbook 相似搜索 |
| **事件/日志** | ClickHouse / Elasticsearch | Agent 决策 trace、用户交互漏斗分析 |
| **文件存储** | 阿里云 OSS / AWS S3 | 高清衣物图、搭配合成图、视频录制缓存 |

### 4.2 数据模型核心概念

```python
# 衣橱单品 (WardrobeItem)
{
  item_id: UUID,
  owner_id: UUID,
  name: str,
  category: enum,          # 上装/下装/鞋/配饰
  attributes: {            # 结构化标签
    color: ["blue", "denim"],
    material: "cotton",
    season: ["spring", "autumn"],
    style_tags: ["casual", "vintage"]
  },
  images: {
    original: s3_url,      # 实拍图
    cutout: s3_url,        # 白底抠图
    embedding: vector      # 视觉向量
  },
  lifecycle: {
    status: "in_closet" | "selling" | "sold" | "donated",
    last_worn: timestamp,
    wear_count: int,
    acquisition_date: timestamp
  },
  price: { original: float, resale_estimate: float }
}

# Lookbook (搭配方案)
{
  lookbook_id: UUID,
  items: [item_id_1, item_id_2, ...],
  occasion: str,           # 场景
  weather_suitability: {}, # 温度/降水适配
  generated_image: s3_url, # 合成搭配图
  user_feedback: { liked: bool, worn: bool }
}
```

---

## 5. 前端与交互层

### 5.1 智能镜 UI

- **技术栈**: **Flutter** (跨平台、GPU 渲染性能好、适合嵌入式 Linux/Android) 或 **React + WebGL** ( если 镜端跑 Chromium)。
- **核心模块**:
  - **Idle 待机界面**: 动态壁纸、天气 Widget、语音助手形象。
  - **Fit Check 模式**: 实时视频流 + AR Overlay (手势光标、 bounding box 高亮新衣)。
  - **Lookbook Gallery**: 卡片式横向滑动交互，支持手势切换。

### 5.2 移动端 App UI

- **技术栈**: **Flutter** (一套代码覆盖 iOS + Android) 或 **React Native**。
- **核心页面**:
  - 推送决策卡片 (闲置转卖)
  - 衣橱浏览器 (二维/三维衣橱)
  - 历史 Lookbook 与穿搭日历
  - 个人风格偏好设置

---

## 6. 外部集成 (Third-Party APIs)

| 能力 | PRD 提及 | 推荐国内集成方案 |
|------|----------|------------------|
| **天气** | 实时天气 | 和风天气 / 心知天气 / 高德天气 |
| **二手平台** | 闲鱼自动发布/交易 | 闲鱼开放平台 (POP) / 爱回收 / 转转 API (需商务入驻) |
| **物流** | 自动下单/预约揽件 | 菜鸟裹裹商家版 / 顺丰同城 / 圆通电子面单 |
| **信用评估** | 买家信用 | 芝麻信用 API (企业接入) / 平台自带信用分 |
| **品牌订阅** | 新品追踪 | 品牌官网 RSS + 爬虫 (合规) / 得物 / 天猫新品 API |
| **语音** | ASR/TTS | 阿里云 (Paraformer + CosyVoice) / 讯飞开放平台 |
| **VLM** | 视觉理解 | 阿里云百炼 (Qwen-VL) / 智谱 GLM-4V / 火山引擎 |

**重要风险**: 闲鱼等 C2C 平台通常不提供完全自动化的发布 API，真实落地可能需要 **RPA (Robotic Process Automation)** 或 **浏览器自动化 (Playwright/Puppeteer)** 结合 OCR 实现，存在平台政策风险。量产阶段建议与平台方签订 B2B 合作协议。

---

## 7. Demo 与 Production 的差异映射

本文档所在的 `liyzhao/demo1` 分支是一个**快速概念验证 (PoC)**，它验证了核心用户故事（闲置衣物自动转卖）的端到端闭环。

| 维度 | Demo (当前分支) | Production (完整产品) |
|------|-----------------|----------------------|
| **Agent 框架** | LangGraph 单文件 4 节点 | LangGraph Supervisor + 多子图 + 持久化记忆 |
| **UI** | Gradio 双面板网页 | 智能镜 Flutter App + 手机 Native App |
| **感知层** | 无 (纯按钮触发) | 语音 ASR + VLM + 手势识别 |
| **数据库** | `database.json` 本地文件 | PostgreSQL + Milvus + Redis 集群 |
| **推荐系统** | 无 (仅转卖) | Fashion-CLIP + 搭配评分模型 + LLM Rerank |
| **外部 API** | `mock_apis.py` 纯随机模拟 | 真实天气/闲鱼/物流/信用 API + RPA |
| **部署** | 本地 `python app.py` | K8s 微服务 + Model Serving + CDN |

---

## 8. 推荐演进路线图

### Phase 1: Demo / PoC (当前)
> 目标: 用最小成本验证交互逻辑与用户价值假设。
- ✅ 已用 LangGraph + Gradio + Mock API 完成转卖流程验证。

### Phase 2: Mirror MVP
> 目标: 让镜子"能看、能说、能推荐"。
- 接入一个可用的 **ASR + TTS** 服务，实现语音唤醒与对话。
- 接入一个 **VLM (如 Qwen-VL)**，实现镜前拍照 + 新衣识别的完整链路。
- 用 **Prompt Engineering + RAG** 实现初代穿搭推荐。

### Phase 3: Agent 平台化
> 目标: 从单流程扩展到完整的 Multi-Agent 生态。
- 拆分 Master Agent 与 Expert Agents。
- 引入 **向量数据库** 沉淀穿搭知识与用户偏好。
- 支持**人机在环断点续行**（用户可在任意步骤接管 Agent）。

### Phase 4: 智能化与规模化
> 目标: 推荐精准度达到"私人造型师"水平。
- 收集真实用户反馈数据，训练 **Outfit Compatibility Model**。
- 接入真实二手平台 API/RPA，实现全自动化交易闭环。
- A/B 测试驱动的 Agent 策略优化（不同用户的决策摘要生成策略）。

---

## 9. 关键技术决策记录 (ADRs)

**ADR-001: 为什么选择 LangGraph 作为 Agent 框架？**
- 原因: 穿搭推荐和闲置转卖都是典型的**长流程人机协同任务**，需要在工作流中明确插入"等待用户确认"的断点。LangGraph 的图结构 + Checkpoint 机制天然支持这种带状态的循环/等待模式，比纯 ReAct 或线性 Chain 更适合。

**ADR-002: 为什么不直接用 LLM 生成搭配图，而是先生成 Lookbook 再合成？**
- 原因: 生成式 AI 直接出搭配图（如 Diffusion）可控性差，难以保证每一件都真实存在于用户衣橱中。先基于真实单品做组合优化，再用拼图/虚拟试衣技术合成，能确保推荐的**可执行性**（用户真的能穿这套出门）。

**ADR-003: 为什么镜端视觉建议用 VLM 而非传统 CV Pipeline？**
- 原因: 为了降低硬件复杂度。传统方案需要分别训练检测、分割、属性分类、场景识别等多个模型，而 VLM（如 GPT-4V / Qwen-VL）可以通过 Prompt 直接输出结构化属性，适合早期快速迭代。量产后可在 VLM 标注数据上蒸馏出边缘端小模型以降低成本。

---

*文档作者: Liyang Zhao*
*最后更新: 2026-04-02*
*关联: FashionClaw_PRD.pdf, `app.py`, `workflow.py`*

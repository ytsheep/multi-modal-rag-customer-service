# 多模态RAG智能客服中台

聚焦工业产品手册、说明书、规则文档查询成本高、PDF 解析噪声多、参数回答易幻觉、页码溯源困难等痛点，搭建面向产品资料场景的智能问答系统。采用“意图识别分流 + 文档解析清洗入库 + LangChain RAG 问答链路”架构，实现闲聊直答、资料问答、引用溯源和自动化评测闭环。系统支持用户上传产品资料，自动解析、清洗、切片并写入 ChromaDB，通过混合召回与重排序 Top3 精准定位原文片段，回答中严格引用文件名与页码，适用于产品参数查询、安装要求确认、型号对比和手册条款检索等场景。

## 技术栈

| 层级 | 技术类型 |
| --- | --- |
| 大模型推理 | DashScope Qwen Plus |
| Embedding | DashScope text-embedding-v4，1024 维 |
| RAG 编排 | LangChain Runnable + PromptTemplate |
| 模型接入 | langchain-community + ChatTongyi |
| 文档解析 | PyMuPDF / python-docx / Markdown 文本解析 |
| OCR | 阿里云 OCR，处理扫描件 PDF |
| 向量库 | ChromaDB，HNSW + cosine 距离 |
| 关键词检索 | 内置 BM25 风格稀疏检索 |
| 重排序 | 向量分、关键词分、标题命中加权重排 |
| 后端框架 | FastAPI + Uvicorn |
| 前端框架 | Vue3 + Vite |
| 数据存储 | SQLite，保存文档记录与问答历史 |
| 会话记忆 | 短期 Memory，仅保留当前问答上下文 |
| 自动评测 | Python 脚本，覆盖意图、检索、内容 |

## 系统运行流程架构

```text
用户输入 / 文件上传
        |
        v
+---------------------------+
| FastAPI 后端服务           |
+---------------------------+
        |
        +---------------- 文件上传 ----------------+
        |                                          |
        v                                          v
+-------------------+                  +----------------------+
| 原文件持久化       |                  | 文档解析              |
| backend/data       |                  | PDF/Word/Markdown/OCR |
+-------------------+                  +----------------------+
                                                |
                                                v
                                      +----------------------+
                                      | 数据清洗管道          |
                                      | 去噪/规范化/结构恢复 |
                                      +----------------------+
                                                |
                                                v
                                      +----------------------+
                                      | Chunk 切分            |
                                      | 512 窗口/100 重叠     |
                                      +----------------------+
                                                |
                                                v
                                      +----------------------+
                                      | ChromaDB 向量入库     |
                                      | 保存页码/文件名元数据 |
                                      +----------------------+

用户问题
   |
   v
+-------------------+
| IntentRouter      |
| 规则意图识别       |
+-------------------+
   | direct_chat                    | rag_qa
   v                                v
+-------------------+       +----------------------------+
| 千问直接回答       |       | LangChain RAG 问答链路      |
+-------------------+       +----------------------------+
                                    |
                                    v
                          +----------------------------+
                          | HybridLangChainRetriever   |
                          | 向量召回 + 关键词召回       |
                          +----------------------------+
                                    |
                                    v
                          +----------------------------+
                          | 加权融合 + 重排序 Top3      |
                          +----------------------------+
                                    |
                                    v
                          +----------------------------+
                          | PromptTemplate 注入上下文   |
                          | 文件名/页码/原文片段        |
                          +----------------------------+
                                    |
                                    v
                          +----------------------------+
                          | Qwen 生成答案 + 引用溯源    |
                          +----------------------------+
```

## 一、文档入库

文档入库流程负责把用户上传的产品资料转成可检索、可追溯的知识片段。

```text
上传文件
  -> 保存原文件
  -> 文档解析
  -> 数据清洗
  -> 结构化恢复
  -> 语义切片
  -> Embedding
  -> ChromaDB 入库
  -> SQLite 记录文档元信息
```

### 1. 文件解析

系统按文件类型选择不同解析方式：

- 普通 PDF：使用 PyMuPDF / fitz 提取文本和页码。
- 扫描件 PDF：文本不足时接入阿里云 OCR。
- Word 文档：使用 python-docx 解析段落文本。
- Markdown / TXT：直接读取文本并做轻量正则处理。

### 2. 数据清洗

入库前会经过清洗管道，减少 PDF 抽取噪声对召回的影响：

- 去噪：删除连续空行、多余空格、制表符和不可见字符。
- 规范化：统一换行符、引号、全角半角和常见标点。
- 内容过滤：过滤过短无效行、纯符号行、页眉页脚。
- 结构恢复：合并分页断裂段落，保留标题和页码信息。
- 轻量处理：不做复杂 NLP，避免破坏产品原文语义。

### 3. 切片策略

当前采用适合产品资料问答的细粒度切片：
```text
chunk_size = 512
chunk_overlap = 100
```

### 4. 元数据设计

每个 chunk 入库时保留以下元数据：

| 字段 | 说明 |
| --- | --- |
| chunk_id | 片段唯一 ID |
| document_id | 文档唯一 ID |
| file_name | 来源文件名 |
| title_path | 章节标题路径 |
| page_start | 起始页码 |
| page_end | 结束页码 |
| content | 清洗后的片段正文 |

元数据用于回答时展示“答案来自哪个文件、第几页、哪段原文”。

## 二、数据检索

系统支持闲聊直答和资料问答分流，避免所有问题都进入 RAG，降低无效检索和响应延迟。

### 1. 意图识别

`IntentRouter` 使用关键词和正则规则，将问题分为两类：
| 意图 | 处理方式 |
| --- | --- |
| direct_chat | 简单问候、感谢、闲聊，直接调用大模型回答 |
| rag_qa | 产品参数、型号、安装、接线、页码依据等问题 |

对于产品资料场景，系统更偏向安全策略：当问题不明显是闲聊时，默认进入 `rag_qa`，让回答尽量基于资料。

### 2. 混合召回

RAG 检索由 `HybridLangChainRetriever` 封装，兼容 LangChain Retriever 接口。

```text
用户问题
  -> Chroma 向量召回 TopN
  -> 关键词稀疏召回 TopN
  -> 0.65 * 向量分 + 0.35 * 关键词分
  -> 标题命中、词项重合、产品线提示加权
  -> 重排序 Top3
```

## 三、效果评测

项目内置自动化评测脚本，覆盖意图识别、检索质量和内容质量三类能力。

### 1. 评测数据

| 文件 | 作用 |
| --- | --- |
| eval/intent_cases.jsonl | 意图识别评测集 |
| eval/rag_knowledge.jsonl | 评测用知识片段 |
| eval/rag_cases.jsonl | 检索评测集 |
| eval/content_cases.jsonl | 回答内容评测集 |

### 2. 评测指标

| 类型 | 指标 | 说明 |
| --- | --- | --- |
| 意图识别 | Accuracy | direct_chat / rag_qa 路由准确率 |
| 检索质量 | Recall@3 | Top3 是否命中目标 chunk |
| 检索质量 | MRR | 目标 chunk 首次出现位置的倒数均值 |
| 内容质量 | Fact Coverage | 关键事实命中比例 |
| 内容质量 | Page Citation Accuracy | 文件名与页码引用准确率 |
| 内容质量 | Latency | 平均延迟与 P95 延迟 |

### 3. 当前样例评测结果

最近一次本地样例评测结果如下：

| 评测项 | 样本数 | 结果 |
| --- | ---: | ---: |
| 意图识别 Accuracy | 20 | 1.0000 |
| 检索 Recall@3 | 20 | 1.0000 |
| 检索 MRR | 20 | 0.9750 |
| 内容 Accuracy | 20 | 0.8500 |
| 平均事实命中率 | 20 | 0.8250 |
| 页码溯源准确率 | 20 | 1.0000 |
| 平均延迟 | 20 | 6.5931s |

## 四、结果展示


## 项目架构

```text
RAG智能问答/
├── backend/
│   ├── app/
│   │   ├── agent/                 # 意图识别与问答服务编排
│   │   ├── api/                   # chat/upload/history 接口
│   │   ├── core/                  # 配置管理
│   │   ├── ingestion/             # 解析、清洗、切片、入库
│   │   ├── langchain_pipeline/    # Prompt 与 RAG Chain
│   │   ├── llm/                   # DashScope 模型与 Embedding
│   │   ├── memory/                # 短期上下文
│   │   ├── retrieval/             # 混合召回与重排序
│   │   ├── storage/               # 文件、Chroma、SQLite 存储
│   │   ├── main.py                # FastAPI 入口
│   │   └── models.py              # 请求响应模型
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── api/                   # 前端请求封装
│   │   ├── components/            # 页面组件
│   │   ├── App.vue
│   │   └── main.js
│   ├── package.json
│   └── vite.config.js
├── eval/
│   ├── intent_cases.jsonl
│   ├── rag_knowledge.jsonl
│   ├── rag_cases.jsonl
│   ├── content_cases.jsonl
│   ├── exports/                   # 导出的 Chroma 切片
│   └── reports/                   # 评测报告
├── scripts/
│   ├── export_chroma_chunks.py    #导出脚本
│   └── run_eval.py                #评测脚本
├── docs/
│   └── EVALUATION.md
├── start-backend.ps1
├── start-frontend.ps1
└── README.md
```

## 快速开始

### 1. 环境准备

推荐环境：
```text
Python 3.12+
Node.js 20+
npm 10+
```

进入项目根目录：
```powershell
cd D:\code\RAG智能问答
```

创建后端环境并安装依赖：
```powershell
python -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

安装前端依赖：
```powershell
cd frontend
npm install
cd ..
```

### 2. 配置环境变量

复制环境变量模板：

```powershell
Copy-Item backend\.env.example backend\.env
```
编辑 `backend/.env`：
```env
DASHSCOPE_API_KEY=你的DashScope API Key
```

如需解析扫描件 PDF，可开启阿里云 OCR：
```env
ALIYUN_OCR_ENABLED=true
ALIYUN_ACCESS_KEY_ID=你的阿里云AccessKey
ALIYUN_ACCESS_KEY_SECRET=你的阿里云Secret
```


### 3. 启动服务

启动后端：
```powershell
.\start-backend.ps1
```
或手动启动：
```powershell
backend\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --app-dir backend
```

启动前端：
```powershell
.\start-frontend.ps1
```
或手动启动：
```powershell
cd frontend
npm run dev
```

### 4. 访问服务

```text
前端页面：http://127.0.0.1:5173
后端接口：http://127.0.0.1:8000/docs
```


### 5. 运行测评

只评测意图识别：
```powershell
python scripts\run_eval.py --mode intent
```

评测检索并重新灌入评测知识库：
```powershell
python scripts\run_eval.py --mode retrieval --seed
```

评测回答内容：
```powershell
python scripts\run_eval.py --mode content --seed
```

运行完整评测：
```powershell
python scripts\run_eval.py --mode all --seed
```

评测报告会生成到：
```text
eval/reports/
```



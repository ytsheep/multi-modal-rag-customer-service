# 产品资料问答助手评测说明

本项目评测只覆盖三类核心能力：意图、检索、内容。评测数据面向产品手册咨询环境，不再使用通用 RAG 知识问答样例。

## 1. 意图识别

文件：`eval/intent_cases.jsonl`

共 20 条：

- `direct_chat`：普通问候、感谢、闲聊、通用概念问题
- `rag_qa`：产品参数、安装、接线、故障、页码引用、两个产品对比

运行：

```powershell
python scripts\run_eval.py --mode intent
```

指标：

```text
Intent Accuracy = 正确路由条数 / 20
```

## 2. 检索评测

文件：

```text
eval/rag_knowledge.jsonl
eval/rag_cases.jsonl
```

`rag_knowledge.jsonl` 提供产品手册样例片段，包含文件名、章节、页码和原文内容。`rag_cases.jsonl` 标注每个问题应该召回的目标 chunk。

运行：

```powershell
python scripts\run_eval.py --mode retrieval --seed
```

指标：

```text
Recall@3 = Top3 中命中任一目标 chunk 的问题数 / 20
MRR = 目标 chunk 首次出现位置的倒数均值
```

## 3. 内容评测

文件：`eval/content_cases.jsonl`

共 20 条，覆盖：

- 产品参数：电压、电流、功率、温度、防护等级
- 安装接线：DIN 导轨、通风距离、端子、拧紧力矩
- 状态故障：ERROR、DC OK、短路过载恢复
- 对比问题：两个产品分别列依据

运行：

```powershell
python scripts\run_eval.py --mode content --seed
```

指标：

```text
Content Accuracy = 意图正确 + 关键事实命中率达标 + 页码引用正确 的样本数 / 20
Avg Fact Coverage = 关键事实平均命中率
Page Citation Accuracy = 结构化引用中页码和文件命中的比例
```

## 注意

样例评测集用于验证项目链路和指标脚本。真实简历项目展示时，建议把样例产品 A/B 替换成你上传的西门子真实手册，并重新标注目标 chunk、关键事实和页码。

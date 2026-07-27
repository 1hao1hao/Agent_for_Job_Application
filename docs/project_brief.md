# Project Brief

## 项目名称

面向实习求职场景的可评测多源 RAG/Agent 系统。

## 一句话目标

构建一个能围绕实习求职问题进行多源检索、可追踪推理、可量化评测、可失败复盘的 Python 项目，而不是一个只会聊天的通用问答机器人。

## 为什么值得做

这个项目服务于实习申请和面试准备，但工程目标更重要：它要展示你对 RAG/Agent 系统的完整理解，包括数据建模、检索、路由、引用、工具调用、可观测性、失败样例沉淀和指标评测。

面试官可以从以下角度深挖：

- 多源知识如何组织、切分和检索。
- Query 如何被识别意图并路由到不同知识源。
- Agent 每一步为什么这样做，是否能被 trace 解释。
- 如何判断系统答得好不好，而不是只看主观体验。
- 失败样例如何进入 regression test set，下一轮如何避免退化。
- 第一版原生 Python 如何保持简单，第二版如何迁移到 LangGraph 或 LlamaIndex。

## 核心知识源

- JD: 目标岗位描述、职责、技能要求、加分项。
- Resume: 个人简历、项目经历、技能栈、教育背景。
- Interview Notes: 八股、面经、常见题、准备材料。
- Project Logs: 本项目的开发日志、设计选择、踩坑记录。
- User Profile: 用户画像、偏好、目标城市、目标岗位、申请节奏。

## 第一版范围

第一版只做最小可运行系统，重点验证工程闭环：

- 原生 Python 实现核心 RAG 流程。
- 支持本地文本数据加载和 chunk 切分。
- 支持基础检索、简单 rerank、答案引用。
- 支持简单 intent/router。
- 输出结构化 agent trace。
- 提供最小 evaluation 脚本和 regression test set。
- 每个模块有最小测试。

第一版暂不追求：

- Web UI。
- 复杂多 Agent 协作。
- 在线爬虫。
- 自动投递。
- 数据库服务化。
- LangGraph/LlamaIndex 重构。
- 多用户权限和生产级部署。

## 推荐项目目录结构

```text
.
├── configs/
│   └──                 # 后续存放模型、检索、评测等配置
├── data/
│   ├── raw/
│   │   ├── interview/  # 八股、面经、题库原始文本
│   │   ├── jd/         # 岗位 JD 原始文本
│   │   ├── project_logs/
│   │   ├── resume/
│   │   └── user_profile/
│   ├── processed/
│   │   └── chunks/     # 后续存放切分后的 chunk 数据
│   └── indexes/        # 后续存放向量索引或倒排索引
├── docs/
│   ├── architecture.md
│   ├── coding_rules.md
│   ├── decision_log.md
│   ├── project_brief.md
│   └── task_board.md
├── notebooks/          # 可选实验，不作为主流程依赖
├── scripts/            # 后续放一次性命令入口
├── src/
│   └── intern_rag/
│       ├── agent/      # Agent 编排与响应生成
│       ├── evaluation/ # 指标计算与评测报告
│       ├── ingestion/  # 数据加载、清洗、切分
│       ├── retrieval/  # 检索、rerank、citation
│       ├── routing/    # intent 识别与知识源路由
│       ├── tracing/    # trace schema 与记录器
│       └── utils/
├── tests/
│   ├── fixtures/       # 测试用小样本
│   ├── regression/     # 失败样例沉淀
│   └── unit/
└── traces/             # 本地运行产生的 agent trace
```

## 关键指标

- Recall@k: 标注相关 chunk 是否出现在前 k 个检索结果中。
- Citation Accuracy: 回答中的引用是否真实支持对应陈述。
- Router Accuracy: intent/router 是否选择了正确知识源或工具。
- Tool Success Rate: 工具调用是否成功返回可用结果。
- Hallucination Rate: 回答中无法由知识源支持的事实比例。
- Latency: 端到端耗时，以及 retrieval、rerank、generation 等阶段耗时。

## 项目展示重点

最终这个项目应该能回答三个问题：

1. 系统为什么这么答？
2. 答案依据在哪里？
3. 下一次如何证明它变好了？

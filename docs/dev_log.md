## 2026-06-15：理解数据切分模块 src/intern_rag/ingestion

这个模块的职责是把多源原始材料统一成后续检索可以消费的“证据单元”。
（不同来源的数据都变成同一种结构：Document 和 Chunk，来源可区分、metadata 不丢失）

### 主要函数：
build_document_from_file() 的实现逻辑是提取本地文件的元数据等信息并将其封装

split_text():
当前切分逻辑是先根据空行进行切分，再将切片合并（不超过max_chars）作为最终切分结果

build_chunks_from_file() ：
先把原始文件封装成 Document，统一正文和 metadata；再用 split_text 切成多个文本片段；最后每个片段继承文档级 metadata，并追加 chunk 自己的 chunk_index、source_file_name、char_count，形成 Chunk 列表。

这个模块的职责是把多源原始材料统一成后续检索可以消费的“证据单元”。

### 问题：
1. Document 表示什么？
2. Chunk 表示什么？
3. 为什么不能直接把整篇文档丢给模型？
4. metadata 是什么时候补全的？
5. 一个 Document 会切成几个 Chunk？
6. Chunk 为什么要继承 source_type、status、company、job_title 这些字段？

## 2026-06-17：理解 检索 模块 src/intern_rag/retrieval

### 核心函数：
1. tokenize_text：把中英文转成关键词集合。英文和数字按连续词提取，中文按单字和相邻双字提取。
案例输入：
tokenize_text("熟悉 FastAPI、Git、Linux 基础使用")
输出：
{
    "fastapi",
    "git",
    "linux",
    "熟", "悉", "基", "础", "使", "用",
    "熟悉", "悉基", "基础", "础使", "使用",
}

2. score_chunk：计算 query token 和 chunk token 的重叠分数。 （ query 和 chunk 都先经过 tokenize_text 处理）
目前的重叠分数用 重叠字符(交集) 在query中的占比表示

3. retrieve_top_k：过滤 source type、排序、返回 top-k 检索结果。
先对 chunk进行初步筛选，主要是过滤来源类型，
再计算匹配分数，再依据分数排序并返回前k个chunk。


## 2026-06-17：

### 理解route模块 src/intern_rag/routing

Intent 是一个问题意图列表；
INTENT_TO_SOURCES 将意图和其所匹配的 source 列表组合起来；INTENT_KEYWORDS 则是将意图和相关关键词组合起来。 
这个routing 模块在项目逻辑上是在 retrieval 模块之前的，缩小检索的文件范围

LLM router:利用LLM来分析问题意图；返回结构化输出
第一版项目不使用 LLM router: 使用自定义规则版：可解释，稳定，无api成本，当前先追求跑通闭环

#### 核心函数
route_query 的作用是找出关键词命中query最多的intent，
返回 RouteDecision（封装了意图，意图匹配的文件路由，匹配依据（命中的意图关键词））

#### Router Accuracy 怎么评测
对一批query人工标注 intent / correct routed_sources，再使用routing模块匹配，对比与标注是否一致

### 梳理项目逻辑
#### 整体逻辑
原始数据
  -> ingestion              
  -> chunks
  -> 用户 query
  -> routing
  -> retrieval
  -> rerank
  -> answer + citations
  -> trace
  -> evaluation / regression

1. 原始数据层  
  data/raw/jd/  
  data/raw/resume/  
  data/raw/interview/  
  data/raw/project_logs/  
  data/raw/user_profile/    
    
    JD 是岗位要求。  
    resume 是个人简历。  
    interview 是面经八股。  
    project_logs 是项目开发过程。  
    user_profile 是用户目标和偏好。  

2. ingestion：把多源原始数据统一成 chunks
3. routing：理解用户 query 要查什么
4. retrieval：在 routed sources 里找相关 chunks
6. answer + citations：基于证据组织回答
7. tracing：记录一次请求的完整过程
8. evaluation / regression：评测和失败复盘

## 2026-07-26：

### 理解trace模块 src/intern_rag/tracing

#### 核心函数
build_agent_trace: 构建traceAgent数据，记录rag流程的 route retrival等阶段的结果(query,intent,source,检索结果,latency,error_type等等) ，相当于构造函数
retrieval_result_to_trace: 将检索得到的 chunk 类型数据转化为 字典类型 ，便于文件读写
write_trace_jsonl：将trace记录（AgentTrace）写入 jsonl 文件  
read_traces_jsonl： 读取 jsonl 文件内容转化为 trace 记录 
#### 当前限制  
1. 当前trace只记录到 retrieval 阶段 （原始数据->ingestion(将原始数据转化为 document 再切分为 chunks -> router(利用query匹配 intent意图 再匹配 source 缩小检索范围，可用标记intent和source的样本检测 routing accuracy)-> 将chunk和query token化，返回top-k个token)  
2. latency 由调用方传入，暂时没有计时器  
3.   JSONL 没有并发写保护
4.  error_type 只是固定枚举

#### 掌握这个模块，你需要会回答：
1. 为什么 trace 用 JSONL？  
  JSONL 是一行一个 JSON ，可以每次追加或读取一行，不用读整个文件
  一行数据又问题不影响其他行
  很适合日志，trace
2. AgentTrace 记录哪些字段？  
    request_id: str 命令id  
    query: str  用户问题  
    intent: str  意图  
    routed_sources: list[str] router路由结果  
    retrieved_chunks: list[TraceDict]  检索结果  
    latency_ms: dict[str, float]  耗时  
    error_type: ErrorType = "none"  错误类型  
    created_at: str = ""  trace 创建时间  
    rerank_results:   重排结果（目前还没有这个模块）  
    tool_calls:    是未来 Agent 调用工具的记录。  
    citations:   目前还没有对应模块
    answer: str = "" llm 输出内容（目前还没有该模块）  
3. trace 和普通日志有什么区别？  
  普通的日常像是记录流水账，而trace是记录结构化的数据，便于测评，统计，复盘完整的决策链路  
4. tracing 为什么不直接执行 router/retrieval？
为了让功能分离化，模块化，增强独立性和可拓展性  
5. error type 有什么用？
便于锁定错误位置，分析错误原因  
6. retrieved chunks 为什么要记录 rank、score、source 和 metadata？  
rank 用于区分 chunk的相关性和重要性， score 是rank的依据， source 提高数据可追溯性；metadata记录了数据的原信息，帮助区分数据

7. 后续 evaluation 如何使用 trace？
Evaluation 就是评测模块，用来回答：
系统到底有没有变好？  
trace 是为 evaluation 提供原材料。

#### 总结
tracing 模块不执行 RAG 流程，它负责把一次 query 的中间结果结构化保存下来。  
实际流程中，router、retrieval、answer 等模块先各自执行，把结果和耗时保存在变量里，最后统一构造一条 AgentTrace 并写入 JSONL。  
这样后续可以复盘系统为什么这么答，也可以基于 trace 做评测和失败样例沉淀。

### 理解 Answer + Citations 模块 src/intern_rag/agent/answer.py
#### 核心函数
compose_answer：基于 top chunks 组织回答和 citations。  
    实际操作：选取 top chunks, 构建引用（citation_from_result）, 整理证据片段(format_evidence_snippet),生成答案(包含answer，citations,chunkid,is_evidence_sufficient)
citation_from_result：从 RetrievalResult 生成 Citation。  
    实际操作：提取 chunk 的 id source title rank score 信息，将其结构化  
format_evidence_snippet：把 chunk 文本整理成回答里的证据片段。  
    实际操作：整合 rank + chunk的部分字符，作为证据片段
#### 掌握这个模块，你需要会回答：
1. 为什么第一版不用 LLM 生成答案？  
因为当前阶段的目标是先跑通可追溯的 RAG 闭环，确保回答只来自 retrieved chunks，并且 citation 能准确指向证据来源。LLM 会引入生成不稳定和幻觉问题，所以第一版先用抽取式回答作为 baseline。等 trace、evaluation 和 citation 结构稳定后，再接 LLM 生成。
2. 什么是 citation，它解决什么问题？  
citation 是回答所依赖证据的结构化引用，记录 chunk_id、source_path、source_type、rank、score 等信息，用来支持溯源、复盘和 citation accuracy 评测。   
3. 为什么回答必须只基于 retrieved chunks？  
retrieved chunks 是系统当前找到的证据来源，但不一定完全正确。因此第一版回答只能基于这些 chunks，是为了避免脱离证据编造；后续还要通过 Recall@k 和 citation accuracy 评测检索和引用是否真的正确。
4. 证据不足时为什么要明确拒答/说明不确定？  
如果没有检索到相关 chunk，系统不能为了回答而编造内容。明确说明证据不足，是降低 hallucination rate 的基本策略。
5. AnswerResult 和 Citation 分别记录什么？  
AnswerResult 记录 答案，引用，引用的chunk的id,证据是否充分；AnswerResult.used_chunk_ids 可以帮助后续检查回答到底用了哪些 chunks，方便和 citation、trace、评测数据对齐。  
Citation 记录 引用的 chunk 的 chunk_id，source_path，source_type，title ，rank，score    
6. 后续如何把 answer 和 citations 写入 trace？  
AgentTrace 已经预留了 answer 和 citations 字段。后续在 agent 编排流程里，调用 compose_answer 后，把 answer_result.answer 和 [citation.to_dict() for citation in answer_result.citations] 填入 trace。  
7. 后续 citation accuracy 应该怎么评测？  
准备一批人工标注样例，每个 query 标注哪些 chunk 能支持答案。系统生成 citations 后，比较 citation 中的 chunk_id 是否出现在人工标注的 gold chunk ids 中。可以计算引用命中率，也可以人工检查引用文本是否真的支持回答中的陈述。   

#### 当前限制：
第一版是抽取式回答，不调用 LLM，不做真正总结推理。  
回答会直接展示 chunk 片段，表达比较朴素。  
citation 只是记录来源，还没有做 citation accuracy 自动评测。  
answer/citations 还没有自动写回 AgentTrace，后续可以在 agent 编排时整合。  
 
#### 总结  
这个模块负责把 retrieval 返回的 top chunks 组织成一个保守回答，并生成结构化 citations。  
第一版不调用 LLM，而是直接抽取 chunk 片段，目的是保证回答不脱离证据。  
Citation 记录 chunk_id、source_path、source_type、rank 和 score，后续可以写入 trace，并用于 citation accuracy 评测。  


### 理解 Evaluation 与 Regression Seed 模块 src/intern_rag/evaluation
#### 核心函数
calculate_recall_at_k 计算检索样例的前k召回率：前k个检索chunk中命中的数量/标注的chunk总数量  
calculate_average_recall_at_k 计算检索样例的平均前k召回率：计算多个检索样例的前k召回率并平均  
calculate_router_accuracy 计算路由准确率：预测 sources 和 intents 与 标注的结果都相同才算准确
evaluate_cases 评估样例：计算一批样例的平均前k召回率以及路由准确率  
load_evaluation_cases 读取 JSONL 中的检索和路由数据进行评估

#### 掌握这个模块，你需要会回答：
1. Recall@k 衡量什么？  
  检索样例的前k召回率：前k个检索chunk中命中的数量/标注的chunk总数量  
2. Router Accuracy 怎么定义？
预测 sources 和 intents 与 标注的结果都相同才算正确，计算 正确的样例数量/样例总数  
3. 为什么需要人工标注 expected？ 
  人工标注 expected 提供一个正确的 检索/路由 基准，便于进行评测  
4. evaluation 和普通单元测试有什么区别？  
  普通单元测试只能测试单元能否跑通，程序能否正常运行，而不能保证结果的准确性。evaluation 通过人工标注的数据可以对检索和路由等阶段进行准确性评测
5. regression seed 的作用是什么？  
  记录失败样例， 你给的解释是 "而是给后续 regression test set（回归测试集）建立格式。" 但我并不是很理解，我看代码时也没注意到 Regression Seed 的相关实现 
6. 为什么当前 demo 指标不能当真实效果？  
  当前还没有形成完整的RAG闭环，没有接入LLM。当前 demo 使用的是最小人工构造样例，只用于验证 evaluation 代码能运行，不代表真实业务效果。
7. 后续如何用 trace 自动生成评测输入？
  load_evaluation_cases 读取 JSON 文件中的 retrieval_cases 和 router_cases，并转成评测样例对象。

### 当前项目主线
我先实现了一个最小但可观测、可评测的 RAG baseline。它能把中文多源求职资料切成 chunks，按 query 路由 source，做关键词检索，生成基于证据的回答和 citation，并把中间过程写入 trace。最后用 Recall@k 和 Router Accuracy 做基础评测，把失败样例沉淀为 regression seed。
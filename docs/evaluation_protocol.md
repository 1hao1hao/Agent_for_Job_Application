# EvalRAG Evaluation Protocol

## 目的

本协议固定 EvalRAG P0 的数据划分、指标公式、运行方式和报告工件。任何正式对比都必须遵守同一口径，否则不能直接比较。

## Dataset Version

P0 正式数据集文件：

```text
data/evaluation/evalrag_v0.1.jsonl
```

`evalrag_v0.1` 已于 2026-07-30 完成 60 条标签审核和正式数据校验：
40 条 dev 用于开发与调参，20 条 test 从此冻结。正式 Keyword dev baseline
为 `reports/runs/keyword-dev-formal-v0.1-20260730/`。

P0-D3 新增 `evalrag_v0.2`：100 份五类文档、310 个自然 Chunk、120 条 Case，
按 80 dev / 40 frozen test 划分，四类 Query 各 30 条。新增 70 份材料明确标记为
`project_authored_synthetic_benchmark`；标签由项目作者委托 Codex 完成全量一致性
审核，不应描述为独立第三方人工标注或真实业务流量。P0-D3 只运行 dev，test
继续冻结。

P0-D3 正式 Dense 配置为 `BAAI/bge-small-zh-v1.5`，固定 revision
`7999e1d3359715c523056ef9478215996d62a620`、512 维；字符 TF-IDF + LSA 只作为
离线 fallback。文档向量离线生成，运行时只编码 Query。

后续如存在 `human_reviewed=false` 的 case，只能运行显式标记
`candidate_not_human_verified` 的 candidate run，不能混入正式对照。

冻结后每次改变以下内容必须升级版本：

- Query 文本。
- split。
- expected intent/sources。
- relevant chunk ids。
- answerable。
- expected points。
- 对应语料和 chunk id。

修正明显标注错误也要记录变更，不静默覆盖。

## Case Schema

```json
{
  "case_id": "multi_001",
  "query": "结合岗位要求和我的项目，我还缺少哪些能力？",
  "category": "multi_source",
  "split": "dev",
  "expected_intent": "match_resume",
  "expected_sources": ["jd", "resume", "project_logs"],
  "relevant_chunk_ids": ["jd-xxx", "resume-xxx", "project_logs-xxx"],
  "answerable": true,
  "expected_points": ["岗位要求", "已有能力", "能力缺口"],
  "notes": "人工核验说明"
}
```

## 数据规模与分割

P0 最少 60 条：

| Category | Dev | Frozen Test | Total |
|---|---:|---:|---:|
| `single_source` | 10 | 5 | 15 |
| `multi_source` | 10 | 5 | 15 |
| `semantic_paraphrase` | 10 | 5 | 15 |
| `unanswerable` | 10 | 5 | 15 |
| Total | 40 | 20 | 60 |

规则：

- dev 用于 Router、Retriever、RRF 和 Evidence 阈值调整。
- test 在最终 P0 Run 前冻结。
- 不根据 test 失败继续调参并仍将同一版本称为 frozen test。
- Query 不得只替换公司名形成近重复样例。
- 同一事实的高度相似问法不能跨 dev/test 泄漏。

## Corpus Manifest

每份文档至少记录：

```text
document_id
source_type
source_platform
source_url
collected_at
public_status
anonymized
content_hash
```

语料统计由脚本生成：

- 文档数量。
- 各 source 数量。
- chunk 数量。
- chunk 字符长度分布。
- 空文档。
- 重复 content hash。

## 固定检索配置

正式消融至少包含：

```text
A: keyword
B: dense
C: hybrid_rrf
```

除当前自变量外保持一致：

- dataset version。
- source routing。
- top-k。
- chunk 集。
- Query 顺序。
- 运行机器。

Hybrid 调参只使用 dev；最终参数写入 RunConfig。

## Routing Metrics

### Router Accuracy

严格口径：

```text
correct =
  predicted_intent == expected_intent
  AND
  set(predicted_sources) == set(expected_sources)

Router Accuracy = correct cases / all routed cases
```

如未来增加宽松 source recall，必须使用新指标名，不能替换严格口径。

## Retrieval Metrics

只在 `answerable=true` 且有 relevant ids 的 case 上计算。

### Recall@k

```text
Recall@k(case) =
  |top_k_retrieved_ids ∩ relevant_ids| / |relevant_ids|

Macro Recall@k =
  sum(Recall@k(case)) / case_count
```

正式报告至少包含 `k=3` 和 `k=5`。

### MRR

```text
RR(case) = 1 / rank_of_first_relevant_chunk
```

没有召回 relevant chunk 时为 0。

```text
MRR = sum(RR(case)) / case_count
```

## Grounding Metrics

### Citation Validity

```text
Citation Validity =
  citations whose chunk_id exists in current context
  / all returned citations
```

answered 但 citation 为空时，该 case validity 记为 0。不可回答且正确拒答的 case 不进入此指标分母。

Citation Validity 不判断引用文本是否语义支持回答。

### Key-Point Coverage

```text
Key-Point Coverage(case) =
  covered expected points / all expected points
```

P0-D5 的规范化字符串包含保留为 `lexical_key_point_coverage` baseline。P0-D6 的
Semantic Grader 对每个 expected point 输出 `covered | not_covered | unknown`、
reason 和答案原文 evidence span：

```text
Semantic Key-Point Coverage(case) =
  covered point verdicts / all expected points
```

只要该 Case 存在 unknown，Case coverage 记为 unavailable，不把 unknown 当作
not_covered 或 covered。Macro Coverage 只对 coverage 已知的 answerable Case 求平均，
报告必须同时给出 unknown case 数。没有 expected points 的 Case 不进入分母。

### Unsupported Answer Rate

由明确记录 grader 类型的逐 claim 审核：

```text
Unsupported Answer Rate =
  answered cases containing at least one unsupported factual claim
  / all answered cases
```

每条 factual claim 必须保存 `supported | unsupported | unknown`、reason、引用
Chunk 和 evidence span。存在至少一条 unsupported claim 的回答记为
`unsupported_answer=true`；全部 claim supported 才记为 false；存在 unknown 且
没有 unsupported 时保持 unknown。UAR 分母只包含 Grounding verdict 已知的
answered Case，并同时报告 answered/known/unknown 数量。模型 Judge 不能描述为
独立人工核验；若只完成抽样核验，必须报告抽样数量。

## P0-D6 Semantic Audit

P0-D6 固定 `p0-d6-key-point-v1` 和 `p0-d6-grounding-v1` 后，只读取 P0-D5 保存的
predictions 与 Trace，不重新运行生成链。source artifact SHA-256、grader call、
token、成本、P50/P95 和错误均保存在：

```text
reports/final/p0-d6-semantic-grounding-v0.2/
```

dev 可用于检查 grader 契约；frozen 只运行固定配置。非法 JSON、字段、citation 或
evidence span 均返回 unknown/unavailable，不允许为了得到完整指标反复重跑 frozen。

## Abstention Metrics

定义正类为“应该拒答”，即 `answerable=false`。

```text
Abstention Accuracy =
  correctly abstained unanswerable cases
  / all unanswerable cases
```

同时报告：

- Unexpected Abstention：可回答但系统拒答。
- Should Abstain：不可回答但系统回答。

只报告 Abstention Accuracy 会掩盖过度拒答，因此必须同时输出这两个失败数。

## End-to-End Success

answerable case 成功需要：

- status 为 `answered`。
- Router 严格正确。
- Recall@5 大于 0。
- Citation Validity 为 1。
- Key-Point Coverage 达到当次 RunConfig 的门槛。
- 人工核验不存在 unsupported factual claim。

unanswerable case 成功需要：

- status 为 `insufficient_evidence`。
- citations 为空。
- 没有 unsupported answer。

```text
End-to-End Success Rate =
  successful cases / all evaluated cases
```

报告同时给出 answerable 与 unanswerable 子集结果。

## Regression Pass Rate

```text
Regression Pass Rate =
  passed fixed regression cases / all fixed regression cases
```

- `open` case 不进入分母。
- `fixed` case 必须有自动化断言。
- 报告 open 数量，避免通过排除困难 case 制造 100%。

## Latency

分别记录：

- router。
- retrieval。
- context。
- generation。
- validation。
- total。

规则：

- 索引构建时间单独记录。
- 在线 query 使用已加载索引。
- 明确 cold/warm。
- 使用同一机器和进程模式对比 Retriever。
- 报告 case 数、P50 和 P95。
- P95 使用排序后最近秩方法，并在实现中固定。
- 不用单次最快耗时代替分布。

## Token 与成本

真实 LLM Run 记录：

- input tokens。
- output tokens。
- total tokens。
- model。
- 当次单价来源或配置时间。
- estimated cost。

Fake LLM 测试不产生真实成本指标。

未调用 LLM 的 deterministic extractive baseline 必须把 token 和成本写为
`unavailable_no_llm_call` / `null`，不能用字符数伪装 token，也不能把 0 写成
真实 API 成本。

## P0-D5 Frozen Test

`evalrag_v0.2/test` 的 40 条样例于 2026-08-04 使用配置
`configs/final/p0_v0.2.json` 一次性运行。配置文件固定 dataset、chunks、Router、
Retriever 和 Evidence config 的 SHA-256；查看 test 结果后不再调参或重跑同一版本。

正式检索对照包括 Keyword、Dense、Hybrid，以及 frozen 前已判定不启用的
Reranker candidate。最终配置为 Hybrid，Reranker 关闭。完整工件位于：

```text
reports/comparisons/p0-d5-v02-frozen-test-20260804/
```

端到端可靠性报告使用 deterministic extractive generator，不调用真实 LLM：

```text
reports/runs/p0-d5-v02-frozen-test-20260804-extractive-e2e/
```

该 Run 的 Unsupported Answer Rate 只表示“回答由已引用 Chunk 摘录组成”的契约级
检查，不满足本协议上文定义的独立人工语义核验口径。因此它不能被描述为真实
LLM 的幻觉率或答案准确率。端到端 token 与成本不可用。

Frozen prediction 落盘后发现一个指标实现错误：可回答但拒答被误记为 unknown。
修复只从原始 `case_results.jsonl` 重算 summary，未重新运行预测；修复前摘要保留为
`summary_before_metric_fix.json`。

## P0-D5 Live LLM Run

真实模型配置为 `deepseek-v4-flash`、非思考模式、JSON Mode、temperature 0、
`p0-deepseek-json-v1`，配置文件为：

```text
configs/llm/deepseek_v4_flash_v1.json
```

在不修改 v0.2 标签及冻结 Router/Retriever/Evidence 配置的情况下，依次运行完整
80 dev 和 40 frozen test。Run 工件：

```text
reports/runs/p0-d5-v02-dev-20260804-deepseek-v4-flash/
reports/runs/p0-d5-v02-frozen-test-20260804-deepseek-v4-flash/
reports/final/p0-d5-live-llm-v0.2/
```

token 直接使用 Provider usage；成本根据 RunConfig 中 2026-08-04 官方美元单价
快照估算。P0-D5 的 Unsupported Answer 标签来自 Codex 审核流程，但没有保存逐
claim/evidence 判定，现标记为 provisional，并由 P0-D6 Claim-Level Grounding
工件替代；两者都不是独立人工审核，报告和简历必须保留这一边界。

## Run Artifacts

每个正式 Run：

```text
reports/runs/<run_id>/
  run_config.json
  summary.json
  case_results.jsonl
  failures.jsonl
  latency.json
```

`run_config.json` 至少包含：

```text
run_id
created_at
git_commit
dataset_version
split
retriever_name
top_k
embedding_model
rrf_k
llm_model
temperature
prompt_version
context_budget
evidence_thresholds
```

## 对照实验规则

- 先运行 baseline，再做改动。
- 一次实验只改变一个主要变量。
- 三种 Retriever 使用相同评测集。
- dev 用于调参，test 用于最终比较。
- 报告提升也报告退化 case 和额外延迟。
- 如果 Hybrid 没有优于 Keyword，如实记录并分析，不修改标签制造提升。

## Failure-to-Regression

每个闭环案例需要：

1. `original_run_id`。
2. `original_trace_id`。
3. failure type。
4. root cause。
5. 修改假设。
6. fixed commit。
7. 修改前后 dev 指标。
8. frozen test 影响。
9. regression case id。

## 简历数字发布规则

一个数字只有同时满足以下条件才能写入简历：

- 来自正式 Run Artifact。
- 关联 dataset version 和 Git commit。
- 指标公式在本协议中定义。
- baseline 和优化方案使用相同数据。
- 能解释样例规模、分割和运行环境。
- 已填写 `docs/resume_evidence.md`。

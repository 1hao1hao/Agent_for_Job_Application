# Evaluation 与 Regression Seed 说明

## 当前评测目标

第一版 evaluation（评测）模块只解决两个最基础的问题：

1. 检索是否把正确证据召回来了。
2. router 是否把 query 路由到了正确 intent 和 source。

它不负责生成答案，也不负责人工判断答案质量。当前模块是一个最小 baseline，用来让项目从“能跑”进入“能量化比较”。

## 当前指标

### Recall@k

Recall@k 表示：人工标注的正确 chunk，有多少出现在系统返回的前 k 个检索结果里。

公式：

```text
Recall@k = 前 k 个 retrieved chunk 中命中的 relevant chunk 数量 / relevant chunk 总数
```

例子：

```text
relevant_chunk_ids = [A, B]
retrieved_chunk_ids = [A, C, D]
k = 2
Recall@2 = 1 / 2 = 0.5
```

### Router Accuracy

Router Accuracy 表示：router 预测的 intent 和 routed sources 是否符合人工标注。

第一版采用严格口径：

```text
predicted_intent == expected_intent
并且
set(predicted_sources) == set(expected_sources)
```

两者都满足才算该 query 路由正确。

## 最小评测样例

当前样例文件：

```text
tests/fixtures/evaluation_cases.json
```

它包含：

- `retrieval_cases`: 检索评测样例。
- `router_cases`: 路由评测样例。

这些样例是为了验证评测代码可以运行，不代表真实线上效果。

## Regression Seed

当前失败样例种子文件：

```text
tests/regression/regression_seed.jsonl
```

JSONL 是一行一个 JSON。每条 seed 记录一个需要后续持续关注的失败类型，例如：

- `router_error`
- `retrieval_miss`
- `citation_error`

它的作用不是现在就修复所有问题，而是给后续 regression test set（回归测试集）建立格式。

## 当前限制

- 评测样例数量很少，只能验证代码闭环，不能代表真实效果。
- Recall@k 依赖人工标注的 relevant chunk ids。
- Router Accuracy 当前只看 intent 和 sources，暂不处理多 intent 或部分正确。
- Citation Accuracy、Hallucination Rate、Tool Success Rate 还没有实现。
- 当前评测没有自动运行完整 RAG 流程，只读取 expected/predicted 样例计算指标。

## 下一步计划

- 用真实 query 和人工标注扩充 `tests/fixtures/evaluation_cases.json`。
- 将 trace 中的 retrieved chunks 自动转成 evaluation 输入。
- 增加 Citation Accuracy，检查 citation 是否真的支持回答内容。
- 把 regression seed 转成可执行的 regression tests。
- 在每次检索、router 或 answer 修改后运行同一批评测，观察指标是否退化。

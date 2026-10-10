# Evidence-Oriented RAG：集中 dev 对照与交付结论

## 结论

四层架构已经接入实际 Pipeline，未修改 HTTP 契约、知识库或评测标签。
**候选未达到默认发布条件：不切换服务默认，不运行 frozen test，不更新简历收益数字。**
新结构提升了可解释调度能力，但完整组装的预算和组选择产生明显质量退化。

代码提交：`bc59e74`。运行开始于其父提交的工作树，最终代码与逐文件 SHA-256
见 [implementation_snapshot.json](implementation_snapshot.json)；不是父提交未经修改的实验。

## 在线架构与实质变化

```text
RagRequest -> Router(软路由)
-> EvidencePlan(事实/来源/关系/时效 Slots + 可信 EvidenceScope)
-> Retrieval Orchestrator(共享调用预算、有限并行、RRF、至多一次 CrossEncoder)
-> EvidenceBundle(槽绑定、完整路径组、跨源组合、出处、整组 Token 装箱)
-> EvidenceVerifier(satisfied / missing / conflicting / unknown)
   -> 缺槽: 定向补救最多一次，不扩大权限
   -> 冲突/耗尽: 拒答
-> ContextPolicy/Engine(原分层记忆保留) -> Context 证据组重验
-> Generator/Model Gateway -> Citation Validator + 引用槽重验 -> RagResponse
```

Planner 不再把 Router 的多个候选来源误当必须覆盖的证据来源。调度器每请求最多
6 次逻辑 Retriever 调用、2 路并行；Hybrid 的内部 BM25/Dense 调用不冒充一次底层模型调用。
全请求只补救一次、最多重排一次，保留每个子查询/通道的召回历史；无重复任务。
Assembler 在 1200 个近似 Evidence Token 内整组保留或丢弃，ContextEngine 继续检查
完整 Prompt 的 1800 Token 预算。Verifier 检查实质正文、锚点/语义通道、必要来源、
完整路径、显式冲突、同岗位同版本差异和岗位状态；结构就绪**不等于**最终事实蕴含。

HTTP `user_id` 不是认证。默认只开放未标注私人归属的共享资料；可信 scope provider
才能授权私有材料。用户/租户、来源和 Chunk ACL 在检索输入与输出均检查。图边出处
必须处于授权子库内。原 Runtime、记忆、Checkpoint、Worker 和 Trace 接口保留。
Planner 支持注入最多一次结构化分解并受控回退，本次实验使用确定性规划，没有调用规划 LLM。

## 一次集中检索对照

`evalrag_v0.3/dev`：160 Case，120 可答、40 不可答。相同共享语料授权、同一 Router、
固定 BGE/CrossEncoder revision。所有预测来自真实 Retriever，gold 仅用于事后评分。
Adaptive 行为是旧首轮基线；其余两组包含按 Gate 缺口触发的一次补救。

| 策略 | Recall@5 | MRR | NDCG@5 | 标注路径完整率 | 可答零召回 | P50 / P95 |
|---|---:|---:|---:|---:|---:|---:|
| 旧 Adaptive 首轮 | 51.25% | 51.07% | 48.11% | 0% | 51 | 366.68 / 1098.88 ms |
| Evidence-Gap | 72.92% | 67.57% | 65.98% | 57.5% | 25 | 336.39 / 1076.83 ms |
| Evidence-Oriented | 56.25% | 56.53% | 55.38% | 12.5% | 50 | 820.45 / 1106.60 ms |

新候选相对 Evidence-Gap：Recall 改善 9 条、退化 38 条、持平 73 条。找回历史
25 条零召回中的 8 条，仍有 17 条未恢复，同时产生其他退化，不能只展示恢复数。
补救调用率从 64.38% 降至 53.75%，平均新增逻辑调用从 1.275 降至 0.7625；
新候选总逻辑调用均值 2.675，100/160 次请求调用重排。

本次统一设置 `OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`。不能把这里约 1.1s 的
P95 与上一报告 7.21s 直接归因于架构优化；线程配置、热缓存和运行负载不同。
新策略返回的是预算装箱后证据，旧策略返回扁平 top-k；这是候选实际输出的对照，
不应将其解释为只改变某个检索模型的单变量实验。

## 失败分析与限制

50 条零召回中，41 条在原始检索历史里曾出现标注证据，后续排序/组选择/装箱丢失；
9 条在已调度候选中也没有命中。总计记录 84 次超预算组、28 次不完整组丢弃。
零召回原因细分：9 条原始候选未命中、8 条排序/组选择、22 条不完整组、11 条预算丢弃。
因此主要问题不只是原始召回，而是组装优先级、路径共享成员与预算不匹配。
具体区分和逐 Case 证据见 [failure_analysis.json](failure_analysis.json)。

历史未恢复的 17 条并非全部证明“知识库缺答案”：部分 Query 使用文件名而非正文标题，
实体别名定位与图路径枚举仍不足；候选未命中也可能是排序/调度问题。
图路径 gold 的存在不保证当前实体链接和两条最短路径枚举能找到它；本次不修改标注，
也不降低门禁来制造提升。开放失败存入 [open_regression_cases.jsonl](open_regression_cases.jsonl)，
不计入 fixed 回归通过率；现有 executable fixed 回归仍由完整测试执行。

剩余技术债：跨路径共享 Chunk 的多组成员表达、过大组的预算选取、文件名/别名实体链接、
真正的 Slot 语义蕴含及未明示的冲突检测。当前事实槽判定是结构 readiness，不能写成事实
支持准确率。日期存在/active 元属性不是官网状态自动刷新，也没有新增鉴权系统。

## 真实 LLM 与发布决策

DeepSeek `deepseek-v4-flash` 单次连接 smoke 成功，使用 53 输入 + 5 输出 = 58 Token。
随后开始最多每类 2 条的 paired dev；旧策略完成 4 个请求：1 条召回拒答、3 条
Provider connection error，连续失败达到上限即停止。Pipeline Generation 调用 3 次，
加 smoke 共 4 次；失败调用未返回 Token usage。API 成本 **NOT AVAILABLE**，配置中的
零价格不能当真实零账单。

**新架构真实 LLM paired E2E：NOT TESTED。完整 paired 对照：NOT TESTED。
Frozen test：NOT TESTED。** 不发布 E2E Success、Unexpected Abstention 或引用提升数字。
逐请求 Trace 和断点保留；重复执行命令只继续未完成项，不重复已保存的请求。
服务默认仍为旧配置；不得把新增候选说成上线默认或已证明可靠。

## 验证与复现

```bash
PYTHONPATH=src python -m pytest tests/unit tests/integration tests/regression -q
PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
  python scripts/run_evidence_oriented_evaluation.py
# DeepSeek key 必须已经在环境中；不会自动提交 .env
PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/run_evidence_oriented_evaluation.py --real-e2e --e2e-only
```

完整离线测试：**293 passed、6 skipped**；跳过的是无实际 PostgreSQL/Redis/stack
连接配置的环境测试。新增 13 项测试覆盖软路由不等于必要来源、规划降级、ACL、
完整路径装箱、冲突、未知、一次补救、Context 丢失和引用槽覆盖。
无新增依赖或数据库 migration。`git diff --check` 通过，历史报告未删除。

工件：[summary](summary.json)、[配置/数据 manifest](manifest.json)、
[新策略逐 Case](evidence_oriented_case_results.jsonl)、[真实 E2E 状态](e2e_status.json)、
[真实调用统计](e2e_summary.json)。公开架构见 [架构图](../../../docs/overview/architecture_diagram.md)。

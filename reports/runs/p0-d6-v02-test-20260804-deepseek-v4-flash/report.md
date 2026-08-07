# Semantic Metrics 与 Claim-Level Grounding Audit

## Run

- Run ID：`p0-d6-v02-test-20260804-deepseek-v4-flash`
- Dataset：`evalrag_v0.2` / `test`
- Source predictions：`p0-d5-v02-frozen-test-20260804-deepseek-v4-flash`
- Prediction reused without generation：`True`

## Key-Point Coverage

- Comparable cases：30
- Lexical macro：68.89%
- Semantic macro：74.44%
- Improved / regressed / agreement：5 / 0 / 25
- Unknown cases：0
- Analysis cases：10

## Grounding 与 E2E

- Unsupported Answer Rate：0.00%
- Key-Point Coverage：74.44%
- End-to-End Success：unavailable
- Grounding known / answered：20 / 23
- Independent human review：False

## Judge Cost

- Calls：46，unavailable：3
- P50 / P95：4059.2326279729605 / 8803.108492866158 ms
- Tokens：54168
- Estimated cost：$0.010669

## 边界

本 Run 复用已保存 predictions，只重新审核和计算指标。Judge 不是独立人工标注，
且与 Generator 使用同一模型家族，可能存在自评偏差；逐 point/claim verdict、reason
和 evidence span 已落盘供复查。unknown 不按 supported 处理。

# Semantic Metrics 与 Claim-Level Grounding Audit

## Run

- Run ID：`p0-d6-v02-dev-20260804-deepseek-v4-flash`
- Dataset：`evalrag_v0.2` / `dev`
- Source predictions：`p0-d5-v02-dev-20260804-deepseek-v4-flash`
- Prediction reused without generation：`True`

## Key-Point Coverage

- Comparable cases：60
- Lexical macro：48.89%
- Semantic macro：54.44%
- Improved / regressed / agreement：9 / 1 / 50
- Unknown cases：0
- Analysis cases：10

## Grounding 与 E2E

- Unsupported Answer Rate：3.03%
- Key-Point Coverage：54.44%
- End-to-End Success：unavailable
- Grounding known / answered：33 / 34
- Independent human review：False

## Judge Cost

- Calls：94，unavailable：1
- P50 / P95：3467.8581757470965 / 10848.30211289227 ms
- Tokens：79739
- Estimated cost：$0.015624

## 边界

本 Run 复用已保存 predictions，只重新审核和计算指标。Judge 不是独立人工标注，
且与 Generator 使用同一模型家族，可能存在自评偏差；逐 point/claim verdict、reason
和 evidence span 已落盘供复查。unknown 不按 supported 处理。

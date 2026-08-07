# 固定 Demo

以下三个 Demo 从同一次 `evalrag_v0.2/test` 真实模型 frozen run 导出：

- `single_source.json`：单来源项目解释，最终状态为 `answered`。
- `multi_source.json`：跨岗位、简历和面经的回答，展示多来源 citations。
- `abstention.json`：语料外问题，Evidence Gate 返回 `insufficient_evidence`。

每个 JSON 都包含 `case -> response -> trace -> config -> provenance`。引用保留
`chunk_id/source_type/source_path`，Trace 保留阶段决策、尝试、延迟与 token，但移除
原始 Chunk 正文和请求标识。

重新导出：

```bash
python scripts/export_fixed_demos.py
```

该命令只读取保存工件，不访问网络、不调用 LLM，也不会改变 frozen prediction。

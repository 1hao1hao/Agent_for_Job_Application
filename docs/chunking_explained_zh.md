# Chunking 模块中文说明

## Document 是什么

`Document` 表示切分前的一整份资料。

在当前项目里，它可以是一份岗位 JD、一份简历、一篇面经、一条项目日志，或者一份用户画像。它包含：

- `source_path`: 原始文件路径或导入来源。
- `title`: 文档标题。
- `text`: 完整正文。
- `metadata`: 文档级元数据。

可以把 `Document` 理解成“还没切开的原始证据”。

## Chunk 是什么

`Chunk` 表示 RAG 检索时真正使用的最小文本片段。

它包含：

- `id`: 可复现的 chunk id。
- `source_type`: 数据来源类型，例如 `jd`、`resume`。
- `source_path`: 原始文档路径。
- `title`: 原始文档标题。
- `text`: 当前 chunk 的文本。
- `metadata`: 继承自 Document 的元数据，并追加 chunk 自己的信息。

可以把 `Chunk` 理解成“可以被检索、引用、评测的一小块证据”。

## 为什么需要 chunk

RAG 不能总是把整篇文档直接拿去检索或塞进模型上下文，原因有三个：

1. 文档可能太长，超过模型上下文或检索粒度。
2. 用户问题通常只需要文档中的一小段证据。
3. 评测时需要知道“哪一块证据被召回了”，而不是只知道“哪篇文档被召回了”。

所以 chunking 的目标不是改写原文，而是把长文本切成大小可控、便于检索和引用的证据单元。

## metadata 为什么要继承

`Document.metadata` 保存的是文档级信息，例如公司、岗位、城市、状态、来源 URL、版本等。

切分之后，每个 `Chunk` 仍然必须知道自己来自哪份文档、对应哪个岗位、这个岗位是否还有效。如果 chunk 不继承 metadata，后续会出现几个问题：

- citation 只能指向一段文本，却不知道原始来源。
- freshness filter 无法判断这个 chunk 是否来自过期岗位。
- regression test 很难定位失败样例对应的数据版本。
- 多平台重复 JD 无法比较来源优先级和版本。

因此当前实现中，`build_chunks_from_file` 会先复制完整的 document metadata，再追加：

- `chunk_index`
- `source_file_name`
- `char_count`

## 岗位时效性字段有什么用

岗位 JD 是动态数据，不是静态知识。一个岗位可能会更新、下架、重复发布，或者在不同平台上内容不一致。

当前 metadata 里的时效性字段用于后续检索和评测：

- `job_id`: 岗位唯一标识，方便去重和版本追踪。
- `company`: 公司名。
- `job_title`: 岗位名。
- `city`: 城市。
- `source_platform`: 来源平台，例如 manual、csv、json，未来可以是官网或招聘平台。
- `source_url`: 原始岗位链接。
- `first_seen_at`: 第一次看到这个岗位的时间。
- `last_seen_at`: 最近一次确认这个岗位存在或更新的时间。
- `status`: `active`、`expired` 或 `unknown`。
- `version`: 岗位版本。
- `source_priority`: 来源优先级，后续可用于排序加权。
- `content_hash`: 岗位正文哈希，方便判断内容是否变化。

这些字段不会在 chunking 阶段做复杂判断，只是被保留下来。真正的过滤和排序可以放到后续 `freshness_filter.py`。

## chunk_id 如何生成

当前 chunk id 由这些信息生成：

```text
source_type + source_path + chunk_index + chunk_text
```

代码会把它们拼成字符串，再计算 SHA1，取前 12 位作为短 hash，最后组成：

```text
{source_type}-{file_stem}-{chunk_index}-{hash}
```

这样做的好处是：

- 同一份文本在同一位置切出来，id 稳定。
- 文本变化后，id 会变化。
- regression test 可以用 chunk id 定位召回错误。
- citation 可以引用具体 chunk，而不是只引用整篇文档。

## 当前模块输入输出是什么

### 输入

`chunking.py` 的主要输入是本地 `.md` 或 `.txt` 文件：

```text
data/raw/{source_type}/some_file.md
```

其中 `source_type` 必须是当前支持的类型：

- `jd`
- `resume`
- `interview`
- `project_logs`
- `user_profile`

文件可以带简单 Markdown front matter：

```text
---
company: Demo Search
job_title: Backend Engineer Intern
status: active
version: 1
---

# Backend Engineer Intern JD
...
```

没有 front matter 也可以加载，缺失字段会使用默认值。

### 输出

主要输出是：

```text
list[Chunk]
```

每个 Chunk 都包含文本、来源信息和继承后的 metadata。

## 建议重点阅读的函数

1. `build_document_from_file`

   负责把文件读成 `Document`，并处理 front matter metadata。

2. `split_text`

   负责把长文本按段落和字符数切成多个片段。

3. `build_chunks_from_file`

   负责把 `Document` 转成 `Chunk` 列表，是 metadata 继承发生的地方。

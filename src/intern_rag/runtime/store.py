from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from intern_rag.runtime.schemas import SpanEvent, StageCheckpoint


class SpanSink(Protocol):
    """Runtime span 输出接口；sink 故障不能覆盖业务结果。"""

    def write(self, event: SpanEvent) -> None: ...


class JsonlSpanSink:
    """把脱敏 span 追加到 JSONL。"""

    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, event: SpanEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")


class CheckpointStore(Protocol):
    def save(self, checkpoint: StageCheckpoint) -> None: ...
    def latest(self, run_id: str) -> StageCheckpoint | None: ...


class FileCheckpointStore:
    """按 run id 保存最近 checkpoint，原子替换避免半写文件。"""

    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, checkpoint: StageCheckpoint) -> None:
        path = self.root / f"{checkpoint.run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(checkpoint.__dict__, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def latest(self, run_id: str) -> StageCheckpoint | None:
        path = self.root / f"{run_id}.json"
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        value["completed_side_effect_keys"] = tuple(value["completed_side_effect_keys"])
        return StageCheckpoint(**value)

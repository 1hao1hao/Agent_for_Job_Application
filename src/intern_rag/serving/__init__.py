"""EvalRAG 的薄 FastAPI 服务层。"""

from intern_rag.serving.api import AppServices, create_app

__all__ = ["AppServices", "create_app"]

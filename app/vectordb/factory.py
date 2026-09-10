"""Vector store factory — điểm duy nhất chọn backend.

Store được tạo một lần lúc startup (xem ``main.py`` lifespan) và chia sẻ cho mọi
request. Điều này BẮT BUỘC với Qdrant embedded: mỗi ``QdrantClient(path=...)``
sẽ cố lấy file lock, nên tạo store per-request là chắc chắn lỗi.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..core.config import vectordb_settings
from .base import VectorStore

_log = logging.getLogger(__name__)

_store: Optional[VectorStore] = None


def build_store() -> VectorStore:
    """Tạo vector store theo ``RAG_BACKEND``."""
    if vectordb_settings.BACKEND == "chroma":
        from .chromadb import ChromaStore

        return ChromaStore()

    from .qdrant import QdrantStore

    return QdrantStore()


def init_store() -> VectorStore:
    """Khởi tạo store dùng chung. Gọi một lần từ lifespan."""
    global _store
    if _store is None:
        _store = build_store()
        _log.info("Vector store initialized (backend=%s)", vectordb_settings.BACKEND)
    return _store


def get_store() -> VectorStore:
    """Lấy store dùng chung. Dùng làm FastAPI dependency."""
    if _store is None:
        raise RuntimeError(
            "Vector store chưa được khởi tạo. init_store() phải chạy trong lifespan."
        )
    return _store


def reset_store() -> None:
    """Xoá store dùng chung — chỉ dùng cho test."""
    global _store
    _store = None

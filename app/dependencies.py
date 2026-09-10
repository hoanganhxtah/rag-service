"""FastAPI dependencies shared by the REST controllers."""

from fastapi import Request

from .services.bm25_service import BM25Index
from .vectordb.qdrant import QdrantStore


def get_store(request: Request) -> QdrantStore:
    """Return the Qdrant instance created once during application startup."""
    try:
        return request.app.state.store
    except AttributeError as exc:
        raise RuntimeError("Qdrant chưa được khởi tạo.") from exc


def get_bm25(request: Request) -> BM25Index:
    try:
        return request.app.state.bm25
    except AttributeError as exc:
        raise RuntimeError("BM25 chưa được khởi tạo.") from exc

"""Stats and readiness endpoints.

``/stats`` expose ``embedding_model`` + ``embedding_dim`` có mục đích: dùng model
khác lúc query so với lúc ingest sẽ làm retrieval sai mà KHÔNG raise lỗi nào.
Endpoint này là cách phát hiện sớm điều đó từ bên ngoài.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ..core.config import vectordb_settings
from ..dependencies import get_bm25, get_store
from ..models.response_schema import ReadyResponse, StatsResponse
from ..services.bm25_service import BM25Index
from ..vectordb.qdrant import QdrantStore

_log = logging.getLogger(__name__)

router = APIRouter(tags=["Diagnostics"])


@router.get(
    "/stats",
    response_model=StatsResponse,
    operation_id="get_knowledge_base_stats",
    summary="Knowledge base statistics",
)
def stats_endpoint(
    store: QdrantStore = Depends(get_store),
    bm25: BM25Index = Depends(get_bm25),
) -> StatsResponse:
    """Trả về thông tin collection: backend, số document, embedding model."""
    return StatsResponse(
        backend=store.backend,
        collection=store.collection_name,
        doc_count=store.count(),
        location=store.location,
        embedding_model=store.embedding_model,
        embedding_dim=store.embedding_dim,
        distance=store.distance,
        bm25_doc_count=bm25.count,
        num_retrieval=vectordb_settings.NUM_RETRIEVAL,
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    operation_id="get_readiness",
    summary="Readiness probe",
)
def ready_endpoint(store: QdrantStore = Depends(get_store)) -> ReadyResponse:
    """Kiểm tra vector store có thực sự truy cập được.

    Khác ``/health`` (chỉ báo process còn sống), endpoint này chạm vào store nên
    phát hiện được các lỗi như Qdrant embedded bị process khác giữ lock, hoặc
    embedding model không load được. Luôn trả HTTP 200 kèm ``ready`` boolean để
    client phân biệt được "service chết" với "service sống nhưng chưa sẵn sàng".
    """
    try:
        return ReadyResponse(
            ready=True,
            backend=store.backend,
            doc_count=store.count(),
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("readiness check thất bại: %s", exc)
        return ReadyResponse(
            ready=False,
            backend="qdrant",
            doc_count=0,
            detail=str(exc),
        )

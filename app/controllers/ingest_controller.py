"""Ingestion endpoint.

Endpoint này GHI vào vector store nên rủi ro cao hơn ``/retrieve``. Nó chưa có
authentication — xem phần Bảo mật trong README trước khi expose ra ngoài
localhost.

Ingest chạy đồng bộ. Với dataset lớn (docling convert khá chậm) request có thể
timeout ở phía client dù server vẫn đang làm việc. Chuyển sang job dạng
background + ``GET /ingest/{job_id}`` là bước nâng cấp hợp lý khi cần.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..models.request_schema import IngestRequest
from ..models.response_schema import IngestResponse
from ..services import ingestion_service
from ..dependencies import get_bm25, get_store
from ..services.bm25_service import BM25Index
from ..vectordb.qdrant import QdrantStore

_log = logging.getLogger(__name__)

router = APIRouter(tags=["Ingestion"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    operation_id="ingest_documents",
    summary="Ingest documents into the knowledge base",
)
def ingest_endpoint(
    request: IngestRequest,
    store: QdrantStore = Depends(get_store),
    bm25: BM25Index = Depends(get_bm25),
) -> IngestResponse:
    """Convert files trong thư mục thành chunk và nạp vào vector store.

    Chạy đồng bộ — có thể mất vài phút với thư mục nhiều file.
    """
    try:
        result = ingestion_service.ingest(
            store,
            folder=request.folder,
            skip_if_not_empty=request.skip_if_not_empty,
        )
        if result.chunks_added:
            bm25.rebuild(store.all_documents())
        return result
    except Exception as exc:  # noqa: BLE001
        _log.exception("ingest thất bại | folder=%r", request.folder)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

"""Retrieval endpoint.

``operation_id`` được đặt tay để tên MCP tool ổn định. Mặc định FastAPI sinh
operation id từ tên function + path + method, cho ra thứ như
``retrieve_api_v1_retrieve_post`` — và nó ĐỔI mỗi khi path đổi. Tên MCP tool
nằm trong prompt của agent, nên nó phải là thứ ta kiểm soát.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..models.request_schema import RetrieveRequest
from ..models.response_schema import RetrieveResponse
from ..services import retrieval_service
from ..dependencies import get_bm25, get_store
from ..vectordb.qdrant import QdrantStore
from ..services.bm25_service import BM25Index

_log = logging.getLogger(__name__)

router = APIRouter(tags=["Retrieval"])


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    operation_id="search_knowledge_base",
    summary="Search the internal knowledge base",
)
def retrieve_endpoint(
    request: RetrieveRequest,
    store: QdrantStore = Depends(get_store),
    bm25: BM25Index = Depends(get_bm25),
) -> RetrieveResponse:
    """Search the internal knowledge base (company documents, policies,
    products, internal procedures) and return the most relevant document chunks
    with relevance scores."""
    try:
        return retrieval_service.retrieve(
            store,
            bm25,
            query=request.query,
            top_k=request.top_k,
        )
    except Exception as exc:  # noqa: BLE001
        _log.exception("retrieve thất bại | query=%r", request.query)
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}")

"""Response schemas — hợp đồng công khai của RAG service.

Đây là artifact bàn giao. Người phát triển ``rag_service`` được tự do đổi
embedding model, chunking strategy, thêm reranker — miễn các shape dưới
đây giữ nguyên.

Ghi chú thiết kế: ``results`` là list PHẲNG, không phải shape lồng
``[[...]]`` như API legacy. Shape lồng đó là chi tiết nội bộ bị rò ra API và
không mang thêm thông tin gì.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class RetrievedDocument(BaseModel):
    """Một chunk tài liệu khớp với query."""

    content: str = Field(..., description="Nội dung text của chunk")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata của chunk: source, file_name, chunk_index, id",
    )
    score: float = Field(
        ...,
        description="RRF score dùng để xếp hạng kết quả hybrid.",
    )


class RetrieveResponse(BaseModel):
    """Kết quả của ``POST /retrieve``."""

    query: str = Field(..., description="Query đã dùng để tìm")
    results: list[RetrievedDocument] = Field(
        ..., description="Các document giảm dần theo RRF score"
    )
    total: int = Field(..., description="Số document trả về")
    raw_count: int = Field(
        ...,
        description=(
            "Số candidate duy nhất sau khi gộp Dense và BM25."
        ),
    )
    best_score: Optional[float] = Field(
        None,
        description="RRF score cao nhất.",
    )
    took_ms: float = Field(..., description="Thời gian xử lý (ms)")


class IngestResponse(BaseModel):
    """Kết quả của ``POST /ingest``."""

    status: str = Field(..., description="'success' hoặc 'no_documents'")
    source: str = Field(..., description="Thư mục hoặc file đã ingest")
    files_processed: int = Field(..., description="Số file đã convert thành công")
    chunks_added: int = Field(..., description="Số chunk đã thêm vào vector store")
    doc_count: int = Field(..., description="Tổng số document trong collection sau khi ingest")
    took_ms: float = Field(..., description="Thời gian xử lý (ms)")


class StatsResponse(BaseModel):
    """Kết quả của ``GET /stats``.

    ``embedding_model`` được expose có mục đích: model lúc ingest khác model lúc
    query sẽ làm retrieval sai mà KHÔNG raise lỗi nào. Đây là cách quan sát điều
    đó từ bên ngoài.
    """

    backend: str = Field(..., description="Luôn là 'qdrant'")
    collection: str = Field(..., description="Tên collection")
    doc_count: int = Field(..., description="Số document trong collection")
    location: str = Field(..., description="Nơi store sống (server url hoặc local path)")
    embedding_model: str = Field(..., description="Embedding model đang dùng")
    embedding_dim: Optional[int] = Field(None, description="Số chiều vector")
    distance: Optional[str] = Field(None, description="Distance metric")
    bm25_doc_count: int = Field(..., description="Số chunk trong BM25 index")
    num_retrieval: int = Field(..., description="top_k mặc định")


class HealthResponse(BaseModel):
    """Kết quả của ``GET /health``."""

    status: str = Field(..., description="'healthy' hoặc 'degraded'")
    service: str = Field(..., description="Tên service")


class ReadyResponse(BaseModel):
    """Kết quả của ``GET /ready``.

    Khác ``/health``: endpoint này thực sự chạm vào vector store. Dùng nó để
    biết service đã sẵn sàng phục vụ traffic, không chỉ là process còn sống.
    """

    ready: bool = Field(..., description="Vector store có truy cập được không")
    backend: str = Field(..., description="Backend đang dùng")
    doc_count: int = Field(..., description="Số document truy cập được")
    detail: Optional[str] = Field(None, description="Lý do khi ready=false")

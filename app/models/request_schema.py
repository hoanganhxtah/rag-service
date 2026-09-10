"""Request schemas for the RAG service."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    """Input của ``POST /retrieve``."""

    query: str = Field(
        ...,
        min_length=1,
        description="Câu truy vấn cần tìm trong knowledge base",
    )
    top_k: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="Số document lấy ra. null → dùng mặc định của service.",
    )


class IngestRequest(BaseModel):
    """Input của ``POST /ingest``.

    Để trống ``folder`` thì dùng ``RAG_INGEST_DATA_FOLDER``.
    """

    folder: Optional[str] = Field(
        None,
        description="Thư mục cần ingest. null → dùng RAG_INGEST_DATA_FOLDER.",
    )
    skip_if_not_empty: bool = Field(
        True,
        description=(
            "Bỏ qua nếu collection đã có document. Mặc định true để tránh "
            "ingest trùng do gọi lặp — hiện chưa có cơ chế upsert theo doc id."
        ),
    )

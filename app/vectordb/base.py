"""Vector store contract for the RAG service.

Trước khi tách, ``qdrant.py`` và ``chromadb.py`` chỉ "giống nhau vì tình cờ":
cùng có ``query`` / ``count`` / ``add_documents`` nhưng không có contract nào
viết ra, và mỗi bên trả shape hơi khác. Module này viết contract đó ra thành
văn để người tiếp nhận service biết chính xác phải implement gì khi thêm backend
mới (pgvector, Milvus, ...).

Điểm khác biệt so với code cũ: ``search()`` trả ``list[ScoredDocument]`` phẳng,
không phải dict shape lồng ``{"documents": [[...]]}``. Shape lồng đó là chi tiết
nội bộ của Chroma bị rò ra khắp nơi, và nó không mang thêm thông tin gì.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from langchain_core.documents import Document


@dataclass(frozen=True)
class ScoredDocument:
    """Một document kèm relevance score.

    Attributes
    ----------
    content:
        Nội dung chunk.
    metadata:
        Metadata của chunk.
    score:
        Relevance 0..1 (cosine). ``None`` nếu backend không cung cấp score.
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None


@dataclass(frozen=True)
class CollectionInfo:
    """Thông tin chẩn đoán về collection đang phục vụ.

    ``embedding_model`` và ``embedding_dim`` được đưa ra ngoài có mục đích: đây
    là cách duy nhất để phát hiện từ bên ngoài rằng model lúc query đã lệch so
    với lúc ingest.
    """

    backend: str
    collection: str
    doc_count: int
    location: str
    embedding_model: str
    embedding_dim: Optional[int] = None
    distance: Optional[str] = None


@runtime_checkable
class VectorStore(Protocol):
    """Contract mà mọi vector-store backend phải thoả."""

    def search(self, query: str, top_k: int) -> list[ScoredDocument]:
        """Trả về tối đa ``top_k`` document liên quan nhất, giảm dần theo score.

        Không áp threshold — lọc là việc của tầng service, để endpoint debug vẫn
        xem được hit dưới ngưỡng.
        """
        ...

    def add_documents(self, documents: list[Document]) -> int:
        """Thêm documents vào store. Trả về số document đã thêm."""
        ...

    def count(self) -> int:
        """Số document trong collection. Trả 0 nếu collection chưa tồn tại."""
        ...

    def is_empty(self) -> bool:
        """Collection có rỗng không."""
        ...

    def info(self) -> CollectionInfo:
        """Thông tin chẩn đoán, phục vụ ``GET /stats``."""
        ...

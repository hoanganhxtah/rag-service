"""Qdrant backend — hỗ trợ cả embedded (local path) và server mode.

Chuyển sang mode nào chỉ phụ thuộc ``RAG_QDRANT_URL``: có giá trị thì kết nối
server, để rỗng thì mở thư mục local. Không có flag thứ hai nào để lệch pha.
Xem ``app.core.config.QdrantLocation``.

Lưu ý vận hành với local mode: Qdrant embedded giữ file lock trên thư mục, nên
CHỈ MỘT process mở được nó cùng lúc. Chạy ``rag_service`` song song với một
process khác đang mở cùng path sẽ lỗi. Đây là lý do chính để chuyển sang server
mode, và code đã sẵn sàng cho việc đó — chỉ cần set env.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.documents import Document

from ..core.config import resolve_path, vectordb_settings
from ..embeddings.embed import build_embeddings
_log = logging.getLogger(__name__)


class QdrantStore:
    """Qdrant store duy nhất của RAG service."""

    backend = "qdrant"

    def __init__(self, collection_name: Optional[str] = None):
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.collection_name = collection_name or vectordb_settings.COLLECTION_NAME
        self.embeddings = build_embeddings()
        self.embedding_model = getattr(self.embeddings, "model_name", "unknown")
        expected_dim = getattr(self.embeddings, "embedding_dim", None)
        self.distance = Distance.COSINE.value

        location = vectordb_settings.qdrant_location()
        self.location = location.describe()

        if location.is_server:
            self.client = QdrantClient(url=location.url)
            _log.info("Qdrant: connected to server at %s", location.url)
        else:
            # Path phải tuyệt đối — chạy service từ cwd khác sẽ trỏ vào một
            # thư mục rỗng và trông như mất hết dữ liệu.
            local_path = resolve_path(location.path)
            local_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(local_path))
            self.location = f"local path={local_path}"
            _log.info("Qdrant: using embedded store at %s", local_path)

        self.embedding_dim: Optional[int] = None
        if self.client.collection_exists(self.collection_name):
            info = self.client.get_collection(self.collection_name)
            self.embedding_dim = self._read_dim(info)
            if (
                expected_dim is not None
                and self.embedding_dim is not None
                and expected_dim != self.embedding_dim
            ):
                raise RuntimeError(
                    "Embedding dimension không khớp collection hiện có: "
                    f"model={expected_dim}, collection={self.embedding_dim}, "
                    f"collection_name={self.collection_name!r}. "
                    "Hãy dùng collection mới và ingest lại."
                )
            _log.info("Qdrant collection '%s' loaded.", self.collection_name)
        else:
            # Chưa có collection — tạo mới, lấy dim từ một lần embed thử.
            if expected_dim is not None:
                self.embedding_dim = expected_dim
            else:
                sample_vec = self.embeddings.embed_query("test")
                self.embedding_dim = len(sample_vec)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            _log.info(
                "Qdrant: created collection '%s' | dim=%d",
                self.collection_name,
                self.embedding_dim,
            )

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        _log.info(
            "Qdrant ready | collection=%s | docs=%d | %s",
            self.collection_name,
            self.count(),
            self.location,
        )

    @staticmethod
    def _read_dim(collection_info) -> Optional[int]:
        """Đọc vector dimension từ collection info.

        Cấu trúc lồng khá sâu và khác nhau giữa các version qdrant-client, nên
        bọc try/except: đây chỉ là thông tin chẩn đoán, không đáng để làm sập
        service.
        """
        try:
            params = collection_info.config.params.vectors
            return getattr(params, "size", None)
        except Exception:  # noqa: BLE001
            return None

    def search(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        return self.vector_store.similarity_search_with_relevance_scores(
            query, k=top_k
        )

    def all_documents(self) -> list[Document]:
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=max(self.count(), 1),
            with_payload=True,
            with_vectors=False,
        )
        return [
            Document(
                page_content=str((point.payload or {}).get("page_content", "")),
                metadata=(point.payload or {}).get("metadata") or {},
            )
            for point in points
        ]

    def add_documents(self, documents: list[Document]) -> int:
        if not documents:
            _log.warning("add_documents: danh sách rỗng — không thêm gì.")
            return 0
        self.vector_store.add_documents(documents)
        _log.info(
            "Qdrant: đã thêm %d docs | tổng=%d | collection=%s",
            len(documents),
            self.count(),
            self.collection_name,
        )
        return len(documents)

    def count(self) -> int:
        try:
            return self.client.get_collection(self.collection_name).points_count
        except Exception:  # noqa: BLE001
            return 0

    def is_empty(self) -> bool:
        return self.count() == 0

    def close(self) -> None:
        """Release the embedded-Qdrant file lock on application shutdown."""
        self.client.close()

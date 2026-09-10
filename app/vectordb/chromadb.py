"""Chroma backend.

Chuyển từ ``app/vectordb/chromadb.py`` với hai sửa đổi:

1. Embedding model lấy từ ``build_embeddings()`` thay vì hardcode
   ``GeminiEmbeddings()``. Bản cũ khiến việc đổi backend cũng âm thầm đổi luôn
   embedding model — hai quyết định không liên quan bị dính vào nhau.
2. Bỏ phần dò ``torch.cuda`` không dùng tới. Việc chọn device thuộc về
   embedding layer, không phải vector store.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.documents import Document

from ..core.config import resolve_path, vectordb_settings
from ..embeddings.embed import build_embeddings
from .base import CollectionInfo, ScoredDocument

_log = logging.getLogger(__name__)


class ChromaStore:
    """``VectorStore`` implementation cho Chroma."""

    def __init__(self, collection_name: Optional[str] = None, path: Optional[str] = None):
        from langchain_chroma import Chroma

        self.collection_name = collection_name or vectordb_settings.COLLECTION_NAME
        self.embeddings = build_embeddings()
        self._embedding_model = getattr(self.embeddings, "model_name", "unknown")

        persist_dir = resolve_path(path or vectordb_settings.CHROMA_PATH)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._location_desc = f"local path={persist_dir}"

        self.collection = Chroma(
            collection_name=self.collection_name,
            persist_directory=str(persist_dir),
            embedding_function=self.embeddings,
        )
        _log.info(
            "Chroma ready | collection=%s | docs=%d | %s",
            self.collection_name,
            self.count(),
            self._location_desc,
        )

    def search(self, query: str, top_k: int) -> list[ScoredDocument]:
        results = self.collection.similarity_search_with_relevance_scores(
            query, k=top_k
        )
        return [
            ScoredDocument(
                content=doc.page_content,
                metadata=doc.metadata or {},
                score=float(score),
            )
            for doc, score in results
        ]

    def add_documents(self, documents: list[Document]) -> int:
        if not documents:
            _log.warning("add_documents: danh sách rỗng — không thêm gì.")
            return 0
        self.collection.add_documents(documents=documents)
        _log.info("Chroma: đã thêm %d docs | tổng=%d", len(documents), self.count())
        return len(documents)

    def count(self) -> int:
        try:
            return self.collection._collection.count()
        except Exception:  # noqa: BLE001
            return 0

    def is_empty(self) -> bool:
        return self.count() == 0

    def info(self) -> CollectionInfo:
        return CollectionInfo(
            backend="chroma",
            collection=self.collection_name,
            doc_count=self.count(),
            location=self._location_desc,
            embedding_model=self._embedding_model,
            embedding_dim=None,
            distance="cosine",
        )

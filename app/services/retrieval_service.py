"""Dense + BM25 retrieval fused with Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
import time
from typing import Optional

from ..core.config import vectordb_settings
from ..models.response_schema import RetrievedDocument, RetrieveResponse
from ..vectordb.qdrant import QdrantStore
from .bm25_service import BM25Index

_log = logging.getLogger(__name__)

CANDIDATES = 20
RRF_K = 60


def retrieve(
    store: QdrantStore,
    bm25: BM25Index,
    query: str,
    top_k: Optional[int] = None,
) -> RetrieveResponse:
    final_k = top_k or vectordb_settings.NUM_RETRIEVAL
    candidate_k = max(CANDIDATES, final_k)

    start = time.perf_counter()
    dense_hits = store.search(query, candidate_k)
    bm25_hits = bm25.search(query, candidate_k)

    documents = {}
    scores: dict[str, float] = {}
    for hits in (dense_hits, bm25_hits):
        for rank, (document, _) in enumerate(hits, start=1):
            key = str(document.metadata.get("id") or document.page_content)
            documents[key] = document
            scores[key] = scores.get(key, 0.0) + 1 / (RRF_K + rank)

    ranked = sorted(scores, key=scores.get, reverse=True)
    selected = ranked[:final_k]
    took_ms = round((time.perf_counter() - start) * 1000, 2)

    _log.info(
        "hybrid retrieve | query=%r | dense=%d | bm25=%d | fused=%d | %.1fms",
        query,
        len(dense_hits),
        len(bm25_hits),
        len(selected),
        took_ms,
    )

    return RetrieveResponse(
        query=query,
        results=[
            RetrievedDocument(
                content=documents[key].page_content,
                metadata=documents[key].metadata or {},
                score=scores[key],
            )
            for key in selected
        ],
        total=len(selected),
        raw_count=len(ranked),
        best_score=scores[selected[0]] if selected else None,
        took_ms=took_ms,
    )

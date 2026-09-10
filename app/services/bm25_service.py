"""Small in-memory BM25 index for Vietnamese and English text."""

from __future__ import annotations

import logging
import re
import unicodedata

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

_log = logging.getLogger(__name__)
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    words = _WORD_RE.findall(unicodedata.normalize("NFKC", text).casefold())
    bigrams = [f"{left}_{right}" for left, right in zip(words, words[1:])]
    return words + bigrams


class BM25Index:
    def __init__(self) -> None:
        self._state: tuple[list[Document], BM25Okapi | None] = ([], None)

    @property
    def count(self) -> int:
        return len(self._state[0])

    def rebuild(self, documents: list[Document]) -> None:
        documents = [doc for doc in documents if doc.page_content.strip()]
        corpus = [tokenize(doc.page_content) for doc in documents]
        self._state = (documents, BM25Okapi(corpus) if corpus else None)
        _log.info("BM25 index ready | docs=%d", len(documents))

    def search(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        documents, index = self._state
        if index is None:
            return []

        scores = index.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [
            (documents[idx], float(score))
            for idx, score in ranked
            if score > 0
        ][:top_k]

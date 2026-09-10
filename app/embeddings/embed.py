"""Embedding models for the RAG service.

Chuyển từ ``app/llm/embeddings/embed.py`` với một thay đổi hành vi có chủ đích:
``LocalEmbeddings`` KHÔNG còn âm thầm tải model từ HuggingFace khi path local
không tồn tại. Xem ``RAG_EMBEDDING_ALLOW_HF_DOWNLOAD``.

Lý do: nếu ingest bằng ``data/models/bge-m3`` rồi query
bằng một model tải từ HF, hai bên nằm ở vector space khác nhau. Không có exception
nào được raise — chỉ là kết quả retrieval tệ đi một cách khó hiểu. Fail sớm và
rõ ràng thì tốt hơn nhiều so với sai âm thầm.
"""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings

from ..core.config import (
    api_key_settings,
    embedding_settings,
    resolve_path,
)

_log = logging.getLogger(__name__)


class LocalEmbeddings(Embeddings):
    """SentenceTransformer chạy local, bọc theo interface LangChain."""

    def __init__(self, model_name: str | None = None, allow_hf_download: bool | None = None):
        model_name = model_name or embedding_settings.MODEL_NAME
        if allow_hf_download is None:
            allow_hf_download = embedding_settings.ALLOW_HF_DOWNLOAD

        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        resolved = resolve_path(model_name)

        if resolved.exists():
            source = str(resolved)
            _log.info("LocalEmbeddings: loading from local path '%s'", resolved)
        else:
            if not allow_hf_download:
                raise FileNotFoundError(
                    f"Embedding model not found locally: {resolved}\n"
                    f"  (RAG_EMBEDDING_MODEL_NAME={model_name!r})\n"
                    f"Tải model về path đó, hoặc set "
                    f"RAG_EMBEDDING_ALLOW_HF_DOWNLOAD=true để cho phép tải từ "
                    f"HuggingFace. Lưu ý: dùng model khác lúc query so với lúc "
                    f"ingest sẽ làm retrieval sai mà không báo lỗi."
                )

            source = model_name.replace("\\", "/")
            self.model_name = source
            _log.warning(
                "LocalEmbeddings: '%s' không có local. Tải '%s' từ HuggingFace "
                "(ALLOW_HF_DOWNLOAD=true). Đảm bảo model này KHỚP với model đã dùng "
                "lúc ingest.",
                model_name,
                source,
            )

        try:
            self.model = SentenceTransformer(
                source,
                device=embedding_settings.DEVICE,
            )
        except Exception as model_err:
            raise OSError(
                f"Không load được embedding model {source!r}: {model_err}"
            ) from model_err

        model_limit = int(self.model.max_seq_length)
        requested_limit = embedding_settings.MAX_SEQ_LENGTH
        if requested_limit > model_limit:
            raise ValueError(
                "RAG_EMBEDDING_MAX_SEQ_LENGTH vượt giới hạn của model: "
                f"requested={requested_limit}, model_limit={model_limit}, "
                f"model={self.model_name!r}."
            )
        self.model.max_seq_length = requested_limit
        self.embedding_dim = int(self.model.get_sentence_embedding_dimension())
        self.batch_size = embedding_settings.BATCH_SIZE
        _log.info(
            "LocalEmbeddings ready | model=%s | device=%s | dim=%d | "
            "max_tokens=%d | batch=%d",
            self.model_name,
            embedding_settings.DEVICE,
            self.embedding_dim,
            requested_limit,
            self.batch_size,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0].tolist()


class GeminiEmbeddings(Embeddings):
    """Google Generative AI embeddings."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        output_dimensionality: int | None = None,
    ):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        api_key = api_key or api_key_settings.GEMINI_API_KEY
        model = model or embedding_settings.MODEL_NAME
        if output_dimensionality is None:
            output_dimensionality = embedding_settings.OUTPUT_DIMENSION

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY chưa được set nhưng "
                "RAG_EMBEDDING_PROVIDER=gemini."
            )

        self.model_name = model
        self.embedding_dim = output_dimensionality
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=api_key,
            output_dimensionality=output_dimensionality,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)


def build_embeddings() -> Embeddings:
    """Tạo embedding model theo ``RAG_EMBEDDING_PROVIDER``.

    Đây là điểm duy nhất chọn embedding provider. Vector store luôn dùng cùng
    instance trả về từ đây, nên ingest và query không thể vô tình lệch model.
    """
    if embedding_settings.PROVIDER == "gemini":
        return GeminiEmbeddings()
    return LocalEmbeddings()

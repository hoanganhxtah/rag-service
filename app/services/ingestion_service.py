"""Document ingestion — convert files thành chunk rồi nạp vào vector store.

Chuyển từ ``app/vectordb/loader.py``. Phần loader CSV/PDF legacy không được mang
sang: nó phục vụ dataset movies-metadata cũ, không còn dùng trong luồng hiện tại.

Định tuyến converter theo extension:
- ``.pdf``  → pypdf (thuần text, không ML/OCR, không nguy cơ OOM)
- còn lại   → Docling (convert sang Markdown)
"""

from __future__ import annotations

import hashlib
import logging
import time
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

from ..core.config import (
    SERVICE_ROOT,
    embedding_settings,
    ingestion_settings,
    resolve_path,
)
from ..models.response_schema import IngestResponse
from ..vectordb.qdrant import QdrantStore

_log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt",
    ".xlsx", ".xls", ".html", ".htm", ".md",
    ".txt", ".rst", ".asciidoc", ".adoc",
}


@lru_cache(maxsize=1)
def _embedding_tokenizer():
    """Load đúng tokenizer của local embedding model, không load model weights."""
    if embedding_settings.PROVIDER != "local":
        raise RuntimeError(
            "Token-aware ingestion hiện yêu cầu RAG_EMBEDDING_PROVIDER=local."
        )

    from transformers import AutoTokenizer

    model_path = resolve_path(embedding_settings.MODEL_NAME)
    if model_path.exists():
        source = str(model_path)
        local_files_only = True
    elif embedding_settings.ALLOW_HF_DOWNLOAD:
        source = embedding_settings.MODEL_NAME.replace("\\", "/")
        local_files_only = False
    else:
        raise FileNotFoundError(
            f"Không tìm thấy tokenizer local tại {model_path}. "
            "Hãy tải model trước khi ingest."
        )

    return AutoTokenizer.from_pretrained(
        source,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )


def _text_splitter():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if ingestion_settings.CHUNK_OVERLAP >= ingestion_settings.CHUNK_SIZE:
        raise ValueError("RAG_INGEST_CHUNK_OVERLAP phải nhỏ hơn CHUNK_SIZE.")
    if ingestion_settings.CHUNK_SIZE > embedding_settings.MAX_SEQ_LENGTH:
        raise ValueError(
            "RAG_INGEST_CHUNK_SIZE không được vượt "
            "RAG_EMBEDDING_MAX_SEQ_LENGTH."
        )

    tokenizer = _embedding_tokenizer()
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=ingestion_settings.CHUNK_SIZE,
        chunk_overlap=ingestion_settings.CHUNK_OVERLAP,
    )
    return splitter, tokenizer


def _relative_source(file_path: Path) -> str:
    """Đường dẫn tương đối so với RAG service root, dùng dấu ``/``.

    Metadata KHÔNG được chứa absolute path. Hai lý do:

    1. ``_stable_id`` băm giá trị này. Băm absolute path nghĩa là cùng một file
       ingest ở máy khác (hoặc trong container) sẽ ra id khác, làm mọi cơ chế
       dedupe/re-index về sau vô dụng.
    2. Absolute path lộ layout thư mục của máy chạy vào metadata, và metadata
       này được truyền tới LLM.

    Dùng ``/`` cố định để id không đổi giữa Windows và Linux.
    """
    try:
        rel = file_path.resolve().relative_to(SERVICE_ROOT)
    except ValueError:
        # File nằm ngoài service root (data folder trỏ ra ổ khác) — đành giữ
        # nguyên, nhưng vẫn chuẩn hoá separator.
        rel = file_path
    return rel.as_posix()


def _stable_id(source: str, chunk_index: int) -> str:
    """Id tiền định từ source path + chunk index.

    ``source`` phải là đường dẫn TƯƠNG ĐỐI (xem ``_relative_source``) để id ổn
    định giữa các máy. Cho phép sau này làm upsert/dedupe: ingest lại cùng file
    sẽ sinh đúng bộ id cũ. Hiện ``add_documents`` chưa dùng id này để dedupe —
    đó là lý do ``/ingest`` mặc định ``skip_if_not_empty=true``.
    """
    return hashlib.md5(f"{source}::{chunk_index}".encode()).hexdigest()


def _convert_pdf(file_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    _log.info("pypdf: '%s' → %d chars", file_path.name, len(text))
    return text


def _convert_with_docling(file_path: Path) -> str:
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        markdown = result.document.export_to_markdown()
        _log.info("docling: '%s' → %d chars", file_path.name, len(markdown))
        return markdown
    except Exception as exc:  # noqa: BLE001
        # Một file lỗi không nên làm sập cả lần ingest — log rồi bỏ qua file đó.
        _log.error("docling thất bại với '%s': %s", file_path.name, exc)
        return ""


def _convert(file_path: Path) -> str:
    if file_path.suffix.lower() == ".pdf":
        return _convert_pdf(file_path)
    return _convert_with_docling(file_path)


def chunk_from_folder(folder: str | None = None) -> tuple[list[Document], int]:
    """Scan folder, convert file, chia chunk.

    Returns
    -------
    (chunks, files_processed)
        ``files_processed`` chỉ đếm file convert ra nội dung không rỗng.
    """
    raw_folder = folder or ingestion_settings.DATA_FOLDER
    folder_path = resolve_path(raw_folder)

    if not folder_path.is_dir():
        _log.error("'%s' không phải thư mục hợp lệ.", folder_path)
        return [], 0

    files = [
        p for p in folder_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        _log.warning("Không tìm thấy file được hỗ trợ trong '%s'.", folder_path)
        return [], 0

    _log.info("Tìm thấy %d file trong '%s'. Bắt đầu convert…", len(files), folder_path)

    splitter, tokenizer = _text_splitter()
    all_chunks: list[Document] = []
    processed = 0

    for file_path in files:
        content = _convert(file_path)
        if not content.strip():
            _log.warning("Bỏ qua '%s' — nội dung rỗng sau convert.", file_path.name)
            continue

        source = _relative_source(file_path)
        chunks = splitter.create_documents(
            texts=[content],
            metadatas=[{
                "source": source,
                "file_name": file_path.name,
                "extension": file_path.suffix.lower(),
            }],
        )
        for idx, chunk in enumerate(chunks):
            token_count = len(
                tokenizer.encode(chunk.page_content, add_special_tokens=True)
            )
            if token_count > embedding_settings.MAX_SEQ_LENGTH:
                raise RuntimeError(
                    f"Chunk vượt embedding limit sau khi split: "
                    f"tokens={token_count}, limit={embedding_settings.MAX_SEQ_LENGTH}, "
                    f"file={file_path.name!r}, chunk={idx}."
                )
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["id"] = _stable_id(source, idx)
            chunk.metadata["token_count"] = token_count

        all_chunks.extend(chunks)
        processed += 1
        _log.info("'%s' → %d chunk", file_path.name, len(chunks))

    _log.info(
        "Ingestion xong: %d/%d file → %d chunk.",
        processed, len(files), len(all_chunks),
    )
    return all_chunks, processed


def ingest(
    store: QdrantStore,
    folder: str | None = None,
    skip_if_not_empty: bool = True,
) -> IngestResponse:
    """Ingest một thư mục vào vector store."""
    start = time.perf_counter()
    source = str(resolve_path(folder or ingestion_settings.DATA_FOLDER))

    if skip_if_not_empty and not store.is_empty():
        count = store.count()
        _log.info("Bỏ qua ingest — collection đã có %d document.", count)
        return IngestResponse(
            status="skipped_not_empty",
            source=source,
            files_processed=0,
            chunks_added=0,
            doc_count=count,
            took_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    chunks, files_processed = chunk_from_folder(folder)
    if not chunks:
        return IngestResponse(
            status="no_documents",
            source=source,
            files_processed=files_processed,
            chunks_added=0,
            doc_count=store.count(),
            took_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    added = store.add_documents(chunks)
    return IngestResponse(
        status="success",
        source=source,
        files_processed=files_processed,
        chunks_added=added,
        doc_count=store.count(),
        took_ms=round((time.perf_counter() - start) * 1000, 2),
    )

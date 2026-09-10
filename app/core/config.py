"""Configuration for the RAG service.

Mọi biến dùng prefix ``RAG_`` để service này không bao giờ đọc trộm config của
``app/``. Nhờ vậy hai service chạy chung một máy, chung một file ``.env`` vẫn
độc lập hoàn toàn — và khi tách repo thật thì không phải đổi tên biến nào.

Điểm thiết kế quan trọng: chọn Qdrant local-path hay server chỉ dựa trên
``RAG_QDRANT_URL`` có được set hay không. Không có flag riêng nào để lệch pha
với nhau. Xem ``QdrantLocation``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings

# Mốc duy nhất cho config và dữ liệu của service. Không phụ thuộc cwd hoặc tên
# thư mục cha khi rag_service được tách thành repository/container riêng.
SERVICE_ROOT = Path(__file__).resolve().parents[2]
SERVICE_ENV_FILE = SERVICE_ROOT / ".env"


@dataclass(frozen=True)
class QdrantLocation:
    """Nơi Qdrant sống — server hoặc thư mục local.

    ``mode`` được suy ra, không cấu hình trực tiếp, nên không thể xảy ra trạng
    thái mâu thuẫn kiểu "mode=server nhưng url rỗng".
    """

    mode: Literal["server", "local"]
    url: Optional[str]
    path: Optional[str]

    @property
    def is_server(self) -> bool:
        return self.mode == "server"

    def describe(self) -> str:
        return f"server url={self.url}" if self.is_server else f"local path={self.path}"


class ServiceSettings(BaseSettings):
    """Server/runtime config.

    Attributes
    ----------
    HOST, PORT:
        Địa chỉ bind. Port 8002 để không đụng app (8000) và mcp_server (8001).

        Không dùng 8002: trên Windows có Hyper-V/WSL, dải port này thường bị
        reserve sẵn (xem ``netsh int ipv4 show excludedportrange protocol=tcp``)
        và uvicorn sẽ báo ``WinError 10013`` — lỗi trông như "port đang bị chiếm"
        nhưng không process nào chiếm cả, rất tốn thời gian để truy.
    CONTEXT_PATH:
        Prefix của REST API + docs. Theo đúng convention của ``mcp_server``.
    LOG_LEVEL:
        Mức log khi chạy trực tiếp ``python -m app.main`` từ service root.
    """

    HOST: str = "127.0.0.1"
    PORT: int = 8002
    RELOAD: bool = True
    LOG_LEVEL: str = "INFO"
    CONTEXT_PATH: str = "/api/v1"

    class Config:
        env_prefix = "RAG_SERVICE_"
        env_file = SERVICE_ENV_FILE
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


class VectorDBSettings(BaseSettings):
    """Vector store config.

    Attributes
    ----------
    COLLECTION_NAME:
        Tên collection.
    QDRANT_URL:
        URL Qdrant server. Set giá trị này là chuyển sang server mode — không
        cần đổi thêm gì. Để rỗng thì dùng ``QDRANT_PATH``.
    QDRANT_PATH:
        Thư mục Qdrant local (embedded). Lưu ý: chỉ MỘT process mở được path
        này cùng lúc do file lock, nên khi chạy song song với ``app/`` phải
        dùng path/collection khác.
    NUM_RETRIEVAL:
        Số hit mặc định trả về mỗi query.
    """

    COLLECTION_NAME: str = "InternalDocDB_bge_m3_v1"
    QDRANT_URL: Optional[str] = None
    QDRANT_PATH: str = "data/runtime/qdrant"
    NUM_RETRIEVAL: int = 5

    class Config:
        env_prefix = "RAG_"
        env_file = SERVICE_ENV_FILE
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"

    def qdrant_location(self) -> QdrantLocation:
        """Suy ra nơi Qdrant sống từ config.

        Chuỗi rỗng/whitespace được coi như chưa set — file ``.env`` rất hay có
        ``RAG_QDRANT_URL=`` để lại, và nó không nên bị hiểu là server mode.
        """
        url = (self.QDRANT_URL or "").strip()
        if url:
            return QdrantLocation(mode="server", url=url, path=None)
        return QdrantLocation(mode="local", url=None, path=self.QDRANT_PATH)


class EmbeddingSettings(BaseSettings):
    """Embedding model config.

    Attributes
    ----------
    PROVIDER:
        ``local`` (sentence-transformers) hoặc ``gemini``.
    MODEL_NAME:
        Với ``local``: đường dẫn thư mục model hoặc HuggingFace model id.
        Với ``gemini``: tên model embedding.
    OUTPUT_DIMENSION:
        Chỉ dùng cho ``gemini``.
    MAX_SEQ_LENGTH:
        Số token tối đa gửi vào local embedding model. Giá trị này phải không
        vượt context thật của model; BGE-M3 hỗ trợ dài hơn nhiều nhưng service
        chủ động dùng 768 để giữ chi phí CPU ổn định.
    BATCH_SIZE:
        Số chunk encode cùng lúc. Laptop CPU dùng 1 để tránh tăng RAM đột biến.
    DEVICE:
        Device cho SentenceTransformer. Máy mục tiêu không có CUDA nên mặc định CPU.
    ALLOW_HF_DOWNLOAD:
        Cho phép tải model từ HuggingFace khi path local không tồn tại.
        Mặc định ``False`` — tải im lặng một model KHÁC sẽ tạo ra vector space
        khác với lúc ingest, làm retrieval sai mà không có lỗi nào được raise.
        Đây là loại bug rất khó truy, nên mặc định là fail-fast.
    """

    PROVIDER: Literal["local", "gemini"] = "local"
    MODEL_NAME: str = "data/models/bge-m3"
    OUTPUT_DIMENSION: int = 1024
    MAX_SEQ_LENGTH: int = 768
    BATCH_SIZE: int = 1
    DEVICE: Literal["cpu", "cuda", "mps"] = "cpu"
    ALLOW_HF_DOWNLOAD: bool = False

    class Config:
        env_prefix = "RAG_EMBEDDING_"
        env_file = SERVICE_ENV_FILE
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


class IngestionSettings(BaseSettings):
    """Document ingestion config.

    Attributes
    ----------
    DATA_FOLDER:
        Thư mục nguồn mặc định cho ``POST /ingest``.
    CHUNK_SIZE, CHUNK_OVERLAP:
        Tham số chia chunk theo TOKEN của embedding tokenizer, không phải ký tự.
    """

    DATA_FOLDER: str = "data/raw"
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 80

    class Config:
        env_prefix = "RAG_INGEST_"
        env_file = SERVICE_ENV_FILE
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


class APIKeySettings(BaseSettings):
    """Khoá API cho embedding provider dạng hosted."""

    GEMINI_API_KEY: Optional[str] = None

    class Config:
        env_file = SERVICE_ENV_FILE
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

service_settings = ServiceSettings()
vectordb_settings = VectorDBSettings()
embedding_settings = EmbeddingSettings()
ingestion_settings = IngestionSettings()
api_key_settings = APIKeySettings()

def resolve_path(raw: str) -> Path:
    """Đổi path tương đối trong config thành path tuyệt đối theo service root.

    Không dùng cwd: service có thể được khởi chạy từ workspace root, thư mục
    ``rag_service``, hoặc process manager mà vẫn dùng đúng dữ liệu của nó.
    """
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (SERVICE_ROOT / path)

"""RAG Service — retrieval + ingestion cho internal knowledge base.

Expose đồng thời:
- REST API tại ``RAG_SERVICE_CONTEXT_PATH`` (default ``/api/v1``) — hợp đồng chính
- MCP Streamable HTTP tại ``/mcp`` — adapter cho MCP client

Chạy:
    python app/main.py                      (từ rag_service/)
    python -m app.main                      (từ rag_service/)

Các import nội bộ là relative theo package ``app`` nên service không phụ thuộc
tên hoặc layout thư mục cha của monorepo.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import sys

# Hỗ trợ cả ``python -m app.main`` (khuyến nghị) và ``python app/main.py``.
# Khi chạy file trực tiếp, Python chỉ thêm ``app/`` vào sys.path; thêm service
# root để package ``app`` có thể được resolve cho các relative import bên dưới.
SERVICE_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(SERVICE_ROOT))
    __package__ = "app"

from fastapi import FastAPI

from .controllers.ingest_controller import router as ingest_router
from .controllers.retrieve_controller import router as retrieve_router
from .controllers.stats_controller import router as stats_router
from .core.config import service_settings, vectordb_settings
from .core.logger import setup_logger
from .core.middleware import LoggingMiddleware
from .models.response_schema import HealthResponse
from .services.bm25_service import BM25Index
from .vectordb.qdrant import QdrantStore

logger = setup_logger()

CONTEXT_PATH = service_settings.CONTEXT_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Store được tạo MỘT LẦN ở đây rồi chia sẻ cho mọi request. Bắt buộc với
    # Qdrant embedded: mỗi client mở local path đều cố lấy file lock, nên tạo
    # store per-request là chắc chắn lỗi.
    store = QdrantStore()
    bm25 = BM25Index()
    bm25.rebuild(store.all_documents())
    app.state.store = store
    app.state.bm25 = bm25
    try:
        doc_count = store.count()
        logger.info(
            "RAG Service ready | port=%d | backend=%s | collection=%s | docs=%d | "
            "%s | embedding=%s",
            service_settings.PORT,
            store.backend,
            store.collection_name,
            doc_count,
            store.location,
            store.embedding_model,
        )
        if doc_count == 0:
            logger.warning(
                "Collection rỗng. Gọi POST %s/ingest để nạp dữ liệu.", CONTEXT_PATH
            )
        yield
    finally:
        store.close()
        del app.state.bm25
        del app.state.store
        logger.info("RAG Service shutting down")


app = FastAPI(
    title="RAG Service",
    summary="Retrieval and ingestion service for the internal knowledge base",
    description=(
        "Service độc lập lo phần retrieval: quản lý vector store, embedding model "
        "và document ingestion. Expose REST API kèm MCP adapter.\n\n"
        "Service trả về documents CÓ CẤU TRÚC kèm relevance score. Việc format "
        "chúng thành prompt cho LLM thuộc phía client — nhờ vậy prompt tuning "
        "không cần deploy lại service này."
    ),
    version="1.0.0",
    docs_url=f"{CONTEXT_PATH}/docs",
    redoc_url=f"{CONTEXT_PATH}/redoc",
    openapi_url=f"{CONTEXT_PATH}/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(LoggingMiddleware)

app.include_router(retrieve_router, prefix=CONTEXT_PATH)
app.include_router(ingest_router, prefix=CONTEXT_PATH)
app.include_router(stats_router, prefix=CONTEXT_PATH)


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health() -> HealthResponse:
    """Liveness probe — không chạm vào vector store.

    Dùng ``GET {CONTEXT_PATH}/ready`` khi cần biết store có truy cập được.
    """
    return HealthResponse(status="healthy", service="rag-service")


# ───────────────────────── MCP adapter ───────────────────────────────
# REST endpoint được fastapi-mcp tự chuyển thành MCP tools. Tên tool lấy từ
# operation_id (đặt tay trong controller) nên nó ổn định qua các version.
def _mount_mcp(fastapi_app: FastAPI) -> None:
    try:
        from fastapi_mcp import FastApiMCP

        mcp = FastApiMCP(fastapi_app, name="RAG Service")
        mcp.mount_http()
        logger.info("MCP mounted tại /mcp")
    except ImportError:
        # REST vẫn dùng được bình thường mà không có fastapi-mcp.
        logger.warning(
            "fastapi-mcp chưa được cài — bỏ qua MCP endpoint. REST API vẫn hoạt động."
        )


_mount_mcp(app)


if __name__ == "__main__":
    import os

    import uvicorn

    # reload_dirs giới hạn ở chính rag_service. Nếu không, uvicorn watch cả
    # workspace root và service sẽ restart mỗi khi ai đó sửa file trong app/ hay
    # ui/ — startup mất vài chục giây (load embedding model) nên restart oan như
    # vậy trông y hệt service bị treo.
    # Reload spawn process phải import được cùng package path. Với chế độ chạy
    # độc lập, cwd về service root để ``app.main`` luôn resolve đúng.
    if __package__ == "app":
        os.chdir(SERVICE_ROOT)
    uvicorn.run(
        f"{__package__}.main:app",
        host=service_settings.HOST,
        port=service_settings.PORT,
        reload=service_settings.RELOAD,
        reload_dirs=[str(SERVICE_ROOT)],
    )

"""Logging setup for the RAG service."""

from __future__ import annotations

import logging
import sys

from .config import service_settings

_FORMAT = "%(asctime)s | %(levelname)-8s | [RAG] %(name)s | %(message)s"


def _force_utf8_stdout() -> None:
    """Đảm bảo stdout ghi được UTF-8.

    Console Windows mặc định dùng cp1252, không encode được tiếng Việt. Log một
    tên file hay query tiếng Việt sẽ ném ``UnicodeEncodeError`` từ trong logging
    handler — traceback dài, khó đọc, và che mất lỗi thật. Tài liệu của service
    này là tiếng Việt nên đây là chuyện xảy ra ngay từ dòng log đầu tiên.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Stream đã bị thay thế (pytest capture, pipe...) — bỏ qua.
            pass


def setup_logger(name: str = "rag-service") -> logging.Logger:
    """Cấu hình root logger và trả về logger của service.

    Dùng ``force=True`` để ghi đè handler mà thư viện khác (sentence-transformers,
    qdrant-client) có thể đã cài trước — nếu không, format log sẽ không đồng nhất.
    """
    _force_utf8_stdout()
    logging.basicConfig(
        level=getattr(logging, service_settings.LOG_LEVEL.upper(), logging.INFO),
        format=_FORMAT,
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger(name)

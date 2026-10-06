import logging
from pathlib import Path
import sys
import uvicorn

# Đảm bảo đường dẫn service root có trong sys.path
SERVICE_ROOT = Path(__file__).resolve().parent
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.core.config import service_settings

if __name__ == "__main__":
    logging.basicConfig(level=getattr(logging, service_settings.LOG_LEVEL.upper(), logging.INFO))
    _log = logging.getLogger(__name__)
    _log.info("Starting RAG Service on %s:%d ...", service_settings.HOST, service_settings.PORT)
    uvicorn.run(
        "app.main:app",
        host=service_settings.HOST,
        port=service_settings.PORT,
        reload=service_settings.RELOAD,
        reload_dirs=[str(SERVICE_ROOT / "app")],
        env_file=".env",
    )

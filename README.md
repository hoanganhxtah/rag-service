# RAG Service

RAG Service là dịch vụ độc lập chịu trách nhiệm nạp tài liệu và tìm kiếm các
đoạn nội dung liên quan trong knowledge base. Service cung cấp REST API và MCP
endpoint để ứng dụng hoặc AI agent có thể sử dụng cùng một pipeline retrieval.

Pipeline tìm kiếm hiện tại kết hợp:

- Dense retrieval bằng embedding và Qdrant.
- Lexical retrieval bằng BM25, hỗ trợ token tiếng Việt và tiếng Anh.
- Reciprocal Rank Fusion (RRF) để gộp và xếp hạng hai nguồn kết quả.

Service chỉ trả về document có cấu trúc cùng relevance score. Việc xây prompt
và gọi LLM thuộc trách nhiệm của client.

## Tính năng chính

- Nạp tài liệu từ một thư mục local.
- Hỗ trợ PDF, Word, PowerPoint, Excel, HTML, Markdown và text.
- Chia chunk theo tokenizer của embedding model.
- Chạy Qdrant ở embedded mode hoặc kết nối Qdrant server.
- Dùng embedding model local hoặc Gemini cho retrieval.
- Cung cấp REST API, OpenAPI/Swagger và MCP Streamable HTTP.
- Cung cấp endpoint health, readiness và thống kê collection.

## Kiến trúc

```text
Document folder
      |
      v
convert -> token-aware chunking -> embedding -> Qdrant
                                             -> BM25 in memory

Client query
      |
      +-> Dense search (Qdrant) --+
      |                           +-> RRF -> ranked documents
      +-> Lexical search (BM25) --+
```

Qdrant store và BM25 index được tạo một lần trong FastAPI lifespan. Khi service
khởi động, BM25 được dựng lại từ các document đang có trong Qdrant. Sau một lần
ingest thành công, BM25 cũng được dựng lại ngay.

## Cấu trúc thư mục

```text
rag_service/
├── app/
│   ├── main.py                    # FastAPI lifespan, routes và MCP
│   ├── controllers/               # REST endpoints
│   ├── services/                  # Ingestion, BM25 và hybrid retrieval
│   ├── embeddings/embed.py        # Local/Gemini embeddings
│   ├── vectordb/qdrant.py         # Qdrant embedded/server
│   ├── models/                    # Request/response schemas
│   └── core/                      # Config, logging và middleware
├── data/
│   ├── raw/                       # Tài liệu nguồn mặc định
│   ├── models/                    # Embedding model local
│   └── runtime/                   # Dữ liệu Qdrant embedded
├── tests/
├── .env.example
├── requirements.txt
└── s3_data.py                     # Script S3 độc lập, chưa nối vào API
```

Các đường dẫn tương đối trong `.env` luôn được resolve theo thư mục
`rag_service/`, không phụ thuộc thư mục hiện tại khi chạy lệnh.

## Yêu cầu

- Python 3.10 trở lên.
- Khoảng trống đĩa đủ cho PyTorch, Docling và embedding model.
- Embedding model local, hoặc kết nối mạng nếu cho phép tải model.
- Qdrant server chỉ bắt buộc khi sử dụng server mode.

## Chạy nhanh

### 1. Tạo môi trường Python

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

### 2. Chuẩn bị embedding model

Mặc định service tìm BGE-M3 tại:

```text
data/models/bge-m3
```

Nếu model đã nằm ở vị trí khác, sửa `RAG_EMBEDDING_MODEL_NAME` trong `.env`.

Để cho phép tải BGE-M3 từ Hugging Face ở lần chạy đầu:

```dotenv
RAG_EMBEDDING_PROVIDER=local
RAG_EMBEDDING_MODEL_NAME=BAAI/bge-m3
RAG_EMBEDDING_ALLOW_HF_DOWNLOAD=true
```

Trong môi trường production nên tải và cố định model trước, sau đó đặt
`RAG_EMBEDDING_ALLOW_HF_DOWNLOAD=false` để tránh vô tình thay đổi vector space.

### 3. Thêm tài liệu

Đặt tài liệu vào `data/raw/`, hoặc cấu hình thư mục khác:

```dotenv
RAG_INGEST_DATA_FOLDER=data/raw
```

### 4. Khởi động service

Từ thư mục `rag_service/`:

```bash
python -m app.main
```

Các URL mặc định:

| Thành phần | URL |
| --- | --- |
| Liveness | `http://localhost:8002/health` |
| Readiness | `http://localhost:8002/api/v1/ready` |
| Swagger UI | `http://localhost:8002/api/v1/docs` |
| ReDoc | `http://localhost:8002/api/v1/redoc` |
| OpenAPI | `http://localhost:8002/api/v1/openapi.json` |
| MCP | `http://localhost:8002/mcp` |

### 5. Nạp và tìm kiếm tài liệu

Nạp thư mục mặc định:

```bash
curl -X POST http://localhost:8002/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"folder": null, "skip_if_not_empty": true}'
```

Tìm kiếm:

```bash
curl -X POST http://localhost:8002/api/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "chính sách thẻ tín dụng", "top_k": 5}'
```

## REST API

| Method | Endpoint | Chức năng |
| --- | --- | --- |
| `POST` | `/api/v1/retrieve` | Hybrid search bằng Dense + BM25 + RRF |
| `POST` | `/api/v1/ingest` | Convert, chunk và nạp tài liệu |
| `GET` | `/api/v1/stats` | Thống kê collection và cấu hình retrieval |
| `GET` | `/api/v1/ready` | Kiểm tra readiness cơ bản của store |
| `GET` | `/health` | Kiểm tra process FastAPI còn hoạt động |

### `POST /api/v1/retrieve`

Request:

```json
{
  "query": "thẻ tín dụng có ưu đãi gì?",
  "top_k": 5
}
```

`top_k` nhận giá trị từ `1` đến `100`. Nếu bỏ trống, service dùng
`RAG_NUM_RETRIEVAL`.

Response rút gọn:

```json
{
  "query": "thẻ tín dụng có ưu đãi gì?",
  "results": [
    {
      "content": "...",
      "metadata": {
        "source": "data/raw/example.docx",
        "file_name": "example.docx",
        "extension": ".docx",
        "chunk_index": 0,
        "id": "...",
        "token_count": 512
      },
      "score": 0.0328
    }
  ],
  "total": 1,
  "raw_count": 1,
  "best_score": 0.0328,
  "took_ms": 18.4
}
```

`score` là RRF score dùng để xếp hạng, không phải cosine similarity và không nằm
trên thang điểm cố định từ `0` đến `1`.

### `POST /api/v1/ingest`

Request:

```json
{
  "folder": null,
  "skip_if_not_empty": true
}
```

- `folder=null`: dùng `RAG_INGEST_DATA_FOLDER`.
- `skip_if_not_empty=true`: bỏ qua nếu collection đã có dữ liệu.
- Ingestion chạy đồng bộ; client cần timeout đủ dài với tập dữ liệu lớn.

Các trạng thái response:

- `success`: đã thêm chunk vào Qdrant.
- `skipped_not_empty`: collection đã có dữ liệu và request yêu cầu bỏ qua.
- `no_documents`: không tìm thấy tài liệu hợp lệ hoặc convert không ra nội dung.

Các định dạng được hỗ trợ:

```text
.pdf .docx .doc .pptx .ppt .xlsx .xls .html .htm
.md .txt .rst .asciidoc .adoc
```

PDF được đọc bằng `pypdf`, không có OCR. Các định dạng còn lại được chuyển sang
Markdown bằng Docling. Một file lỗi sẽ được log và bỏ qua thay vì làm hỏng toàn
bộ lần ingest.

### `GET /api/v1/stats`

Endpoint trả về backend, collection, document count, vị trí Qdrant, embedding
model/dimension, distance metric, BM25 document count và `top_k` mặc định.

```bash
curl http://localhost:8002/api/v1/stats
```

### Health và readiness

- `/health` là liveness probe, không truy cập vector store.
- `/api/v1/ready` trả payload có trường `ready` và hiện luôn dùng HTTP `200` cho
  cả trạng thái sẵn sàng lẫn chưa sẵn sàng.

Không nên chỉ kiểm tra HTTP status của `/ready` trong production. Healthcheck
cần đọc giá trị `ready` trong JSON; implementation hiện tại vẫn là kiểm tra cơ
bản và chưa nên là tín hiệu duy nhất để điều phối traffic.

## MCP

Khi `fastapi-mcp` được cài đặt, service mount MCP Streamable HTTP tại `/mcp`.
Các REST operation chính được ánh xạ thành MCP tools:

| MCP tool | REST endpoint |
| --- | --- |
| `search_knowledge_base` | `POST /api/v1/retrieve` |
| `ingest_documents` | `POST /api/v1/ingest` |
| `get_knowledge_base_stats` | `GET /api/v1/stats` |
| `get_readiness` | `GET /api/v1/ready` |

Ví dụ cấu hình MCP client:

```json
{
  "mcpServers": {
    "rag-service": {
      "url": "http://localhost:8002/mcp"
    }
  }
}
```

Các `operation_id` là hợp đồng công khai vì chúng quyết định tên MCP tool.
Thay đổi chúng có thể làm hỏng prompt hoặc client đang sử dụng service.

## Cấu hình

### HTTP server

| Biến | Mặc định | Mô tả |
| --- | --- | --- |
| `RAG_SERVICE_HOST` | `127.0.0.1` | Địa chỉ bind |
| `RAG_SERVICE_PORT` | `8002` | Cổng HTTP |
| `RAG_SERVICE_RELOAD` | `true` | Tự reload khi code thay đổi |
| `RAG_SERVICE_LOG_LEVEL` | `INFO` | Mức log |
| `RAG_SERVICE_CONTEXT_PATH` | `/api/v1` | Prefix của REST API và docs |

### Qdrant và retrieval

| Biến | Mặc định | Mô tả |
| --- | --- | --- |
| `RAG_COLLECTION_NAME` | `InternalDocDB_bge_m3_v1` | Tên collection |
| `RAG_QDRANT_URL` | rỗng | Có giá trị thì dùng Qdrant server |
| `RAG_QDRANT_PATH` | `data/runtime/qdrant` | Dữ liệu Qdrant embedded |
| `RAG_NUM_RETRIEVAL` | `5` | Số kết quả mặc định |

Nếu `RAG_QDRANT_URL` rỗng, service dùng embedded mode. Embedded mode chỉ cho
một process mở cùng một data path tại một thời điểm vì Qdrant giữ file lock.

### Embedding

| Biến | Mặc định | Mô tả |
| --- | --- | --- |
| `RAG_EMBEDDING_PROVIDER` | `local` | `local` hoặc `gemini` |
| `RAG_EMBEDDING_MODEL_NAME` | `data/models/bge-m3` | Local path, Hugging Face ID hoặc Gemini model |
| `RAG_EMBEDDING_OUTPUT_DIMENSION` | `1024` | Output dimension cho Gemini |
| `RAG_EMBEDDING_MAX_SEQ_LENGTH` | `768` | Giới hạn token của local model |
| `RAG_EMBEDDING_BATCH_SIZE` | `1` | Batch size khi encode local |
| `RAG_EMBEDDING_DEVICE` | `cpu` | `cpu`, `cuda` hoặc `mps` |
| `RAG_EMBEDDING_ALLOW_HF_DOWNLOAD` | `false` | Cho phép tải model từ Hugging Face |
| `GEMINI_API_KEY` | rỗng | Bắt buộc khi provider là Gemini |

Retrieval có thể sử dụng Gemini embeddings. Tuy nhiên ingestion hiện dùng
tokenizer local để chia chunk nên yêu cầu `RAG_EMBEDDING_PROVIDER=local`.

### Ingestion

| Biến | Mặc định | Mô tả |
| --- | --- | --- |
| `RAG_INGEST_DATA_FOLDER` | `data/raw` | Thư mục tài liệu nguồn |
| `RAG_INGEST_CHUNK_SIZE` | `600` | Kích thước chunk theo token |
| `RAG_INGEST_CHUNK_OVERLAP` | `80` | Số token overlap |

`RAG_INGEST_CHUNK_OVERLAP` phải nhỏ hơn `RAG_INGEST_CHUNK_SIZE`, và chunk size
không được vượt `RAG_EMBEDDING_MAX_SEQ_LENGTH`.

## Kiểm thử

Chạy unit tests bằng thư viện chuẩn của Python:

```bash
python -m unittest discover -s tests -v
```

Test hiện tại tập trung vào tokenizer BM25 và cách RRF gộp Dense/BM25 results.

## S3

`s3_data.py` là script thử nghiệm độc lập để đọc một object từ S3. Script này:

- Chưa được gọi bởi FastAPI hoặc endpoint `/ingest`.
- Không tự nạp nội dung đọc được vào Qdrant.
- Cần thêm `boto3` và `python-docx` nếu muốn chạy với file DOCX.

Luồng ingestion chính hiện chỉ đọc file từ thư mục local.

## Giới hạn và lưu ý bảo mật

- API hiện chưa có authentication hoặc authorization.
- `/ingest` là endpoint ghi dữ liệu; không expose service trực tiếp ra Internet.
- Ingestion chạy đồng bộ và có thể vượt timeout của client với dữ liệu lớn.
- Chưa có upsert/deduplication thực sự. Gọi ingest lặp với
  `skip_if_not_empty=false` có thể tạo dữ liệu trùng.
- PDF scan không có text layer cần OCR trước khi ingest.
- BM25 là in-memory index và được rebuild khi service khởi động hoặc ingest xong.
- Cùng một collection phải luôn dùng cùng embedding model/vector space. Nếu đổi
  model, hãy tạo collection mới và ingest lại dữ liệu.

## Xử lý lỗi thường gặp

### Không tìm thấy embedding model

Kiểm tra `RAG_EMBEDDING_MODEL_NAME`. Nếu muốn tải từ Hugging Face, dùng model ID
hợp lệ như `BAAI/bge-m3` và bật `RAG_EMBEDDING_ALLOW_HF_DOWNLOAD=true`.

### Qdrant báo file đang bị lock

Một process khác đang mở cùng `RAG_QDRANT_PATH`. Dừng process đó, dùng data path
khác, hoặc chuyển sang Qdrant server bằng `RAG_QDRANT_URL`.

### Embedding dimension không khớp

Collection được tạo bằng model có vector dimension khác. Không tái sử dụng
collection đó; đổi `RAG_COLLECTION_NAME` và ingest lại.

### Ingest trả `no_documents`

Kiểm tra thư mục nguồn tồn tại, có định dạng được hỗ trợ và file convert ra nội
dung không rỗng. PDF dạng ảnh sẽ không đọc được nếu chưa OCR.

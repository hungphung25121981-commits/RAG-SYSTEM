# README 4 — APP (Backend Engine)

 **App backend**

---

## 1. APP — Backend Engine (FastAPI + Docker Compose)

### 1.1 Vai trò

`backend/app/main.py` là một **lớp vỏ API** (API wrapper) bọc quanh pipeline RAG (`cli_pipeline.py` + các module `src/phase*`), để lộ ra chuẩn **OpenAI-compatible API** (`/v1/chat/completions`). Nhờ chuẩn này, bất kỳ giao diện chat nào hỗ trợ OpenAI API — điển hình là **Open WebUI** — đều có thể "nói chuyện" trực tiếp với hệ thống VideoRAG mà không cần code thêm giao diện riêng.

### 1.2 Công nghệ sử dụng

| Thành phần | Công nghệ | Vai trò |
|---|---|---|
| Web framework | **FastAPI** + **Uvicorn** | Expose REST API, hot-reload khi dev (`--reload`) |
| Giao tiếp streaming | **SSE (Server-Sent Events)** | Trả lời từng từ một (giống hiệu ứng gõ chữ của ChatGPT) |
| Chuẩn dữ liệu | **Pydantic** (`BaseModel`) | Validate request/response theo schema OpenAI |
| CORS | `CORSMiddleware` | Cho phép Open WebUI (chạy ở domain/port khác) gọi API |
| Điều phối container | **Docker Compose** | Dựng đồng thời 3 service: `vector_db`, `rag-api`, `open-webui` |

### 1.3 Kiến trúc 3 container (`docker-compose.yaml`)

```
┌──────────────┐      HTTP :3000       ┌──────────────────┐
│  open-webui   │ ────────────────────▶   rag-api       
│ (giao diện)   │  OPENAI_API_BASE_URL │ (FastAPI :8000)  │
└──────────────┘                       └──────┬───────────┘
                                              │ đọc/ghi vector
                                              ▼
                                       ┌───────────────┐
                                       │  vector_db    │
                                       │ (Qdrant :8080)│
                                       └───────────────┘
```

- **`vector_db`**: image `qdrant/qdrant:v1.8.0`, lưu embeddings, mount volume `./backend/data/qdrant_db`.
- **`rag-api`**: build từ `backend/Dockerfile`, chạy `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`.
- **`open-webui`**: image có sẵn `ghcr.io/open-webui/open-webui`, trỏ `OPENAI_API_BASE_URL` về `http://rag-api:8000/v1`, dùng `OPENAI_API_KEY` giả (`sk-dummy-key`) vì hệ thống không cần xác thực thật.
- Có thêm service `rag_engine` dùng để chạy `cli_pipeline.py` như một CLI container độc lập (phục vụ batch job, không phải web server).

### 1.4 Cách khởi động

```bash
# 1. Copy file env mẫu và điền API key nếu cần (Gemini, W&B...)
cp .env.example .env

# 2. Dựng toàn bộ hệ thống
docker compose up --build

# 3. Truy cập giao diện chat tại:
http://localhost:3000
```

Khi mở Open WebUI, model **`video-rag-v1`** sẽ tự động xuất hiện trong dropdown chọn model — vì `rag-api` đã tự khai báo nó qua endpoint `/v1/models`.

### 1.5 Các endpoint hiện có

| Method | Path | Mục đích |
|---|---|---|
| `GET` | `/` | Health check — kiểm tra server sống hay chết |
| `GET` | `/v1/models` | Trả danh sách model để Open WebUI hiển thị dropdown |
| `POST` | `/v1/chat/completions` | Endpoint chính, nhận câu hỏi và trả lời dạng stream (SSE) |

**Ví dụ gọi trực tiếp bằng `curl`:**

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "video-rag-v1",
        "messages": [{"role": "user", "content": "Ai ghi bàn thắng quyết định?"}],
        "stream": true
      }'
```

Phản hồi trả về theo từng dòng `data: {...}` chuẩn SSE, kết thúc bằng `data: [DONE]` — đúng định dạng OpenAI Chat Completions Chunk.

### 1.6 Lưu ý quan trọng (đang ở trạng thái "khung sườn")

`main.py` hiện tại **chưa nối logic RAG thật** — hàm `dummy_rag_streamer()` chỉ echo lại câu hỏi kèm câu trả lời giả để test giao diện. Ngoài ra file đang có **2 lỗi cần sửa trước khi chạy được**:

1. Class `ChatMessage` bị định nghĩa lồng sai — nó vừa là model tin nhắn, vừa cố gán field `messages: List[ChatMessage]` (tự tham chiếu chính nó) và `model`, `stream` bên trong cùng 1 class. Cần tách thành 2 class riêng: `ChatMessage` (role, content) và `ChatCompletionRequest` (model, messages, stream, temperature).
2. Hàm `chat_completions()` nhận tham số `request: ChatCompletionRequest` nhưng class này **chưa từng được khai báo** trong file — sẽ gây `NameError` khi chạy.

**Việc cần làm ở bước tiếp theo:** thay `dummy_rag_streamer()` bằng lời gọi thật tới `Pipeline.run_phase3()` (đã có sẵn trong `cli_pipeline.py`) để API thực sự truy vấn Qdrant + Kùzu + sinh câu trả lời bằng Qwen2.5-VL, thay vì trả lời giả lập.

---

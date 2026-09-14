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



## 3. DATA — Cấu Trúc Thư Mục Dữ Liệu

```
backend/data/
├── raw_videos/          # (input) Video gốc người dùng đưa vào, .mp4/.mkv/.mov/.avi
│   └── stress_test.mp4  #   video 10s dùng riêng cho --dummy_test (kiểm tra VRAM)
│
├── temp_workspace/       # (trung gian) Không gian làm việc tạm cho từng video
│   └── <ten_video>/
│       ├── frames/            # Keyframe trích từ video (ảnh .jpg)
│       ├── audio/              # Audio đã tách: giọng nói sạch + tiếng nền
│       ├── audio_raw.wav       # Audio thô tách bằng ffmpeg (48kHz mono)
│       ├── chunks.json         # Kết quả Hybrid Chunking (Phase 1) — đơn vị dữ liệu
│       │                       #   sẽ được embed ở Phase 2
│       ├── audio_tags.json     # Nhãn âm thanh nền (CLAP) — ví dụ "tiếng cổ vũ", "còi"
│       └── scenes.json         # Danh sách các cảnh (scene) đã cắt
│
├── qdrant_db/            # Vector database (Qdrant, chế độ embedded/local)
│   ├── collections/          #   mỗi domain = 1 collection riêng
│   ├── aliases/
│   └── raft_state.json
│
├── kuzu_graph/            # Graph database (Kùzu) — phục vụ GraphRAG / Graph Hop
│                          #   lưu quan hệ giữa các chunk (parent-child, cùng scene...)
│
├── eval_logs/             # "Sổ nhật ký" phục vụ Phase 4 + vận hành
│   ├── eval_queue.jsonl        # Log [query, context, answer] từ Phase 3 (nhịp 1)
│   │                           #   -> bị XOÁ sau khi Phase 4 tiêu thụ xong, tránh chấm trùng
│   ├── eval_queue_deep.jsonl   # Các case "nghi ngờ" của Tầng 1, chờ Prometheus-2 (Tầng 2)
│   ├── eval_report.jsonl       # Kết quả chấm điểm cuối cùng (pass/fail + điểm số)
│   ├── tracker_state.sqlite3   # DB theo dõi tiến độ pipeline (Dual Tracker, Phase 0)
│   └── search_export_*.csv     # Kết quả xuất CSV khi dùng cờ -outcsv/--export_csv
│
└── models/ (tuỳ chọn, người dùng tự tải về)
    └── prometheus2-7b.Q4_K_M.gguf   # Model Judge dùng cho Tầng 2 của Phase 4
```

### 3.1 Vai trò của từng nhóm

| Thư mục | Được ghi bởi | Được đọc bởi | Có nên commit vào Git không? |
|---|---|---|---|
| `raw_videos/` | Người dùng upload | Phase 0, Phase 1 | Không (video nặng) |
| `temp_workspace/` | Phase 1 | Phase 2 | Không (dữ liệu tạm, có thể xoá & tái tạo) |
| `qdrant_db/` | Phase 2 | Phase 3, Phase 4 (tune_weights) | Không (binary DB, đổi liên tục) |
| `kuzu_graph/` | Phase 2 (`--build_graph`) | Phase 3 (`--use_graph`) | Không |
| `eval_logs/` | Phase 3 (log) + Phase 4 (chấm) | Phase 4, người vận hành xem báo cáo | Có thể commit `eval_report.jsonl` để lưu lịch sử chất lượng |
| `config/settings.yaml` | `--save_domain` (tune_weights) hoặc chỉnh tay | Toàn bộ Phase 1-4 | **Có** — đây là "single source of truth" cho mọi hyper-parameter |

### 3.2 File cấu hình trung tâm: `config/settings.yaml`

Đây là **nguồn cấu hình duy nhất** cho toàn bộ hệ thống (mô hình dùng, ngưỡng, đường dẫn...). Cấu trúc quan trọng nhất cho Phase 3 & 4:

```yaml
domains:
  hub_kien_thuc:            # tên domain = tên collection trong Qdrant
    alpha: 0.50              # trọng số semantic search (tune bằng --tune_weights)
    beta: 0.30                # trọng số visual search
    # keyword weight (BM25) = 1 - alpha - beta, ngầm định
    gamma_matrix:             # ngưỡng OCR, calibrate 1 lần ở Phase 1 (không đụng vào khi tune alpha/beta)
      variance_boundary: 100.0
      low_variance: 70.0
      high_variance: 85.0

phase4_eval:
  reranker_model: "BAAI/bge-reranker-base"
  nli_model: "cross-encoder/nli-deberta-v3-base"
  cascade_low_reject: 0.4
  cascade_high_pass: 0.7
```

Mỗi khi thêm một domain nội dung mới (ví dụ chuyển từ video bóng đá sang video nấu ăn), chỉ cần thêm 1 block domain mới ở đây — không cần sửa code.

---

## 4. Bảng Tổng Hợp Công Nghệ (Tech Stack Overview)

| Nhóm | Công nghệ |
|---|---|
| API / Backend | FastAPI, Uvicorn, Pydantic, SSE |
| Điều phối hệ thống | Docker Compose, Open WebUI |
| Vector DB | Qdrant (embedded local, có thể chạy container riêng) |
| Graph DB | Kùzu (GraphRAG, graph hop) |
| Audio | faster-whisper (ASR), Silero-VAD, DeepFilterNet (khử ồn), LAION-CLAP (gắn nhãn âm thanh nền) |
| Vision | SigLIP2, YOLOv8 (Ultralytics), RapidOCR + EasyOCR (fallback), PySceneDetect |
| Embedding văn bản | BGE-M3 (dense) + BM25 (sparse) |
| Sinh câu trả lời (VLM) | Qwen2.5-VL-3B-Instruct (quantize NF4) |
| Rerank | BGE-Reranker-base (Cross-Encoder) |
| Đánh giá Tầng 1 | BGE-Reranker + NLI DeBERTa-v3 (100% CPU) |
| Đánh giá Tầng 2 | Prometheus-2 (GGUF 4-bit qua llama-cpp-python) |
| MLOps / theo dõi | Weights & Biases (W&B) |

---

### Tóm tắt luồng end-to-end

```
Video → Phase 0 (hash+tracker) → Phase 1 (trích xuất+chunking)
      → Phase 2 (embedding → Qdrant + Kùzu)
      → Phase 3 (tìm kiếm + trả lời qua API/CLI, log câu hỏi)
      → Phase 4 (chấm điểm batch, tối ưu trọng số)
```

Người dùng cuối chỉ tương tác qua **Open WebUI (cổng 3000)** hoặc gọi thẳng **API `/v1/chat/completions`**; toàn bộ Phase 0-2 và Phase 4 chạy nền qua `cli_pipeline.py`, không lộ ra ngoài giao diện chat.

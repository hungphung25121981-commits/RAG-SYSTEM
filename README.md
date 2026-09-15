# HUB KIẾN THỨC —VIDEO-RAG System

> Hệ thống RAG (Retrieval-Augmented Generation) đa phương thức cho video, xử lý hoàn toàn cục bộ (local-first), domain-agnostic, tối ưu để chạy trên phần cứng phổ thông (16GB VRAM).

---

## 1. Dự Án Này Là Gì?

Đây là một **backend xử lý video thành tri thức có thể truy vấn**: đưa vào một thư mục video thô, hệ thống sẽ tự động trích xuất âm thanh, hình ảnh, văn bản trong khung hình, xây dựng cơ sở dữ liệu vector + đồ thị tri thức, rồi cho phép người dùng đặt câu hỏi bằng ngôn ngữ tự nhiên và nhận câu trả lời có **trích dẫn chính xác đến từng mốc thời gian** trong video gốc.

Điểm khác biệt cốt lõi so với các hệ thống RAG video thông thường:

| Đặc tính | Cách tiếp cận của dự án |
|---|---|
| **Domain** | Hoàn toàn domain-agnostic — không hard-code cho một lĩnh vực cụ thể (thể thao, giáo dục, tin tức...), tự calibrate theo từng bộ dữ liệu qua `--save_domain` |
| **Phần cứng** | Chạy được trên GPU 16GB VRAM, kiểm soát peak < 10GB bằng Tiered Caching nghiêm ngặt |
| **Chi phí vận hành** | 0đ — không phụ thuộc bất kỳ API Cloud trả phí nào (kể cả bước đánh giá chất lượng RAG) |
| **Đa phương thức thật sự** | Không chỉ OCR text — kết hợp Audio (giọng nói + âm thanh môi trường), Visual (keyframe + metadata), Text (OCR), và Graph (quan hệ nhân quả/thời gian) |
| **Có thể kiểm chứng** | Mọi câu trả lời đều đi kèm `trake` (chunk_id, timestamp, keyframe_path) để người dùng tự đối chiếu lại video gốc |

---

## 2. Kiến Trúc Tổng Thể

Hệ thống được chia thành **5 Phase độc lập**, có thể chạy rời hoặc nối tiếp qua CLI. Dữ liệu chảy tuyến tính qua các phase, mỗi phase ghi trạng thái ra đĩa để phase sau (hoặc lần chạy resume sau) có thể tiếp tục mà không mất dữ liệu.

```
                      ───────────────────────────────────────────────────
                     │                  data/raw_videos/                 │
                     └───────────────────────┬───────────────────────────┘
                                              ▼
   ┌─────────────┐   ┌─────────────────┐   ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │   PHASE 0    ──▶     PHASE 1      ──▶      PHASE 2      ──▶      PHASE 3      ──▶     PHASE 4        
   │ Foundation  │   │ Extract &       │   │ Vector & Graph  │   │ Search &         │   │ 100% Local       │
   │ & MLOps     │   │ Chunking        │   │ Construction    │   │ Generation       │   │ Evaluation       │
   └─────────────┘   └─────────────────┘   └─────────────────┘   └──────────────────┘   └──────────────────┘
   Hash ID,           Audio Residual        SigLIP2 + BGE-M3       DeBERTa Router,        Cascade 2 tầng:
   Rollback,          Subtraction,          → Qdrant Tri-Search    RRF + Graph Hop,       BGE-Reranker + NLI
   Stress Test        OCR động (γ),         + Kuzu Graph DB        Knapsack DP Cut-off,   → Prometheus-2
                       Hybrid Chunking                              Qwen 2.5 VL             (VRAM Swap)
```

Chi tiết kỹ thuật đầy đủ của từng phase (thuật toán, model, cờ CLI, ví dụ chạy) nằm trong README riêng của từng thư mục — xem mục [4. Bản Đồ Tài Liệu](#4-bản-đồ-tài-liệu).

---

## 3. Cấu Trúc Thư Mục Dự Án

```
project-root/
├── backend/                     # Toàn bộ logic xử lý (Python, CLI-driven)
│   ├── cli_pipeline.py          # Điểm vào duy nhất — điều phối cả 5 phase qua argparse
│   ├── requirements.txt         # Toàn bộ dependency, quản lý version tường minh
│   ├── src/
│   │   ├── phase0_ops/          # Hash ID, Dependency check, Transactional Rollback
│   │   ├── phase1_extract/      # Audio/Vision extraction, Hybrid Chunking
│   │   ├── phase2_build/        # Embedding, Qdrant, Kuzu Graph
│   │   ├── phase3_search/       # Router, RRF, Graph Hop, Knapsack, VLM
│   │   └── phase4_eval/         # Cascade Evaluation (Cross-Encoder + Prometheus-2)
│   └── README.md
│
├── data/                         # Toàn bộ dữ liệu runtime — KHÔNG commit lên git
│   ├── raw_videos/               # Video đầu vào
│   ├── temp_workspace/           # File tạm theo từng phase, tự dọn khi rollback
│   ├── qdrant_db/                 # Vector DB local (theo Collection = domain)
│   ├── kuzu_graph/                # Graph DB local (Kuzu)
│   ├── eval_queue.jsonl           # Hàng đợi log [Query, Context, Answer] cho Phase 4
│   └── README.md
│
├── config/
│   └── settings.yaml              # α, β, gamma_matrix — lưu theo từng domain
│
├── app/                           # Lớp giao diện người dùng (frontend)
│   ├── docker-compose.yaml        # WebUI tạm thời (hiện tại)
│   └── README.md                  # Trạng thái hiện tại + lộ trình frontend chính thức
│
├── .env                           # API keys (W&B...), KHÔNG commit
├── .gitignore                     # Bắt buộc chứa data/ và .env
└── README.md                      # ← Bạn đang đọc file này
```

---

## 4. Bản Đồ Tài Liệu

| Tài liệu | Nội dung |
|---|---|
| [`backend/README.md`](./backend/README.md) | Cài đặt môi trường, ma trận cờ CLI đầy đủ, ví dụ chạy toàn pipeline |
| [`backend/src/phase0_ops/README.md`](./backend/src/phase0_ops/README.md) | Hash ID, Rollback, Stress Test |
| [`backend/src/phase1_extract/README.md`](./backend/src/phase1_extract/README.md) | Audio Residual Subtraction, OCR động, Chunking |
| [`backend/src/phase2_build/README.md`](./backend/src/phase2_build/README.md) | Embedding, Qdrant, Kuzu Graph |
| [`backend/src/phase3_search/README.md`](./backend/src/phase3_search/README.md) | Router, RRF, Graph Hop, Knapsack, VLM |
| [`backend/src/phase4_eval/README.md`](./backend/src/phase4_eval/README.md) | Cascade Evaluation 2 tầng |
| [`data/README.md`](./data/README.md) | Giải thích từng thư mục con trong `data/`, quy tắc dọn dẹp |
| [`app/README.md`](./app/README.md) | Docker WebUI tạm thời + Lộ trình frontend chính thức |

---

## 5. Cài Đặt Nhanh (Quick Start)

```bash
# 1. Clone dự án
git clone <repo_url>
cd project-root

# 2. Tạo virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Cài dependency (không pip install lẻ trong code — mọi thứ nằm trong requirements.txt)
pip install -r backend/requirements.txt

# 4. Khai báo API key (chỉ cần cho W&B — hệ thống KHÔNG cần key LLM cloud)
cp .env.example .env
# Mở .env, điền WANDB_API_KEY, QDRANT_API_KEY (nếu dùng Qdrant Cloud thay vì local)

# 5. Kiểm tra hệ thống trước khi chạy thật
python backend/cli_pipeline.py --run_phase 0 --health_check --dummy_test

# 6. Chạy toàn bộ pipeline cho một thư mục video
python backend/cli_pipeline.py --run_phase all --input_dir ./data/raw_videos --batch_size 4
```

Xem chi tiết đầy đủ (từng cờ, từng phase, xử lý lỗi) trong [`backend/README.md`](./backend/README.md).

---

## 6. Trạng Thái Frontend Hiện Tại

Dự án **chưa có frontend chính thức**. Hiện đang dùng tạm một **Docker WebUI** (giao diện chat generic, không tùy biến) để test nhanh khả năng truy vấn của backend trong lúc phát triển.

```bash
cd app
docker compose up -d
```

Đây là giải pháp **tạm thời**, chỉ phục vụ mục đích kiểm thử nội bộ. Chi tiết cấu hình và giới hạn của WebUI tạm, cùng lộ trình xây dựng frontend chính thức (kiến trúc dự kiến, công nghệ, mốc phát triển) nằm đầy đủ trong [`app/README.md`](./app/README.md).

---

## 7. Nguyên Tắc Vận Hành Chung (áp dụng cho mọi phase)

- **Không hardcode secret** — mọi API key qua `python-dotenv`, đọc từ `.env`.
- **Không `pip install` trong code** — mọi dependency khai báo tường minh trong `requirements.txt`.
- **`data/` và `.env` luôn nằm trong `.gitignore`.**
- **Mọi model nặng tuân thủ Tiered Caching**: `Load → xử lý toàn batch → torch.cuda.empty_cache() → Unload`, không nạp xen kẽ nhiều model cùng lúc trên GPU (trừ Qwen 2.5 VL được ghim cố định ở Phase 3).
- **Mọi I/O qua CLI** (`argparse`), output là JSON tinh gọn hoặc CSV — không in log rác ra terminal.

---

## 8. Đóng Góp & Quy Ước Code

- Mỗi Phase là một module độc lập trong `backend/src/`, không import chéo trực tiếp giữa các phase — giao tiếp qua file trạng thái trên đĩa (Qdrant, Kuzu, `eval_queue.jsonl`) để đảm bảo tính resume-able.
- Thay đổi cờ CLI mới phải cập nhật đồng thời: `cli_pipeline.py`, README của phase liên quan, và bảng Ma Trận Cờ Lệnh trong `backend/README.md`.
- Trước khi merge, chạy tối thiểu:
  ```bash
  python backend/cli_pipeline.py --run_phase 0 --health_check --dummy_test
  ```
## 9. Định Hướng Phát Triển

- Kiến Trúc đã được xây dựng tương thích để đóng gói có thể chạy Docker, cũng như việc có thể deploy tạo thành 1 hệ thống CI/CD
- Có thể Sẽ phát triển thêm phần giao diện người dùng trong tương lai nếu như đã có đủ tìm hiểu về mảng
- Có thể tối ưu hơn về các kĩ thuật cũng như các mô hình công nghệ sẽ được áp dụng trong tương lai

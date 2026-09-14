# Backend VIDEO-RAG System (Kiến trúc V8.0 Final Consolidated)

## 1. Tổng Quan

Backend là toàn bộ "bộ não" xử lý của hệ thống — biến video thô thành tri thức có thể truy vấn bằng ngôn ngữ tự nhiên. Kiến trúc tuân thủ nghiêm ngặt:

- **Domain-agnostic**: không hard-code logic riêng cho một lĩnh vực, tự calibrate qua `--save_domain`.
- **Giới hạn phần cứng**: chạy trên GPU 16GB VRAM, peak thực tế luôn giữ dưới 10GB.
- **100% Local**: không gọi API Cloud cho LLM sinh câu trả lời lẫn đánh giá chất lượng (Phase 4 dùng Prometheus-2 local, không dùng Groq/Gemini).
- **Bảo mật**: mọi secret nạp qua `python-dotenv`, không hardcode; `data/` và `.env` luôn trong `.gitignore`.
- **I/O chuẩn hóa**: toàn bộ tương tác qua CLI (`argparse`), trả JSON tinh gọn hoặc CSV, không log rác.

## 2. Cấu Trúc Thư Mục `backend/`

```
backend/
├── cli_pipeline.py           # Entry point duy nhất — điều phối 5 phase
├── requirements.txt          # Toàn bộ dependency (pin version)
├── src/
│   ├── phase0_ops/           # Ops & Foundation
│   ├── phase1_extract/       # Extraction & Chunking
│   ├── phase2_build/         # Vector & Graph Construction
│   ├── phase3_search/        # Search & Generation
│   └── phase4_eval/          # 100% Local Evaluation
└── README.md                 # File này
```

## 3. Công Nghệ Sử Dụng Trong Toàn Hệ Thống

| Nhóm | Công nghệ | Vai trò | Chạy ở |
|---|---|---|---|
| Audio | DeepFilterNet | Tách giọng nói sạch khỏi audio gốc | Phase 1 |
| Audio | Fast Whisper | Speech-to-text + timestamp | Phase 1 |
| Audio | CLAP | Phân loại sự kiện âm thanh môi trường | Phase 1 |
| Vision | YOLOv8 (ONNX) | Lọc metadata thị giác | Phase 1 |
| Vision | RapidOCR / EasyOCR | Trích xuất văn bản trong khung hình | Phase 1 |
| Vision | OpenCV | Tiền xử lý ảnh (CLAHE, Denoise, Lanczos4) | Phase 1 |
| Chunking | PySceneDetect | Phát hiện mốc chuyển cảnh | Phase 1 |
| Embedding | SigLIP 2 | Vector hóa ảnh (keyframe) | Phase 2 |
| Embedding | BGE-M3 | Vector hóa văn bản (dense + sparse/BM25) | Phase 2 |
| Vector DB | Qdrant (local) | Tri-Search: Dense + Sparse + Metadata | Phase 2, 3 |
| Graph DB | Kuzu | Knowledge Graph, hỗ trợ update gia tăng | Phase 2, 3 |
| Routing | DeBERTa | Micro-Router phân loại modal truy vấn | Phase 3 |
| Ranking | Cross-Encoder (rerank) | Chấm lại Top-K | Phase 3, 4 |
| Generation | Qwen 2.5 VL (3B INT4) | Sinh câu trả lời từ ảnh + văn bản | Phase 3 |
| Evaluation | BGE-Reranker + NLI (`nli-deberta-v3-base`) | Fast pre-filter RAG Triad (CPU) | Phase 4 |
| Evaluation | Prometheus-2 (GGUF, `llama-cpp-python`) | Deep Judge cho case biên | Phase 4 |
| Eval Framework | TruLens | Điều phối chấm điểm RAG Triad | Phase 4 |
| MLOps | Weights & Biases | Theo dõi tiến độ, resume session | Phase 0 |
| Config | PyYAML | Lưu/đọc `config/settings.yaml` theo domain | Phase 1, 3 |

## 4. Cài Đặt Môi Trường

```bash
# Từ thư mục gốc dự án
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r backend/requirements.txt
```

**File `.env` cần có (đặt ở gốc dự án):**
```ini
WANDB_API_KEY=your_wandb_key_here
QDRANT_API_KEY=your_qdrant_key_here     # để trống nếu dùng Qdrant local, không cần Cloud
```

> Lưu ý: hệ thống **không cần** bất kỳ key LLM cloud nào (OpenAI/Groq/Gemini) — toàn bộ generation (Qwen 2.5 VL) và evaluation (Prometheus-2) đều chạy local.

## 5. Ma Trận Cờ Lệnh Hệ Thống (CLI Flags) — Đầy Đủ

Hệ thống điều khiển qua `cli_pipeline.py`. Dưới đây là toàn bộ cờ, nhóm theo mục đích sử dụng.

### 5.1. Vận Hành & Điều Phối Phase

| Cờ | Mô tả | Giá trị mặc định |
|---|---|---|
| `--run_phase [0-4/all]` | Chọn phase chạy độc lập hoặc toàn bộ pipeline | bắt buộc |
| `--resume_phase [id]` | Tiếp tục từ phase bị lỗi | — |
| `--input_dir` | Thư mục chứa video đầu vào | `./data/raw_videos` |
| `--batch_size` | Kích thước batch xử lý | 4 |
| `--health_check` | Kiểm tra `pip check` + trạng thái hệ thống | flag |
| `--dummy_test` | Chạy stress test bằng `stress_test.mp4` (10s, đa cảnh, đa giọng) | flag |
| `--wandb_key` | API key Weights & Biases | từ `.env` |
| `--id_pro` | ID project W&B để nối tiếp session | — |

**Ví dụ:**
```bash
# Chạy toàn bộ pipeline (Phase 0 → 4) cho một thư mục video
python cli_pipeline.py --run_phase all --input_dir ./data/raw_videos --batch_size 4

# Chỉ kiểm tra hệ thống trước khi xử lý dữ liệu thật
python cli_pipeline.py --run_phase 0 --health_check --dummy_test
```

### 5.2. Tìm Kiếm & GraphRAG (Phase 3)

| Cờ | Mô tả | Giá trị mặc định |
|---|---|---|
| `--query [text]` | Câu hỏi người dùng | bắt buộc khi chạy Phase 3 |
| `-s / --search_only` | Tắt VLM, chỉ trả bối cảnh thô | flag |
| `-qa / --question_answer` | Dùng sau `-s`, lấy rank 1 làm câu hỏi | flag |
| `-t / --top_k` | Số candidate lấy từ mỗi Collection | 20 |
| `-f / --filter_meta` | Lọc theo điều kiện payload (VD: `speaker=A`) | — |
| `--use_graph` | Bật nhảy cóc qua Kuzu Graph | flag |
| `--hop_limit [int]` | Số bước nhảy (clamp nội bộ 1–5) | 2 |
| `-rr / --rerank` | Chấm lại Top-K bằng Cross-Encoder (CPU) | flag |
| `--max_context_images [int]` | Capacity cho Knapsack DP — số ảnh tối đa nạp vào VLM | 4 |
| `--qdrant_path [path]` | Đường dẫn Qdrant local | `./data/qdrant_db` |
| `--kuzu_graph_path [path]` | Đường dẫn Kuzu Graph local | `./data/kuzu_graph` |

**Ví dụ:**
```bash
# Tìm kiếm dùng Graph (hop=2), giới hạn 4 ảnh gửi vào VLM
python cli_pipeline.py --run_phase 3 \
  --query "Phân tích chiến thuật bù giờ" \
  --use_graph --hop_limit 2 --max_context_images 4

# Chỉ lấy bối cảnh thô (không sinh câu trả lời), xuất kèm truy vết
python cli_pipeline.py --run_phase 3 \
  --qdrant_path ./data/qdrant_db \
  --query "Hiệp 1 có lỗi nào?" -s -tr
```

**Output mẫu (JSON):**
```json
{
  "status": "success",
  "answer": "Ở phút 89, đội chủ nhà thực hiện pressing tầm cao...",
  "trake": [
    {"chunk_id": "c_0042", "timestamp": "01:29:03", "keyframe_path": "data/temp_workspace/kf_0042.jpg"},
    {"chunk_id": "c_0043", "timestamp": "01:29:11", "keyframe_path": "data/temp_workspace/kf_0043.jpg"}
  ]
}
```

### 5.3. Tối Ưu Trọng Số Động (Grid Search)

| Cờ | Mô tả | Giá trị mặc định |
|---|---|---|
| `--tune_weights` | Bật Grid Search 2D cho `(α, β)` | flag |
| `--target_kf [JSON]` | Tối thiểu 5 cặp `[query, keyframe]` để tính Mean MRR | bắt buộc khi tune |
| `--save_domain [tên]` | Lưu `α, β, gamma_matrix` vào `config/settings.yaml` | bắt buộc khi tune |

**Ví dụ:**
```bash
python cli_pipeline.py --run_phase 3 --tune_weights \
  --target_kf '[["cầu thủ ghi bàn", "kf_0012.jpg"], ["thẻ đỏ", "kf_0088.jpg"], ["phạt góc", "kf_0140.jpg"], ["thay người", "kf_0210.jpg"], ["còi kết thúc", "kf_0300.jpg"]]' \
  --save_domain football_analytics
```

### 5.4. Đánh Giá (Phase 4 — Cascade 2 tầng, 100% local)

| Cờ | Mô tả | Giá trị mặc định |
|---|---|---|
| `--eval_sample_rate [float]` | Tỷ lệ mẫu đưa vào Tầng 1 (Fast Pre-Filter) | 0.1 |
| `--deep_eval_threshold_low [float]` | Biên dưới vùng nghi ngờ → đẩy lên Tầng 2 | 0.4 |
| `--deep_eval_threshold_high [float]` | Biên trên vùng nghi ngờ | 0.7 |

**Ví dụ:**
```bash
python cli_pipeline.py --run_phase 4 \
  --eval_sample_rate 0.2 \
  --deep_eval_threshold_low 0.4 --deep_eval_threshold_high 0.7
```

### 5.5. Định Dạng Output

| Cờ | Mô tả |
|---|---|
| `-tr / --trake_mode` | Xuất mảng truy vết (`chunk_id`, `timestamp`, `keyframe_path`) |
| `-outcsv / --export_csv` | Xuất log kết quả dạng CSV |
| `-c / --chat_session` | Lưu/khôi phục ID ngữ cảnh hội thoại |

## 6. Quy Trình Chạy Đầy Đủ Từ Đầu Đến Cuối (End-to-End)

```bash
# Bước 1 — Kiểm tra hệ thống
python cli_pipeline.py --run_phase 0 --health_check --dummy_test

# Bước 2 — Trích xuất dữ liệu từ video thô
python cli_pipeline.py --run_phase 1 --input_dir ./data/raw_videos --batch_size 4

# Bước 3 — Build Vector DB + Graph
python cli_pipeline.py --run_phase 2 --build_graph

# Bước 4 (tùy chọn) — Tinh chỉnh trọng số cho domain cụ thể
python cli_pipeline.py --run_phase 3 --tune_weights \
  --target_kf '[[...]]' --save_domain <ten_domain>

# Bước 5 — Truy vấn
python cli_pipeline.py --run_phase 3 --query "..." --use_graph

# Bước 6 (cuối ngày/cuối batch) — Đánh giá chất lượng
python cli_pipeline.py --run_phase 4 --eval_sample_rate 0.1
```

## 7. Xử Lý Sự Cố Thường Gặp

| Vấn đề | Nguyên nhân khả dĩ | Cách xử lý |
|---|---|---|
| `pip check` fail ở Phase 0 | Xung đột version dependency | Xóa `venv`, cài lại đúng theo `requirements.txt` |
| OOM khi chạy Phase 1/2 | `batch_size` quá lớn so với VRAM | Giảm `--batch_size`, kiểm tra không có model nào bị ghim thừa qua `nvidia-smi` |
| Resume không dọn sạch dữ liệu cũ | `VID_Hash` của video bị trùng do sửa `Middle_1MB` logic | Kiểm tra `commit_status` trong Qdrant, chạy lại `--resume_phase 0` |
| Phase 3 trả về quá ít context | `α, β` chưa được tune cho domain hiện tại | Chạy `--tune_weights` với tối thiểu 5 cặp `target_kf` |
| Phase 4 chạy chậm | Rơi vào Tầng 2 (Prometheus-2) quá nhiều case | Nới `deep_eval_threshold_low/high` để giảm vùng biên, hoặc kiểm tra chất lượng Context Relevance ở Tầng 1 |

# Kiến Trúc Backend RAG Đa Phương Thức — HUB KIẾN THỨC
## Phiên bản V8.0 — Final Consolidated (sau phản biện kỹ thuật đầy đủ)

---

## 0. Ràng Buộc Toàn Cục

| Hạng mục | Yêu cầu |
|---|---|
| Domain | Hoàn toàn domain-agnostic |
| Phần cứng | Giới hạn 16GB VRAM, peak thực tế phải giữ < 10GB |
| Quản lý secret | Toàn bộ API Key (WANDB_API_KEY, QDRANT_API_KEY, Gemini/Groq) nạp qua `python-dotenv`. Không hardcode. |
| Quản lý dependency | Chỉ dùng `requirements.txt`; tuyệt đối không `pip install` trong code |
| Bảo mật file | `data/` (video, temp, qdrant_db) và `.env` nằm trong `.gitignore` |
| I/O | Điều khiển 100% qua `argparse` CLI. Output JSON tinh gọn (`status`, `answer`, `trake`) hoặc CSV. Không log rác. |

**Nguyên tắc bất biến xuyên suốt:** mọi model nặng (SigLIP 2, BGE-M3, DeepFilterNet, Whisper) tuân thủ **Tiered Caching nghiêm ngặt**: `Load → Xử lý toàn bộ batch → torch.cuda.empty_cache() → Unload`, không xen kẽ model, không gọi `empty_cache()` trong inner-loop (tránh memory fragmentation). Chỉ VLM Qwen 2.5 VL (3B INT4 (QLORA)) được ghim cố định trên GPU ở khâu suy luận.

---

## Phase 0 — Foundation & MLOps (Bảo vệ luồng)

### 0.1 Dependency Hard-Exit
- Chạy `subprocess` gọi `pip check`. Nếu phát hiện xung đột thư viện → ngắt hệ thống ngay lập tức, không cho chạy tiếp.

### 0.2 Định danh Tĩnh (Deterministic ID) — Chống Đụng Độ Hash
- **Công thức:** `VID_Hash = MD5(Middle_1MB_of_file + File_Size + Duration)`
- Lấy 1MB **giữa file** (không phải đầu file) để loại trừ trường hợp đụng độ do các video dùng chung template intro/outro giống nhau nhưng nội dung khác nhau.
- Phớt lờ hoàn toàn tên file vật lý.

### 0.3 Transactional Rollback (Tính toàn vẹn giao dịch)
- Mỗi vector upsert vào Qdrant mang payload field `commit_status: pending`.
- Sau khi toàn bộ batch của một `VID_Hash` ghi xong và xác nhận sạch → cập nhật `commit_status: committed`.
- **Nếu sập nguồn:** khi resume, Tracker tự động:
  1. Tra `VID_Hash` đang dở, xóa sạch file rác trong `temp_workspace` của phase đang chạy.
  2. Query Qdrant, xóa toàn bộ vector còn ở trạng thái `pending` của `VID_Hash` đó trước khi upsert lại.
- **Graph không build song song với ingest vector.** Kùzu/NetworkX chỉ được phép đọc và nối Edge **sau khi** toàn bộ vector của `VID_Hash` đã ở trạng thái `committed`. Nếu bước build Graph sập giữa chừng, Qdrant vẫn giữ nguyên dữ liệu sạch — Tracker chỉ cần gọi lại lệnh nối Edge ở lần chạy kế tiếp, không cần rollback vector.

### 0.4 Giám sát W&B & Stress Test Thực Chiến
- Thay dummy-test video rỗng bằng file `stress_test.mp4` (10s, cắt cảnh liên tục, 3 giọng nói chồng lấn, text dày đặc) để ép VRAM chạm đỉnh thực sự (peak load), phát hiện lỗi OOM đáng tin cậy hơn video rỗng 3s.
- Đồng bộ tiến độ ngầm lên Weights & Biases qua `--wandb_key` và `--id_pro` để nối tiếp session.

---

## Phase 1 — Extract & Chunking

> **Lưu ý gốc:** SigLIP 2 không tham gia phase này để tránh thắt cổ chai CPU–GPU.

### 1.1 Audio Pipeline — Kiến trúc "Mù Domain" (Residual Subtraction)
Không dùng cơ chế gating theo phát hiện nhạc nền (Demucs) — thay bằng thuật toán trừ tín hiệu, hoạt động đúng trên mọi domain mà không cần phán đoán trước:

1. **Bước 1 (DeepFilterNet):** Chạy trên toàn bộ audio của batch → xuất `clean_speech.wav` (giọng người đã cô lập).
2. **Bước 2 (Residual — Zero Model):** `Audio_gốc − Audio_clean = Audio_môi_trường` (numpy/scipy) → lưu `background_noise.wav`.
   - **Chuẩn hóa bắt buộc trước khi trừ:** đảm bảo hai tín hiệu cùng sample rate (resample về gốc nếu DeepFilterNet xử lý nội bộ ở 48kHz) và cùng thang giá trị (float32, [-1, 1]) để tránh lệch pha/artifact.
3. **Audio RMS Gate:** Trước khi đưa `background_noise.wav` vào CLAP, tính RMS trên tín hiệu đã normalize. Nếu `RMS < epsilon (1e-4)` → coi là gần như câm, **bỏ qua CLAP** để tránh quét nhầm "musical noise" (artifact giả do STFT sinh ra khi audio gốc gần như thuần giọng nói).
4. **Bước 3 (Giao việc tuần tự, Tiered Caching):**
   `Load Whisper → quét clean_speech.wav (Text + Timestamp) → empty_cache() → Load CLAP → quét background_noise.wav (đã qua RMS gate) → empty_cache()`

### 1.2 Vision — OCR với Ngưỡng Động (γ)
- **YOLOv8 (ONNX):** lọc metadata thị giác.
- **RapidOCR** bóc chữ. Nếu confidence < γ (ngưỡng động, xem 1.3) → tạo bản sao qua OpenCV (CLAHE, Denoising, Lanczos4) → ép quét lại bằng EasyOCR.
- **Cap retry:** tối đa **3 lần** cho vòng lặp OpenCV, tránh nghẽn vô hạn.

### 1.3 Calibration Ngưỡng OCR Động (γ) — Tách biệt khỏi Query-time Tuning
- **Đo độ mờ:** `cv2.Laplacian(frame, cv2.CV_64F).var()` chạy trực tiếp trên ma trận điểm ảnh gốc — **không** cần gọi RapidOCR trước để đo (blur là thuộc tính ảnh, độc lập với OCR).
- **Bảng Calibration theo Domain** (lưu trong `config/settings.yaml`, gắn với `--save_domain`, không hard-code toàn cục):

```yaml
domains:
  football_analytics:
    alpha: 0.85
    beta: 0.10
    gamma_matrix:
      variance_boundary: 100   # Laplacian var < 100 → nhóm "low", >= 100 → nhóm "high"
      low_variance: 70         # Ngưỡng OCR (%) cho frame mờ
      high_variance: 85        # Ngưỡng OCR (%) cho frame nét
      ...
```

- **γ chỉ calibrate một lần ở đầu Phase 1** (quét mẫu 5-10 frame đại diện, tra bảng), không nằm trong vòng lặp `--tune_weights` — vì thay đổi γ kéo theo re-extract toàn bộ corpus (chi phí cực cao), trong khi α, β chỉ cần re-rank trên dữ liệu đã có sẵn trong Qdrant (chi phí thấp).

### 1.4 Hybrid Adaptive Chunking
- Trục thời gian chính = nhịp câu nói của Whisper, đối chiếu mốc chuyển cảnh của PySceneDetect.
- Nếu một câu vắt ngang 2 cảnh → tạo **2 sub-chunk vật lý riêng** (mỗi sub-chunk 1 keyframe .jpg), gắn chung một `parent_chunk_id` trong payload Qdrant.
- **Không dùng Mean Pooling** để gộp vector của 2 keyframe (sẽ làm loãng đặc trưng) — giữ 2 vector độc lập, gộp lại ở tầng retrieval qua `parent_chunk_id`.

---

## Phase 2 — Vector & Graph

### 2.1 Qdrant Tri-Search DB
- Payload bắt buộc gồm: Dense Vector (BGE-M3 text, SigLIP 2 ảnh), Sparse Vector (BM25), Metadata YOLO, `keyframe_path`, `commit_status`, `parent_chunk_id`.
- **Cô lập Domain:** mỗi `domain` là một **Collection Qdrant riêng biệt** (không dùng payload filter chung chạ) — đảm bảo cô lập dữ liệu vật lý và query nhanh hơn.

### 2.2 Strict Sequential Batching (Chống Fragmentation)
```
Load SigLIP 2 → Encode toàn bộ Image Batch → Unload + empty_cache()
        ↓ (hoàn tất 100% trước khi bắt đầu bước sau)
Load BGE-M3 → Encode toàn bộ Text Batch → Unload + empty_cache()
```
- Không nạp xen kẽ 2 model. `empty_cache()` chỉ gọi ở **biên giới chuyển giao model**, không gọi trong inner-loop.

### 2.3 GraphRAG — Kùzu (thay NetworkX in-memory thuần)
- Chuyển từ lưu `.graphml` toàn cục sang CSDL đồ thị nhúng **Kùzu**, cho phép update/append gia tăng (incremental) khi có video mới, không cần rebuild toàn bộ mỗi lần.
- Kích hoạt bằng `--build_graph`. Nodes = Thực thể/Sự kiện; Edges = liên kết Thời gian/Không gian.
- Build **sau khi** vector `commit_status = committed` (xem 0.3), đảm bảo transactional integrity.

---

## Phase 3 — Search & Generate

### 3.1 Micro-Router (CPU — DeBERTa)
- Phân tích câu hỏi, kích hoạt cờ định tuyến modal (visual / audio / caption).
- **Multi-label Fallback:** nếu độ tin cậy < 0.4 cho mọi cờ → tự động bật **cả 3 cờ** (quét diện rộng), chấp nhận dư còn hơn bỏ sót.

### 3.2 Tri-Search RRF — 2 Bậc Tự Do
Công thức chuẩn:

```
S_total = α · S_semantic + β · S_visual + (1 − α − β) · S_keyword
```

- Grid Search quét trên mặt phẳng `(α, β)`, tối ưu **Mean MRR** trên tập `target_kf` (JSON array, tối thiểu **5 cặp** `[query, keyframe]` — tránh overfit vào 1 điểm dữ liệu duy nhất).
- γ (OCR threshold) **không** nằm trong vòng grid search này (xem 1.3) — `--tune_weights` giờ chỉ chạy DP 2 chiều thuần túy trên vector đã có sẵn trong Qdrant, siêu nhẹ.

### 3.3 Graph Multi-hop — Cypher Traversal (Kùzu)
- `--hop_limit` được **clamp nghiêm ngặt** trước khi build query: `safe_hop = max(1, min(int(args.hop_limit), 5))`.
- Truyền qua **Parameterized Query** của thư viện Kùzu (không f-string / nối chuỗi trực tiếp) → triệt tiêu Cypher Injection.
- Query mẫu dạng: `MATCH (a)-[:NEXT_EVENT*1..{safe_hop}]->(b)` — trả về danh sách ID trước khi tra Qdrant lấy ảnh vật lý.

### 3.4 DAG Xử Lý Bất Biến (Chống Nổ Context)
Thứ tự **bắt buộc**, không phụ thuộc số cờ Router kích hoạt:

1. **Quét diện rộng** (Router/Fallback) — chọc vào tối đa 3 Collection (Semantic, Visual, Sparse).
2. **Nhảy cóc** (Graph Hop) — gom thêm node lân cận qua Kùzu Cypher.
3. **Xếp hạng toàn cục** (Global Rank) — gộp toàn bộ candidate hỗn tạp, chấm lại điểm bằng RRF với `(α, β)` đã tune. (An toàn về mặt toán học: RRF là rank-based nên việc pool phình to ở bước 1–2 không phá công thức ở bước 3.)
4. **Cắt gọt** (Cut-off) — Knapsack 0/1 DP, xem 3.5.

### 3.5 Cut-off bằng 0/1 Knapsack (Quy hoạch động — thay Greedy)
- **Capacity:** `W = --max_context_images` (mặc định 4).
- **Item = Logical Chunk:**
  - Chunk đơn: `Weight = 1`, `Value = điểm RRF của chunk đó`.
  - Chunk cặp (vắt cảnh, có partner qua `parent_chunk_id`): `Weight = 2`, `Value = điểm RRF của sub-chunk được match độc lập (Chunk A)` — partner B không có RRF riêng nên không tự đóng góp Value.
- DP với `dp[n+1][W+1]`, truy vết cho ra bộ chunk khít đúng W slot với tổng Value cao nhất — đảm bảo dùng tối đa ngân sách 4 ảnh, không bỏ sót slot như thuật toán greedy-by-rank trước đó.
- Kết quả: VLM Qwen 2.5 VL **không bao giờ** nhận quá 4 ảnh, dù Graph Hop hay Fallback mở rộng candidate pool đến đâu.

### 3.6 VLM Chain-of-Thought
- Móc `keyframe_path` từ kết quả Cut-off, dùng `PIL.Image` bọc `{type: image}` truyền vào Qwen 2.5 VL (đã ghim GPU) để sinh đáp án bám mốc thời gian.

---

## Phase 4 — Kiểm Định Chất Lượng Toàn Diện (100% Local — Cascade 2 Tầng)

> **Thay đổi kiến trúc quan trọng:** loại bỏ hoàn toàn phụ thuộc cloud API (Groq/Gemini) cho LLM-as-judge. Toàn bộ RAG Triad được chấm cục bộ, chi phí 0đ, không còn mâu thuẫn với triết lý "vận hành độc lập 16GB VRAM" đã đặt ra từ đầu.

### 4.0 Tách nhịp Phase 3 / Phase 4 (Nền tảng cho Cascade)
- **Nhịp 1 (Phase 3, runtime):** Qwen 2.5 VL ghim VRAM để trả lời truy vấn như bình thường. Thay vì chấm điểm ngay, hệ thống log bộ ba `[Query, Context, Answer]` vào `eval_queue.jsonl` trên đĩa.
- **Nhịp 2 (Phase 4, batch — kích hoạt qua `--run_phase 4`, thường chạy cuối ngày/cuối batch):** xử lý toàn bộ `eval_queue.jsonl` qua 2 tầng bên dưới.

### 4.1 Tầng 1 — Fast Pre-Filter bằng Cross-Encoder (CPU, chạy trên 100% sample đã lọc theo `--eval_sample_rate`)

Dùng **2 model CPU riêng biệt**, không dùng chung 1 model cho 2 nhiệm vụ khác bản chất:

| Tiêu chí RAG Triad | Model | Bài toán |
|---|---|---|
| **Context Relevance** | BGE-Reranker (tái sử dụng từ `-rr/--rerank`) | Relevance ranking: `[Query, Context]` → score |
| **Groundedness** | `cross-encoder/nli-deberta-v3-base` (model NLI riêng) | Natural Language Inference: `[Context, Answer]` → Entailment / Neutral / Contradiction |

> **Lưu ý kỹ thuật:** BGE-Reranker **không** phải model NLI, không thể dùng để suy luận Entailment. Groundedness bắt buộc phải qua một model NLI chuyên biệt, dù cũng chạy CPU và cũng miễn phí — đây là 2 model độc lập chạy song song, không phải "dùng lại một model cho cả hai".

**Luồng quyết định:**
```
score_relevance = BGE-Reranker(query, context)
score_ground    = NLI(context, answer)   # xác suất nhãn "Entailment"

NẾU score_relevance > 0.7  VÀ  score_ground > 0.7:
    → Ghi PASS trực tiếp vào eval_report.jsonl (không cần chấm sâu)
NẾU 0.4 ≤ score bất kỳ ≤ 0.7  (vùng biên, không chắc chắn)
    → Đẩy sang eval_queue_deep.jsonl (chờ Tầng 2)
NẾU score < 0.4:
    → Ghi FAIL trực tiếp (đủ rõ ràng để không cần Judge sâu)
```

- **Answer Relevance** (tiêu chí thứ 3 của RAG Triad) được xử lý cùng cơ chế: BGE-Reranker chấm `[Query, Answer]`.
- Toàn bộ Tầng 1 chạy trên CPU, không tốn VRAM, không ảnh hưởng đến Qwen 2.5 VL đang ghim GPU nếu Phase 4 chạy chồng lấn thời gian với truy vấn thực tế.

### 4.2 Tầng 2 — Deep Judge bằng Prometheus-2 (VRAM Swapping, chỉ chấm phần nghi ngờ)

Chỉ kích hoạt cho các case rơi vào `eval_queue_deep.jsonl` (vùng biên của Tầng 1) — **không phải toàn bộ tập sample**, giảm mạnh số lần phải load/unload model nặng.

**Quy trình gỡ Qwen — nạp Judge:**
```python
# 1. Gỡ Qwen 2.5 VL khỏi VRAM (empty_cache() không đủ — phải xóa tham chiếu)
del qwen_model
gc.collect()
torch.cuda.empty_cache()

# 2. Nạp Prometheus-2 (7B/8B GGUF 4-bit) qua llama-cpp-python
judge_model = load_prometheus2_gguf()

# 3. TruLens gọi thẳng judge_model (local) để chấm RAG Triad
#    cho từng bản ghi trong eval_queue_deep.jsonl

# 4. Chấm xong toàn bộ deep queue → giải phóng VRAM
del judge_model
gc.collect()
torch.cuda.empty_cache()
```

- Vì Phase 4 chạy như một batch riêng biệt cuối ngày (không xen kẽ liên tục với Phase 3), chi phí swap I/O (đọc model GGUF từ đĩa) chỉ phát sinh 1 lần/batch, không cộng dồn.
- Prometheus-2 là model chuyên huấn luyện cho đánh giá RAG, độ chính xác ngang GPT-4-as-judge cho các tiêu chí Groundedness/Relevance — phù hợp đúng cho các case biên mà Cross-Encoder không đủ khả năng suy luận chuỗi (chain-of-thought) để phân xử.

### 4.3 Sampling (giữ nguyên, áp dụng cho Tầng 1)
- Cờ `--eval_sample_rate 0.1` — chỉ đưa 10% chunk ngẫu nhiên vào toàn bộ luồng Cascade (Tầng 1 + Tầng 2 nếu rơi vào vùng biên), giữ tính đại diện thống kê.

### 4.4 Cờ điều khiển ngưỡng Cascade
- `--deep_eval_threshold_low 0.4` / `--deep_eval_threshold_high 0.7` — biên dưới/trên định nghĩa "vùng nghi ngờ" cần đẩy lên Tầng 2. Có thể tune theo domain nếu cần độ nhạy khác nhau.

> **Kết quả:** Circuit Breaker và `ProviderErrorHandler` (xử lý lỗi 429/quota cloud API) của thiết kế trước đây **không còn cần thiết** — vì Phase 4 giờ hoàn toàn local, không còn phụ thuộc rate-limit hay quota của Groq/Gemini. Toàn bộ rủi ro "hết quota giữa batch" bị loại bỏ tận gốc.

---

## Ma Trận Cờ Lệnh Hệ Thống (CLI Feature Flags)

**Vận hành & Điều phối Phase**
```
--run_phase [0-4/all]        --resume_phase [id]
--input_dir                  --batch_size
--health_check                --dummy_test
--wandb_key                  --id_pro
--build_graph                 --kuzu_graph_path [path]
--qdrant_path [path] (truyền file vào Qdrant local ex: --run_phase 3 --db_path my_dataset/qdrant_db --query "...")
--meta_path [path] (truyền file vào metadata local ex: --run_phase 3 --meta_path my_dataset/metadata.json --query "...") (nếu có)
```

**Tìm kiếm & Nhảy cóc (GraphRAG)**
```
--query [text]
-s  / --search_only        (tắt VLM, chỉ lấy bối cảnh thô)
-qa / --question_answer    (dùng sau -s, lấy rank 1 làm câu hỏi)
-t  / --top_k
-f  / --filter_meta [payload_condition]
--use_graph
--hop_limit [int]           (clamp nội bộ: 1–5)
-rr / --rerank              (Cross-Encoder CPU chấm lại Top K)
--max_context_images [int]  (mặc định: 4 — capacity cho Knapsack DP)
```

**Tối ưu Trọng số Động**
```
--tune_weights
--target_kf [JSON array, ≥5 cặp [query, keyframe]]
--save_domain [tên_miền]     (lưu α, β, gamma_matrix vào config/settings.yaml)
```

**Đánh giá (Phase 4 — Cascade 2 tầng, 100% local)**
```
--eval_sample_rate [float]          (mặc định 0.1)
--deep_eval_threshold_low [float]   (mặc định 0.4 — biên dưới vùng nghi ngờ)
--deep_eval_threshold_high [float]  (mặc định 0.7 — biên trên vùng nghi ngờ)
```

**Giao thức Dữ liệu Đầu Ra**
```
-tr    / --trake_mode        (mảng truy vết: chunk_id, timestamp, keyframe_path)
-outcsv/ --export_csv
-c     / --chat_session
```

---

## Sơ Đồ Cấu Trúc `config/settings.yaml`

```yaml
domains:
  <ten_domain>:
    alpha: <float>              # trọng số semantic (query-time, tune bằng grid search)
    beta: <float>                # trọng số visual  (query-time)
    # keyword weight = 1 - alpha - beta (ngầm định)
    gamma_matrix:                # extraction-time, calibrate 1 lần
      variance_boundary: <float> # ngưỡng Laplacian variance phân "low"/"high"
      low_variance: <float>      # % ngưỡng OCR cho frame mờ
      high_variance: <float>     # % ngưỡng OCR cho frame nét
```

---

## Tóm Tắt Các Quyết Định Kiến Trúc Then Chốt

| # | Vấn đề gốc | Giải pháp chốt |
|---|---|---|
| 1 | Dummy test không đại diện tải thực | `stress_test.mp4` 10s đa giọng, đa cảnh |
| 2 | Hash đụng độ do intro/outro giống nhau | MD5 lấy 1MB **giữa** file |
| 3 | Rollback không xử lý vector nửa vời | `commit_status: pending/committed` + Graph build sau commit |
| 4 | Audio nhị phân triệt tiêu lẫn nhau | Residual Subtraction (DeepFilterNet + trừ tín hiệu) — mù domain |
| 5 | Musical noise giả khi audio gần như câm | RMS Gate (ε = 1e-4) trước khi vào CLAP |
| 6 | OCR ngưỡng cứng 85% không phù hợp mọi domain | γ động, calibrate qua Laplacian variance + bảng domain-aware |
| 7 | Mean Pooling làm loãng vector chunk vắt cảnh | Sub-chunk riêng + `parent_chunk_id` |
| 8 | Fragmentation do gọi empty_cache() tùy tiện | Strict Sequential Batching, cache-clear chỉ ở biên giới model |
| 9 | `.graphml` không scale, không incremental | Kùzu graph DB, update gia tăng |
| 10 | RRF 1 biến không đủ cho 3 không gian | `S = α·S_sem + β·S_vis + (1-α-β)·S_kw`, grid search 2D |
| 11 | Overfit khi tune trên 1 target_kf | Tối thiểu 5 cặp, tối ưu Mean MRR |
| 12 | Cypher Injection qua `--hop_limit` | Clamp `int` [1,5] + Parameterized Query |
| 13 | Context nổ khi Graph hop nhiều | Cap cứng `--max_context_images`, DAG bất biến |
| 14 | Greedy cut-off bỏ sót slot tối ưu | 0/1 Knapsack DP đúng chuẩn |
| 15 | Eval phụ thuộc cloud mâu thuẫn "cục bộ" | **Cascade 2 tầng 100% local**: Cross-Encoder pre-filter (CPU) → Prometheus-2 deep judge (VRAM Swap, chỉ case biên) |
| 16 | Chi phí API + rủi ro hết quota giữa batch | Loại bỏ hoàn toàn — không còn cloud call nên không còn rủi ro 429/quota |
| 17 | BGE-Reranker không phải model NLI | Groundedness dùng model NLI riêng (`cross-encoder/nli-deberta-v3-base`), tách biệt khỏi Reranker |

---

*Tài liệu này là bản tổng hợp cuối cùng sau các vòng phản biện kỹ thuật, sẵn sàng làm cơ sở để triển khai code theo từng Phase. Có thể sẽ cân nhắc thay RapidOCR thành DocTR model, chuyển siglip2 base sang siglip2 400m*

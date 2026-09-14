## 2. PHASE 4 — Đánh Giá Chất Lượng (Eval Cascade 2 Tầng, 100% Local)

### 2.1 Ý tưởng cốt lõi

Thay vì chấm điểm chất lượng câu trả lời ngay khi người dùng hỏi (tốn thời gian, tốn VRAM), hệ thống tách làm **2 nhịp**:

- **Nhịp 1 (lúc chạy Phase 3 — real-time):** mỗi lần trả lời xong, hệ thống chỉ **ghi log** bộ ba `[Câu hỏi, Ngữ cảnh, Câu trả lời]` vào file `eval_queue.jsonl`, không chấm điểm ngay.
- **Nhịp 2 (Phase 4 — chạy batch, ví dụ cuối ngày):** đọc toàn bộ `eval_queue.jsonl` và chấm điểm hàng loạt qua cơ chế **Cascade 2 tầng**.

Ưu điểm: không làm chậm trải nghiệm chat, và tận dụng được việc gom nhiều câu hỏi lại để chấm cùng lúc, tiết kiệm chi phí tải model.

### 2.2 Tầng 1 — Fast Pre-Filter (chạy trên CPU, không tốn VRAM)

Dùng 2 model nhỏ, chuyên biệt hoá, **khác nhiệm vụ nhau**:

| Tiêu chí (RAG Triad) | Model | Đo cái gì |
|---|---|---|
| **Context Relevance** | `BAAI/bge-reranker-base` | Ngữ cảnh lấy về có liên quan tới câu hỏi không |
| **Groundedness** | `cross-encoder/nli-deberta-v3-base` (NLI) | Câu trả lời có thực sự "bắt nguồn" từ ngữ cảnh, hay bịa (hallucination) |
| **Answer Relevance** | `BAAI/bge-reranker-base` (dùng lại) | Câu trả lời có đúng trọng tâm câu hỏi không |

**Luồng quyết định:**

```
điểm thấp nhất trong 3 tiêu chí

  > 0.7 (cả relevance & groundedness)  →  PASS   → eval_report.jsonl
  0.4 – 0.7 (bất kỳ tiêu chí nào)      →  NGHI NGỜ → eval_queue_deep.jsonl (chờ Tầng 2)
  < 0.4                                →  FAIL   → eval_report.jsonl
```

Nhờ vậy, phần lớn câu trả lời (rõ đúng hoặc rõ sai) được xử lý xong ngay ở Tầng 1, rẻ và nhanh.

### 2.3 Tầng 2 — Deep Judge bằng Prometheus-2 (chỉ chấm case biên)

Chỉ áp dụng cho các câu hỏi rơi vào vùng "nghi ngờ" (`eval_queue_deep.jsonl`) — thường chỉ là một phần nhỏ trong tổng số.

```
1. Gỡ model Qwen2.5-VL khỏi VRAM (nếu đang ghim)  → gc.collect() → torch.cuda.empty_cache()
2. Nạp Prometheus-2 (7B, GGUF 4-bit) qua llama-cpp-python
3. Chấm điểm 1–5 cho từng case theo rubric Groundedness + Relevance
   (điểm ≥ 4 → pass, ngược lại → fail)
4. Chấm xong toàn bộ hàng đợi → gỡ Prometheus-2, giải phóng VRAM
```

Kỹ thuật này gọi là **VRAM Swapping**: máy chỉ có 1 GPU 16GB nên không thể giữ cả Qwen2.5-VL (dùng để trả lời) và Prometheus-2 (dùng để chấm điểm) cùng lúc — phải tráo đổi model ra/vào VRAM theo từng giai đoạn.

### 2.4 Đối chiếu Ground Truth (tuỳ chọn)

Nếu có file "đáp án chuẩn" (`ground_truth.json`) dạng:

```json
[
  {"query": "Ai ghi bàn thắng quyết định?", "expected_keyframe": "frames/scene12_003.jpg", "expected_answer": "Cầu thủ số 10 ghi bàn ở phút 89"}
]
```

hệ thống sẽ tự tính:
- **Keyframe Hit-Rate**: tỉ lệ tìm đúng đúng khung hình mong đợi
- **Mean Reciprocal Rank (MRR)**: khung hình đúng đứng ở hạng bao nhiêu trong kết quả
- **Answer Overlap Ratio**: độ trùng từ vựng giữa câu trả lời sinh ra và đáp án mẫu

### 2.5 Cách chạy Phase 4

```bash
# Chấm cơ bản, chỉ Tầng 1, lấy toàn bộ log đã tích luỹ ở Phase 3
python cli_pipeline.py --run_phase 4

# Chấm 20% mẫu, bật cả Tầng 2 (Prometheus-2), và đối chiếu ground truth
python cli_pipeline.py --run_phase 4 \
  --eval_sample_rate 0.2 \
  --deep_eval_threshold_low 0.3 \
  --deep_eval_threshold_high 0.8 \
  --prometheus_gguf_path data/models/prometheus2-7b.Q4_K_M.gguf \
  --ground_truth_path data/eval_logs/ground_truth.json
```

**Giải thích cờ:**

| Cờ | Ý nghĩa |
|---|---|
| `--eval_sample_rate 0.1` | Chỉ lấy ngẫu nhiên 10% số câu hỏi đã log để chấm (đủ đại diện thống kê, đỡ tốn thời gian) |
| `--deep_eval_threshold_low / _high` | Chỉnh biên dưới/trên của "vùng nghi ngờ" cần đẩy lên Tầng 2 |
| `--prometheus_gguf_path` | Đường dẫn tới file GGUF của Prometheus-2 — nếu bỏ trống, hệ thống **chỉ chạy Tầng 1** |
| `--ground_truth_path` | File JSON đáp án chuẩn để tính Hit-Rate / MRR |

**Đầu ra ví dụ (in ra console dạng JSON):**

```json
{
  "tier1_stats": {"pass": 34, "fail": 5, "deep_queue": 11},
  "sampled": 50,
  "total_logged": 500,
  "tier2_stats": {"pass": 7, "fail": 4},
  "ground_truth_metrics": {
    "matched_queries": 20,
    "keyframe_hit_rate": 0.85,
    "mean_reciprocal_rank": 0.71,
    "mean_answer_overlap": 0.63
  }
}
```

### 2.6 Tính năng liên quan: Tự động tối ưu trọng số (`--tune_weights`)

Cùng nằm trong module `phase4_eval` (file `hyper_tuner.py`), tính năng này giúp tìm bộ trọng số **alpha (semantic) / beta (visual)** tối ưu cho tri-search ở Phase 3, bằng **Grid Search 2 chiều** tối đa hoá **Mean Reciprocal Rank**.

```bash
python cli_pipeline.py --tune_weights \
  --qdrant_path data/qdrant_db \
  --domain football_analytics \
  --target_kf data/eval_logs/target_kf.json \
  --save_domain football_analytics
```

Trong đó `target_kf.json` cần **tối thiểu 5 cặp** `[câu hỏi, đường dẫn keyframe đúng]` để tránh overfit vào 1 điểm dữ liệu. Kết quả `alpha`, `beta` tối ưu sẽ được **ghi thẳng vào `config/settings.yaml`** dưới đúng domain tương ứng, không đụng vào `gamma_matrix` (tham số calibrate OCR riêng ở Phase 1).

> Lưu ý về hiệu năng: toàn bộ vòng lặp thử nghiệm (α, β) chỉ là **phép cộng số học thuần** trên dữ liệu đã cache sẵn từ 1 lần truy vấn Qdrant duy nhất mỗi câu hỏi — **không** query lại vector DB hay dùng GPU cho mỗi cặp trọng số thử nghiệm, nên chạy rất nhanh dù quét cả trăm tổ hợp.

---

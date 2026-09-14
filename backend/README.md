# # Backend - MultiLayer-VIDEORAG-SYSTEM (Kiến Trúc V8.0 Final Consolidated)

## 1. Tổng Quan (Overview)
Backend của hệ thống MultiLayer-VIDEORAG-SYSTEM được xây dựng theo kiến trúc V8.0 Final Consolidated [cite: 2]. Điểm cốt lõi của kiến trúc này là hoàn toàn domain-agnostic và giới hạn nghiêm ngặt VRAM dưới 10GB trên phần cứng 16GB VRAM, thực thi 100% local (không gọi API Cloud cho LLM/Eval) [cite: 2]. Hệ thống quản lý dependency qua `requirements.txt`, tuyệt đối không cài đặt ngầm trong code và ẩn mọi API key qua `python-dotenv` [cite: 2].

## 2. Cấu Trúc Hệ Thống Các Phase
Hệ thống được chia thành 5 giai đoạn (Phase 0 đến Phase 4), mỗi phase có thể chạy độc lập hoặc chạy nối tiếp. Mọi luồng I/O được điều khiển 100% qua CLI (`argparse`) [cite: 2]. Output trả về JSON tinh gọn hoặc CSV để tránh sinh log rác [cite: 2].

## 3. Hướng Dẫn Sử Dụng & Ví Dụ Chạy Cờ (CLI Flags)
Hệ thống tương tác qua `cli_pipeline.py` bằng ma trận cờ lệnh hệ thống [cite: 2]. Dưới đây là cách sử dụng chi tiết và các ví dụ thực tế.

### 3.1. Vận hành & Điều phối Phase
* `--run_phase [0-4/all]`: Lựa chọn phase để chạy độc lập hoặc toàn bộ pipeline [cite: 2].
* `--resume_phase [id]`: Chạy tiếp từ phase bị lỗi dựa trên ID [cite: 2].
* `--input_dir`: Thư mục chứa video đầu vào [cite: 2].
* `--batch_size`: Kích thước batch xử lý [cite: 2].
* `--health_check`, `--dummy_test`: Kiểm tra hệ thống. Nên dùng kèm `stress_test.mp4` 10s để ép tải VRAM thay vì video rỗng [cite: 2].
* `--wandb_key`, `--id_pro`: Khóa Weights & Biases để đồng bộ ngầm [cite: 2].

**Ví dụ:**
```bash
# Chạy toàn bộ pipeline cho một thư mục video
python cli_pipeline.py --run_phase all --input_dir ./data/raw_videos --batch_size 4

# Chạy kiểm tra hệ thống với stress test
python cli_pipeline.py --run_phase 0 --health_check --dummy_test
```

### 3.2. Tìm Kiếm & GraphRAG (Thường dùng ở Phase 3)
* `--query [text]`: Câu hỏi của người dùng [cite: 2].
* `-s / --search_only`: Tắt sinh văn bản (VLM), chỉ lấy bối cảnh thô [cite: 2].
* `-qa / --question_answer`: Dùng sau `-s`, dùng rank 1 làm câu hỏi [cite: 2].
* `--use_graph`, `--hop_limit [int]`: Bật duyệt đồ thị đa trạm, `hop_limit` được giới hạn từ 1-5 trạm [cite: 2].
* `-rr / --rerank`: Chấm lại Top K bằng model Cross-Encoder chạy trên CPU [cite: 2].
* `--max_context_images [int]`: Sức chứa cho thuật toán 0/1 Knapsack DP (mặc định 4) để giới hạn ảnh nạp vào VLM [cite: 2].
* Đường dẫn dữ liệu (local): `--build_graph`, `--kuzu_graph_path [path]`, `--qdrant_path [path]`, `--meta_path [path]` [cite: 2].

**Ví dụ:**
```bash
# Tìm kiếm dùng đồ thị (hop=2) và giới hạn 4 ảnh gửi vào VLM
python cli_pipeline.py --run_phase 3 --query "Phân tích chiến thuật bù giờ" --use_graph --hop_limit 2 --max_context_images 4

# Chỉ tìm kiếm bối cảnh, không dùng VLM, sử dụng CSDL local
python cli_pipeline.py --run_phase 3 --qdrant_path ./my_db --query "Hiệp 1 có lỗi nào?" -s
```

### 3.3. Tối ưu Trọng Số & Đánh Giá (Thường dùng ở Phase 4)
* `--tune_weights`: Bật tính năng tối ưu trọng số tự động bằng Grid Search 2D [cite: 2].
* `--target_kf [JSON array]`: Cung cấp tối thiểu 5 cặp [query, keyframe] để tính toán RRF chống overfit [cite: 2].
* `--save_domain [tên_miền]`: Lưu các trọng số $\alpha, \beta, \gamma$ vào file `config/settings.yaml` [cite: 2].
* Đánh giá tự động: `--eval_sample_rate [float]` (mặc định 0.1), `--deep_eval_threshold_low [float]` (mặc định 0.4), `--deep_eval_threshold_high [float]` (mặc định 0.7) [cite: 2].

**Ví dụ:**
```bash
# Chạy đánh giá 100% local với tỷ lệ mẫu 20%
python cli_pipeline.py --run_phase 4 --eval_sample_rate 0.2 --deep_eval_threshold_low 0.4 --deep_eval_threshold_high 0.7
```

### 3.4. Định dạng Output
* `-tr / --trake_mode`: Xuất output dạng mảng truy vết (chunk_id, timestamp, keyframe_path) [cite: 2].
* `-outcsv / --export_csv`: Xuất định dạng CSV [cite: 2].
* `-c / --chat_session`: Bật luồng chat session [cite: 2].

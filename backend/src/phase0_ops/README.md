# Phase 0 — Foundation & MLOps

## 1. Mục Tiêu

Phase 0 là "người gác cổng" của toàn hệ thống — chạy **trước mọi phase khác**, đảm bảo:
- Môi trường sạch (không xung đột dependency).
- Mỗi video có một định danh duy nhất, ổn định, không phụ thuộc tên file.
- Nếu hệ thống sập nguồn giữa chừng ở bất kỳ phase nào, lần chạy sau có thể tự dọn dẹp và tiếp tục an toàn (không tạo dữ liệu rác hoặc trùng lặp).

**Input:** danh sách video trong `--input_dir` (chưa xử lý gì).
**Output:** `VID_Hash` cho mỗi video, trạng thái hệ thống (`healthy` / `conflict`), log tiến độ trên W&B.

## 2. Sơ Đồ Luồng Xử Lý

```
input_dir/*.mp4
      │
      ▼
┌────────────────────────┐     conflict     ┌──────────────┐
│ pip check              │ ────────────────▶ HARD EXIT    
│ (Dependency Hard-Exit) │                  └──────────────┘
└──────────┬─────────────┘
           │ ok
           ▼
┌──────────────────────────────┐
│ VID_Hash = MD5(              │
│   Middle_1MB(file) +         │
│   File_Size + Duration)      │
└──────────┬───────────────────┘
           ▼
┌───────────────────────────────┐      pending còn sót       ┌────────────────────────┐
│ Kiểm tra Dual-State Tracker   │ ─────────────────────────▶   Xóa vector "pending"   
│ (có phiên chạy dở không?)     │                            │ trong Qdrant + dọn     │
└──────────┬────────────────────┘                            │ temp_workspace         │
           │ sạch                                            └────────────────────────┘
           ▼
┌──────────────────────────────┐
│ (tùy chọn) Stress Test       │
│ stress_test.mp4 (10s)        │
│ → đo peak VRAM thực tế       │
└──────────┬───────────────────┘
           ▼
     Sẵn sàng cho Phase 1
```

## 3. Công Nghệ & Thuật Toán Áp Dụng

| Thành phần | Công nghệ | Chi tiết |
|---|---|---|
| Dependency check | `subprocess` + `pip check` | Nếu phát hiện xung đột thư viện → ngắt hệ thống ngay (Hard-Exit), không cho chạy tiếp |
| Định danh video | MD5 tùy biến | `MD5(Middle_1MB_of_file + File_Size + Duration)` — lấy 1MB **giữa file** (không phải đầu file) để tránh đụng độ giữa các video dùng chung intro/outro nhưng nội dung khác nhau; phớt lờ hoàn toàn tên file |
| Transactional Rollback | Payload field `commit_status` | Vector Qdrant mang trạng thái `pending` → `committed`; nếu sập giữa chừng, resume sẽ xóa sạch phần `pending` của `VID_Hash` đó trước khi upsert lại |
| MLOps tracking | Weights & Biases | Đồng bộ tiến độ ngầm qua `--wandb_key` và `--id_pro`, cho phép nối tiếp session giữa các lần chạy |
| Stress Test | `stress_test.mp4` (10s) | Video có cắt cảnh liên tục + 3 giọng nói chồng lấn + text dày đặc — mô phỏng tải đỉnh (peak load) thực tế, đáng tin cậy hơn nhiều so với video rỗng |

> **Nguyên tắc quan trọng:** Graph (Kuzu) **không** được build song song với việc ingest vector. Nếu build Graph sập giữa chừng, Qdrant vẫn giữ nguyên dữ liệu sạch (đã `committed`) — Tracker chỉ cần gọi lại lệnh nối Edge ở lần chạy sau, không cần rollback vector.

## 4. Cách Chạy Độc Lập

```bash
# Kiểm tra xung đột dependency + chạy stress test ép tải VRAM
python cli_pipeline.py --run_phase 0 --health_check --dummy_test

# Bật đồng bộ tiến độ lên W&B
python cli_pipeline.py --run_phase 0 --wandb_key "YOUR_KEY" --id_pro "Project_A"

# Chạy Phase 0 cho một thư mục video cụ thể (chỉ cấp Hash ID, không xử lý)
python cli_pipeline.py --run_phase 0 --input_dir ./data/raw_videos
```

**Output mẫu (JSON):**
```json
{
  "status": "healthy",
  "dependency_check": "passed",
  "vram_peak_stress_test_mb": 8214,
  "videos_hashed": 12,
  "rollback_actions": []
}
```

## 5. Lưu Ý Vận Hành

- Nếu `dependency_check` báo `conflict`, hệ thống dừng ngay — **không** tự động sửa `requirements.txt`, cần người vận hành kiểm tra thủ công và cài lại đúng version.
- `--dummy_test` nên chạy lại mỗi khi thay đổi cấu hình model (đổi độ lớn batch, đổi GPU) để xác nhận VRAM peak vẫn nằm dưới ngưỡng 10GB.
- `VID_Hash` là khóa xuyên suốt toàn hệ thống — nếu đổi công thức hash, **toàn bộ dữ liệu cũ trong Qdrant/Kuzu sẽ không còn liên kết được** với video gốc theo cách cũ, cần có kế hoạch migrate riêng.

# Phase 0: Foundation & MLOps

## 1. Tổng Quan (Overview)
Phase 0 đóng vai trò bảo vệ hệ thống trước khi bắt đầu xử lý khối dữ liệu lớn. Phase này đảm nhiệm việc cấp mã băm định danh tĩnh (Deterministic ID) để chống đụng độ giữa các video có cùng intro/outro, kiểm tra xung đột thư viện, và thiết lập cơ chế tính toàn vẹn giao dịch (Transactional Rollback) để dọn rác tự động nếu hệ thống bị sập giữa chừng [cite: 2].

## 2. Công Nghệ & Thuật Toán Áp Dụng
* **Thuật toán Hash ID:** MD5 lấy 1MB ở **giữa file** cộng với kích thước và thời lượng, phớt lờ tên file gốc để định danh chính xác [cite: 2].
* **Quản lý MLOps:** Weights & Biases (W&B) để theo dõi tiến độ ngầm [cite: 2].
* **Dependency Check:** Thư viện hệ thống Python (`subprocess` gọi `pip check`) dùng làm Dependency Hard-Exit [cite: 2].
* **Stress Test VRAM:** Kiểm tra bằng video cấu trúc phức tạp `stress_test.mp4` dài 10s (chứa đa giọng nói, đa cảnh) để kích hoạt tải peak thực tế [cite: 2].

## 3. Cách Chạy Độc Lập
Phase 0 thường được kích hoạt đầu tiên để dọn dẹp không gian tạm (`temp_workspace`) và kiểm tra máy chủ.
```bash
# Kiểm tra xung đột và chạy stress test ép tải VRAM
python cli_pipeline.py --run_phase 0 --health_check --dummy_test

# Bật đồng bộ W&B
python cli_pipeline.py --run_phase 0 --wandb_key "YOUR_KEY" --id_pro "Project_A"
```

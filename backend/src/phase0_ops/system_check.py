"""
PHASE 0 — System Check (spec V8.0 §0.1)

Chạy trước toàn bộ pipeline để đảm bảo:
1. FFmpeg & FFprobe đã cài đặt (sống còn cho Phase 0 Hashing & Phase 1 Audio).
2. CUDA / PyTorch khả dụng.
3. Các thư viện core (qdrant_client, kuzu) có thể import.
"""

import subprocess
import sys
import logging

logger = logging.getLogger(__name__)

def check_command_exists(cmd: str) -> bool:
    try:
        # Tương thích cả Windows (where) và Linux (which)
        check_cmd = "where" if sys.platform == "win32" else "which"
        subprocess.run([check_cmd, cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def run_system_check() -> None:
    """Thực hiện chuỗi kiểm tra hệ thống. Văng lỗi (sys.exit) nếu thiếu dependency cốt lõi."""
    logger.info("=== Bắt đầu kiểm tra hệ thống (Phase 0.1) ===")
    
    # 1. Kiểm tra FFmpeg & FFprobe
    if not check_command_exists("ffmpeg"):
        logger.error("LỖI CHÍ MẠNG: Không tìm thấy 'ffmpeg' trong hệ thống. Vui lòng cài đặt FFmpeg và thêm vào PATH.")
        sys.exit(1)
    if not check_command_exists("ffprobe"):
        logger.error("LỖI CHÍ MẠNG: Không tìm thấy 'ffprobe' trong hệ thống.")
        sys.exit(1)
    logger.info("✅ FFmpeg & FFprobe hợp lệ.")

    # 2. Kiểm tra PyTorch & CUDA
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✅ PyTorch nhận diện GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("⚠️ CẢNH BÁO: PyTorch không nhận diện được CUDA. Hệ thống sẽ chạy trên CPU (Rất chậm).")
    except ImportError:
        logger.error("LỖI CHÍ MẠNG: Thư viện 'torch' chưa được cài đặt.")
        sys.exit(1)

    # 3. Kiểm tra DB Clients
    try:
        import qdrant_client
        logger.info("✅ Thư viện qdrant-client hợp lệ.")
    except ImportError:
        logger.error("LỖI CHÍ MẠNG: Thiếu thư viện 'qdrant-client'.")
        sys.exit(1)
        
    try:
        import kuzu
        logger.info("✅ Thư viện kuzu hợp lệ.")
    except ImportError:
        logger.error("LỖI CHÍ MẠNG: Thiếu thư viện 'kuzu'.")
        sys.exit(1)

    logger.info("=== Hệ thống ĐẠT chuẩn. Sẵn sàng khởi chạy Pipeline. ===")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_system_check()


# src/phase0_ops/hashing.py
"""
PHASE 0 — Định danh Tĩnh (Deterministic ID)
Công thức (spec V8.0 §0.2):
    VID_Hash = MD5( Middle_1MB_of_file_bytes + File_Size + Duration )

Lý do lấy 1MB ở GIỮA file (không phải đầu file): nhiều video dùng chung
intro/outro template giống hệt nhau -> nếu băm ở đầu file sẽ đụng độ hash
dù nội dung thực tế khác nhau. Vùng giữa file có xác suất trùng lặp gần
như bằng 0 vì đó là phần nội dung "thân bài" duy nhất cho từng video.

Tên file vật lý bị phớt lờ hoàn toàn — hash chỉ phụ thuộc vào byte thực
+ metadata vật lý (size, duration), đảm bảo tính bất biến khi người dùng
đổi tên / di chuyển file.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import json
from pathlib import Path
from typing import Optional

CHUNK_SIZE = 8 * 1024  # 8KB — đọc file theo chunk để tránh nạp toàn bộ video vào RAM
MIDDLE_WINDOW_BYTES = 1 * 1024 * 1024  # 1MB vùng giữa file


def _get_duration_seconds(file_path: str) -> float:
    """
    Lấy duration (giây) của video bằng ffprobe (đi kèm ffmpeg).
    Không dùng OpenCV VideoCapture vì frame_count/fps của một số codec
    (VFR) cho ra duration sai lệch; ffprobe đọc trực tiếp container metadata,
    chính xác hơn cho mục đích băm.
    """
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            file_path,
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=30, check=True,
        )
        payload = json.loads(result.stdout.decode("utf-8"))
        duration = float(payload["format"]["duration"])
        return round(duration, 3)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            KeyError, ValueError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"[hashing] Không thể trích xuất duration từ '{file_path}' qua ffprobe: {exc}"
        ) from exc


def _read_middle_1mb(file_path: str, file_size: int) -> bytes:
    """
    Đọc chính xác 1MB (hoặc toàn bộ file nếu file nhỏ hơn 1MB) tại vị trí
    giữa file, theo chunk 8KB để tránh spike RAM với video dung lượng lớn.
    """
    if file_size <= MIDDLE_WINDOW_BYTES:
        # File quá nhỏ để lấy 1MB giữa -> lấy toàn bộ file
        offset = 0
        window = file_size
    else:
        offset = (file_size // 2) - (MIDDLE_WINDOW_BYTES // 2)
        offset = max(0, offset)
        window = MIDDLE_WINDOW_BYTES

    buf = bytearray()
    remaining = window
    with open(file_path, "rb") as f:
        f.seek(offset)
        while remaining > 0:
            to_read = min(CHUNK_SIZE, remaining)
            data = f.read(to_read)
            if not data:
                break  # EOF sớm hơn dự kiến (file bị corrupt/truncated)
            buf.extend(data)
            remaining -= len(data)
    return bytes(buf)


def compute_vid_hash(file_path: str, duration_seconds: Optional[float] = None) -> str:
    """
    Tính VID_Hash = MD5(Middle_1MB_bytes + file_size + duration).

    Args:
        file_path: đường dẫn tuyệt đối/tương đối tới file video.
        duration_seconds: nếu đã biết trước (VD: được truyền từ pipeline
            upstream) thì bỏ qua bước gọi ffprobe để tiết kiệm thời gian.

    Returns:
        Chuỗi hex 32 ký tự (MD5 digest).
    """
    path_obj = Path(file_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"[hashing] Không tìm thấy file: {file_path}")

    file_size = path_obj.stat().st_size
    if file_size == 0:
        raise ValueError(f"[hashing] File rỗng (0 byte), không thể băm: {file_path}")

    if duration_seconds is None:
        duration_seconds = _get_duration_seconds(str(path_obj))

    middle_bytes = _read_middle_1mb(str(path_obj), file_size)

    hasher = hashlib.md5()
    hasher.update(middle_bytes)
    # File_Size và Duration được đưa vào dạng byte tường minh, tách biệt
    # bằng dấu '|' để tránh trường hợp nối số gây "ảo giác" trùng khớp
    # (VD: size=12, duration=3.4 vs size=1, duration=23.4).
    meta_tag = f"|SIZE={file_size}|DUR={duration_seconds:.3f}|".encode("utf-8")
    hasher.update(meta_tag)

    return hasher.hexdigest()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tính VID_Hash cho 1 file video.")
    parser.add_argument("video_path", type=str, help="Đường dẫn file video cần băm")
    args = parser.parse_args()

    vid_hash = compute_vid_hash(args.video_path)
    print(json.dumps({"file": args.video_path, "VID_Hash": vid_hash}, ensure_ascii=False))
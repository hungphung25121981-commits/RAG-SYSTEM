# Phase 1 — Extraction & Chunking

## 1. Mục Tiêu

Phase 1 "mổ xẻ" video thô thành các mảnh dữ liệu đa phương thức: văn bản (từ giọng nói và từ chữ trong khung hình), sự kiện âm thanh môi trường, và keyframe ảnh — tất cả được gom thành các **chunk** có timestamp chính xác, sẵn sàng cho việc vector hóa ở Phase 2.

**Input:** video đã có `VID_Hash` từ Phase 0.
**Output:** danh sách chunk (mỗi chunk gồm text, timestamp, keyframe_path, metadata), lưu tạm trong `data/temp_workspace/`.

> Nguyên tắc: SigLIP 2 **không** tham gia phase này (tránh nghẽn CPU–GPU) — model embedding ảnh chỉ chạy ở Phase 2.

## 2. Sơ Đồ Luồng Xử Lý

### 2.1 Audio Pipeline — Kiến trúc "Mù Domain" (Residual Subtraction)

```
audio_gốc.wav
     │
     ▼
┌────────────────────┐
│  DeepFilterNet     │  → clean_speech.wav (giọng người đã cô lập)
└─────────┬──────────┘
          │
          ▼
┌──────────────────────────────────────────┐
│ Residual: Audio_gốc − Audio_clean        │  → background_noise.wav
│ (numpy/scipy, đã resample + normalize)   │
└─────────┬────────────────────────────────┘
          ▼
┌──────────────────────┐        RMS < 1e-4          ┌──────────────────────┐
│  RMS Gate            │ ─────────────────────────▶ │ Bỏ qua CLAP          
│  (kiểm tra độ ồn)    │        (gần như câm)       │ (tránh musical noise)│
└─────────┬────────────┘                            └──────────────────────┘
          │ RMS ≥ 1e-4
          ▼
┌──────────────────────────────────────────┐
│ Tiered Caching (tuần tự, không xen kẽ):  │
│  Load Whisper → quét clean_speech.wav    │
│  → Text + Timestamp → empty_cache()      │
│  Load CLAP → quét background_noise.wav   │
│  → Nhãn sự kiện (gió, xe, động vật...)   │
│  → empty_cache()                         │
└──────────────────────────────────────────┘
```

### 2.2 Vision / OCR — Ngưỡng Động (γ)

```
keyframe.jpg
     │
     ▼
YOLOv8 (ONNX) → lọc metadata thị giác
     │
     ▼
cv2.Laplacian(frame, CV_64F).var() → đo độ mờ
     │
     ▼
Tra config/settings.yaml[domain].gamma_matrix
     │  (so với variance_boundary → chọn low_variance hoặc high_variance làm γ)
     ▼
RapidOCR (ngưỡng γ)
     │
     ├── confidence ≥ γ → nhận kết quả
     │
     └── confidence < γ → OpenCV (CLAHE, Denoise, Lanczos4)
                           → EasyOCR (tối đa 3 lần retry)
```

### 2.3 Hybrid Adaptive Chunking

```
Whisper timestamps  +  PySceneDetect scene markers
              │
              ▼
   Câu nói vắt ngang 2 cảnh?
        │              │
      Không            Có
        │              │
        ▼              ▼
  1 chunk,        2 sub-chunk (2 keyframe .jpg riêng)
  1 keyframe       chung parent_chunk_id
                   (KHÔNG dùng Mean Pooling)
```

## 3. Công Nghệ & Thuật Toán Áp Dụng

| Thành phần | Công nghệ | Ghi chú kỹ thuật |
|---|---|---|
| Tách giọng nói | DeepFilterNet | Xuất `clean_speech.wav` |
| Tách nhiễu môi trường | Residual Subtraction (numpy/scipy) | `Audio_gốc − Audio_clean`; phải resample + normalize (float32, [-1,1]) trước khi trừ để tránh lệch pha |
| Chống musical noise | RMS Gate (ε = 1e-4) | Ngăn CLAP nhận input gần-như-câm bị nhiễu artifact từ STFT |
| Speech-to-Text | Fast Whisper | Xuất text + timestamp chính xác |
| Phân loại âm thanh | CLAP | Nhãn hóa `background_noise.wav` |
| Metadata thị giác | YOLOv8 (ONNX) | Chạy nhẹ, không cần GPU nặng |
| OCR chính | RapidOCR | Ngưỡng tin cậy động γ (không hard-code 85%) |
| OCR dự phòng | EasyOCR | Chỉ kích hoạt khi RapidOCR dưới ngưỡng γ, tối đa 3 lần retry |
| Tiền xử lý ảnh | OpenCV (CLAHE, Denoising, Lanczos4) | Tăng chất lượng ảnh trước khi OCR lại |
| Đo độ mờ | `cv2.Laplacian().var()` | Độc lập với OCR — chạy trực tiếp trên ma trận điểm ảnh |
| Phát hiện cảnh | PySceneDetect | Xác định mốc chuyển cảnh cho Hybrid Chunking |

## 4. Ngưỡng OCR Động (γ) — Cách Hoạt Động

γ được calibrate **một lần** ở đầu Phase 1 cho mỗi domain (không nằm trong vòng lặp `--tune_weights` của Phase 3 vì đổi γ kéo theo re-extract toàn corpus, chi phí rất cao). Cấu hình lưu trong `config/settings.yaml`:

```yaml
domains:
  football_analytics:
    gamma_matrix:
      variance_boundary: 100   # Laplacian var < 100 → "low", >= 100 → "high"
      low_variance: 70          # % ngưỡng OCR cho frame mờ (VD: cảnh di chuyển nhanh)
      high_variance: 85         # % ngưỡng OCR cho frame nét (VD: slide, bảng điểm)
```

## 5. Cách Chạy Độc Lập

```bash
# Chỉ chạy trích xuất, lưu kết quả vào temp_workspace
python cli_pipeline.py --run_phase 1 --input_dir ./data/raw_videos --batch_size 2
```

**Output mẫu (JSON):**
```json
{
  "status": "success",
  "video_hash": "a1b2c3d4e5f6...",
  "chunks_created": 214,
  "audio": {
    "speech_segments": 187,
    "environment_events": 34,
    "rms_gate_skipped": 9
  },
  "vision": {
    "ocr_primary_pass": 178,
    "ocr_fallback_easyocr": 22,
    "ocr_failed_after_retry": 3
  }
}
```

## 6. Lưu Ý Vận Hành

- Nếu `ocr_fallback_easyocr` chiếm tỷ lệ lớn bất thường, kiểm tra lại `variance_boundary` trong `gamma_matrix` — có thể domain hiện tại cần ngưỡng khác với mặc định.
- Nếu `rms_gate_skipped` gần bằng tổng số chunk (video gần như không có tiếng ồn môi trường), đây là hành vi bình thường — không phải lỗi.
- Video không có thoại (chỉ nhạc nền/tiếng động) vẫn xử lý đúng: `clean_speech.wav` sẽ gần như trống, Whisper trả về ít/không có timestamp, phần lớn tín hiệu dồn vào `background_noise.wav` cho CLAP phân loại.

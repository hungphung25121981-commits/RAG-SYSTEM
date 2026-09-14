import os
import numpy as np
import soundfile as sf

# 1. Kéo ConfigLoader xịn sò vừa viết vào
from backend.src.common.config_loader import config

# 2. Bốc đúng key phase1_audio theo chuẩn file settings.yaml của bạn
_RMS_EPSILON = config.get("phase1_audio", {}).get("rms_gate_epsilon", 0.01)

def _normalize_audio(audio_array: np.ndarray) -> np.ndarray:
    """Chuẩn hóa âm thanh về thang [-1.0, 1.0] để tính RMS chính xác, bất chấp video to hay nhỏ."""
    max_val = np.max(np.abs(audio_array))
    if max_val > 0:
        return audio_array / max_val
    return audio_array

def _compute_residual_background(raw_path: str, clean_path: str, output_dir: str) -> tuple:
    """
    Trừ phần Clean Speech ra khỏi Raw Audio để lấy tạp âm nền.
    Trả về: (đường_dẫn_file_nền, giá_trị_rms, boolean_nên_gọi_CLAP_không)
    """
    raw_audio, sr = sf.read(raw_path)
    clean_audio, _ = sf.read(clean_path)
    if raw_audio.ndim > 1:
        raw_audio = raw_audio.mean(axis=1)
    if clean_audio.ndim > 1:
        clean_audio = clean_audio.mean(axis=1)
    
    # Cân bằng độ dài (phòng trường hợp DeepFilterNet làm suy hao vài frame)
    min_len = min(len(raw_audio), len(clean_audio))
    raw_audio = raw_audio[:min_len]
    clean_audio = clean_audio[:min_len]
    
    # Trừ dư ảnh (Residual Subtraction)
    residual_audio = raw_audio - clean_audio
    
    # Chuẩn hóa (Normalization) để tránh lỗi video âm lượng quá nhỏ
    normalized_residual = _normalize_audio(residual_audio)
    
    # Tính Năng lượng (RMS) trên bản đã chuẩn hóa
    rms = np.sqrt(np.mean(normalized_residual**2))
    
    # Lưu lại file residual để dự phòng
    bg_path = os.path.join(output_dir, "residual_background.wav")
    sf.write(bg_path, residual_audio, sr) 
    
    # Gate Logic (Đã ăn theo đúng ngưỡng Epsilon trong settings.yaml)
    should_call_clap = bool(rms >= _RMS_EPSILON)
    
    return bg_path, rms, should_call_clap
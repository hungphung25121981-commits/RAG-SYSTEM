from __future__ import annotations

import gc
import json
import os
import subprocess # THÊM subprocess cho ffmpeg
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Kéo config vào để loại bỏ hardcode
from backend.src.common.config_loader import config
_AUDIO_CFG = config.get("phase1_audio", {})

RMS_EPSILON = float(_AUDIO_CFG.get("rms_gate_epsilon", 1e-4))
EDGE_DENSITY_THRESHOLD = 0.06 
INFO_BEARING_CLASSES = {"laptop", "tv", "cell phone", "book"}


# src/phase1_extract/video_processor.py
"""
PHASE 1 — Vision & Audio Extraction

Vision (spec §1.2):
    - PySceneDetect xác định biên cảnh.
    - YOLOv8 (ONNX/PT) lọc khung hình MANG THÔNG TIN (bảng, slide, màn hình
      trình chiếu) tại mốc giữa mỗi cảnh -> tránh chụp tràn lan mọi frame.
      Heuristic "info-bearing": (a) YOLO phát hiện các lớp đối tượng liên
      quan tới màn hình/trình chiếu (laptop, tv, cell phone, book) HOẶC
      (b) mật độ biên cạnh (Canny edge density) cao — đặc trưng của
      bảng/text dày đặc, độc lập với việc YOLO có nhận diện được lớp
      "slide" hay không (COCO gốc không có lớp này).

Audio — Residual Subtraction "Mù Domain" (spec §1.1):
    1. DeepFilterNet chạy trên audio gốc -> clean_speech.wav
    2. Audio_gốc(chuẩn hoá) - Audio_clean(chuẩn hoá) = Audio_môi_trường
       (numpy, cùng sample rate, cùng thang float32 [-1,1])
    3. RMS Gate (epsilon=1e-4): nếu RMS(background) < epsilon -> bỏ qua CLAP
       (tránh "musical noise" giả do STFT artifact).
    4. Tiered Caching tuần tự: Whisper KHÔNG được nạp ở đây (thuộc
       text_processor.py) — DeepFilterNet và CLAP tuân thủ
       Load -> xử lý toàn batch -> empty_cache() -> Unload, không xen kẽ.

Toàn bộ hàm unload đều gọi `del <ref>; gc.collect(); torch.cuda.empty_cache()`
theo đúng nguyên tắc "empty_cache() không đủ, phải xoá tham chiếu" của spec.
"""





@dataclass
class Keyframe:
    scene_index: int
    timestamp_sec: float
    frame_idx: int
    keyframe_path: str
    yolo_detections: List[str] = field(default_factory=list)
    edge_density: float = 0.0


class VideoProcessor:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self._yolo_model = None
        self._clap_model = None
        self._clap_processor = None

    # ------------------------------------------------------------------ #
    # Vision
    # ------------------------------------------------------------------ #
    def _load_yolo(self, weights_path: str = "yolov8n.pt") -> None:
        from ultralytics import YOLO
        self._yolo_model = YOLO(weights_path)
        self._yolo_model.to(self.device)

    def _unload_yolo(self) -> None:
        if self._yolo_model is not None:
            del self._yolo_model
            self._yolo_model = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _detect_scenes(self, video_path: str) -> List[Tuple[float, float]]:
        """PySceneDetect: trả về danh sách (start_sec, end_sec) của mỗi cảnh."""
        from scenedetect import detect, ContentDetector
        scene_list = detect(video_path, ContentDetector())
        if not scene_list:
            # Video không có cắt cảnh rõ rệt -> coi toàn bộ là 1 cảnh
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.release()
            duration = total_frames / fps if fps else 0.0
            return [(0.0, duration)]
        return [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]

    @staticmethod
    def _edge_density(frame_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return float(np.count_nonzero(edges)) / float(edges.size)

    def _is_information_frame(self, frame_bgr: np.ndarray, conf_threshold: float) -> Tuple[bool, List[str], float]:
        results = self._yolo_model.predict(frame_bgr, conf=conf_threshold, verbose=False)
        detected_labels: List[str] = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            names = results[0].names
            cls_ids = results[0].boxes.cls.detach().cpu().numpy().astype(int)
            detected_labels = [names[c] for c in cls_ids]

        class_hit = bool(set(detected_labels) & INFO_BEARING_CLASSES)
        density = self._edge_density(frame_bgr)
        density_hit = density > EDGE_DENSITY_THRESHOLD

        return (class_hit or density_hit), detected_labels, density

    def extract_keyframes(
        self,
        video_path: str,
        output_dir: str,
        conf_threshold: float = 0.35,
        yolo_weights: str = "yolov8n.pt",
    ) -> List[Keyframe]:
        """
        Chạy PySceneDetect -> lấy 1 frame đại diện/cảnh -> lọc qua YOLOv8.
        Chỉ những frame "mang thông tin" mới được lưu ra .jpg vật lý.
        """
        os.makedirs(output_dir, exist_ok=True)
        scenes = self._detect_scenes(video_path)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        self._load_yolo(yolo_weights)
        keyframes: List[Keyframe] = []
        try:
            for scene_idx, (start_sec, end_sec) in enumerate(scenes):
                mid_sec = (start_sec + end_sec) / 2.0
                frame_idx = int(mid_sec * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                is_info, labels, density = self._is_information_frame(frame, conf_threshold)
                if not is_info:
                    continue

                out_path = str(Path(output_dir) / f"scene_{scene_idx:04d}_f{frame_idx}.jpg")
                cv2.imwrite(out_path, frame)
                keyframes.append(Keyframe(
                    scene_index=scene_idx,
                    timestamp_sec=round(mid_sec, 3),
                    frame_idx=frame_idx,
                    keyframe_path=out_path,
                    yolo_detections=labels,
                    edge_density=round(density, 4),
                ))
        finally:
            cap.release()
            self._unload_yolo()  # Tiered Caching: unload ngay sau khi xử lý xong toàn bộ batch

        return keyframes
    def extract_audio_from_video(self, video_path: str, output_wav_path: str, sample_rate: int = 48000) -> str:
        """Tách audio bằng FFmpeg trước khi đưa vào luồng DeepFilterNet."""
        os.makedirs(os.path.dirname(output_wav_path) or ".", exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", str(sample_rate), "-acodec", "pcm_s16le", output_wav_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return output_wav_path

    # ------------------------------------------------------------------ #
    # Audio — Residual Subtraction Pipeline
    # ------------------------------------------------------------------ #
    def denoise_and_split_audio(self, audio_path: str, output_dir: str) -> Dict[str, str]:
        """
        Bước 1 (DeepFilterNet) + Bước 2 (Residual Subtraction, numpy thuần).
        Trả về path của clean_speech.wav và background_noise.wav.
        """
        from df.enhance import enhance, init_df, load_audio, save_audio

        os.makedirs(output_dir, exist_ok=True)
        clean_path = str(Path(output_dir) / "clean_speech.wav")
        background_path = str(Path(output_dir) / "background_noise.wav")

        # --- Load DeepFilterNet, chạy full batch ---
        df_model, df_state, _ = init_df()
        model_sr = df_state.sr()

        # load_audio của DeepFilterNet tự resample về model_sr nội bộ (thường 48kHz)
        audio_tensor, _ = load_audio(audio_path, sr=model_sr)
        enhanced_tensor = enhance(df_model, df_state, audio_tensor)
        save_audio(clean_path, enhanced_tensor, model_sr)

        # --- Unload DeepFilterNet ngay sau khi xử lý xong (Tiered Caching) ---
        del df_model
        df_state = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # --- Bước 2: Residual Subtraction ---
        # Chuẩn hoá: cùng sample rate (model_sr) + cùng thang float32 [-1,1]
        import librosa
        original_audio, _ = librosa.load(audio_path, sr=model_sr, mono=True)
        original_audio = original_audio.astype(np.float32)

        clean_np = enhanced_tensor.detach().cpu().numpy().astype(np.float32)
        if clean_np.ndim > 1:
            clean_np = clean_np.mean(axis=0)  # gộp về mono nếu multi-channel

        # Căn chỉnh chiều dài (DeepFilterNet có thể pad/crop vài sample)
        min_len = min(len(original_audio), len(clean_np))
        original_audio = original_audio[:min_len]
        clean_np = clean_np[:min_len]

        # Clip về [-1, 1] để tránh overflow trước khi trừ
        original_audio = np.clip(original_audio, -1.0, 1.0)
        clean_np = np.clip(clean_np, -1.0, 1.0)

        background_audio = (original_audio - clean_np).astype(np.float32)

        import soundfile as sf
        sf.write(background_path, background_audio, model_sr)

        return {
            "clean_speech_path": clean_path,
            "background_noise_path": background_path,
            "sample_rate": model_sr,
        }

    @staticmethod
    def _compute_rms(audio_path: str) -> float:
        import soundfile as sf
        data, _ = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return float(np.sqrt(np.mean(np.square(data)) + 1e-12))

    def _load_clap(self, model_name: str = "laion/clap-htsat-unfused") -> None:
        from transformers import ClapModel, ClapProcessor
        self._clap_model = ClapModel.from_pretrained(model_name).to(self.device)
        self._clap_model.eval()
        self._clap_processor = ClapProcessor.from_pretrained(model_name)

    def _unload_clap(self) -> None:
        if self._clap_model is not None:
            del self._clap_model
            self._clap_model = None
        self._clap_processor = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def tag_background_audio(
        self,
        background_noise_path: str,
        candidate_labels: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> Dict[str, object]:
        """
        Bước 3 + RMS Gate: chỉ chạy CLAP nếu background_noise không gần-câm.
        candidate_labels domain-agnostic mặc định, có thể override theo domain.
        """
        rms = self._compute_rms(background_noise_path)
        if rms < RMS_EPSILON:
            return {"skipped_clap": True, "reason": "rms_below_epsilon", "rms": rms, "tags": []}

        if candidate_labels is None:
            candidate_labels = [
                "music", "applause", "crowd noise", "silence", "traffic noise",
                "mechanical noise", "wind noise", "laughter", "background chatter",
                "alarm or siren", "nature sounds",
            ]

        import librosa
        audio_arr, sr = librosa.load(background_noise_path, sr=48000, mono=True)

        self._load_clap()
        try:
            inputs = self._clap_processor(
                text=candidate_labels,
                audios=[audio_arr],
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self._clap_model(**inputs)
                logits_per_audio = outputs.logits_per_audio  # shape [1, num_labels]
                probs = torch.softmax(logits_per_audio, dim=-1)[0].detach().cpu().numpy()

            ranked = sorted(zip(candidate_labels, probs.tolist()), key=lambda x: x[1], reverse=True)
            tags = [{"label": lbl, "score": round(score, 4)} for lbl, score in ranked[:top_k]]
        finally:
            self._unload_clap()  # Tiered Caching: unload ngay sau batch này

        return {"skipped_clap": False, "rms": rms, "tags": tags}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1 Video/Audio processor (standalone test).")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    vp = VideoProcessor()
    kf = vp.extract_keyframes(args.video_path, os.path.join(args.output_dir, "frames"))
    audio_paths = vp.denoise_and_split_audio(args.audio_path, os.path.join(args.output_dir, "audio"))
    clap_result = vp.tag_background_audio(audio_paths["background_noise_path"])

    print(json.dumps({
        "keyframes": [kf_i.__dict__ for kf_i in kf],
        "audio": audio_paths,
        "clap": clap_result,
    }, ensure_ascii=False, indent=2))
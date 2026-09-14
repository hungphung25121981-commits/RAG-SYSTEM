from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import torch

# 1. Kéo config vào để gỡ bỏ Hardcode
from backend.src.common.config_loader import config
_AUDIO_CFG = config.get("phase1_audio", {})

# Regex nhận diện dấu kết câu trên từng từ (word-level)
_SENTENCE_TERMINATORS = re.compile(r"[.!?…]+")


@dataclass
class Sentence:
    text: str
    start: float
    end: float


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    sentences: List[Sentence] = field(default_factory=list)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


class TextProcessor:
    def __init__(self, device: Optional[str] = None):
        self.device = device or _AUDIO_CFG.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self._compute_type = _AUDIO_CFG.get("compute_type", "float16" if self.device == "cuda" else "int8")
        self._model = None

    def _load_model(self, model_size: str) -> None:
        from faster_whisper import WhisperModel
        self._model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=self._compute_type,
        )

    def _unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    @staticmethod
    def _build_sentences_from_words(words: Any, default_text: str, default_start: float, default_end: float) -> List[Sentence]:
        """
        Tách câu dựa trên timestamp CỦA TỪNG TỪ (word-level) từ Whisper.
        Đảm bảo nhịp thời gian chính xác 100% để đối chiếu với mốc chuyển cảnh.
        """
        words = list(words) if words else []
        if not words:
            text = default_text.strip()
            if text:
                return [Sentence(text=text, start=default_start, end=default_end)]
            return []

        sentences: List[Sentence] = []
        buffer_words: List[str] = []
        buffer_start: Optional[float] = None
        buffer_end: Optional[float] = None

        for w in words:
            if buffer_start is None:
                buffer_start = w.start
            buffer_end = w.end
            buffer_words.append(w.word)

            if _SENTENCE_TERMINATORS.search(w.word):
                # Vá lỗi dính chữ (Word Spacing)
                text = " ".join([x.strip() for x in buffer_words if x.strip()]).strip()
                if text:
                    sentences.append(Sentence(text=text, start=buffer_start, end=buffer_end))
                buffer_words = []
                buffer_start = None
                buffer_end = None

        if buffer_words:
            text = " ".join([x.strip() for x in buffer_words if x.strip()]).strip()
            if text:
                sentences.append(Sentence(text=text, start=buffer_start, end=buffer_end))

        return sentences

    def transcribe(
        self,
        clean_speech_path: str,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
        beam_size: Optional[int] = None,
        vad_filter: Optional[bool] = None,
    ) -> Dict[str, object]:
        """
        Bóc băng toàn bộ file clean_speech.wav. Tuân thủ Tiered Caching.
        Bật word_timestamps=True bắt buộc để phục vụ Hybrid Chunking.
        """
        model_size = model_size or _AUDIO_CFG.get("whisper_model_size", "large-v3")
        beam_size = beam_size or _AUDIO_CFG.get("beam_size", 5)
        vad_filter = vad_filter if vad_filter is not None else _AUDIO_CFG.get("vad_filter", True)
        vad_min_silence_duration_ms = _AUDIO_CFG.get("vad_min_silence_duration_ms", 500)
        vad_parameters = dict(min_silence_duration_ms=vad_min_silence_duration_ms) if vad_filter else None

        self._load_model(model_size)
        try:
            segments_iter, info = self._model.transcribe(
                clean_speech_path,
                language=language,
                beam_size=beam_size,
                vad_filter=vad_filter,
                vad_parameters=vad_parameters,
                word_timestamps=True,  # BẮT BUỘC ĐỂ CHUNKING CHÍNH XÁC
            )

            segments: List[Segment] = []
            full_text_parts: List[str] = []
            
            for idx, seg in enumerate(segments_iter):
                # 2. Tạo câu dựa trên word-level timestamps
                sentences = self._build_sentences_from_words(seg.words, seg.text, seg.start, seg.end)
                
                segments.append(Segment(
                    id=idx,
                    start=round(seg.start, 3),
                    end=round(seg.end, 3),
                    text=seg.text.strip(),
                    sentences=sentences,
                    avg_logprob=round(getattr(seg, "avg_logprob", 0.0), 4),
                    no_speech_prob=round(getattr(seg, "no_speech_prob", 0.0), 4),
                ))
                full_text_parts.append(seg.text.strip())

            result = {
                "language": info.language,
                "language_probability": round(float(info.language_probability), 4),
                "duration": round(float(info.duration), 3),
                "full_text": " ".join(full_text_parts).strip(),
                "segments": [
                    {
                        "id": s.id, "start": s.start, "end": s.end, "text": s.text,
                        "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
                        "sentences": [
                            {"text": sent.text, "start": sent.start, "end": sent.end}
                            for sent in s.sentences
                        ],
                    }
                    for s in segments
                ],
            }
        finally:
            # Tiered Caching: unload NGAY sau khi xử lý xong toàn bộ file
            self._unload_model()

        return result
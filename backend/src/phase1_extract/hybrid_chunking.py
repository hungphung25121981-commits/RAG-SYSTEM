# src/phase1_extract/hybrid_chunking.py
"""
PHASE 1 — Hybrid Adaptive Chunking (spec V8.0 §1.4)

Nguyên tắc:
    - Trục thời gian CHÍNH = nhịp câu nói của Whisper (đầu ra text_processor.py),
      đối chiếu với mốc chuyển cảnh của PySceneDetect (đầu ra video_processor.py).
    - Chunk được gom "adaptive": tích luỹ liên tiếp các câu trong CÙNG một
      cảnh cho tới khi chạm ngưỡng độ dài (ký tự) hoặc ngưỡng thời lượng,
      hoặc cho tới khi gặp biên chuyển cảnh -> chốt chunk ngay tại đó
      (không gộp câu xuyên cảnh vào cùng 1 chunk).
    - Nếu MỘT CÂU vắt ngang 2 cảnh (start nằm cảnh A, end nằm cảnh B):
        -> tạo 2 SUB-CHUNK VẬT LÝ riêng biệt, mỗi sub-chunk gắn 1 keyframe
           .jpg riêng (của cảnh tương ứng).
        -> cả 2 sub-chunk dùng chung 1 `parent_chunk_id` (uuid) để tầng
           retrieval biết chúng thuộc cùng 1 phát ngôn logic.
        -> KHÔNG Mean Pooling vector của 2 keyframe (sẽ làm loãng đặc
           trưng) — việc gộp vector là trách nhiệm của phase2 (embedding
           độc lập từng ảnh) + phase3 (gộp lại qua parent_chunk_id ở tầng
           retrieval), module này chỉ chịu trách nhiệm SINH ra đúng cấu
           trúc chunk + parent_chunk_id, không đụng vào vector.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple
from backend.src.common.config_loader import config
_CHUNK_CFG = config.get("phase1_chunking", {}) 

DEFAULT_MAX_CHUNK_CHARS = int(_CHUNK_CFG.get("max_chunk_chars", 400))
DEFAULT_MAX_CHUNK_DURATION = float(_CHUNK_CFG.get("max_chunk_duration", 15.0))
DEFAULT_MIN_CHUNK_CHARS = int(_CHUNK_CFG.get("min_chunk_chars", 40))


@dataclass
class SentenceIn:
    text: str
    start: float
    end: float


@dataclass
class KeyframeRef:
    scene_index: int
    keyframe_path: str
    timestamp_sec: float


@dataclass
class Chunk:
    chunk_id: str
    parent_chunk_id: Optional[str]  # None nếu chunk độc lập, không vắt cảnh
    is_cross_scene_subchunk: bool
    scene_index: int
    text: str
    start: float
    end: float
    keyframe_path: Optional[str]
    sentence_count: int
    weight: int = 1  # dùng cho Knapsack DP ở Phase 3 (§3.5): chunk đơn=1, cặp vắt cảnh=2 (tổng)


def _find_scene_index(t: float, scenes: Sequence[Tuple[float, float]]) -> int:
    """Trả về index cảnh chứa thời điểm t. Nếu t rơi ra ngoài mọi cảnh (do
    sai số làm tròn), gán vào cảnh gần nhất."""
    for idx, (s, e) in enumerate(scenes):
        if s <= t < e:
            return idx
    if not scenes:
        return 0
    # fallback: cảnh cuối cùng nếu t >= end của video, hoặc cảnh đầu nếu t < 0
    if t >= scenes[-1][1]:
        return len(scenes) - 1
    return 0


def _find_nearest_keyframe(scene_index: int, keyframes: Sequence[KeyframeRef]) -> Optional[str]:
    same_scene = [kf for kf in keyframes if kf.scene_index == scene_index]
    if same_scene:
        return same_scene[0].keyframe_path
    return None


def _split_sentence_by_scene_boundary(
    sentence: SentenceIn,
    scene_a_end: float,
) -> Tuple[SentenceIn, SentenceIn]:
    """
    Câu vắt ngang 2 cảnh -> chia text theo tỉ lệ thời gian (không có
    word-alignment nên dùng xấp xỉ theo ký tự, đồng bộ với cách
    text_processor.py nội suy timestamp câu từ segment).
    """
    total_duration = max(sentence.end - sentence.start, 1e-6)
    ratio_a = max(0.0, min(1.0, (scene_a_end - sentence.start) / total_duration))
    split_char_idx = max(1, int(len(sentence.text) * ratio_a))
    split_char_idx = min(split_char_idx, len(sentence.text) - 1) if len(sentence.text) > 1 else len(sentence.text)

    text_a = sentence.text[:split_char_idx].strip()
    text_b = sentence.text[split_char_idx:].strip()

    part_a = SentenceIn(text=text_a or sentence.text, start=sentence.start, end=scene_a_end)
    part_b = SentenceIn(text=text_b or sentence.text, start=scene_a_end, end=sentence.end)
    return part_a, part_b


class HybridChunker:
    def __init__(
        self,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        max_chunk_duration: float = DEFAULT_MAX_CHUNK_DURATION,
        min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.max_chunk_duration = max_chunk_duration
        self.min_chunk_chars = min_chunk_chars

    @staticmethod
    def _flatten_sentences(whisper_result: Dict[str, object]) -> List[SentenceIn]:
        sentences: List[SentenceIn] = []
        for seg in whisper_result.get("segments", []):
            for sent in seg.get("sentences", []):
                if sent["text"].strip():
                    sentences.append(SentenceIn(
                        text=sent["text"].strip(),
                        start=float(sent["start"]),
                        end=float(sent["end"]),
                    ))
        sentences.sort(key=lambda s: s.start)
        return sentences

    def _finalize_buffer(
        self,
        buffer: List[SentenceIn],
        scene_index: int,
        keyframes: Sequence[KeyframeRef],
    ) -> Optional[Chunk]:
        if not buffer:
            return None
        text = " ".join(s.text for s in buffer).strip()
        if not text:
            return None
        return Chunk(
            chunk_id=str(uuid.uuid4()),
            parent_chunk_id=None,
            is_cross_scene_subchunk=False,
            scene_index=scene_index,
            text=text,
            start=buffer[0].start,
            end=buffer[-1].end,
            keyframe_path=_find_nearest_keyframe(scene_index, keyframes),
            sentence_count=len(buffer),
            weight=1,
        )

    def chunk(
        self,
        whisper_result: Dict[str, object],
        scenes: Sequence[Tuple[float, float]],
        keyframes: Sequence[KeyframeRef],
    ) -> List[Chunk]:
        """
        Thuật toán Adaptive Chunking chính.

        Args:
            whisper_result: output của TextProcessor.transcribe().
            scenes: danh sách (start_sec, end_sec) từ PySceneDetect.
            keyframes: danh sách keyframe đã trích ở video_processor.py,
                mỗi keyframe gắn với 1 scene_index.

        Returns:
            Danh sách Chunk (đơn hoặc cặp sub-chunk vắt cảnh).
        """
        sentences = self._flatten_sentences(whisper_result)
        if not sentences or not scenes:
            return []

        chunks: List[Chunk] = []
        buffer: List[SentenceIn] = []
        buffer_scene_index: Optional[int] = None

        for sentence in sentences:
            scene_start_idx = _find_scene_index(sentence.start, scenes)
            scene_end_idx = _find_scene_index(max(sentence.end - 1e-6, sentence.start), scenes)

            if scene_start_idx != scene_end_idx:
                # ---- CÂU VẮT NGANG 2 CẢNH: chốt buffer hiện tại trước ----
                finalized = self._finalize_buffer(buffer, buffer_scene_index or scene_start_idx, keyframes)
                if finalized:
                    chunks.append(finalized)
                buffer, buffer_scene_index = [], None

                scene_a_end = scenes[scene_start_idx][1]
                part_a, part_b = _split_sentence_by_scene_boundary(sentence, scene_a_end)

                parent_id = str(uuid.uuid4())
                sub_a = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    parent_chunk_id=parent_id,
                    is_cross_scene_subchunk=True,
                    scene_index=scene_start_idx,
                    text=part_a.text,
                    start=part_a.start,
                    end=part_a.end,
                    keyframe_path=_find_nearest_keyframe(scene_start_idx, keyframes),
                    sentence_count=1,
                    weight=2,  # Weight=2 cho cặp, theo §3.5 Knapsack (Chunk A mang Value RRF)
                )
                sub_b = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    parent_chunk_id=parent_id,
                    is_cross_scene_subchunk=True,
                    scene_index=scene_end_idx,
                    text=part_b.text,
                    start=part_b.start,
                    end=part_b.end,
                    keyframe_path=_find_nearest_keyframe(scene_end_idx, keyframes),
                    sentence_count=1,
                    weight=0,  # Partner B không tự đóng góp Value RRF riêng (§3.5)
                )
                chunks.append(sub_a)
                chunks.append(sub_b)
                continue

            # ---- CÂU NẰM TRỌN TRONG 1 CẢNH ----
            if buffer_scene_index is None:
                buffer_scene_index = scene_start_idx

            crosses_scene_change = scene_start_idx != buffer_scene_index
            buffer_text_len = sum(len(s.text) for s in buffer) + len(sentence.text)
            buffer_duration = (buffer[-1].end - buffer[0].start) if buffer else 0.0
            exceeds_limits = (
                buffer_text_len > self.max_chunk_chars
                or (sentence.start - (buffer[0].start if buffer else sentence.start)) > self.max_chunk_duration
            )

            if crosses_scene_change or (exceeds_limits and buffer):
                finalized = self._finalize_buffer(buffer, buffer_scene_index, keyframes)
                if finalized:
                    chunks.append(finalized)
                buffer = [sentence]
                buffer_scene_index = scene_start_idx
            else:
                buffer.append(sentence)

        # Chốt buffer còn lại cuối cùng
        finalized = self._finalize_buffer(buffer, buffer_scene_index if buffer_scene_index is not None else 0, keyframes)
        if finalized:
            chunks.append(finalized)

        return chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1 Hybrid Adaptive Chunking (standalone test).")
    parser.add_argument("--whisper_json", required=True, help="Path tới JSON output của text_processor.py")
    parser.add_argument("--scenes_json", required=True, help='JSON list [[start,end], ...]')
    parser.add_argument("--keyframes_json", required=True, help='JSON list [{"scene_index":0,"keyframe_path":"...","timestamp_sec":1.2}]')
    args = parser.parse_args()

    with open(args.whisper_json, "r", encoding="utf-8") as f:
        whisper_result = json.load(f)
    with open(args.scenes_json, "r", encoding="utf-8") as f:
        scenes = [tuple(s) for s in json.load(f)]
    with open(args.keyframes_json, "r", encoding="utf-8") as f:
        kf_raw = json.load(f)
    keyframes = [KeyframeRef(**kf) for kf in kf_raw]

    chunker = HybridChunker()
    result_chunks = chunker.chunk(whisper_result, scenes, keyframes)

    print(json.dumps([asdict(c) for c in result_chunks], ensure_ascii=False, indent=2))
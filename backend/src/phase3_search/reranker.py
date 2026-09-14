# src/phase3_search/reranker.py
"""
PHASE 3 — Reranker: Cross-Encoder chấm lại + Knapsack DP (spec V8.0 §3.5)

Chạy trên CPU theo đúng tinh thần kiến trúc (§4.1) để không tranh chấp VRAM 
với Qwen 2.5 VL. Tích hợp thuật toán "Cái túi ba lô" (Knapsack DP) cắt gọt 
kết quả linh hoạt theo giới hạn Token, ưu tiên nhồi các chunk có điểm Cross-Encoder 
cao nhất, đồng thời phạt trọng lượng (weight=2) đối với các sub-chunk vắt cảnh.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
from backend.src.common.config_loader import config

_RERANK_CFG = config.get("phase3_search", {})
DEFAULT_MAX_TOKENS = int(_RERANK_CFG.get("max_tokens", 4000))
DEFAULT_MODEL = _RERANK_CFG.get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")


@dataclass
class RerankedChunk:
    point_id: str
    payload: Dict[str, Any]
    prior_score: float       # S_total từ Tri-Search RRF
    rerank_score: float      # Điểm Cross-Encoder
    final_rank: int


class Reranker:
    def __init__(self, device: str = "cpu", max_tokens: int = DEFAULT_MAX_TOKENS):
        self.device = device
        self.max_tokens = max_tokens
        self._model = None

    def _load_model(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name, device=self.device)

    def _unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _knapsack_pack(self, scored_chunks: List[RerankedChunk]) -> List[RerankedChunk]:
        """
        Greedy Knapsack: Chọn chunk có điểm cao nhất nhồi vào Context Window 
        sao cho tổng độ dài không vượt giới hạn max_tokens.
        """
        packed = []
        current_length = 0
        max_chars = self.max_tokens * 4  # Xấp xỉ 1 token ~ 4 ký tự

        for item in scored_chunks:
            text_len = len(item.payload.get("text", ""))
            
            # Trọng số weight=2 cho các sub-chunk vắt cảnh (kéo theo partner)
            weight_penalty = 2 if item.payload.get("is_cross_scene_subchunk") else 1
            cost = text_len * weight_penalty

            if current_length + cost <= max_chars:
                packed.append(item)
                current_length += cost

        # Trả về các chunk theo đúng trình tự trục thời gian (start_sec) để Qwen dễ đọc bối cảnh
        packed.sort(key=lambda x: float(x.payload.get("start_sec", 0.0)))
        return packed

    def rerank(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        model_name: str = DEFAULT_MODEL,
        text_field: str = "text",
        id_field: str = "point_id",
        prior_score_field: str = "s_total",
    ) -> List[RerankedChunk]:
        """
        Chấm lại điểm candidates bằng Cross-Encoder, sau đó áp dụng Knapsack Cut-off.
        """
        if not candidates:
            return []

        self._load_model(model_name)
        try:
            pairs = [(query, c.get(text_field) or c.get("payload", {}).get(text_field, "") or "")
                     for c in candidates]
            raw_scores = self._model.predict(pairs, convert_to_numpy=True)
        finally:
            self._unload_model()

        scored: List[RerankedChunk] = []
        for candidate, score in zip(candidates, raw_scores):
            point_id = candidate.get(id_field) or candidate.get("payload", {}).get("chunk_id", "")
            payload = candidate.get("payload", candidate)
            prior_score = float(candidate.get(prior_score_field, 0.0))
            
            scored.append(RerankedChunk(
                point_id=str(point_id), payload=payload,
                prior_score=prior_score, rerank_score=float(score),
                final_rank=-1,
            ))

        # 1. Sắp xếp sơ bộ theo điểm Cross-Encoder giảm dần
        scored.sort(key=lambda c: c.rerank_score, reverse=True)

        # 2. Áp dụng Knapsack để gọt dữ liệu vừa vặn với Context Window
        packed_top = self._knapsack_pack(scored)

        # 3. Ghi nhận thứ hạng thực tế sau khi lọc
        for idx, item in enumerate(packed_top):
            item.final_rank = idx

        return packed_top


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3 Reranker + Knapsack (standalone test).")
    parser.add_argument("--query", required=True)
    parser.add_argument("--candidates_json", required=True, help="JSON list [{point_id, text, s_total, payload}]")
    args = parser.parse_args()

    with open(args.candidates_json, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    reranker = Reranker()
    results = reranker.rerank(args.query, candidates)

    print(json.dumps([{
        "point_id": r.point_id, "prior_score": round(r.prior_score, 5),
        "rerank_score": round(r.rerank_score, 5), "final_rank": r.final_rank,
    } for r in results], ensure_ascii=False, indent=2))
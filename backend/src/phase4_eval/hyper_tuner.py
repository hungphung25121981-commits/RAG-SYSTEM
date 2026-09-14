# src/phase4_eval/hyper_tuner.py
"""
PHASE 4 — Hyper Tuner: Grid Search (alpha, beta) tối ưu Mean MRR (spec V8.0 §3.2)

Nguyên tắc "siêu nhẹ" (đúng ghi chú §1.3 / §3.2):
    - γ (ngưỡng OCR) KHÔNG nằm trong vòng grid search này — vì thay đổi γ
      kéo theo re-extract toàn bộ corpus (chi phí cực cao). γ chỉ
      calibrate 1 lần ở Phase 1.
    - --tune_weights chỉ chạy DP 2 chiều THUẦN TUÝ trên vector đã có sẵn
      trong Qdrant: mỗi query chỉ cần query 3 không gian (semantic,
      visual, keyword) ĐÚNG 1 LẦN để lấy raw hits + RRF component score
      -> sau đó toàn bộ vòng lặp (α, β) chỉ là PHÉP CỘNG SỐ HỌC thuần
      trên dữ liệu đã cache, không re-query Qdrant, không tốn GPU.

--target_kf: JSON array tối thiểu 5 cặp [query, expected_keyframe_path]
             (bắt buộc >=5 để tránh overfit vào 1 điểm dữ liệu duy nhất,
             theo Tóm Tắt Quyết Định #11).

--save_domain: sau khi tìm được (α, β) tối ưu, ghi đè vào
               config/settings.yaml dưới đúng domain tương ứng, giữ
               nguyên gamma_matrix hiện có (không đụng vào calibration
               OCR, tách biệt hoàn toàn 2 loại tham số theo §1.3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

MIN_TARGET_KF_PAIRS = 5
DEFAULT_GRID_STEP = 0.1
DEFAULT_TOP_K = 20


@dataclass
class QueryComponents:
    """Cache raw RRF component cho 1 query — tính 1 lần, dùng lại cho mọi (α, β)."""
    query: str
    expected_keyframe: str
    semantic_rrf: Dict[str, float]
    visual_rrf: Dict[str, float]
    keyword_rrf: Dict[str, float]
    keyframe_by_id: Dict[str, Optional[str]]  # point_id -> keyframe_path (từ payload)


class HyperTuner:
    def __init__(self, settings_yaml_path: Optional[str] = None):
        from backend.src.common.config_loader import config
        # Lấy từ config, ưu tiên biến settings_yaml_path nếu được truyền vào
        self.settings_yaml_path = settings_yaml_path or config.get("paths", {}).get("settings_yaml", "config/settings.yaml")

    # ------------------------------------------------------------------ #
    # --target_kf validation & parsing
    # ------------------------------------------------------------------ #
    @staticmethod
    def load_target_kf(target_kf_arg: str) -> List[Tuple[str, str]]:
        """
        Chấp nhận `target_kf_arg` là JSON string trực tiếp hoặc path tới
        file JSON. Format: [[query, keyframe_path], ...], tối thiểu 5 cặp.
        """
        raw: Any
        if Path(target_kf_arg).is_file():
            with open(target_kf_arg, "r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            raw = json.loads(target_kf_arg)

        pairs = [(str(item[0]), str(item[1])) for item in raw]
        if len(pairs) < MIN_TARGET_KF_PAIRS:
            raise ValueError(
                f"[hyper_tuner] --target_kf cần tối thiểu {MIN_TARGET_KF_PAIRS} cặp "
                f"[query, keyframe] để tránh overfit; hiện chỉ có {len(pairs)}."
            )
        return pairs

    # ------------------------------------------------------------------ #
    # Precompute — mỗi query chỉ query Qdrant 1 LẦN DUY NHẤT
    # ------------------------------------------------------------------ #
    def precompute_query_components(
        self,
        retriever: Any,  # phase3_search.retriever.Retriever
        target_kf: Sequence[Tuple[str, str]],
        text_dense_encoder,  # callable(str) -> List[float]
        image_dense_query: Optional[Sequence[float]] = None,
        sparse_encoder=None,  # callable(str) -> Dict[int,float] | None
        top_k: int = DEFAULT_TOP_K,
        filter_meta: Optional[str] = None,
    ) -> List[QueryComponents]:
        user_filter = retriever.parse_filter_meta(filter_meta)
        qdrant_filter = retriever._build_committed_filter(user_filter)

        components: List[QueryComponents] = []
        for query, expected_kf in target_kf:
            text_vec = text_dense_encoder(query)
            semantic_hits = retriever._search_named_vector("text_dense", text_vec, top_k, qdrant_filter)

            visual_hits: List[Tuple[str, Dict[str, Any]]] = []
            if image_dense_query is not None:
                visual_hits = retriever._search_named_vector("image_dense", image_dense_query, top_k, qdrant_filter)

            keyword_hits: List[Tuple[str, Dict[str, Any]]] = []
            if sparse_encoder is not None:
                sparse_q = sparse_encoder(query)
                if sparse_q:
                    keyword_hits = retriever._search_sparse(sparse_q, top_k, qdrant_filter)

            semantic_rrf = retriever._rrf_scores([pid for pid, _ in semantic_hits])
            visual_rrf = retriever._rrf_scores([pid for pid, _ in visual_hits])
            keyword_rrf = retriever._rrf_scores([pid for pid, _ in keyword_hits])

            keyframe_by_id: Dict[str, Optional[str]] = {}
            for pid, payload in semantic_hits + visual_hits + keyword_hits:
                keyframe_by_id[pid] = payload.get("keyframe_path")

            components.append(QueryComponents(
                query=query, expected_keyframe=expected_kf,
                semantic_rrf=semantic_rrf, visual_rrf=visual_rrf, keyword_rrf=keyword_rrf,
                keyframe_by_id=keyframe_by_id,
            ))

        return components

    # ------------------------------------------------------------------ #
    # Grid Search 2 chiều (α, β) — thuần arithmetic trên cache
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mrr_for_weights(components: Sequence[QueryComponents], alpha: float, beta: float) -> float:
        gamma = max(0.0, 1.0 - alpha - beta)
        reciprocal_ranks: List[float] = []

        for comp in components:
            all_ids = set(comp.semantic_rrf) | set(comp.visual_rrf) | set(comp.keyword_rrf)
            scored = [
                (pid, alpha * comp.semantic_rrf.get(pid, 0.0)
                      + beta * comp.visual_rrf.get(pid, 0.0)
                      + gamma * comp.keyword_rrf.get(pid, 0.0))
                for pid in all_ids
            ]
            scored.sort(key=lambda x: x[1], reverse=True)

            rank_found = 0
            for rank, (pid, _) in enumerate(scored, start=1):
                if comp.keyframe_by_id.get(pid) == comp.expected_keyframe:
                    rank_found = rank
                    break

            reciprocal_ranks.append(1.0 / rank_found if rank_found > 0 else 0.0)

        return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    def grid_search(
        self,
        components: Sequence[QueryComponents],
        step: float = DEFAULT_GRID_STEP,
    ) -> Dict[str, Any]:
        """
        Quét mặt phẳng (α, β) với α, β ∈ [0,1], α+β<=1, bước nhảy `step`.
        Trả về (alpha, beta) tối ưu theo Mean MRR + toàn bộ lưới kết quả
        (phục vụ debug/visualize nếu cần).
        """
        best_alpha, best_beta, best_mrr = 0.0, 0.0, -1.0
        grid_results: List[Dict[str, float]] = []

        alpha_values = np.round(np.arange(0.0, 1.0 + 1e-9, step), 3)
        for alpha in alpha_values:
            beta_values = np.round(np.arange(0.0, 1.0 - alpha + 1e-9, step), 3)
            for beta in beta_values:
                mean_mrr = self._mrr_for_weights(components, float(alpha), float(beta))
                grid_results.append({"alpha": float(alpha), "beta": float(beta), "mean_mrr": round(mean_mrr, 5)})
                if mean_mrr > best_mrr:
                    best_alpha, best_beta, best_mrr = float(alpha), float(beta), mean_mrr

        return {
            "best_alpha": round(best_alpha, 3),
            "best_beta": round(best_beta, 3),
            "best_gamma": round(max(0.0, 1.0 - best_alpha - best_beta), 3),
            "best_mean_mrr": round(best_mrr, 5),
            "grid_results": grid_results,
        }

    # ------------------------------------------------------------------ #
    # --save_domain: ghi (α, β) vào config/settings.yaml, giữ nguyên γ
    # ------------------------------------------------------------------ #
    def save_domain_weights(self, domain: str, alpha: float, beta: float) -> None:
        settings_path = Path(self.settings_yaml_path)
        settings_path.parent.mkdir(parents=True, exist_ok=True)

        if settings_path.is_file():
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = yaml.safe_load(f) or {}
        else:
            settings = {}

        settings.setdefault("domains", {})
        domain_cfg = settings["domains"].setdefault(domain, {})

        # Chỉ cập nhật alpha/beta (query-time) — KHÔNG đụng vào gamma_matrix
        # (extraction-time, đã calibrate riêng ở Phase 1 §1.3).
        domain_cfg["alpha"] = round(float(alpha), 3)
        domain_cfg["beta"] = round(float(beta), 3)
        domain_cfg.setdefault("gamma_matrix", {
            "variance_boundary": 100,
            "low_variance": 70,
            "high_variance": 85,
        })

        with open(settings_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(settings, f, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))  # cho phép import phase3_search, phase2_build

    parser = argparse.ArgumentParser(description="Phase 4 Hyper Tuner (standalone CLI).")
    parser.add_argument("--qdrant_path", required=True)
    parser.add_argument("--collection_name", required=True)
    parser.add_argument("--target_kf", required=True, help="JSON string hoặc path file, >=5 cặp [query, keyframe]")
    parser.add_argument("--save_domain", default=None, help="Tên domain để lưu α, β vào config/settings.yaml")
    parser.add_argument("-f", "--filter_meta", default=None)
    parser.add_argument("-t", "--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--step", type=float, default=DEFAULT_GRID_STEP)
    parser.add_argument("--settings_yaml_path", default="config/settings.yaml")
    args = parser.parse_args()

    from phase3_search.retriever import Retriever
    from phase2_build.embedding import EmbeddingEngine

    target_kf_pairs = HyperTuner.load_target_kf(args.target_kf)

    retriever = Retriever(args.qdrant_path, args.collection_name)
    embedder = EmbeddingEngine()

    def _text_encoder(q: str):
        result = embedder.encode_texts([q], return_sparse=False)
        return result.dense_text_vectors[0].tolist()

    tuner = HyperTuner(args.settings_yaml_path)
    components = tuner.precompute_query_components(
        retriever, target_kf_pairs, text_dense_encoder=_text_encoder,
        top_k=args.top_k, filter_meta=args.filter_meta,
    )
    retriever.close()

    search_result = tuner.grid_search(components, step=args.step)

    if args.save_domain:
        tuner.save_domain_weights(args.save_domain, search_result["best_alpha"], search_result["best_beta"])

    print(json.dumps({
        "best_alpha": search_result["best_alpha"],
        "best_beta": search_result["best_beta"],
        "best_gamma": search_result["best_gamma"],
        "best_mean_mrr": search_result["best_mean_mrr"],
        "saved_to_domain": args.save_domain,
    }, ensure_ascii=False, indent=2))
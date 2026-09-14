# src/phase3_search/retriever.py
"""
PHASE 3 — Retriever: Tri-Search trên Qdrant Local (spec V8.0 §2.1, §3.2)

Trách nhiệm:
    1. Query 3 "không gian" trong CÙNG 1 collection (mỗi domain = 1
       collection riêng, §2.1) bằng 3 named-vector khác nhau:
           - "text_dense"  (BGE-M3)   -> S_semantic
           - "image_dense" (SigLIP 2) -> S_visual
           - "text_sparse" (BM25)     -> S_keyword
    2. Mỗi không gian được chấm điểm theo Reciprocal Rank (1/(k+rank))
       thay vì similarity thô — giúp công thức RRF ổn định về mặt toán
       học khi pool candidate phình to ở các bước Router/Graph-hop phía
       sau (§3.4 điểm 3): "RRF là rank-based nên việc pool phình to ở
       bước 1–2 không phá công thức ở bước 3".
    3. Áp dụng công thức:
           S_total = α · S_semantic + β · S_visual + (1 − α − β) · S_keyword
    4. `-f / --filter_meta` BẮT BUỘC được resolve thành Qdrant Filter và
       áp dụng trực tiếp lên payload (VD: chặn cross-domain / cross-môn
       học) — filter luôn kết hợp AND với điều kiện `commit_status =
       committed` để không bao giờ trả về dữ liệu dở dang (pending).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

RRF_K = 60  # hằng số chuẩn của Reciprocal Rank Fusion (Cormack et al.)


@dataclass
class CandidateChunk:
    point_id: str
    payload: Dict[str, Any]
    s_semantic: float = 0.0
    s_visual: float = 0.0
    s_keyword: float = 0.0
    s_total: float = 0.0
    source_flags: List[str] = field(default_factory=list)


class Retriever:
    def __init__(self, qdrant_path: Optional[str] = None, collection_name: str = "default"):
        from qdrant_client import QdrantClient
        from backend.src.common.config_loader import config
        
        self.qdrant_path = qdrant_path or config.get("paths", {}).get("qdrant_db", "backend/data/qdrant_db")
        self.collection_name = collection_name
        self.client = QdrantClient(path=self.qdrant_path)

    def close(self) -> None:
        self.client.close()

    # ------------------------------------------------------------------ #
    # --filter_meta parsing
    # ------------------------------------------------------------------ #
    @staticmethod
    def parse_filter_meta(filter_meta: Optional[str]):
        """
        Chuyển chuỗi --filter_meta thành qdrant_client Filter.
        Hỗ trợ 2 cú pháp:
            1. JSON:        '{"domain":"football_analytics","category":"highlight"}'
            2. key=value:   'domain=football_analytics,category=highlight'
        Mọi điều kiện được nối bằng AND (must). Trả về None nếu không có
        filter nào được truyền (chỉ áp commit_status=committed).
        """
        from qdrant_client.http import models as qmodels

        if not filter_meta:
            return None

        conditions: Dict[str, str] = {}
        try:
            parsed = json.loads(filter_meta)
            if isinstance(parsed, dict):
                conditions = {str(k): str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            for pair in filter_meta.split(","):
                if "=" not in pair:
                    continue
                key, value = pair.split("=", 1)
                conditions[key.strip()] = value.strip()

        if not conditions:
            return None

        must_conditions = [
            qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v))
            for k, v in conditions.items()
        ]
        return qmodels.Filter(must=must_conditions)

    def _build_committed_filter(self, user_filter):
        """Luôn AND thêm điều kiện commit_status=committed vào filter người dùng."""
        from qdrant_client.http import models as qmodels

        committed_cond = qmodels.FieldCondition(
            key="commit_status", match=qmodels.MatchValue(value="committed")
        )
        if user_filter is None:
            return qmodels.Filter(must=[committed_cond])

        merged_must = list(user_filter.must) if user_filter.must else []
        merged_must.append(committed_cond)
        return qmodels.Filter(
            must=merged_must,
            should=user_filter.should,
            must_not=user_filter.must_not,
        )

    # ------------------------------------------------------------------ #
    # Single-space search
    # ------------------------------------------------------------------ #
    def _search_named_vector(
        self,
        vector_name: str,
        query_vector: Sequence[float],
        top_k: int,
        qdrant_filter,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        from qdrant_client.http import models as qmodels

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=qmodels.NamedVector(name=vector_name, vector=list(query_vector)),
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )
        return [(hit.id, hit.payload) for hit in hits]

    def _search_sparse(
        self,
        sparse_query: Dict[int, float],
        top_k: int,
        qdrant_filter,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        from qdrant_client.http import models as qmodels

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=qmodels.NamedSparseVector(
                name="text_sparse",
                vector=qmodels.SparseVector(
                    indices=list(sparse_query.keys()), values=list(sparse_query.values())
                ),
            ),
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )
        return [(hit.id, hit.payload) for hit in hits]

    # ------------------------------------------------------------------ #
    # Tri-Search RRF (§3.2)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rrf_scores(ranked_ids: List[str], k: int = RRF_K) -> Dict[str, float]:
        return {pid: 1.0 / (k + rank + 1) for rank, pid in enumerate(ranked_ids)}

    def tri_search(
        self,
        text_dense_query: Optional[Sequence[float]],
        image_dense_query: Optional[Sequence[float]],
        sparse_query: Optional[Dict[int, float]],
        alpha: float,
        beta: float,
        top_k: int = 20,
        filter_meta: Optional[str] = None,
        active_flags: Optional[Sequence[str]] = None,
    ) -> List[CandidateChunk]:
        """
        Quét diện rộng lên tối đa 3 không gian (semantic/visual/keyword),
        chỉ quét những không gian có cờ tương ứng BẬT trong `active_flags`
        (kết quả từ Micro-Router §3.1; None = quét cả 3).
        """
        user_filter = self.parse_filter_meta(filter_meta)
        qdrant_filter = self._build_committed_filter(user_filter)

        flags = set(active_flags) if active_flags else {"semantic", "visual", "keyword"}

        semantic_hits: List[Tuple[str, Dict[str, Any]]] = []
        visual_hits: List[Tuple[str, Dict[str, Any]]] = []
        keyword_hits: List[Tuple[str, Dict[str, Any]]] = []

        if "semantic" in flags and text_dense_query is not None:
            semantic_hits = self._search_named_vector("text_dense", text_dense_query, top_k, qdrant_filter)
        if "visual" in flags and image_dense_query is not None:
            visual_hits = self._search_named_vector("image_dense", image_dense_query, top_k, qdrant_filter)
        if "keyword" in flags and sparse_query:
            keyword_hits = self._search_sparse(sparse_query, top_k, qdrant_filter)

        semantic_rrf = self._rrf_scores([pid for pid, _ in semantic_hits])
        visual_rrf = self._rrf_scores([pid for pid, _ in visual_hits])
        keyword_rrf = self._rrf_scores([pid for pid, _ in keyword_hits])

        payload_by_id: Dict[str, Dict[str, Any]] = {}
        for pid, payload in semantic_hits + visual_hits + keyword_hits:
            payload_by_id[pid] = payload

        gamma = max(0.0, 1.0 - alpha - beta)  # trọng số keyword ngầm định

        candidates: Dict[str, CandidateChunk] = {}
        for pid, payload in payload_by_id.items():
            s_sem = semantic_rrf.get(pid, 0.0)
            s_vis = visual_rrf.get(pid, 0.0)
            s_kw = keyword_rrf.get(pid, 0.0)
            s_total = alpha * s_sem + beta * s_vis + gamma * s_kw

            source_flags = []
            if pid in semantic_rrf: source_flags.append("semantic")
            if pid in visual_rrf: source_flags.append("visual")
            if pid in keyword_rrf: source_flags.append("keyword")

            candidates[pid] = CandidateChunk(
                point_id=pid, payload=payload,
                s_semantic=s_sem, s_visual=s_vis, s_keyword=s_kw,
                s_total=s_total, source_flags=source_flags,
            )

        ranked = sorted(candidates.values(), key=lambda c: c.s_total, reverse=True)
        return ranked


if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Phase 3 Retriever (standalone test).")
    parser.add_argument("--qdrant_path", required=True)
    parser.add_argument("--collection_name", required=True)
    parser.add_argument("-f", "--filter_meta", default=None)
    parser.add_argument("-t", "--top_k", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.2)
    args = parser.parse_args()

    retriever = Retriever(args.qdrant_path, args.collection_name)
    # Vector giả lập chỉ để test đường ống (production: lấy từ embedding.py)
    dummy_text_vec = np.random.rand(1024).astype(np.float32)
    results = retriever.tri_search(
        text_dense_query=dummy_text_vec.tolist(),
        image_dense_query=None,
        sparse_query=None,
        alpha=args.alpha, beta=args.beta, top_k=args.top_k,
        filter_meta=args.filter_meta,
    )
    retriever.close()

    print(json.dumps([{
        "point_id": c.point_id, "s_total": round(c.s_total, 5),
        "source_flags": c.source_flags,
        "text_preview": (c.payload.get("text") or "")[:80],
    } for c in results], ensure_ascii=False, indent=2))
# src/phase3_search/graph_router.py
"""
PHASE 3 — Graph Router: Multi-hop qua Kùzu (spec V8.0 §3.3, §3.4 bước 2)

- `--hop_limit` được clamp NGHIÊM NGẶT trước khi build query:
      safe_hop = max(1, min(int(hop_limit), 5))
- Toàn bộ giá trị động (id, tham số) truyền qua Parameterized Query của
  Kùzu (`parameters={...}`) — KHÔNG f-string / nối chuỗi trực tiếp giá
  trị người dùng, triệt tiêu Cypher Injection. Riêng con số lặp `*1..N`
  là literal bắt buộc theo cú pháp Cypher (Kùzu không cho tham số hoá số
  lần lặp), nhưng N đã được ép kiểu `int()` + clamp về [1,5] NGAY TRƯỚC
  khi chèn vào chuỗi query nên không có bề mặt injection nào còn lại.

- DAG bước 2 (§3.4): Graph Hop chỉ có nhiệm vụ "nhảy cóc" gom thêm ứng
  viên node lân cận — KHÔNG tự chấm điểm; điểm số cuối cùng do
  reranker.py + Global Rank (RRF) đảm nhiệm ở bước 3.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Set


def clamp_hop_limit(hop_limit: int) -> int:
    """safe_hop = max(1, min(int(hop_limit), 5)) — đúng nguyên văn spec §3.3."""
    try:
        value = int(hop_limit)
    except (TypeError, ValueError):
        value = 1
    return max(1, min(value, 5))


class GraphRouter:
    def __init__(self, kuzu_graph_path: Optional[str] = None):
        import kuzu
        from backend.src.common.config_loader import config

        # Ưu tiên path truyền vào, nếu không có thì lấy từ settings.yaml
        if not kuzu_graph_path:
            kuzu_graph_path = config.get("paths", {}).get("kuzu_graph", "backend/data/kuzu_graph")
            
        self.db = kuzu.Database(kuzu_graph_path)
        self.conn = kuzu.Connection(self.db)

    def close(self) -> None:
        self.conn = None
        self.db = None

    # ------------------------------------------------------------------ #
    # NEXT_EVENT — nhảy cóc theo trục THỜI GIAN, đa hop, clamp [1,5]
    # ------------------------------------------------------------------ #
    def hop_temporal(self, seed_chunk_id: str, hop_limit: int) -> List[str]:
        safe_hop = clamp_hop_limit(hop_limit)
        # safe_hop là int đã clamp -> an toàn khi chèn vào literal *1..N của Cypher.
        query = f"""
        MATCH (a:Chunk {{chunk_id: $seed_id}})-[:NEXT_EVENT*1..{safe_hop}]->(b:Chunk)
        RETURN DISTINCT b.chunk_id AS chunk_id
        """
        result = self.conn.execute(query, parameters={"seed_id": seed_chunk_id})
        return self._collect_ids(result)

    # ------------------------------------------------------------------ #
    # SAME_SCENE — nhảy cóc theo trục KHÔNG GIAN (1 hop trực tiếp)
    # ------------------------------------------------------------------ #
    def hop_spatial(self, seed_chunk_id: str) -> List[str]:
        query = """
        MATCH (a:Chunk {chunk_id: $seed_id})-[:SAME_SCENE]->(b:Chunk)
        RETURN DISTINCT b.chunk_id AS chunk_id
        """
        result = self.conn.execute(query, parameters={"seed_id": seed_chunk_id})
        return self._collect_ids(result)

    # ------------------------------------------------------------------ #
    # CROSS_SCENE_PAIR — lấy partner của sub-chunk vắt cảnh (1 hop)
    # ------------------------------------------------------------------ #
    def hop_cross_scene_partner(self, seed_chunk_id: str) -> List[str]:
        query = """
        MATCH (a:Chunk {chunk_id: $seed_id})-[:CROSS_SCENE_PAIR]->(b:Chunk)
        RETURN DISTINCT b.chunk_id AS chunk_id
        """
        result = self.conn.execute(query, parameters={"seed_id": seed_chunk_id})
        return self._collect_ids(result)

    @staticmethod
    def _collect_ids(result) -> List[str]:
        ids: List[str] = []
        while result.has_next():
            row = result.get_next()
            ids.append(row[0])
        return ids

    # ------------------------------------------------------------------ #
    # Orchestration — gom toàn bộ neighbor cho 1 tập seed
    # ------------------------------------------------------------------ #
    def expand_context(
        self,
        seed_chunk_ids: Sequence[str],
        hop_limit: int,
        use_temporal: bool = True,
        use_spatial: bool = True,
        use_cross_scene: bool = True,
    ) -> List[str]:
        """
        Với mỗi seed chunk (Top-K từ retriever.py), nhảy cóc gom node lân
        cận qua 3 loại edge. Trả về danh sách chunk_id MỚI (không trùng
        seed) để đưa vào Global Rank (§3.4 bước 3).
        """
        safe_hop = clamp_hop_limit(hop_limit)
        seed_set = set(seed_chunk_ids)
        expanded: Set[str] = set()

        for seed_id in seed_chunk_ids:
            if use_temporal:
                expanded.update(self.hop_temporal(seed_id, safe_hop))
            if use_spatial:
                expanded.update(self.hop_spatial(seed_id))
            if use_cross_scene:
                expanded.update(self.hop_cross_scene_partner(seed_id))

        # Chỉ trả về node MỚI, chưa nằm trong seed ban đầu
        return list(expanded - seed_set)

    # ------------------------------------------------------------------ #
    # Tra cứu metadata node (dùng để join lại với Qdrant nếu cần)
    # ------------------------------------------------------------------ #
    def get_node_metadata(self, chunk_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        if not chunk_ids:
            return {}
        query = """
        UNWIND $ids AS cid
        MATCH (c:Chunk {chunk_id: cid})
        RETURN c.chunk_id AS chunk_id, c.vid_hash AS vid_hash, c.domain AS domain,
               c.text AS text, c.start_sec AS start_sec, c.end_sec AS end_sec,
               c.scene_index AS scene_index, c.keyframe_path AS keyframe_path,
               c.parent_chunk_id AS parent_chunk_id
        """
        result = self.conn.execute(query, parameters={"ids": list(chunk_ids)})
        metadata: Dict[str, Dict[str, Any]] = {}
        while result.has_next():
            row = result.get_next()
            metadata[row[0]] = {
                "vid_hash": row[1], "domain": row[2], "text": row[3],
                "start_sec": row[4], "end_sec": row[5], "scene_index": row[6],
                "keyframe_path": row[7], "parent_chunk_id": row[8],
            }
        return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3 Graph Router (standalone test).")
    parser.add_argument("--kuzu_graph_path", required=True)
    parser.add_argument("--seed_chunk_ids", required=True, help="JSON list of chunk_id")
    parser.add_argument("--hop_limit", type=int, default=2)
    args = parser.parse_args()

    seed_ids = json.loads(args.seed_chunk_ids)
    router = GraphRouter(args.kuzu_graph_path)
    neighbor_ids = router.expand_context(seed_ids, args.hop_limit)
    metadata = router.get_node_metadata(neighbor_ids)
    router.close()

    print(json.dumps({
        "safe_hop_used": clamp_hop_limit(args.hop_limit),
        "neighbor_count": len(neighbor_ids),
        "neighbors": metadata,
    }, ensure_ascii=False, indent=2))
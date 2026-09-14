# src/phase2_build/kuzu_builder.py
"""
PHASE 2 — GraphRAG Builder bằng Kùzu (spec V8.0 §2.3)

Thay thế hoàn toàn việc lưu `.graphml` toàn cục bằng CSDL đồ thị nhúng
Kùzu, ghi trực tiếp xuống ổ cứng tại `data/kuzu_graph/`, hỗ trợ
update/append gia tăng (incremental) khi có video mới mà KHÔNG cần
rebuild toàn bộ graph mỗi lần.

Ràng buộc bắt buộc (đã enforce ở tầng gọi — cli_pipeline.py):
    - Graph CHỈ được build SAU KHI toàn bộ vector của 1 VID_Hash đã ở
      trạng thái `commit_status = committed` trong Qdrant (§0.3). Module
      này không tự kiểm tra Qdrant — nó tin tưởng caller (dual_tracker +
      qdrant_manager.confirm_committed) đã xác nhận điều đó trước khi
      gọi build_graph_for_video().

Schema:
    Node  Chunk(chunk_id STRING PK, vid_hash, domain, text, start_sec,
                 end_sec, scene_index, keyframe_path, parent_chunk_id)
    Edge  NEXT_EVENT      : liên kết THỜI GIAN — chunk kế tiếp theo timeline
                             của cùng 1 video (gap_sec = khoảng cách thời gian).
    Edge  SAME_SCENE       : liên kết KHÔNG GIAN — các chunk cùng scene_index.
    Edge  CROSS_SCENE_PAIR : liên kết cặp sub-chunk vắt cảnh, cùng parent_chunk_id.

Toàn bộ câu Cypher đều dùng THAM SỐ HOÁ (`parameters={...}`) của thư viện
Kùzu — không nối chuỗi/f-string trực tiếp giá trị người dùng vào query,
triệt tiêu rủi ro Cypher Injection (đúng nguyên tắc §3.3, áp dụng luôn ở
tầng ghi dữ liệu này để nhất quán toàn hệ thống).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class KuzuGraphBuilder:
    def __init__(self, kuzu_graph_path: Optional[str] = None):
        import kuzu
        from backend.src.common.config_loader import config
        
        # Tự động lấy path từ config nếu không truyền tay
        if not kuzu_graph_path:
            kuzu_graph_path = config.get("paths", {}).get("kuzu_graph", "backend/data/kuzu_graph")
            
        Path(kuzu_graph_path).mkdir(parents=True, exist_ok=True)
        self.db = kuzu.Database(kuzu_graph_path)
        self.conn = kuzu.Connection(self.db)
        self._init_schema()

    # ------------------------------------------------------------------ #
    # Schema — idempotent (chạy nhiều lần không lỗi, phục vụ incremental)
    # ------------------------------------------------------------------ #
    def _init_schema(self) -> None:
        node_stmt = """
        CREATE NODE TABLE IF NOT EXISTS Chunk(
            chunk_id STRING,
            vid_hash STRING,
            domain STRING,
            text STRING,
            start_sec DOUBLE,
            end_sec DOUBLE,
            scene_index INT64,
            keyframe_path STRING,
            parent_chunk_id STRING,
            PRIMARY KEY (chunk_id)
        );
        """
        self.conn.execute(node_stmt)

        rel_statements = [
            """
            CREATE REL TABLE IF NOT EXISTS NEXT_EVENT(
                FROM Chunk TO Chunk,
                gap_sec DOUBLE
            );
            """,
            """
            CREATE REL TABLE IF NOT EXISTS SAME_SCENE(
                FROM Chunk TO Chunk
            );
            """,
            """
            CREATE REL TABLE IF NOT EXISTS CROSS_SCENE_PAIR(
                FROM Chunk TO Chunk,
                parent_chunk_id STRING
            );
            """,
        ]
        for stmt in rel_statements:
            self.conn.execute(stmt)

    # ------------------------------------------------------------------ #
    # Nodes — Thực thể/Chunk
    # ------------------------------------------------------------------ #
    def upsert_chunk_nodes(self, chunks: Sequence[Dict[str, Any]], vid_hash: str, domain: str) -> int:
        """
        MERGE từng Chunk node (idempotent — chạy lại không tạo trùng),
        tham số hoá 100% để chống Cypher Injection.
        """
        query = """
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c.vid_hash = $vid_hash,
            c.domain = $domain,
            c.text = $text,
            c.start_sec = $start_sec,
            c.end_sec = $end_sec,
            c.scene_index = $scene_index,
            c.keyframe_path = $keyframe_path,
            c.parent_chunk_id = $parent_chunk_id
        """
        count = 0
        for chunk in chunks:
            self.conn.execute(
                query,
                parameters={
                    "chunk_id": chunk["chunk_id"],
                    "vid_hash": vid_hash,
                    "domain": domain,
                    "text": chunk.get("text", ""),
                    "start_sec": float(chunk.get("start", 0.0)),
                    "end_sec": float(chunk.get("end", 0.0)),
                    "scene_index": int(chunk.get("scene_index", -1)),
                    "keyframe_path": chunk.get("keyframe_path") or "",
                    "parent_chunk_id": chunk.get("parent_chunk_id") or "",
                },
            )
            count += 1
        return count

    # ------------------------------------------------------------------ #
    # Edges — Thời gian (NEXT_EVENT)
    # ------------------------------------------------------------------ #
    def build_temporal_edges(self, chunks: Sequence[Dict[str, Any]]) -> int:
        """Nối các chunk liên tiếp theo trục thời gian (cùng 1 video) bằng NEXT_EVENT."""
        sorted_chunks = sorted(chunks, key=lambda c: float(c.get("start", 0.0)))
        query = """
        MATCH (a:Chunk {chunk_id: $a_id}), (b:Chunk {chunk_id: $b_id})
        MERGE (a)-[r:NEXT_EVENT]->(b)
        SET r.gap_sec = $gap_sec
        """
        count = 0
        for i in range(len(sorted_chunks) - 1):
            a, b = sorted_chunks[i], sorted_chunks[i + 1]
            gap_sec = float(b.get("start", 0.0)) - float(a.get("end", 0.0))
            self.conn.execute(
                query,
                parameters={"a_id": a["chunk_id"], "b_id": b["chunk_id"], "gap_sec": gap_sec},
            )
            count += 1
        return count

    # ------------------------------------------------------------------ #
    # Edges — Không gian (SAME_SCENE)
    # ------------------------------------------------------------------ #
    def build_spatial_edges(self, chunks: Sequence[Dict[str, Any]]) -> int:
        """Nối các chunk cùng scene_index (liên kết không gian trong cùng 1 cảnh quay)."""
        groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for c in chunks:
            groups[int(c.get("scene_index", -1))].append(c)

        query = """
        MATCH (a:Chunk {chunk_id: $a_id}), (b:Chunk {chunk_id: $b_id})
        MERGE (a)-[:SAME_SCENE]->(b)
        """
        count = 0
        for scene_idx, group in groups.items():
            if scene_idx < 0 or len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    self.conn.execute(
                        query,
                        parameters={"a_id": group[i]["chunk_id"], "b_id": group[j]["chunk_id"]},
                    )
                    count += 1
        return count

    # ------------------------------------------------------------------ #
    # Edges — Cặp sub-chunk vắt cảnh (CROSS_SCENE_PAIR)
    # ------------------------------------------------------------------ #
    def build_cross_scene_pair_edges(self, chunks: Sequence[Dict[str, Any]]) -> int:
        """Nối cặp sub-chunk chia sẻ cùng parent_chunk_id (câu vắt ngang 2 cảnh, §1.4)."""
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for c in chunks:
            pid = c.get("parent_chunk_id")
            if pid:
                groups[pid].append(c)

        query = """
        MATCH (a:Chunk {chunk_id: $a_id}), (b:Chunk {chunk_id: $b_id})
        MERGE (a)-[r:CROSS_SCENE_PAIR]->(b)
        SET r.parent_chunk_id = $parent_chunk_id
        """
        count = 0
        for parent_id, group in groups.items():
            if len(group) < 2:
                continue
            # Thông thường mỗi cặp chỉ có đúng 2 phần tử (A, B) theo thiết kế hybrid_chunking.py
            a, b = group[0], group[1]
            self.conn.execute(
                query,
                parameters={"a_id": a["chunk_id"], "b_id": b["chunk_id"], "parent_chunk_id": parent_id},
            )
            count += 1
        return count

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def build_graph_for_video(
        self,
        chunks: Sequence[Dict[str, Any]],
        vid_hash: str,
        domain: str,
    ) -> Dict[str, int]:
        """
        Entry-point chính: build toàn bộ Node + Edge cho 1 video đã có
        vector `committed`. Có thể gọi lại an toàn (idempotent nhờ MERGE)
        nếu bước build trước đó bị gián đoạn giữa chừng.
        """
        n_nodes = self.upsert_chunk_nodes(chunks, vid_hash, domain)
        n_temporal = self.build_temporal_edges(chunks)
        n_spatial = self.build_spatial_edges(chunks)
        n_pairs = self.build_cross_scene_pair_edges(chunks)

        return {
            "nodes_upserted": n_nodes,
            "next_event_edges": n_temporal,
            "same_scene_edges": n_spatial,
            "cross_scene_pair_edges": n_pairs,
        }

    # ------------------------------------------------------------------ #
    # Multi-hop query tiện ích (dùng lại ở Phase 3 graph_router.py)
    # ------------------------------------------------------------------ #
    def multi_hop_from(self, start_chunk_id: str, hop_limit: int = 2) -> List[str]:
        """
        Truy vấn multi-hop qua NEXT_EVENT, clamp hop_limit về [1,5] (§3.3)
        và dùng Parameterized Query cho start_chunk_id (id vẫn tham số
        hoá được; chỉ riêng con số lặp `*1..N` trong Cypher của Kùzu bắt
        buộc phải là literal nên được clamp CHẶT bằng int() trước khi
        chèn — không bao giờ chèn trực tiếp giá trị thô từ người dùng).
        """
        safe_hop = max(1, min(int(hop_limit), 5))
        query = f"""
        MATCH (a:Chunk {{chunk_id: $start_id}})-[:NEXT_EVENT*1..{safe_hop}]->(b:Chunk)
        RETURN DISTINCT b.chunk_id AS chunk_id
        """
        result = self.conn.execute(query, parameters={"start_id": start_chunk_id})
        chunk_ids: List[str] = []
        while result.has_next():
            row = result.get_next()
            chunk_ids.append(row[0])
        return chunk_ids

    def close(self) -> None:
        # Kùzu tự flush khi Connection/Database bị garbage-collected; xoá
        # tham chiếu tường minh để giải phóng ngay, tránh giữ file lock.
        self.conn = None
        self.db = None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 Kùzu Graph Builder (standalone test).")
    parser.add_argument("--kuzu_graph_path", required=True)
    parser.add_argument("--chunks_json", required=True, help="Path tới JSON list chunk (từ hybrid_chunking.py)")
    parser.add_argument("--vid_hash", required=True)
    parser.add_argument("--domain", default="unknown")
    args = parser.parse_args()

    with open(args.chunks_json, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    builder = KuzuGraphBuilder(args.kuzu_graph_path)
    stats = builder.build_graph_for_video(chunks, args.vid_hash, args.domain)
    builder.close()

    print(json.dumps({"status": "ok", "stats": stats}, ensure_ascii=False, indent=2))
# src/phase2_build/qdrant_manager.py
"""
PHASE 2 — Qdrant Tri-Search DB Manager (spec V8.0 §2.1, liên kết §0.3)

Trách nhiệm:
    1. Đọc `source_meta.json` (đường dẫn truyền qua --meta_mapping) — file
       này nằm cạnh video gốc trong data/raw_videos/, chứa metadata cấp
       video (title, domain, source_url, category, ...).
    2. Merge metadata video + metadata chunk (parent_chunk_id, keyframe_path,
       timestamp, yolo_detections, ...) thành MỘT payload JSON thống nhất
       cho mỗi point trước khi upsert.
    3. Cô lập Domain: mỗi `domain` = 1 Collection Qdrant riêng biệt (§2.1),
       không dùng payload filter chung.
    4. Transactional Rollback (§0.3): mọi point ghi vào với
       `commit_status: "pending"` trước, chỉ chuyển "committed" sau khi
       TOÀN BỘ batch của 1 VID_Hash ghi xong và xác nhận sạch.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class QdrantManager:
    def __init__(self, qdrant_path: Optional[str] = None, collection_name: str = "default"):
        from qdrant_client import QdrantClient
        from backend.src.common.config_loader import config # Kéo config vào

        # Ưu tiên path truyền vào, nếu không có thì lấy từ settings.yaml, mặc định là "data/qdrant_db"
        self.qdrant_path = qdrant_path or config.get("paths", {}).get("qdrant_db", "backend/data/qdrant_db")
        self.collection_name = collection_name
        self.client = QdrantClient(path=self.qdrant_path)

    def close(self) -> None:
        self.client.close()

    # ------------------------------------------------------------------ #
    # Collection lifecycle — mỗi domain là 1 collection riêng (§2.1)
    # ------------------------------------------------------------------ #
    def ensure_collection(
        self,
        text_dense_dim: int = 1024,   # BGE-M3 default dim
        image_dense_dim: int = 768,   # SigLIP2-base default dim
    ) -> None:
        from qdrant_client.http import models as qmodels

        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "text_dense": qmodels.VectorParams(size=text_dense_dim, distance=qmodels.Distance.COSINE),
                "image_dense": qmodels.VectorParams(size=image_dense_dim, distance=qmodels.Distance.COSINE),
            },
            sparse_vectors_config={
                "text_sparse": qmodels.SparseVectorParams(
                    index=qmodels.SparseIndexParams(on_disk=False)
                ),
            },
        )
        # Index các trường payload dùng để filter thường xuyên (§3.1 -f/--filter_meta)
        for field_name, schema in [
            ("vid_hash", qmodels.PayloadSchemaType.KEYWORD),
            ("commit_status", qmodels.PayloadSchemaType.KEYWORD),
            ("parent_chunk_id", qmodels.PayloadSchemaType.KEYWORD),
            ("domain", qmodels.PayloadSchemaType.KEYWORD),
        ]:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=schema,
            )

    # ------------------------------------------------------------------ #
    # source_meta.json (--meta_mapping)
    # ------------------------------------------------------------------ #
    @staticmethod
    def load_source_meta(meta_path: str) -> Dict[str, Any]:
        """
        Đọc file source_meta.json nằm cạnh video gốc. Cấu trúc kỳ vọng
        tối thiểu:
            {
              "domain": "football_analytics",
              "title": "...",
              "source_url": "...",
              "category": "...",
              "extra": { ... }   # mọi field tuỳ biến khác được giữ nguyên
            }
        Nếu file không tồn tại/không hợp lệ -> trả về dict rỗng, pipeline
        vẫn tiếp tục chạy với metadata tối thiểu (domain='unknown').
        """
        path_obj = Path(meta_path)
        if not path_obj.is_file():
            return {"domain": "unknown"}
        try:
            with open(path_obj, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "domain" not in data:
                data["domain"] = "unknown"
            return data
        except (json.JSONDecodeError, OSError):
            return {"domain": "unknown"}

    @staticmethod
    def build_unified_payload(
        chunk: Dict[str, Any],
        source_meta: Dict[str, Any],
        vid_hash: str,
        commit_status: str = "pending",
    ) -> Dict[str, Any]:
        """
        Hợp nhất metadata video (source_meta) + metadata chunk vào 1
        payload JSON duy nhất theo đúng danh sách bắt buộc trong §2.1:
        keyframe_path, commit_status, parent_chunk_id, metadata YOLO, ...
        """
        payload: Dict[str, Any] = {
            # --- Định danh & trạng thái giao dịch ---
            "vid_hash": vid_hash,
            "chunk_id": chunk.get("chunk_id"),
            "parent_chunk_id": chunk.get("parent_chunk_id"),
            "is_cross_scene_subchunk": chunk.get("is_cross_scene_subchunk", False),
            "commit_status": commit_status,

            # --- Nội dung & định vị thời gian ---
            "text": chunk.get("text", ""),
            "start_sec": chunk.get("start"),
            "end_sec": chunk.get("end"),
            "scene_index": chunk.get("scene_index"),
            "keyframe_path": chunk.get("keyframe_path"),
            "yolo_detections": chunk.get("yolo_detections", []),

            # --- Metadata cấp video (từ source_meta.json) ---
            "domain": source_meta.get("domain", "unknown"),
            "title": source_meta.get("title"),
            "source_url": source_meta.get("source_url"),
            "category": source_meta.get("category"),
        }
        # Giữ nguyên mọi field tuỳ biến khác trong source_meta (nếu có)
        extra = source_meta.get("extra")
        if isinstance(extra, dict):
            payload["extra"] = extra

        return payload

    # ------------------------------------------------------------------ #
    # Upsert theo batch — commit_status = pending
    # ------------------------------------------------------------------ #
    def upsert_pending(
        self,
        chunks: Sequence[Dict[str, Any]],
        text_dense_vectors: Optional[np.ndarray],
        image_dense_vectors: Optional[np.ndarray],
        sparse_vectors: Optional[List[Dict[int, float]]],
        source_meta: Dict[str, Any],
        vid_hash: str,
    ) -> List[str]:
        """
        Upsert toàn bộ chunk của 1 VID_Hash với commit_status='pending'.
        Mỗi chunk map 1-1 với point ID (uuid4) được sinh mới; text_dense
        vector lấy theo index tương ứng trong text_dense_vectors (đã
        encode theo đúng thứ tự `chunks`), image_dense vector tương tự
        (nếu chunk có keyframe -> có ảnh -> có vector ảnh, ngược lại
        dùng vector 0 để giữ schema nhất quán).
        """
        from qdrant_client.http import models as qmodels

        self.ensure_collection(
            text_dense_dim=int(text_dense_vectors.shape[1]) if text_dense_vectors is not None and text_dense_vectors.size else 1024,
            image_dense_dim=int(image_dense_vectors.shape[1]) if image_dense_vectors is not None and image_dense_vectors.size else 768,
        )

        points: List[Any] = []
        point_ids: List[str] = []

        for i, chunk in enumerate(chunks):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)

            text_vec = (
                text_dense_vectors[i].tolist()
                if text_dense_vectors is not None and i < len(text_dense_vectors)
                else [0.0] * (text_dense_vectors.shape[1] if text_dense_vectors is not None and text_dense_vectors.size else 1024)
            )
            image_vec = (
                image_dense_vectors[i].tolist()
                if image_dense_vectors is not None and i < len(image_dense_vectors)
                else [0.0] * (image_dense_vectors.shape[1] if image_dense_vectors is not None and image_dense_vectors.size else 768)
            )

            vector_payload: Dict[str, Any] = {
                "text_dense": text_vec,
                "image_dense": image_vec,
            }
            if sparse_vectors is not None and i < len(sparse_vectors):
                sparse = sparse_vectors[i]
                vector_payload["text_sparse"] = qmodels.SparseVector(
                    indices=list(sparse.keys()),
                    values=list(sparse.values()),
                )

            payload = self.build_unified_payload(chunk, source_meta, vid_hash, commit_status="pending")

            points.append(qmodels.PointStruct(id=point_id, vector=vector_payload, payload=payload))

        # Upsert theo batch nội bộ 64 point/lần để tránh request quá lớn
        BATCH = 64
        for start in range(0, len(points), BATCH):
            self.client.upsert(collection_name=self.collection_name, points=points[start:start + BATCH])

        return point_ids

    def confirm_committed(self, vid_hash: str) -> int:
        """
        Sau khi toàn bộ batch của VID_Hash ghi xong sạch (§0.3), chuyển
        commit_status: pending -> committed cho toàn bộ point của
        VID_Hash đó. Trả về số point đã cập nhật.
        """
        from qdrant_client.http import models as qmodels

        result = self.client.set_payload(
            collection_name=self.collection_name,
            payload={"commit_status": "committed"},
            points=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key="vid_hash", match=qmodels.MatchValue(value=vid_hash)),
                    qmodels.FieldCondition(key="commit_status", match=qmodels.MatchValue(value="pending")),
                ]
            ),
        )
        # Đếm lại số point committed để trả về (xác nhận sạch)
        count_result = self.client.count(
            collection_name=self.collection_name,
            count_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key="vid_hash", match=qmodels.MatchValue(value=vid_hash)),
                    qmodels.FieldCondition(key="commit_status", match=qmodels.MatchValue(value="committed")),
                ]
            ),
        )
        return count_result.count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 Qdrant Manager (standalone test / debug).")
    parser.add_argument("--qdrant_path", required=True)
    parser.add_argument("--collection_name", required=True)
    parser.add_argument("--meta_mapping", required=True, help="Path tới source_meta.json")
    parser.add_argument("--vid_hash", required=True)
    parser.add_argument("--chunks_json", required=True, help="Path tới JSON list chunk (từ hybrid_chunking.py)")
    parser.add_argument("--confirm_commit", action="store_true")
    args = parser.parse_args()

    manager = QdrantManager(args.qdrant_path, args.collection_name)
    source_meta = manager.load_source_meta(args.meta_mapping)

    with open(args.chunks_json, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if args.confirm_commit:
        n_committed = manager.confirm_committed(args.vid_hash)
        print(json.dumps({"status": "ok", "committed_points": n_committed}, ensure_ascii=False))
    else:
        # Test upsert với vector giả lập (chỉ để kiểm thử schema, không dùng production)
        dummy_text_vecs = np.zeros((len(chunks), 1024), dtype=np.float32)
        dummy_image_vecs = np.zeros((len(chunks), 768), dtype=np.float32)
        ids = manager.upsert_pending(chunks, dummy_text_vecs, dummy_image_vecs, None, source_meta, args.vid_hash)
        print(json.dumps({"status": "ok", "upserted_pending": len(ids)}, ensure_ascii=False))

    manager.close()
import argparse
import json
import shutil
from __future__ import annotations
import sqlite3
import sys
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

# Nạp config chung để gỡ bỏ hardcode
from backend.src.common.config_loader import config

STATUS_PROCESSING = "processing"
STATUS_FAILED = "failed"
STATUS_COMMITTED = "committed"
VALID_STATUSES = {STATUS_PROCESSING, STATUS_FAILED, STATUS_COMMITTED}

# Lấy đường dẫn động từ settings.yaml, có fallback mặc định
DEFAULT_DB_PATH = config.get("paths", {}).get(
    "tracker_db", "data/eval_logs/tracker_state.sqlite3"
)
# src/phase0_ops/dual_tracker.py
"""
PHASE 0 — Transactional Rollback / Dual Tracker (spec V8.0 §0.3)

Theo dõi trạng thái xử lý của từng VID_Hash qua 3 trạng thái:
    - processing : đang xử lý dở (chưa commit xong toàn bộ batch vector)
    - failed     : lần chạy trước bị crash / lỗi giữa chừng
    - committed  : toàn bộ vector của VID_Hash đã upsert + xác nhận sạch

Cơ chế "Dual" nằm ở chỗ tracker đồng thời quản lý 2 loại tài nguyên cần
rollback khi resume:
    1. File rác vật lý trong data/temp_workspace/<VID_Hash>/
    2. Vector còn ở trạng thái payload commit_status=pending trong Qdrant

Khi hệ thống sập nguồn và người dùng chạy lại với --resume_phase:
    a. Tracker tra toàn bộ VID_Hash có status != 'committed'.
    b. Với mỗi VID_Hash dở dang: xoá sạch temp_workspace tương ứng,
       xoá vector 'pending' còn sót trong Qdrant, rồi đưa VID_Hash đó
       trở lại hàng đợi xử lý từ đầu.
    c. Các VID_Hash đã 'committed' được bỏ qua hoàn toàn (skip), tránh
       xử lý lại tốn kém.

Graph (Kùzu) KHÔNG nằm trong phạm vi rollback của tracker này — vì Graph
chỉ được build sau khi vector đã committed (§0.3), nên nếu graph-build
sập giữa chừng, Qdrant vẫn sạch, ta chỉ cần gọi lại bước nối Edge, không
cần rollback vector.
"""
@dataclass
class TrackerRecord:
    vid_hash: str
    phase: int
    status: str
    temp_workspace_path: str
    source_video_path: str
    updated_at: str
    error_message: Optional[str] = None


class DualTracker:
    """Quản lý trạng thái xử lý VID_Hash qua SQLite, hỗ trợ --resume_phase."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------ #
    # Kết nối / Schema
    # ------------------------------------------------------------------ #
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")  # an toàn hơn khi crash giữa ghi
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tracker (
                    vid_hash            TEXT NOT NULL,
                    phase               INTEGER NOT NULL,
                    status              TEXT NOT NULL,
                    temp_workspace_path TEXT NOT NULL DEFAULT '',
                    source_video_path   TEXT NOT NULL DEFAULT '',
                    error_message       TEXT,
                    updated_at          TEXT NOT NULL,
                    PRIMARY KEY (vid_hash, phase)
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tracker_status ON tracker(status);"
            )

    # ------------------------------------------------------------------ #
    # Ghi trạng thái
    # ------------------------------------------------------------------ #
    def mark_processing(
        self,
        vid_hash: str,
        phase: int,
        temp_workspace_path: str = "",
        source_video_path: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tracker (vid_hash, phase, status, temp_workspace_path,
                                      source_video_path, error_message, updated_at)
                VALUES (?, ?, ?, ?, ?, NULL, ?)
                ON CONFLICT(vid_hash, phase) DO UPDATE SET
                    status=excluded.status,
                    temp_workspace_path=excluded.temp_workspace_path,
                    source_video_path=excluded.source_video_path,
                    error_message=NULL,
                    updated_at=excluded.updated_at;
                """,
                (vid_hash, phase, STATUS_PROCESSING, temp_workspace_path,
                 source_video_path, now),
            )

    def mark_committed(self, vid_hash: str, phase: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE tracker SET status = ?, updated_at = ?, error_message = NULL
                WHERE vid_hash = ? AND phase = ?;
                """,
                (STATUS_COMMITTED, now, vid_hash, phase),
            )
            if cur.rowcount == 0:
                raise ValueError(
                    f"[dual_tracker] Không thể commit: chưa có record processing "
                    f"cho VID_Hash={vid_hash}, phase={phase}."
                )

    def mark_failed(self, vid_hash: str, phase: int, error_message: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE tracker SET status = ?, updated_at = ?, error_message = ?
                WHERE vid_hash = ? AND phase = ?;
                """,
                (STATUS_FAILED, now, error_message, vid_hash, phase),
            )

    # ------------------------------------------------------------------ #
    # Đọc trạng thái
    # ------------------------------------------------------------------ #
    def get_record(self, vid_hash: str, phase: int) -> Optional[TrackerRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tracker WHERE vid_hash = ? AND phase = ?;",
                (vid_hash, phase),
            ).fetchone()
        if row is None:
            return None
        return TrackerRecord(
            vid_hash=row["vid_hash"], phase=row["phase"], status=row["status"],
            temp_workspace_path=row["temp_workspace_path"],
            source_video_path=row["source_video_path"],
            updated_at=row["updated_at"], error_message=row["error_message"],
        )

    def get_incomplete_records(self, phase: int) -> List[TrackerRecord]:
        """Trả về mọi VID_Hash chưa 'committed' ở phase chỉ định (processing/failed)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM tracker
                WHERE phase = ? AND status != ?
                ORDER BY updated_at ASC;
                """,
                (phase, STATUS_COMMITTED),
            ).fetchall()
        return [
            TrackerRecord(
                vid_hash=r["vid_hash"], phase=r["phase"], status=r["status"],
                temp_workspace_path=r["temp_workspace_path"],
                source_video_path=r["source_video_path"],
                updated_at=r["updated_at"], error_message=r["error_message"],
            )
            for r in rows
        ]

    def is_committed(self, vid_hash: str, phase: int) -> bool:
        rec = self.get_record(vid_hash, phase)
        return rec is not None and rec.status == STATUS_COMMITTED

    # ------------------------------------------------------------------ #
    # Resume / Rollback
    # ------------------------------------------------------------------ #
    def resume_phase(
        self,
        phase: int,
        qdrant_path: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> List[str]:
        """
        Thực thi quy trình rollback cho toàn bộ VID_Hash dở dang ở `phase`:
          1. Xoá temp_workspace vật lý.
          2. Xoá vector commit_status='pending' tương ứng trong Qdrant
             (nếu qdrant_path + collection_name được cung cấp).
          3. Reset record về trạng thái 'failed' để pipeline biết cần
             xử lý lại từ đầu.

        Returns:
            Danh sách VID_Hash đã được rollback và sẵn sàng chạy lại.
        """
        incomplete = self.get_incomplete_records(phase)
        rolled_back: List[str] = []

        for rec in incomplete:
            # 1) Dọn rác temp_workspace
            if rec.temp_workspace_path:
                ws_path = Path(rec.temp_workspace_path)
                if ws_path.exists() and ws_path.is_dir():
                    shutil.rmtree(ws_path, ignore_errors=True)
                    ws_path.mkdir(parents=True, exist_ok=True)

            # 2) Xoá vector pending còn sót trong Qdrant
            if qdrant_path and collection_name:
                self._purge_pending_vectors(rec.vid_hash, qdrant_path, collection_name)

            # 3) Reset trạng thái -> failed, chờ pipeline enqueue lại
            self.mark_failed(
                rec.vid_hash, phase,
                error_message="rolled_back_by_resume_phase",
            )
            rolled_back.append(rec.vid_hash)

        return rolled_back

    @staticmethod
    def _purge_pending_vectors(vid_hash: str, qdrant_path: str, collection_name: str) -> None:
        """
        Xoá toàn bộ point trong Qdrant local có payload
        {vid_hash: <VID_Hash>, commit_status: 'pending'}.
        Import qdrant_client cục bộ (lazy) để module này không bắt buộc
        phụ thuộc Qdrant khi chỉ dùng cho mục đích tracking thuần tuý.
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qmodels
        except ImportError as exc:
            raise RuntimeError(
                "[dual_tracker] Cần cài 'qdrant-client' để rollback vector pending."
            ) from exc

        client = QdrantClient(path=qdrant_path)
        try:
            if collection_name not in [c.name for c in client.get_collections().collections]:
                return  # Collection chưa tồn tại -> không có gì để xoá
            client.delete(
                collection_name=collection_name,
                wait=True,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="vid_hash",
                                match=qmodels.MatchValue(value=vid_hash),
                            ),
                            qmodels.FieldCondition(
                                key="commit_status",
                                match=qmodels.MatchValue(value="pending"),
                            ),
                        ]
                    )
                ),
            )
        finally:
            client.close()


# -------------------------------------------------------------------------- #
# CLI Entry-point độc lập cho tracker (dùng khi debug / gọi trực tiếp)
# -------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Dual Tracker CLI — quản lý resume/rollback.")
    parser.add_argument("--db_path", default=DEFAULT_DB_PATH)
    parser.add_argument("--resume_phase", type=int, help="Phase cần resume/rollback")
    parser.add_argument("--qdrant_path", type=str, default=None)
    parser.add_argument("--collection_name", type=str, default=None)
    args = parser.parse_args()

    tracker = DualTracker(db_path=args.db_path)

    if args.resume_phase is not None:
        rolled_back = tracker.resume_phase(
            phase=args.resume_phase,
            qdrant_path=args.qdrant_path,
            collection_name=args.collection_name,
        )
        print(json.dumps({
            "status": "ok",
            "phase": args.resume_phase,
            "rolled_back_vid_hashes": rolled_back,
            "count": len(rolled_back),
        }, ensure_ascii=False, indent=2))
    else:
        print("Không có hành động nào được yêu cầu. Dùng --resume_phase <id>.", file=sys.stderr)


if __name__ == "__main__":
    main()
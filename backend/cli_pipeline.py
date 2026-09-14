#!/usr/bin/env python3
# cli_pipeline.py  (đặt tại thư mục gốc hub_kien_thuc_v8/)
"""
HUB KIẾN THỨC — Entry-point trung tâm điều phối toàn dự án (spec V8.0)

Vai trò: một "bảng mạch" duy nhất — mọi Phase (0→4), mọi bước nhỏ (hash,
tracker, OCR-free vision, audio residual, chunking, embedding, Qdrant,
Kùzu, tri-search, graph hop, rerank, knapsack, VLM, eval cascade, tune
weights) đều được gọi thông qua CÙNG MỘT tập cờ `argparse`, không cần
import thủ công từng module.

    python cli_pipeline.py --run_phase all --input_dir data/raw_videos --build_graph
    python cli_pipeline.py --run_phase 3 --qdrant_path data/qdrant_db --domain football_analytics \
        --query "Ai ghi bàn thắng quyết định?" -f "category=highlight" -rr --use_graph --hop_limit 2 -tr
    python cli_pipeline.py --tune_weights --qdrant_path data/qdrant_db --domain football_analytics \
        --target_kf data/eval_logs/target_kf.json --save_domain football_analytics
    python cli_pipeline.py --run_phase 4 --prometheus_gguf_path data/models/prometheus2-7b.Q4_K_M.gguf

Toàn bộ import thư viện nặng đều LAZY (nằm trong hàm/phương thức của
từng module con) — cli_pipeline.py chỉ import các module `src.*` ở đầu
file (nhẹ, không tốn VRAM cho tới khi phase tương ứng thực sự chạy).
"""

from __future__ import annotations

import argparse
import csv
import glob
import gc
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Đảm bảo `src` nằm trên sys.path dù script được gọi từ đâu
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent    # trỏ về thư mục cha của cli_pipeline.py (root hub_kien_thuc_v8/backend)
SRC_DIR = ROOT_DIR /"src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase0_ops.hashing import compute_vid_hash                        # noqa: E402
from phase0_ops.dual_tracker import DualTracker, STATUS_COMMITTED       # noqa: E402
from phase0_ops import system_check                                     # noqa: E402

DEFAULT_QDRANT_PATH = str(ROOT_DIR / "data" / "qdrant_db")
DEFAULT_KUZU_PATH = str(ROOT_DIR / "data" / "kuzu_graph")
DEFAULT_INPUT_DIR = str(ROOT_DIR / "data" / "raw_videos")
DEFAULT_TEMP_WORKSPACE = str(ROOT_DIR / "data" / "temp_workspace")
DEFAULT_EVAL_LOGS_DIR = str(ROOT_DIR / "data" / "eval_logs")
DEFAULT_SETTINGS_YAML = str(ROOT_DIR / "config" / "settings.yaml")
STRESS_TEST_VIDEO = str(ROOT_DIR / "data" / "raw_videos" / "stress_test.mp4")
DEFAULT_ALPHA = 0.6
DEFAULT_BETA = 0.2



# =========================================================================== #
# Bootstrap: .env secrets + W&B
# =========================================================================== #
def load_env_secrets() -> None:
    """
    Nạp toàn bộ API Key (WANDB_API_KEY, QDRANT_API_KEY, GEMINI_API_KEY,
    GROQ_API_KEY) qua python-dotenv — KHÔNG hardcode (spec §0, hàng
    "Quản lý secret").
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=ROOT_DIR / ".env")
    except ImportError:
        # python-dotenv không bắt buộc phải cài nếu người dùng tự export
        # biến môi trường thủ công — không hard-exit vì đây không phải
        # xung đột dependency (khác pip check ở §0.1).
        pass


def setup_wandb(wandb_key: Optional[str], id_pro: Optional[str]):
    """
    Đồng bộ tiến độ ngầm lên W&B qua --wandb_key / WANDB_API_KEY +
    --id_pro để nối tiếp session bị gián đoạn (spec §0.4).
    Trả về wandb Run object hoặc None nếu không có key.
    """
    key = wandb_key or os.environ.get("WANDB_API_KEY")
    if not key:
        return None
    try:
        import wandb
        wandb.login(key=key, relogin=False)
        run = wandb.init(
            project="hub_kien_thuc_v8",
            id=id_pro,
            resume="allow" if id_pro else None,
        )
        return run
    except ImportError:
        print("[cli_pipeline] Cảnh báo: chưa cài `wandb`, bỏ qua đồng bộ W&B.", file=sys.stderr)
        return None


def wandb_log(run, payload: Dict[str, Any]) -> None:
    if run is not None:
        run.log(payload)


# =========================================================================== #
# Tiện ích I/O nhỏ dùng chung
# =========================================================================== #
def _discover_videos(input_dir: str) -> List[str]:
    patterns = ["*.mp4", "*.mkv", "*.mov", "*.avi"]
    videos: List[str] = []
    for pattern in patterns:
        videos.extend(sorted(glob.glob(str(Path(input_dir) / pattern))))
    return videos


def _extract_audio_track(video_path: str, out_wav_path: str) -> str:
    """Tách audio track thô từ video bằng ffmpeg, chuẩn hoá 48kHz mono PCM16."""
    Path(out_wav_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1",
        out_wav_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    return out_wav_path


def _resolve_meta_path_for_video(video_path: str, cli_meta_path: Optional[str]) -> str:
    """
    Ưu tiên --meta_path do người dùng truyền tay (dùng chung cho cả batch);
    nếu không có, tự động tìm file `<ten_video>.source_meta.json` hoặc
    `source_meta.json` nằm CẠNH video trong data/raw_videos/.
    """
    if cli_meta_path:
        return cli_meta_path

    video_p = Path(video_path)
    sibling_named = video_p.with_suffix(".source_meta.json")
    if sibling_named.is_file():
        return str(sibling_named)

    sibling_generic = video_p.parent / "source_meta.json"
    if sibling_generic.is_file():
        return str(sibling_generic)

    return str(sibling_generic)  # trả về path không tồn tại -> load_source_meta() tự fallback domain='unknown'


def _load_settings_yaml(settings_yaml_path: str) -> Dict[str, Any]:
    import yaml
    path_obj = Path(settings_yaml_path)
    if not path_obj.is_file():
        return {"domains": {}}
    with open(path_obj, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"domains": {}}


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _export_csv(rows: List[Dict[str, Any]], prefix: str = "export") -> str:
    Path(DEFAULT_EVAL_LOGS_DIR).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = str(Path(DEFAULT_EVAL_LOGS_DIR) / f"{prefix}_{ts}.csv")
    if not rows:
        Path(out_path).touch()
        return out_path
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


# =========================================================================== #
# PIPELINE — điều phối toàn bộ Phase 0-4
# =========================================================================== #
class Pipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.tracker = DualTracker(db_path=str(Path(DEFAULT_EVAL_LOGS_DIR) / "tracker_state.sqlite3"))
        self.wandb_run = setup_wandb(args.wandb_key, args.id_pro)

    # ------------------------------------------------------------------ #
    # --health_check / --dummy_test
    # ------------------------------------------------------------------ #
    def do_health_check(self) -> Dict[str, Any]:
        report = system_check.run_health_check(skip_dep_check=self.args.skip_dep_check)
        wandb_log(self.wandb_run, {"health_check": report["status"]})
        return report

    def do_dummy_test(self) -> Dict[str, Any]:
        """
        Stress test thực chiến (spec §0.4): chạy toàn bộ Phase 0-2 trên
        `stress_test.mp4` (10s, cắt cảnh liên tục, giọng chồng lấn, text
        dày đặc) để ép VRAM chạm đỉnh thực sự, phát hiện OOM sớm.
        """
        if not Path(STRESS_TEST_VIDEO).is_file():
            return {"status": "error", "reason": f"Không tìm thấy {STRESS_TEST_VIDEO}. "
                                                    f"Hãy đặt file stress test 10s vào đúng path này."}

        import torch
        peak_before = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        vid_hash_map = self.run_phase0([STRESS_TEST_VIDEO])
        chunks_map = self.run_phase1(vid_hash_map)
        self.run_phase2(chunks_map)
        elapsed = round(time.time() - t0, 2)

        peak_after = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

        result = {
            "status": "ok", "elapsed_sec": elapsed,
            "peak_vram_gb": round(peak_after, 3),
            "vram_within_budget": peak_after < 10.0,  # peak thực tế phải giữ < 10GB (§0)
        }
        wandb_log(self.wandb_run, {"dummy_test_peak_vram_gb": result["peak_vram_gb"]})
        return result

    # ------------------------------------------------------------------ #
    # --resume_phase
    # ------------------------------------------------------------------ #
    def do_resume_phase(self, phase: int) -> Dict[str, Any]:
        rolled_back = self.tracker.resume_phase(
            phase=phase,
            qdrant_path=self.args.qdrant_path,
            collection_name=self.args.domain,
        )
        result = {"phase": phase, "rolled_back_vid_hashes": rolled_back, "count": len(rolled_back)}
        wandb_log(self.wandb_run, {"resume_phase": phase, "rolled_back_count": len(rolled_back)})
        return result

    # ------------------------------------------------------------------ #
    # PHASE 0 — Hashing + Tracker Init
    # ------------------------------------------------------------------ #
    def run_phase0(self, videos: Optional[Sequence[str]] = None) -> Dict[str, str]:
        videos = list(videos) if videos is not None else _discover_videos(self.args.input_dir)
        vid_hash_map: Dict[str, str] = {}

        for video_path in videos:
            temp_ws = str(Path(DEFAULT_TEMP_WORKSPACE) / Path(video_path).stem)
            try:
                vid_hash = compute_vid_hash(video_path)
            except Exception as exc:  # noqa: BLE001 — log lỗi hash & bỏ qua video hỏng, không sập batch
                print(f"[phase0] Lỗi hash {video_path}: {exc}", file=sys.stderr)
                continue

            if self.tracker.is_committed(vid_hash, phase=0):
                vid_hash_map[video_path] = vid_hash
                continue

            self.tracker.mark_processing(vid_hash, phase=0, temp_workspace_path=temp_ws, source_video_path=video_path)
            Path(temp_ws).mkdir(parents=True, exist_ok=True)
            self.tracker.mark_committed(vid_hash, phase=0)
            vid_hash_map[video_path] = vid_hash

        wandb_log(self.wandb_run, {"phase0_videos_processed": len(vid_hash_map)})
        return vid_hash_map

    # ------------------------------------------------------------------ #
    # PHASE 1 — Extract & Hybrid Chunking
    # ------------------------------------------------------------------ #
    def run_phase1(self, vid_hash_map: Dict[str, str]) -> Dict[str, str]:
        from phase1_extract.video_processor import VideoProcessor
        from phase1_extract.text_processor import TextProcessor
        from phase1_extract.hybrid_chunking import HybridChunker, KeyframeRef

        chunks_path_map: Dict[str, str] = {}

        for video_path, vid_hash in vid_hash_map.items():
            if self.tracker.is_committed(vid_hash, phase=1):
                temp_ws = Path(DEFAULT_TEMP_WORKSPACE) / Path(video_path).stem
                existing = temp_ws / "chunks.json"
                if existing.is_file():
                    chunks_path_map[vid_hash] = str(existing)
                continue

            temp_ws = Path(DEFAULT_TEMP_WORKSPACE) / Path(video_path).stem
            self.tracker.mark_processing(vid_hash, phase=1, temp_workspace_path=str(temp_ws), source_video_path=video_path)

            try:
                raw_audio_wav = str(temp_ws / "audio_raw.wav")
                _extract_audio_track(video_path, raw_audio_wav)

                vp = VideoProcessor()
                keyframes = vp.extract_keyframes(
                    video_path, str(temp_ws / "frames"), yolo_weights=self.args.yolo_weights
                )
                scenes = vp._detect_scenes(video_path)
                audio_paths = vp.denoise_and_split_audio(raw_audio_wav, str(temp_ws / "audio"))

                tp = TextProcessor()
                whisper_result = tp.transcribe(
                    audio_paths["clean_speech_path"], model_size=self.args.whisper_model_size
                )
                clap_result = vp.tag_background_audio(audio_paths["background_noise_path"])

                kf_refs = [
                    KeyframeRef(scene_index=kf.scene_index, keyframe_path=kf.keyframe_path, timestamp_sec=kf.timestamp_sec)
                    for kf in keyframes
                ]
                chunker = HybridChunker()
                chunks = chunker.chunk(whisper_result, scenes, kf_refs)

                chunks_json = [c.__dict__ for c in chunks]
                chunks_path = temp_ws / "chunks.json"
                with open(chunks_path, "w", encoding="utf-8") as f:
                    json.dump(chunks_json, f, ensure_ascii=False, indent=2)

                # Lưu kèm audio-tag & scene list để phase2/debug dùng lại nếu cần
                with open(temp_ws / "audio_tags.json", "w", encoding="utf-8") as f:
                    json.dump(clap_result, f, ensure_ascii=False, indent=2)
                with open(temp_ws / "scenes.json", "w", encoding="utf-8") as f:
                    json.dump(scenes, f, ensure_ascii=False, indent=2)

                self.tracker.mark_committed(vid_hash, phase=1)
                chunks_path_map[vid_hash] = str(chunks_path)

            except Exception as exc:  # noqa: BLE001
                self.tracker.mark_failed(vid_hash, phase=1, error_message=str(exc))
                print(f"[phase1] Lỗi xử lý {video_path} (VID_Hash={vid_hash}): {exc}", file=sys.stderr)
                continue

        wandb_log(self.wandb_run, {"phase1_videos_chunked": len(chunks_path_map)})
        return chunks_path_map

    # ------------------------------------------------------------------ #
    # PHASE 2 — Embedding + Qdrant + Kùzu
    # ------------------------------------------------------------------ #
    def run_phase2(self, chunks_path_map: Dict[str, str]) -> None:
        from phase2_build.embedding import EmbeddingEngine
        from phase2_build.qdrant_manager import QdrantManager
        import numpy as np

        # video_path gốc không còn trong chunks_path_map (key là vid_hash) — tra ngược qua tracker
        for vid_hash, chunks_path in chunks_path_map.items():
            if self.tracker.is_committed(vid_hash, phase=2):
                continue

            record = self.tracker.get_record(vid_hash, phase=1)
            video_path = record.source_video_path if record else ""
            temp_ws = Path(chunks_path).parent

            self.tracker.mark_processing(vid_hash, phase=2, temp_workspace_path=str(temp_ws), source_video_path=video_path)

            try:
                with open(chunks_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)

                meta_path = _resolve_meta_path_for_video(video_path, self.args.meta_path)
                qdrant_mgr = QdrantManager(self.args.qdrant_path, collection_name="_bootstrap_")
                source_meta = qdrant_mgr.load_source_meta(meta_path)
                domain = source_meta.get("domain", "unknown")
                qdrant_mgr.collection_name = domain  # mỗi domain = 1 collection riêng (§2.1)

                engine = EmbeddingEngine()
                image_paths = [c["keyframe_path"] for c in chunks if c.get("keyframe_path")]
                texts = [c.get("text", "") for c in chunks]

                embed_result = engine.encode_batch(
                    image_paths, texts,
                    siglip_model_name=self.args.siglip_model,
                    bge_model_name=self.args.bge_model,
                )

                # Căn lại ma trận ảnh về đúng vị trí trong `chunks` (không phải mọi
                # chunk đều có keyframe -> chèn vector 0 cho chunk không ảnh).
                img_dim = embed_result.dense_image_vectors.shape[1] if embed_result.dense_image_vectors.size else 768
                image_vectors_full = np.zeros((len(chunks), img_dim), dtype=np.float32)
                cursor = 0
                for i, c in enumerate(chunks):
                    if c.get("keyframe_path"):
                        image_vectors_full[i] = embed_result.dense_image_vectors[cursor]
                        cursor += 1

                point_ids = qdrant_mgr.upsert_pending(
                    chunks,
                    embed_result.dense_text_vectors,
                    image_vectors_full,
                    embed_result.sparse_text_vectors,
                    source_meta,
                    vid_hash,
                )
                n_committed = qdrant_mgr.confirm_committed(vid_hash)
                qdrant_mgr.close()

                if self.args.build_graph:
                    from phase2_build.kuzu_builder import KuzuGraphBuilder
                    kuzu_builder = KuzuGraphBuilder(self.args.kuzu_graph_path)
                    kuzu_builder.build_graph_for_video(chunks, vid_hash, domain)
                    kuzu_builder.close()

                self.tracker.mark_committed(vid_hash, phase=2)
                wandb_log(self.wandb_run, {"phase2_points_committed": n_committed, "vid_hash": vid_hash})

            except Exception as exc:  # noqa: BLE001
                self.tracker.mark_failed(vid_hash, phase=2, error_message=str(exc))
                print(f"[phase2] Lỗi embedding/upsert VID_Hash={vid_hash}: {exc}", file=sys.stderr)
                continue

    # ------------------------------------------------------------------ #
    # PHASE 3 — Search & Generate
    # ------------------------------------------------------------------ #
    def run_phase3(self) -> Dict[str, Any]:
        args = self.args
        if not args.query:
            return {"status": "error", "reason": "--query là bắt buộc cho --run_phase 3"}

        domain = self._resolve_domain()
        if not domain:
            return {"status": "error", "reason": "Không xác định được --domain (truyền tay hoặc qua --meta_path)."}

        from phase3_search.retriever import Retriever
        from phase2_build.embedding import EmbeddingEngine

        from src.common.config_loader import config
        domain_cfg = config.get("domains", {}).get(domain, {})
        alpha = domain_cfg.get("alpha", DEFAULT_ALPHA)
        beta = domain_cfg.get("beta", DEFAULT_BETA)

        retriever = Retriever(args.qdrant_path, domain)
        engine = EmbeddingEngine()

        embed_res = engine.encode_texts([args.query], return_sparse=True)
        text_vec = embed_res.dense_text_vectors[0].tolist()
        sparse_vec = embed_res.sparse_text_vectors[0] if embed_res.sparse_text_vectors else None

        candidates = retriever.tri_search(
            text_dense_query=text_vec, image_dense_query=None, sparse_query=sparse_vec,
            alpha=alpha, beta=beta, top_k=args.top_k, filter_meta=args.filter_meta,
        )

        candidate_dicts = [
            {"point_id": c.point_id, "payload": c.payload, "text": c.payload.get("text", ""), "s_total": c.s_total}
            for c in candidates
        ]

        # --- Graph Hop (--use_graph) — DAG bước 2 (§3.4) ---
        if args.use_graph:
            candidate_dicts.extend(self._graph_expand(candidate_dicts, args))

        # --- Rerank (-rr) — Cross-Encoder chấm lại Top-K ---
        if args.rerank:
            from phase3_search.reranker import Reranker
            reranker = Reranker()
            reranked = reranker.rerank(args.query, candidate_dicts, top_k=args.top_k)
            final_candidates = [
                {"point_id": r.point_id, "payload": r.payload, "s_total": r.prior_score, "rerank_score": r.rerank_score}
                for r in reranked
            ]
        else:
            final_candidates = sorted(candidate_dicts, key=lambda c: c.get("s_total", 0.0), reverse=True)[: args.top_k]

        retriever.close()

        # --- -s / --search_only ---
        if args.search_only:
            output: Dict[str, Any] = {
                "status": "ok",
                "results": [
                    {"chunk_id": c["payload"].get("chunk_id"), "text": c["payload"].get("text"),
                     "keyframe_path": c["payload"].get("keyframe_path"), "score": c.get("rerank_score", c.get("s_total"))}
                    for c in final_candidates
                ],
            }
            self._maybe_export_csv(output["results"], args)
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return output

        # --- -qa / --question_answer: lấy rank 1 làm câu hỏi ---
        effective_query = args.query
        if args.question_answer and final_candidates:
            effective_query = final_candidates[0]["payload"].get("text", args.query)

        from phase3_search.vlm_generator import VLMGenerator
        generator = VLMGenerator()
        gen_result = generator.generate(
            effective_query, final_candidates, max_context_images=args.max_context_images,
        )

        output = {"status": "ok", "answer": gen_result["answer"]}
        if args.trake_mode:
            output["trake"] = gen_result["trake"]

        # --- Log nhịp 1 cho Phase 4 (§4.0) ---
        context_joined = "\n".join(c["payload"].get("text", "") for c in final_candidates)
        _append_jsonl(str(Path(DEFAULT_EVAL_LOGS_DIR) / "eval_queue.jsonl"), {
            "query": effective_query, "context": context_joined, "answer": gen_result["answer"],
        })

        self._maybe_export_csv([{"query": effective_query, **output}], args)

        if args.chat_session:
            self._run_chat_loop(generator, retriever_domain=domain, alpha=alpha, beta=beta, args=args)

        generator.unload()
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return output

    def _resolve_domain(self) -> Optional[str]:
        args = self.args
        if args.domain:
            return args.domain
        if args.meta_path and Path(args.meta_path).is_file():
            try:
                with open(args.meta_path, "r", encoding="utf-8") as f:
                    return json.load(f).get("domain")
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _graph_expand(self, candidate_dicts: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
        from phase3_search.graph_router import GraphRouter

        seed_ids = [c["payload"].get("chunk_id") for c in candidate_dicts if c["payload"].get("chunk_id")]
        if not seed_ids:
            return []

        router = GraphRouter(args.kuzu_graph_path)
        neighbor_ids = router.expand_context(seed_ids, args.hop_limit)
        neighbor_meta = router.get_node_metadata(neighbor_ids)
        router.close()

        # Candidate mở rộng qua Graph Hop không có RRF riêng — gán s_total
        # thấp (0.0) để không lấn điểm candidate matched trực tiếp, nhưng
        # vẫn tham gia Cut-off Knapsack như ngữ cảnh bổ trợ (DAG §3.4).
        expanded = []
        for chunk_id, meta in neighbor_meta.items():
            expanded.append({
                "point_id": chunk_id,
                "payload": {
                    "chunk_id": chunk_id, "text": meta.get("text", ""),
                    "keyframe_path": meta.get("keyframe_path"),
                    "parent_chunk_id": meta.get("parent_chunk_id"),
                    "scene_index": meta.get("scene_index"),
                },
                "text": meta.get("text", ""),
                "s_total": 0.0,
            })
        return expanded

    def _run_chat_loop(self, generator, retriever_domain: str, alpha: float, beta: float, args: argparse.Namespace) -> None:
        """
        -c / --chat_session: giữ Qwen 2.5 VL ghim GPU xuyên suốt session,
        chỉ retriever/embedding được mở lại mỗi lượt hỏi (chi phí thấp,
        không nạp lại model nặng).
        """
        from phase3_search.retriever import Retriever
        from phase2_build.embedding import EmbeddingEngine

        print("\n[chat_session] Nhập câu hỏi tiếp theo (hoặc 'exit' để thoát):")
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break
            if user_input.lower() in {"exit", "quit"}:
                break
            if not user_input:
                continue

            retriever = Retriever(args.qdrant_path, retriever_domain)
            engine = EmbeddingEngine()
            embed_res = engine.encode_texts([user_input], return_sparse=True)
            text_vec = embed_res.dense_text_vectors[0].tolist()
            sparse_vec = embed_res.sparse_text_vectors[0] if embed_res.sparse_text_vectors else None

            candidates = retriever.tri_search(
                text_dense_query=text_vec, image_dense_query=None, sparse_query=sparse_vec,
                alpha=alpha, beta=beta, top_k=args.top_k, filter_meta=args.filter_meta,
            )
            candidate_dicts = [
                {"point_id": c.point_id, "payload": c.payload, "s_total": c.s_total} for c in candidates[: args.top_k]
            ]
            retriever.close()

            gen_result = generator.generate(user_input, candidate_dicts, max_context_images=args.max_context_images)
            print(f"Claude(Qwen2.5-VL): {gen_result['answer']}\n")

            _append_jsonl(str(Path(DEFAULT_EVAL_LOGS_DIR) / "eval_queue.jsonl"), {
                "query": user_input,
                "context": "\n".join(c["payload"].get("text", "") for c in candidate_dicts),
                "answer": gen_result["answer"],
            })

    def _maybe_export_csv(self, rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
        if args.export_csv:
            out_path = _export_csv(rows, prefix="search_export")
            print(f"[cli_pipeline] Đã xuất CSV: {out_path}", file=sys.stderr)

    # ------------------------------------------------------------------ #
    # PHASE 4 — Eval Cascade 2 Tầng
    # ------------------------------------------------------------------ #
    def run_phase4(self) -> Dict[str, Any]:
        args = self.args
        from phase4_eval import metric_scorer as metric_scorer_module
        from phase4_eval.metric_scorer import MetricScorer

        # Cờ --deep_eval_threshold_low/high override ngưỡng module-level
        # (glue-level override, không sửa file gốc phase4_eval/metric_scorer.py)
        metric_scorer_module.DEFAULT_THRESHOLD_LOW = args.deep_eval_threshold_low
        metric_scorer_module.DEFAULT_THRESHOLD_HIGH = args.deep_eval_threshold_high

        eval_queue_path = Path(DEFAULT_EVAL_LOGS_DIR) / "eval_queue.jsonl"
        eval_report_path = str(Path(DEFAULT_EVAL_LOGS_DIR) / "eval_report.jsonl")
        eval_queue_deep_path = str(Path(DEFAULT_EVAL_LOGS_DIR) / "eval_queue_deep.jsonl")

        if not eval_queue_path.is_file():
            return {"status": "error", "reason": f"Không tìm thấy {eval_queue_path} — chưa có truy vấn nào được log ở Phase 3."}

        with open(eval_queue_path, "r", encoding="utf-8") as f:
            all_records = [json.loads(line) for line in f if line.strip()]

        # --- --eval_sample_rate: chỉ lấy % ngẫu nhiên, giữ tính đại diện thống kê ---
        sample_size = max(1, int(len(all_records) * args.eval_sample_rate))
        sampled = random.sample(all_records, min(sample_size, len(all_records)))

        scorer = MetricScorer()
        tier1_stats = scorer.run_cascade_tier1(sampled, eval_report_path, eval_queue_deep_path)
        result: Dict[str, Any] = {"tier1_stats": tier1_stats, "sampled": len(sampled), "total_logged": len(all_records)}

        if args.prometheus_gguf_path:
            tier2_stats = scorer.deep_judge_tier2(
                eval_queue_deep_path, eval_report_path, args.prometheus_gguf_path,
            )
            result["tier2_stats"] = tier2_stats

        if args.ground_truth_path:
            generated_results = [
                {"query": r["query"], "answer": r["answer"], "trake": r.get("trake", [])} for r in sampled
            ]
            result["ground_truth_metrics"] = scorer.compare_with_ground_truth(generated_results, args.ground_truth_path)

        # Sau khi đã tiêu thụ, xoá eval_queue.jsonl để tránh chấm trùng ở batch sau
        eval_queue_path.unlink(missing_ok=True)

        wandb_log(self.wandb_run, {"eval_tier1": tier1_stats})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    # ------------------------------------------------------------------ #
    # --tune_weights
    # ------------------------------------------------------------------ #
    def run_tune_weights(self) -> Dict[str, Any]:
        args = self.args
        if not args.target_kf:
            return {"status": "error", "reason": "--target_kf là bắt buộc khi dùng --tune_weights"}

        domain = self._resolve_domain() or args.save_domain
        if not domain:
            return {"status": "error", "reason": "Cần --domain hoặc --save_domain để biết collection nào cần tune."}

        from phase3_search.retriever import Retriever
        from phase2_build.embedding import EmbeddingEngine
        from phase4_eval.hyper_tuner import HyperTuner

        retriever = Retriever(args.qdrant_path, domain)
        engine = EmbeddingEngine()

        def _text_encoder(q: str) -> List[float]:
            res = engine.encode_texts([q], return_sparse=False)
            return res.dense_text_vectors[0].tolist()

        tuner = HyperTuner(args.settings_yaml_path)
        target_pairs = tuner.load_target_kf(args.target_kf)
        components = tuner.precompute_query_components(
            retriever, target_pairs, text_dense_encoder=_text_encoder,
            top_k=args.top_k, filter_meta=args.filter_meta,
        )
        retriever.close()

        search_result = tuner.grid_search(components)

        if args.save_domain:
            tuner.save_domain_weights(args.save_domain, search_result["best_alpha"], search_result["best_beta"])

        output = {
            "best_alpha": search_result["best_alpha"], "best_beta": search_result["best_beta"],
            "best_gamma": search_result["best_gamma"], "best_mean_mrr": search_result["best_mean_mrr"],
            "saved_to_domain": args.save_domain,
        }
        wandb_log(self.wandb_run, output)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return output


# =========================================================================== #
# ARGPARSE — Ma Trận Cờ Lệnh Hệ Thống (spec V8.0, mục cuối tài liệu)
# =========================================================================== #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli_pipeline.py",
        description="RAG V8.0  — Entry-point điều phối Phase 0-4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Vận hành & Điều phối Phase ---
    op = parser.add_argument_group("Vận hành & Điều phối Phase")
    op.add_argument("--run_phase", default=None, help="0 | 1 | 2 | 3 | 4 | all")
    op.add_argument("--resume_phase", type=int, default=None, help="Rollback + resume phase bị gián đoạn")
    op.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)
    op.add_argument("--batch_size", type=int, default=4)
    op.add_argument("--health_check", action="store_true")
    op.add_argument("--dummy_test", action="store_true")
    op.add_argument("--wandb_key", default=None)
    op.add_argument("--id_pro", default=None, help="W&B run id để resume session")
    op.add_argument("--build_graph", action="store_true")
    op.add_argument("--kuzu_graph_path", default=DEFAULT_KUZU_PATH)
    op.add_argument("--qdrant_path", default=DEFAULT_QDRANT_PATH)
    op.add_argument("--db_path", dest="qdrant_path", help=argparse.SUPPRESS)  # alias theo ví dụ trong spec
    op.add_argument("--meta_path", default=None)
    op.add_argument("--meta_mapping", dest="meta_path", help=argparse.SUPPRESS)  # alias
    op.add_argument("--domain", default=None,
                     help="Tên domain -> chọn Qdrant Collection (mỗi domain = 1 collection, §2.1). "
                          "Nếu bỏ trống ở Phase 3, tự suy ra từ --meta_path.")
    op.add_argument("--skip_dep_check", action="store_true", help="Bỏ qua pip-check hard-exit (chỉ dùng khi debug)")
    
    # --- Tham số model (override mặc định) ---
    mp = parser.add_argument_group("Model overrides")
    mp.add_argument("--yolo_weights", default="yolov8n.pt")
    mp.add_argument("--whisper_model_size", default="large-v3")
    mp.add_argument("--siglip_model", default="google/siglip2-base-patch16-256")
    mp.add_argument("--bge_model", default="BAAI/bge-m3")
    mp.add_argument("--settings_yaml_path", default=DEFAULT_SETTINGS_YAML)
    op.add_argument("--hf_token", default=None, help="Truyền trực tiếp HF Token (dùng cho Kaggle)")
    # --- Tìm kiếm & Nhảy cóc (GraphRAG) ---
    sg = parser.add_argument_group("Tìm kiếm & Nhảy cóc (GraphRAG)")
    sg.add_argument("--query", default=None)
    sg.add_argument("-s", "--search_only", action="store_true", help="Tắt VLM, chỉ lấy bối cảnh thô")
    sg.add_argument("-qa", "--question_answer", action="store_true", help="Dùng sau -s, lấy rank 1 làm base để trả lời ")
    sg.add_argument("-t", "--top_k", type=int, default=20)
    sg.add_argument("-f", "--filter_meta", default=None, help='JSON hoặc "key=value,key2=value2"')
    sg.add_argument("--use_graph", action="store_true")
    sg.add_argument("--hop_limit", type=int, default=2, help="Clamp nội bộ [1,5]")
    sg.add_argument("-rr", "--rerank", action="store_true", help="Cross-Encoder CPU chấm lại Top K")
    sg.add_argument("--max_context_images", type=int, default=4, help="Capacity cho Knapsack DP")

    # --- Tối ưu Trọng số Động ---
    tw = parser.add_argument_group("Tối ưu Trọng số Động")
    tw.add_argument("--tune_weights", action="store_true")
    tw.add_argument("--target_kf", default=None, help="JSON string hoặc path, >=5 cặp [query, keyframe]")
    tw.add_argument("--save_domain", default=None, help="Lưu alpha/beta vào config/settings.yaml")

    # --- Đánh giá (Phase 4) ---
    ev = parser.add_argument_group("Đánh giá (Phase 4)")
    ev.add_argument("--eval_sample_rate", type=float, default=0.1)
    ev.add_argument("--deep_eval_threshold_low", type=float, default=0.4)
    ev.add_argument("--deep_eval_threshold_high", type=float, default=0.7)
    ev.add_argument("--prometheus_gguf_path", default=None, help="Kích hoạt Tầng 2 Deep Judge")
    ev.add_argument("--ground_truth_path", default=None, help="JSON Ground Truth để đối chiếu")

    # --- Giao thức Dữ liệu Đầu Ra ---
    out = parser.add_argument_group("Giao thức Dữ liệu Đầu Ra")
    out.add_argument("-tr", "--trake_mode", action="store_true", help="Trả kèm mảng truy vết chunk_id/timestamp/keyframe_path")
    out.add_argument("-outcsv", "--export_csv", action="store_true", help="Xuất kết quả ra CSV trong data/eval_logs/")
    out.add_argument("-c", "--chat_session", action="store_true", help="Giữ Qwen ghim GPU, hỏi liên tục qua REPL")

    return parser


def main() -> None:
    load_env_secrets()
    parser = build_arg_parser()
    args = parser.parse_args()

    pipeline = Pipeline(args)
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    # --health_check / --dummy_test có thể chạy độc lập, không cần --run_phase
    if args.health_check:
        print(json.dumps(pipeline.do_health_check(), ensure_ascii=False, indent=2))
        return

    if args.dummy_test:
        print(json.dumps(pipeline.do_dummy_test(), ensure_ascii=False, indent=2))
        return

    # Dependency Hard-Exit (§0.1) — chạy TRƯỚC mọi phase thực thi thật sự
    if not args.skip_dep_check:
        system_check.check_pip_conflicts(hard_exit=True)

    if args.resume_phase is not None:
        print(json.dumps(pipeline.do_resume_phase(args.resume_phase), ensure_ascii=False, indent=2))
        # Sau rollback, tiếp tục chạy lại phase đó nếu --run_phase cũng được truyền
        if not args.run_phase:
            return

    if args.tune_weights:
        pipeline.run_tune_weights()
        return

    phase_arg = (args.run_phase or "").lower().strip()
    if not phase_arg:
        if args.query:
            phase_arg = "3"  # tiện dụng: chỉ truyền --query mà quên --run_phase vẫn chạy được Phase 3
        else:
            parser.print_help()
            return

    run_all = phase_arg == "all"

    if run_all or phase_arg == "0":
        vid_hash_map = pipeline.run_phase0()
    else:
        vid_hash_map = pipeline.run_phase0() if phase_arg in {"1", "2"} else {}

    if run_all or phase_arg == "1":
        chunks_map = pipeline.run_phase1(vid_hash_map)
    else:
        chunks_map = {}

    if run_all or phase_arg == "2":
        if not chunks_map:
            # Phase 2 độc lập: dò lại chunks.json đã có sẵn từ Phase 1 trước đó
            vid_hash_map = vid_hash_map or pipeline.run_phase0()
            chunks_map = {
                vh: str(Path(DEFAULT_TEMP_WORKSPACE) / Path(vp).stem / "chunks.json")
                for vp, vh in vid_hash_map.items()
                if (Path(DEFAULT_TEMP_WORKSPACE) / Path(vp).stem / "chunks.json").is_file()
            }
        pipeline.run_phase2(chunks_map)

    if run_all or phase_arg == "3":
        pipeline.run_phase3()

    if run_all or phase_arg == "4":
        pipeline.run_phase4()

    if pipeline.wandb_run is not None:
        pipeline.wandb_run.finish()


if __name__ == "__main__":
    main()
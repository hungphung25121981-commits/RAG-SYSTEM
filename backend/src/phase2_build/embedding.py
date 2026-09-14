# src/phase2_build/embedding.py
"""
PHASE 2 — Nhúng vector: SigLIP 2 (ảnh) + BGE-M3 (văn bản)

Strict Sequential Batching (spec V8.0 §2.2) — BẮT BUỘC:
    Load SigLIP 2 -> Encode TOÀN BỘ Image Batch -> Unload + empty_cache()
            |  (hoàn tất 100% trước khi bắt đầu bước sau)
            v
    Load BGE-M3   -> Encode TOÀN BỘ Text Batch  -> Unload + empty_cache()

    - Không nạp xen kẽ 2 model cùng lúc trên VRAM.
    - empty_cache() CHỈ được gọi ở biên giới chuyển giao model (sau khi
      unload), TUYỆT ĐỐI không gọi trong inner-loop của vòng lặp encode
      từng batch nhỏ — vì gọi liên tục sẽ gây memory fragmentation, phản
      tác dụng so với mục tiêu tiết kiệm VRAM.
    - Sparse vector (BM25) được tính riêng bằng rank_bm25 (CPU-only,
      không tốn VRAM) nên không nằm trong ràng buộc Tiered Caching này.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from backend.src.common.config_loader import config
_VECTOR_CFG = config.get("phase2_vector", {})

DEFAULT_IMAGE_BATCH_SIZE = int(_VECTOR_CFG.get("siglip_batch_size", 16))
DEFAULT_TEXT_BATCH_SIZE = int(_VECTOR_CFG.get("embedding_batch_size", 32))
DEFAULT_MAX_LENGTH = int(_VECTOR_CFG.get("embedding_max_length", 512)) # Cứu tinh VRAM ở đây
DEFAULT_SIGLIP_MODEL = _VECTOR_CFG.get("siglip_model_id", "google/siglip2-base-patch16-256")
DEFAULT_BGE_MODEL = _VECTOR_CFG.get("embedding_model_id", "BAAI/bge-m3")

@dataclass
class EmbeddingResult:
    dense_image_vectors: Optional[np.ndarray]  # shape [N_img, dim_siglip]
    dense_text_vectors: Optional[np.ndarray]   # shape [N_txt, dim_bgem3]
    sparse_text_vectors: Optional[List[Dict[int, float]]]  # BM25-style sparse per text


class EmbeddingEngine:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self._siglip_model = None
        self._siglip_processor = None
        self._bge_model = None

    # ------------------------------------------------------------------ #
    # SigLIP 2 — Image Embedding
    # ------------------------------------------------------------------ #
    def _load_siglip(self, model_name: str = "google/siglip2-base-patch16-256") -> None:
        from transformers import AutoModel, AutoProcessor
        self._siglip_model = AutoModel.from_pretrained(model_name).to(self.device)
        self._siglip_model.eval()
        self._siglip_processor = AutoProcessor.from_pretrained(model_name)

    def _unload_siglip(self) -> None:
        if self._siglip_model is not None:
            del self._siglip_model
            self._siglip_model = None
        self._siglip_processor = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()  # empty_cache() ở BIÊN GIỚI chuyển giao model, đúng spec §2.2

    def encode_images(
        self,
        image_paths: Sequence[str],
        model_name: str =DEFAULT_SIGLIP_MODEL,
        batch_size: int = DEFAULT_IMAGE_BATCH_SIZE,
    ) -> np.ndarray:
        """Nhúng toàn bộ batch ảnh bằng SigLIP 2, trả về ma trận vector đã L2-normalize."""
        if not image_paths:
            return np.zeros((0, 0), dtype=np.float32)

        self._load_siglip(model_name)
        all_vectors: List[np.ndarray] = []
        try:
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                images = [Image.open(p).convert("RGB") for p in batch_paths]
                inputs = self._siglip_processor(images=images, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    image_features = self._siglip_model.get_image_features(**inputs)
                    image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)

                all_vectors.append(image_features.detach().cpu().numpy().astype(np.float32))
                # KHÔNG gọi empty_cache() ở đây (inner-loop) — chỉ gọi sau khi unload toàn bộ model.
        finally:
            self._unload_siglip()

        return np.concatenate(all_vectors, axis=0) if all_vectors else np.zeros((0, 0), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # BGE-M3 — Text Embedding (Dense)
    # ------------------------------------------------------------------ #
    def _load_bge(self, model_name: str = "BAAI/bge-m3") -> None:
        from FlagEmbedding import BGEM3FlagModel
        use_fp16 = self.device == "cuda"
        self._bge_model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=self.device)

    def _unload_bge(self) -> None:
        if self._bge_model is not None:
            del self._bge_model
            self._bge_model = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()  # biên giới chuyển giao model

    def encode_texts(
        self,
        texts: Sequence[str],
        model_name: str = "BAAI/bge-m3",
        batch_size: int = DEFAULT_TEXT_BATCH_SIZE,
        return_sparse: bool = True,
    ) -> EmbeddingResult:
        """
        Nhúng toàn bộ batch text bằng BGE-M3 (dense) + sinh sparse vector
        (lexical weights) để phục vụ Tri-Search RRF (§2.1, §3.2).
        """
        if not texts:
            return EmbeddingResult(dense_image_vectors=None, dense_text_vectors=np.zeros((0, 0)), sparse_text_vectors=[])

        self._load_bge(model_name)
        try:
            output = self._bge_model.encode(
                list(texts),
                batch_size=batch_size,
                max_length=DEFAULT_MAX_LENGTH,
                return_dense=True,
                return_sparse=return_sparse,
                return_colbert_vecs=False,
            )
            dense_vectors = np.asarray(output["dense_vecs"], dtype=np.float32)

            sparse_vectors: Optional[List[Dict[int, float]]] = None
            if return_sparse and "lexical_weights" in output:
                sparse_vectors = []
                for lw in output["lexical_weights"]:
                    # lexical_weights: dict[token_id(str) -> weight(float)] -> chuẩn hoá key thành int
                    sparse_vectors.append({int(k): float(v) for k, v in lw.items()})
        finally:
            self._unload_bge()

        return EmbeddingResult(
            dense_image_vectors=None,
            dense_text_vectors=dense_vectors,
            sparse_text_vectors=sparse_vectors,
        )

    # ------------------------------------------------------------------ #
    # Orchestration — tuân thủ nghiêm ngặt thứ tự Sequential Batching
    # ------------------------------------------------------------------ #
    def encode_batch(
        self,
        image_paths: Sequence[str],
        texts: Sequence[str],
        siglip_model_name: str = "google/siglip2-base-patch16-256",
        bge_model_name: str = "BAAI/bge-m3",
        image_batch_size: int = DEFAULT_IMAGE_BATCH_SIZE,
        text_batch_size: int = DEFAULT_TEXT_BATCH_SIZE,
    ) -> EmbeddingResult:
        """
        Entry-point chính của Phase 2 embedding: đảm bảo SigLIP 2 chạy
        XONG HOÀN TOÀN (encode + unload) trước khi BGE-M3 được nạp.
        """
        image_vectors = self.encode_images(image_paths, siglip_model_name, image_batch_size)
        text_result = self.encode_texts(texts, bge_model_name, text_batch_size)

        return EmbeddingResult(
            dense_image_vectors=image_vectors,
            dense_text_vectors=text_result.dense_text_vectors,
            sparse_text_vectors=text_result.sparse_text_vectors,
        )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Phase 2 Embedding Engine (standalone test).")
    parser.add_argument("--image_dir", required=False, default=None)
    parser.add_argument("--texts_json", required=False, default=None, help="JSON list of strings")
    args = parser.parse_args()

    import glob
    image_paths = sorted(glob.glob(f"{args.image_dir}/*.jpg")) if args.image_dir else []
    texts: List[str] = []
    if args.texts_json:
        with open(args.texts_json, "r", encoding="utf-8") as f:
            texts = json.load(f)

    engine = EmbeddingEngine()
    result = engine.encode_batch(image_paths, texts)
    print(json.dumps({
        "image_vectors_shape": list(result.dense_image_vectors.shape) if result.dense_image_vectors is not None else None,
        "text_vectors_shape": list(result.dense_text_vectors.shape) if result.dense_text_vectors is not None else None,
        "sparse_count": len(result.sparse_text_vectors) if result.sparse_text_vectors else 0,
    }, ensure_ascii=False, indent=2))